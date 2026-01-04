
#!/usr/bin/env python3
"""
Minimal HPCEnv for proof-of-concept RL.
- State: first K jobs' leakage-safe features [req_procs, age_in_queue_s, submit_hour, submit_dow] flattened, plus backlog size.
- Action: pick an index in [0..K-1] to start next (if that slot is empty/no job, it's a no-op).
- Transition: remove the chosen job from the queue, advance time by predicted runtime, accumulate predicted energy.
- Reward: negative energy_kwh and queue penalty; small penalty for no-op.
This is intentionally simple so you can confirm wiring end-to-end quickly.
Replace with your simulator integration later.
"""
import math
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import gymnasium as gym
from gymnasium import spaces


def _f32(x): return np.asarray(x, dtype=np.float32)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class HPCEnv(gym.Env):
    """
    Minimal but functional scheduler env:
      • State  = per-job features for top-K + backlog scalar
      • Action = pick which job (0..K-1) to start now
      • Step   = remove chosen job, advance time by predicted runtime
      • Reward = scaled penalty on (kWh + backlog) → O(−1…−10) per step

    Surrogates:
      • runtime model -> predicts runtime_s
      • energy  model -> predicts energy_j
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jobs_csv: str,
        K: int = 8,
        runtime_model_path: str = "models/runtime_surrogate_rf.joblib",
        energy_model_path: str = "models/energy_surrogate_rf.joblib",
        max_steps: int = 1000,
        reward_scale: float = 50.0,  # stronger signal; VecNormalize will stabilize
    ):
        super().__init__()
        self.jobs_csv = jobs_csv
        self.K = int(K)
        self.max_steps = int(max_steps)
        self.reward_scale = float(reward_scale)

        # Load surrogates with memory mapping to avoid per-process duplication
        self.rt_model = joblib.load(runtime_model_path, mmap_mode="r")
        self.en_model = joblib.load(energy_model_path, mmap_mode="r")

        # Avoid nested parallelism warnings/oversubscription
        for m in (self.rt_model, self.en_model):
            try:
                if hasattr(m, "set_params"):
                    m.set_params(n_jobs=1)
            except Exception:
                pass

        self.rt_cols = getattr(self.rt_model, "feature_names_in_", None)
        self.en_cols = getattr(self.en_model, "feature_names_in_", None)

        # Per-job features include predicted energy
        self.per_job_feats = [
            "req_procs", "age_in_queue_s", "submit_hour", "submit_dow",
            "pred_runtime_s", "pred_energy_j",
        ]
        self.obs_dim = self.K * len(self.per_job_feats) + 1  # + backlog scalar
        self.observation_space = spaces.Box(
            low=0.0, high=1e9, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K)

        self.steps = 0
        self.t_now = 0.0
        self.queue: pd.DataFrame = pd.DataFrame()
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)

        self._load_jobs()

    # ---------- core helpers ----------

    def _load_jobs(self):
        df = pd.read_csv(self.jobs_csv)

        # Minimal leakage-safe columns
        for col in ["submit", "req_procs", "req_time", "status"]:
            if col not in df.columns:
                df[col] = 0

        df["submit"] = df["submit"].fillna(0)
        df["req_procs"] = df["req_procs"].fillna(1)

        # Initialize derived fields
        df = df.sort_values("submit").reset_index(drop=True)
        self.queue = df
        self.steps = 0
        self.t_now = 0.0

        self._refresh_predictions()
        self._obs = self._make_obs()

    def _predict_model(self, model, cols, feat: Dict[str, float]) -> float:
        if cols is not None:
            X = pd.DataFrame([{c: _safe_float(feat.get(c, 0.0)) for c in cols}], columns=cols)
        else:
            X = np.array([list(feat.values())], dtype=np.float32)
        return float(model.predict(X)[0])

    def _refresh_predictions(self):
        """Recompute leakage-safe features and surrogate predictions for each job."""
        feats = []
        for _, r in self.queue.iterrows():
            f = {
                "req_procs": _safe_float(r.get("req_procs", 0)),
                "age_in_queue_s": max(0.0, self.t_now - _safe_float(r.get("submit", 0))),
                "submit_hour": ((_safe_float(r.get("submit", 0)) % (24 * 3600)) // 3600),
                "submit_dow":  ((_safe_float(r.get("submit", 0)) // (24 * 3600)) % 7),
            }
            pr = self._predict_model(self.rt_model, self.rt_cols, f)
            ej = self._predict_model(self.en_model, self.en_cols, f)
            f["pred_runtime_s"] = max(1.0, pr)
            f["pred_energy_j"] = max(0.0, ej)
            feats.append(f)

        if len(self.queue) > 0:
            dfp = pd.DataFrame(feats)
            for c in dfp.columns:
                self.queue.loc[:, c] = dfp[c].values

    def _top_k_indices(self) -> List[int]:
        if len(self.queue) == 0:
            return []
        # Oldest first by age_in_queue
        order = np.argsort(-self.queue["age_in_queue_s"].to_numpy())
        return order[: self.K].tolist()

    def _make_obs(self) -> np.ndarray:
        vec: List[float] = []
        idxs = self._top_k_indices()
        for j in range(self.K):
            if j < len(idxs):
                r = self.queue.iloc[idxs[j]]
                vec.extend([
                    _safe_float(r.get("req_procs")),
                    _safe_float(r.get("age_in_queue_s")),
                    _safe_float(r.get("submit_hour")),
                    _safe_float(r.get("submit_dow")),
                    _safe_float(r.get("pred_runtime_s")),
                    _safe_float(r.get("pred_energy_j")),
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vec.append(float(len(self.queue)))  # backlog size
        return _f32(vec)

    def _select_from_top_k(self, action: int) -> int:
        idxs = self._top_k_indices()
        if len(idxs) == 0:
            return -1
        action = int(action)
        if action < 0 or action >= len(idxs):
            action = 0
        return int(idxs[action])

    # ---------- Gymnasium API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load_jobs()
        return self._obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        pick_idx = self._select_from_top_k(action)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        reward = 0.0
        if pick_idx >= 0:
            job_row = self.queue.iloc[pick_idx]
            pr = _safe_float(job_row.get("pred_runtime_s"), 1.0)
            ej = _safe_float(job_row.get("pred_energy_j"), 0.0)
            kwh = ej / 3.6e6

            # advance time and remove job
            self.t_now += max(1.0, pr)
            self.queue = self.queue.drop(self.queue.index[pick_idx]).reset_index(drop=True)

            # dense penalty = energy (kWh) + backlog pressure, then scaled
            dense_pen = 0.5 * kwh + 0.01 * float(len(self.queue))
            reward = -dense_pen * self.reward_scale

            info.update({"kwh": kwh, "backlog_pen": 0.01 * float(len(self.queue))})

        self.steps += 1
        if self.steps >= self.max_steps or len(self.queue) == 0:
            terminated = (len(self.queue) == 0)
            truncated = (self.steps >= self.max_steps)

        if len(self.queue) > 0:
            self._refresh_predictions()
        self._obs = self._make_obs()
        return self._obs, float(reward), bool(terminated), bool(truncated), info