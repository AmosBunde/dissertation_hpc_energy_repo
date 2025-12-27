
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
import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import gymnasium as gym
from gymnasium import spaces


def _safe_get(d: Dict[str, float], k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default


class HPCEnv(gym.Env):
    """
    Minimal but *functional* scheduler env:
      - State = features of top-K jobs in queue + simple backlog stats
      - Action = pick which job (0..K-1) to start now
      - Transition = remove chosen job, advance time by predicted runtime
      - Reward = -(0.5 * kWh + backlog_penalty), scaled to O(1..10)
    Uses two surrogate models:
      - runtime_surrogate_rf.joblib: predicts runtime_s
      - energy_surrogate_rf.joblib:  predicts energy_j
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jobs_csv: str,
        K: int = 8,
        runtime_model_path: str = "models/runtime_surrogate_rf.joblib",
        energy_model_path: str = "models/energy_surrogate_rf.joblib",
        max_steps: int = 1000,
    ):
        super().__init__()
        self.jobs_csv = jobs_csv
        self.K = int(K)
        self.max_steps = int(max_steps)

        # Load surrogates
        self.rt_model = joblib.load(runtime_model_path)
        self.en_model = joblib.load(energy_model_path)
        self.rt_cols = getattr(self.rt_model, "feature_names_in_", None)
        self.en_cols = getattr(self.en_model, "feature_names_in_", None)

        # Observation: (top-K * per-job-feats) + 1 backlog scalar
        self.per_job_feats = ["req_procs", "age_in_queue_s", "submit_hour", "submit_dow", "pred_runtime_s"]
        self.obs_dim = self.K * len(self.per_job_feats) + 1
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)

        self.steps = 0
        self.t_now = 0.0
        self.queue: pd.DataFrame = pd.DataFrame()
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)

        self._load_jobs()

    # ---------- core helpers ----------

    def _load_jobs(self):
        df = pd.read_csv(self.jobs_csv)
        # Minimal leakage-safe features. If missing, default to zeros.
        for col in ["submit", "req_procs", "req_time", "status"]:
            if col not in df.columns:
                df[col] = 0
        df["age_in_queue_s"] = 0.0
        df["submit_hour"] = (df["submit"] % (24 * 3600)) // 3600
        df["submit_dow"] = (df["submit"] // (24 * 3600)) % 7

        # At reset we sort by submit time to emulate FIFO arrival order
        self.queue = df.sort_values("submit").reset_index(drop=True)
        # when a job starts, we will drop it from queue
        self.steps = 0
        self.t_now = 0.0
        self._refresh_predicted_runtime_energy()
        self._obs = self._make_obs()

    def _predict_model(self, model, cols, feat: Dict[str, float]) -> float:
        if cols is not None:
            row = {c: _safe_get(feat, c, 0.0) for c in cols}
            X = pd.DataFrame([row], columns=cols)
        else:
            X = np.array([list(feat.values())], dtype=np.float32)
        return float(model.predict(X)[0])

    def _refresh_predicted_runtime_energy(self):
        # For the current queue, compute predicted runtime for each job (used in obs and heuristics)
        feats = []
        for _, r in self.queue.iterrows():
            f = {
                "req_procs": float(r.get("req_procs", 0)),
                "age_in_queue_s": float(self.t_now - float(r.get("submit", 0))),
                "submit_hour": float(r.get("submit_hour", 0)),
                "submit_dow": float(r.get("submit_dow", 0)),
            }
            pr = self._predict_model(self.rt_model, self.rt_cols, f)
            f["pred_runtime_s"] = pr
            feats.append(f)
        # store new columns
        if len(self.queue) > 0:
            dfp = pd.DataFrame(feats)
            for c in dfp.columns:
                self.queue.loc[:, c] = dfp[c].values

    def _top_k_indices(self) -> List[int]:
        # pick top-K by "age_in_queue_s" (oldest first); could swap to priority ordering
        if len(self.queue) == 0:
            return []
        order = np.argsort(-self.queue["age_in_queue_s"].to_numpy())
        return order[: self.K].tolist()

    def _make_obs(self) -> np.ndarray:
        vec: List[float] = []
        idxs = self._top_k_indices()
        for j in range(self.K):
            if j < len(idxs):
                r = self.queue.iloc[idxs[j]]
                vec.extend([
                    float(r.get("req_procs", 0)),
                    float(r.get("age_in_queue_s", 0.0)),
                    float(r.get("submit_hour", 0.0)),
                    float(r.get("submit_dow", 0.0)),
                    float(r.get("pred_runtime_s", 0.0)),
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        # backlog scalar (number of waiting jobs)
        vec.append(float(len(self.queue)))
        return np.asarray(vec, dtype=np.float32)

    def _select_from_top_k(self, action: int) -> int:
        idxs = self._top_k_indices()
        if len(idxs) == 0:
            return -1
        if action < 0 or action >= len(idxs):
            action = 0  # clamp if invalid
        return int(idxs[action])

    # ---------- Gymnasium API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load_jobs()
        return self._obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        # choose job among top-K
        pick_idx = self._select_from_top_k(action)
        terminated = False
        truncated = False

        reward = 0.0
        if pick_idx >= 0:
            job_row = self.queue.iloc[pick_idx]

            feat = {
                "req_procs": float(job_row.get("req_procs", 0)),
                "age_in_queue_s": float(job_row.get("age_in_queue_s", 0.0)),
                "submit_hour": float(job_row.get("submit_hour", 0.0)),
                "submit_dow": float(job_row.get("submit_dow", 0.0)),
            }
            runtime_s = self._predict_model(self.rt_model, self.rt_cols, feat)
            energy_j = self._predict_model(self.en_model, self.en_cols, feat)

            # advance time by predicted runtime
            self.t_now += max(runtime_s, 1.0)
            # remove job from queue
            self.queue = self.queue.drop(self.queue.index[pick_idx]).reset_index(drop=True)

            # backlog penalty: small per waiting job
            backlog_pen = 0.01 * float(len(self.queue))
            kwh = energy_j / 3.6e6
            reward = -(0.5 * kwh + backlog_pen)  # O(1) scale

        self.steps += 1
        if self.steps >= self.max_steps or len(self.queue) == 0:
            terminated = (len(self.queue) == 0)
            truncated = (self.steps >= self.max_steps)

        # refresh derived features and obs
        if len(self.queue) > 0:
            self._refresh_predicted_runtime_energy()
        self._obs = self._make_obs()
        return self._obs, float(reward), bool(terminated), bool(truncated), {}

