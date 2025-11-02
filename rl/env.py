
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
import numpy as np
import pandas as pd
import joblib
import gym
from gym import spaces

class HPCEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs_csv="data/processed/llnl_jobs.csv", K=8,
                 runtime_model_path="models/runtime_surrogate_rf.joblib",
                 energy_model_path="models/energy_surrogate_rf.joblib",
                 seed=42):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.K = K
        self.jobs_df = pd.read_csv(jobs_csv)
        # Build the same leakage-safe features used during training
        self.jobs_df = self._build_features(self.jobs_df)
        # Load surrogates
        self.rt_model = joblib.load(runtime_model_path)
        self.en_model = joblib.load(energy_model_path)
        # Observation: K * 4 features + 1 backlog scalar
        self.obs_dim = K * 4 + 1
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(K)  # choose which slot to schedule next
        self.reset()

    def _build_features(self, df):
        df = df.copy()
        for c in ["req_procs", "age_in_queue_s", "submit_hour", "submit_dow"]:
            if c not in df.columns:
                df[c] = 0
        # Normalize a bit for stability
        df["req_procs"] = df["req_procs"].fillna(0).clip(0, 4096)
        df["age_in_queue_s"] = df["age_in_queue_s"].fillna(0).clip(0, 7*24*3600)
        df["submit_hour"] = df["submit_hour"].fillna(0).clip(0, 23)
        df["submit_dow"] = df["submit_dow"].fillna(0).clip(0, 6)
        return df[["req_procs","age_in_queue_s","submit_hour","submit_dow"]]

    def _get_obs(self):
        # take first K jobs (pad with zeros)
        view = self.queue[:self.K]
        if len(view) < self.K:
            pad = np.zeros((self.K - len(view), 4), dtype=np.float32)
            feats = np.vstack([view, pad])
        else:
            feats = view
        backlog = np.array([len(self.queue)], dtype=np.float32)
        return np.concatenate([feats.flatten().astype(np.float32), backlog])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        # shuffle jobs to make episodes varied
        X = self.jobs_df.sample(frac=1.0, random_state=self.rng.randint(0, 1_000_000)).reset_index(drop=True)
        self.queue = X.values  # numpy array N x 4
        self.t = 0.0
        self.energy_kwh = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        if len(self.queue) == 0:
            done = True
            return self._get_obs(), 0.0, done, info

        # Select job features at slot 'action' or no-op if slot empty
        if action >= len(self.queue):
            # no-op penalty
            reward -= 0.1
        else:
            x = self.queue[action]
            # Predict runtime (seconds) and convert to hours
            rt = float(self.rt_model.predict([x])[0])
            rt = max(rt, 1.0)
            # Predict energy (J) then convert to kWh (1 kWh = 3.6e6 J)
            eJ = float(self.en_model.predict([x])[0])
            ekwh = max(eJ / 3.6e6, 0.0)
            self.energy_kwh += ekwh
            # Advance time and remove job from queue
            self.t += rt
            self.queue = np.delete(self.queue, action, axis=0)
            # Reward: negative energy and backlog pressure
            reward -= ekwh
        # Backlog penalty
        reward -= 0.001 * len(self.queue)

        # Termination conditions
        if len(self.queue) == 0 or self.steps >= 10_000:
            done = True
            info["energy_kwh"] = self.energy_kwh
            info["time_s"] = self.t

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        print(f"t={self.t:.1f}s, backlog={len(self.queue)}, energy={self.energy_kwh:.4f} kWh")
