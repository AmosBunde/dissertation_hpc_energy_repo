import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from rl.env import HPCEnv

class ObservationProjector(gym.Wrapper):
    """
    Projects Box observations to the expected dimension by taking
    the first D features. Use ONLY for quick compatibility checks.
    """
    def __init__(self, env, expected_dim: int):
        super().__init__(env)
        cur_dim = env.observation_space.shape[0]
        assert isinstance(env.observation_space, spaces.Box)
        low  = env.observation_space.low[:expected_dim]
        high = env.observation_space.high[:expected_dim]
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
        self.expected_dim = expected_dim
        self.cur_dim = cur_dim

    def observation(self, obs):
        # If upstream wrappers call observation(), ensure slicing happens there too
        return obs[:self.expected_dim]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[:self.expected_dim], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[:self.expected_dim], reward, terminated, truncated, info


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate PPO vs FIFO baseline")
    ap.add_argument("--jobs", required=True, type=str)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--baseline", type=str, default="ppo", choices=["ppo", "fifo"])
    ap.add_argument("--model", type=str, default="models/ppo_hpcenv.zip")
    ap.add_argument("--K", type=int, default=8, help="Top-K window (must match training)")
    return ap.parse_args()


def eval_policy(model_path: str | None, jobs_csv: str, episodes: int, K: int):
    # Build env first (current feature set)
    env = HPCEnv(jobs_csv=jobs_csv, K=K)

    if model_path is None:
        # FIFO baseline
        total_rewards = []
        for _ in range(episodes):
            obs, info = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                # FIFO: pick the earliest-submitted job among available
                # Our simple env uses "index action"; 0 corresponds to the first in window
                action = 0
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                ep_ret += r
            total_rewards.append(ep_ret)
        return float(np.mean(total_rewards)), float(np.std(total_rewards))

    # Load PPO model (gets expected obs dim from the policy)
    model = PPO.load(model_path)
    expected_dim = model.observation_space.shape[0]
    current_dim = env.observation_space.shape[0]
    if current_dim != expected_dim:
        # Project observations to the expected dimension
        env = ObservationProjector(env, expected_dim)
        print(f"[warn] Projecting obs from {current_dim} -> {expected_dim} for compatibility.")

    # Evaluate PPO
    rets = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            done = term or trunc
            ep_ret += r
        rets.append(ep_ret)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    args = parse_args()
    if args.baseline == "ppo":
        mean, std = eval_policy(args.model, args.jobs, args.episodes, args.K)
        print(f"PPO:  mean={mean:.3f} ± {std:.3f}")
    else:
        mean, std = eval_policy(None, args.jobs, args.episodes, args.K)
        print(f"FIFO: mean={mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    # be kind to BLAS threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
