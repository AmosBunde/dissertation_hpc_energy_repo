import argparse
import numpy as np
from stable_baselines3 import PPO

from rl.env import HPCEnv


def eval_policy(policy_path: str | None, jobs_csv: str, episodes: int = 5, K: int = 8):
    env = HPCEnv(jobs_csv=jobs_csv, K=K)
    model = PPO.load(policy_path) if policy_path else None

    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        total = 0.0
        while True:
            if model:
                a, _ = model.predict(obs, deterministic=True)
            else:
                # FIFO baseline = pick oldest among top-K = index 0
                a = 0
            obs, r, term, trunc, info = env.step(int(a))
            total += r
            if term or trunc:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    ppo_mean, ppo_std = eval_policy("models/ppo_hpcenv.zip", args.jobs, args.episodes, args.K)
    fifo_mean, fifo_std = eval_policy(None, args.jobs, args.episodes, args.K)
    print(f"PPO:  mean={ppo_mean:.3f} ± {ppo_std:.3f}")
    print(f"FIFO: mean={fifo_mean:.3f} ± {fifo_std:.3f}")


if __name__ == "__main__":
    main()
