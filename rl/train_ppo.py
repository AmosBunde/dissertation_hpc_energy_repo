import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def make_env_fn(jobs_csv, K):
    def _f():
        from .env import HPCEnv
        return HPCEnv(jobs_csv=jobs_csv, K=K)
    return _f


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="Path to jobs CSV (from swf_to_csv.py)")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--K", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()

    # Parallel envs + normalization (more stable learning)
    n_envs = 2
    venv = SubprocVecEnv([make_env_fn(args.jobs, args.K) for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO config tuned a notch up
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        learning_rate=5e-4,
        clip_range=0.2,
        verbose=1,
    )

    # SB3 CSV + TensorBoard logging
    from stable_baselines3.common.logger import configure
    os.makedirs("logs/ppo_run", exist_ok=True)
    new_logger = configure("logs/ppo_run", ["csv", "tensorboard"])
    model.set_logger(new_logger)

    # Optional Parquet logger (guarded)
    cb = None
    try:
        from .callbacks import ParquetLogger
        os.makedirs("logs", exist_ok=True)
        cb = ParquetLogger("logs/ppo_metrics.parquet")
    except Exception as e:
        print("[warn] parquet logger disabled:", e)

    if cb:
        model.learn(total_timesteps=args.timesteps, callback=cb)
    else:
        model.learn(total_timesteps=args.timesteps)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_hpcenv.zip")
    print("[OK] Saved PPO policy -> models/ppo_hpcenv.zip")


if __name__ == "__main__":
    main()
