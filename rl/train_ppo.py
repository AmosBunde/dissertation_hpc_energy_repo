
#!/usr/bin/env python3
"""
Train PPO on the minimal HPCEnv using the saved surrogate predictors.
Ensure you've trained the surrogates first:
  make train
Then run:
  python rl/train_ppo.py --jobs data/processed/llnl_jobs.csv --timesteps 20000
"""
import argparse
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.env import HPCEnv

def make_env(jobs_csv, K):
    return lambda: HPCEnv(jobs_csv=jobs_csv, K=K)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='data/processed/llnl_jobs.csv')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--timesteps', type=int, default=20000)
    ap.add_argument('--out', default='models/ppo_hpcenv.zip')
    args = ap.parse_args()

    env = DummyVecEnv([make_env(args.jobs, args.K)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved policy -> {args.out}")

if __name__ == '__main__':
    main()
