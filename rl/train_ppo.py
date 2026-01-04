import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback


def make_env_fn(jobs_csv, K, runtime_model, energy_model, reward_scale):
    def _f():
        from .env import HPCEnv
        return HPCEnv(
            jobs_csv=jobs_csv,
            K=K,
            runtime_model_path=runtime_model,
            energy_model_path=energy_model,
            reward_scale=reward_scale,
        )
    return _f


class Heartbeat(BaseCallback):
    """Print a simple heartbeat after each rollout."""
    def _on_rollout_end(self):
        print(f"[hb] timesteps={self.num_timesteps}")

    def _on_step(self) -> bool:
        # required by BaseCallback; return True to continue training
        return True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="Path to jobs CSV")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--runtime_model", default="models/runtime_surrogate_rf.joblib")
    ap.add_argument("--energy_model", default="models/energy_surrogate_rf.joblib")
    ap.add_argument("--reward_scale", type=float, default=50.0)
    return ap.parse_args()

def main():
    args = parse_args()

    # Build vec env: DummyVecEnv when n_envs=1, else SubprocVecEnv
    if args.n_envs == 1:
        venv = DummyVecEnv([make_env_fn(args.jobs, args.K, args.runtime_model, args.energy_model, args.reward_scale)])
    else:
        venv = SubprocVecEnv([
            make_env_fn(args.jobs, args.K, args.runtime_model, args.energy_model, args.reward_scale)
            for _ in range(args.n_envs)
        ])

    # Normalize obs and rewards for stability
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,          # shorter rollouts → more frequent logs
        batch_size=256,        # more frequent updates
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,         # slightly lower to encourage policy movement
        learning_rate=5e-4,
        clip_range=0.2,
        verbose=1,
    )

    # Configure SB3 logging: CSV + TensorBoard
    os.makedirs("logs/ppo_run", exist_ok=True)
    model.set_logger(configure("logs/ppo_run", ["csv", "tensorboard"]))

    # Optional Parquet logger (if available)
    callbacks = [Heartbeat()]
    try:
        from .callbacks import ParquetLogger
        callbacks.append(ParquetLogger("logs/ppo_metrics.parquet"))
    except Exception as e:
        print("[warn] parquet logger disabled:", e)

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_hpcenv.zip")
    venv.save("models/vecnorm.pkl")  # normalization stats for inference/eval
    print("[OK] Saved PPO policy -> models/ppo_hpcenv.zip")


if __name__ == "__main__":
    main()

