import time
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

from stable_baselines3.common.callbacks import BaseCallback


class ParquetLogger(BaseCallback):
    def __init__(self, out_path="logs/ppo_metrics.parquet", verbose=0):
        super().__init__(verbose)
        self.out_path = out_path
        self.rows = []
        self.t0 = None

    def _on_training_start(self) -> None:
        self.t0 = time.time()

    def _on_rollout_end(self) -> None:
        g = self.model.logger.name_to_value.get
        row = {
            "timesteps": self.num_timesteps,
            "time_elapsed_s": time.time() - self.t0,
            "approx_kl": g("train/approx_kl"),
            "clip_fraction": g("train/clip_fraction"),
            "entropy_loss": g("train/entropy_loss"),
            "explained_variance": g("train/explained_variance"),
            "policy_gradient_loss": g("train/policy_gradient_loss"),
            "value_loss": g("train/value_loss"),
            "learning_rate": g("train/learning_rate"),
        }
        self.rows.append(row)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if not self.rows or pq is None:
            return
        df = pd.DataFrame(self.rows)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.out_path)
