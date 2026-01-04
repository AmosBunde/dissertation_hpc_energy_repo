

# === Feature building and training shortcuts ===
features:
	python scripts/build_features.py --jobs_csv data/processed/llnl_jobs.csv --synthetic-energy

train-runtime:
	python scripts/train_runtime_surrogate.py --csv data/processed/runtime_features.csv --target runtime_s

train-energy:
	python scripts/train_energy_surrogate.py --csv data/processed/energy_features.csv --target energy_j

train: features train-runtime train-energy
	@echo "All surrogates trained."


# ====== RL training ======
rl-train:
	$(PYTHON) rl/train_ppo.py --jobs data/processed/llnl_jobs.csv --timesteps 20000

rl-train-fast:
	$(PYTHON) rl/train_ppo.py --jobs data/processed/llnl_jobs.csv --timesteps 5000

.PHONY: rl-train rl-train-fast
