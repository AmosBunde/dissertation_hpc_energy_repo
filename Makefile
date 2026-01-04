VENV ?= .venv
PYTHON := $(VENV)/bin/python

.PHONY: rl-train rl-train-fast eval

# Cap BLAS threads to avoid oversubscription
ENVVARS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.

rl-train:
	$(ENVVARS) $(PYTHON) -m rl.train_ppo --jobs data/processed/llnl_jobs.csv --timesteps 300000 --n_envs 2

rl-train-fast:
	$(ENVVARS) $(PYTHON) -m rl.train_ppo --jobs data/processed/llnl_jobs.csv --timesteps 20000 --n_envs 1

eval:
	$(ENVVARS) $(PYTHON) -m scripts.eval_policy --jobs data/processed/llnl_jobs.csv --episodes 5
