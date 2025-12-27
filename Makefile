VENV ?= .venv
PYTHON := $(VENV)/bin/python

.PHONY: rl-train rl-train-fast eval

rl-train:
	$(PYTHON) -m rl.train_ppo --jobs data/processed/llnl_jobs.csv --timesteps 200000

rl-train-fast:
	$(PYTHON) -m rl.train_ppo --jobs data/processed/llnl_jobs.csv --timesteps 20000

eval:
	$(PYTHON) scripts/eval_policy.py --jobs data/processed/llnl_jobs.csv --episodes 5
