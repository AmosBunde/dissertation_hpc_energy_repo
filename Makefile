
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt matplotlib jupyter nbformat nbconvert

simulate:
	@echo "==> Simulating with config: batsim/$(CONFIG).json"
	@bash batsim/run_replay.sh batsim/$(CONFIG).json

simulate-LLNL:
	@$(MAKE) CONFIG=config_llnl simulate

simulate-KTH:
	@$(MAKE) CONFIG=config_kth simulate

plot:
	@$(PYTHON) - <<'PY'
import os
nb = 'notebooks/plot_backlog_energy_co2e.ipynb'
os.system(f'$(VENV)/bin/jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=240 {nb} --output notebooks/plot_report.html')
print('Rendered notebooks/plot_report.html')
PY

clean:
	@rm -rf notebooks/*.html batsim/results_* .ipynb_checkpoints

.PHONY: setup simulate simulate-LLNL simulate-KTH plot clean
