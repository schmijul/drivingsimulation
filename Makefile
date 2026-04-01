.PHONY: run demo-3d test test-workflows docs-media docs-media-clean docs-media-check train train-auto train-anyone train-rl eval eval-compare eval-report eval-history eval-best experiment-history install install-dev install-rl
PY := PYTHONUNBUFFERED=1 PYTHONPATH=src python3
MODE ?= live
TRAIN_ANYONE_ARGS ?=
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

install:
	@if [ ! -x "$(VENV_PY)" ]; then python3 -m venv $(VENV); fi
	$(VENV_PIP) install -U pip
	$(VENV_PIP) install .

install-dev:
	@if [ ! -x "$(VENV_PY)" ]; then python3 -m venv $(VENV); fi
	$(VENV_PIP) install -U pip
	$(VENV_PIP) install -e .[dev]

install-rl:
	@if [ ! -x "$(VENV_PY)" ]; then python3 -m venv $(VENV); fi
	$(VENV_PIP) install -U pip
	$(VENV_PIP) install -e .[dev,rl]

run:
	$(PY) -m drivesim.main

test:
	$(PY) -m pytest

test-workflows:
	$(PY) -m pytest tests/test_train.py tests/test_models.py tests/test_agent.py tests/test_eval.py tests/test_curriculum.py tests/test_experiment_history.py

docs-media:
	$(PY) scripts/generate_readme_media.py

docs-media-clean:
	rm -rf imgs/readme

docs-media-check:
	test -f imgs/readme/drive_chase_ground_truth.png
	test -f imgs/readme/drive_iso_ground_truth.png
	test -f imgs/readme/drive_topdown_ground_truth.png
	test -f imgs/readme/map_ground_truth.png
	test -f imgs/readme/map_sensor_driven.png
	test -f imgs/readme/train_live_status.png

train:
ifeq ($(MODE),live)
	DRIVESIM_START_MODE=train-live $(PY) -m drivesim.main
else ifeq ($(MODE),anyone)
	$(PY) -m drivesim.ml.train_anyone $(TRAIN_ANYONE_ARGS)
else ifeq ($(MODE),auto)
	$(PY) -m drivesim.ml.train_auto
else
	$(PY) -m drivesim.ml.train
endif

train-auto:
	$(PY) -m drivesim.ml.train_auto

train-anyone:
	$(PY) -m drivesim.ml.train_anyone $(TRAIN_ANYONE_ARGS)

train-rl:
	$(PY) -m drivesim.ml.train_rl

demo-3d:
	DRIVESIM_START_VIEW=3d $(PY) -m drivesim.main

eval:
	$(PY) -m drivesim.ml.eval

eval-compare:
	$(PY) -m drivesim.ml.eval --policy both

eval-report:
	$(PY) -m drivesim.ml.eval --policy both --json-auto

eval-history:
	$(PY) -m drivesim.ml.eval_history --limit 15

eval-best:
	$(PY) -m drivesim.ml.eval_history --best

experiment-history:
	$(PY) -m drivesim.ml.experiment_history --limit 15
