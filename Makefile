.PHONY: run demo-3d test test-workflows train train-auto train-anyone eval eval-compare eval-report eval-history eval-best install install-dev
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

run:
	$(PY) -m drivesim.main

test:
	$(PY) -m pytest

test-workflows:
	$(PY) -m pytest tests/test_train.py tests/test_models.py tests/test_agent.py tests/test_eval.py

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
