.PHONY: run test train train-auto eval eval-compare install-dev
PY := PYTHONUNBUFFERED=1 PYTHONPATH=src python3
MODE ?= live

install-dev:
	python3 -m pip install -e .[dev]

run:
	$(PY) -m drivesim.main

test:
	$(PY) -m pytest

train:
ifeq ($(MODE),live)
	DRIVESIM_START_MODE=train-live $(PY) -m drivesim.main
else ifeq ($(MODE),auto)
	$(PY) -m drivesim.ml.train_auto
else
	$(PY) -m drivesim.ml.train
endif

train-auto:
	$(PY) -m drivesim.ml.train_auto

eval:
	$(PY) -m drivesim.ml.eval

eval-compare:
	$(PY) -m drivesim.ml.eval --policy both
