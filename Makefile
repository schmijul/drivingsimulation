.PHONY: run test train train-auto install-dev
PY := PYTHONUNBUFFERED=1 PYTHONPATH=src python3

install-dev:
	python3 -m pip install -e .[dev]

run:
	$(PY) -m drivesim.main

test:
	$(PY) -m pytest

train:
	$(PY) -m drivesim.ml.train

train-auto:
	$(PY) -m drivesim.ml.train_auto
