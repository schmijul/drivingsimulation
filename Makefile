.PHONY: run test train train-auto install-dev

install-dev:
	python -m pip install -e .[dev]

run:
	drivesim-run

test:
	pytest

train:
	drivesim-train

train-auto:
	drivesim-train-auto
