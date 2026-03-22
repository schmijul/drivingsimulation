.PHONY: run test train install-dev

install-dev:
	python -m pip install -e .[dev]

run:
	drivesim-run

test:
	pytest

train:
	drivesim-train
