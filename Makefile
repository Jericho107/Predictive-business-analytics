.PHONY: install lint test smoke reverse-test validate

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

smoke:
	python -m predictive_analytics.cli smoke

reverse-test:
	python -m predictive_analytics.cli reverse-test

validate: lint test smoke reverse-test
