.PHONY: install test lint mypy clean format serve

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check .

mypy:
	poetry run mypy .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

format:
	poetry run black .
	poetry run isort .

serve:
	poetry run ./run_server.sh

# Default target when just running 'make'
all: lint mypy test