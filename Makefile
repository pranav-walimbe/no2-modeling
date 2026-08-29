UV ?= uv
PYTHON_VERSION ?= 3.11
VENV ?= .venv
UV_PROJECT := UV_PROJECT_ENVIRONMENT=$(VENV) UV_PYTHON_DOWNLOADS=never $(UV)

.PHONY: help setup check clean

help:
	@echo "Available targets:"
	@echo "  setup    Sync the locked environment with uv"
	@echo "  check    Run Ruff and compile Python sources"
	@echo "  clean    Remove local Python caches and build artifacts"

setup:
	$(UV_PROJECT) sync --locked --python $(PYTHON_VERSION)

check:
	$(UV_PROJECT) run --locked ruff check src
	$(UV_PROJECT) run --locked python -m compileall -q src

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
