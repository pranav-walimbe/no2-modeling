PYTHON ?= python3
VENV ?= .venv

VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: help setup install check clean

help:
	@echo "Available targets:"
	@echo "  setup    Create the virtual environment and install dependencies"
	@echo "  install  Install or update dependencies in the virtual environment"
	@echo "  check    Compile Python sources to catch syntax errors"
	@echo "  clean    Remove local Python caches and build artifacts"

setup: $(VENV_PYTHON)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install: setup

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

check:
	$(PYTHON) -m compileall -q src

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist .pytest_cache .ruff_cache
	find . -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
