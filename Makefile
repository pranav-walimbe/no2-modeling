PYTHON ?= python3
VENV ?= .venv

VENV_PYTHON := $(VENV)/bin/python
SETUP_STAMP := $(VENV)/.setup

.PHONY: help setup check clean

help:
	@echo "Available targets:"
	@echo "  setup    Create the virtual environment and install dependencies"
	@echo "  check    Run Ruff and compile Python sources"
	@echo "  clean    Remove local Python caches and build artifacts"

setup: $(SETUP_STAMP)

$(SETUP_STAMP): requirements.txt | $(VENV_PYTHON)
	$(VENV_PYTHON) -m ensurepip --upgrade
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	touch $(SETUP_STAMP)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

check: setup
	$(VENV_PYTHON) -m ruff check src
	$(VENV_PYTHON) -m compileall -q src

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist .pytest_cache .ruff_cache
