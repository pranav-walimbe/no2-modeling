UV ?= uv
PYTHON_VERSION ?= 3.11
VENV ?= .venv
UV_PROJECT := UV_PROJECT_ENVIRONMENT=$(VENV) UV_PYTHON_DOWNLOADS=never $(UV)
USER ?= $(shell id -un)
CODEX ?= $(HOME)/.local/bin/codex
CODEX_HOME ?= $(HOME)/.codex
CODEX_LOCAL ?= /tmp/$(USER)/.codex
CODEX_SEED ?= $(HOME)/.codex_seed
# Durable directories the seed owns and the node-local tree reaches by symlink
CODEX_SHARED := packages plugins skills sessions cache
# Small files mirrored both ways so every node keeps a usable copy
CODEX_MIRRORED := auth.json config.toml installation_id

.PHONY: help setup check clean fix-codex

help:
	@echo "Available targets:"
	@echo "  setup    Sync the locked environment with uv"
	@echo "  check    Run Ruff and compile Python sources"
	@echo "  clean    Remove local Python caches and build artifacts"
	@echo "  fix-codex Move Codex SQLite state off NFS and clear stale locks"

setup:
	$(UV_PROJECT) sync --locked --python $(PYTHON_VERSION)

check:
	$(UV_PROJECT) run --locked ruff check src
	$(UV_PROJECT) run --locked python -m compileall -q src

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache

# Codex stores runtime state in SQLite and NFS home directories cannot provide
# the file locks SQLite needs. That is the "(code: 15) locking protocol"
# startup failure. $(CODEX_HOME) also holds the installed release binaries so
# the tree is split rather than relocated wholesale. $(CODEX_SEED) keeps the
# durable parts on NFS. $(CODEX_LOCAL) holds the SQLite databases on node-local
# disk. $(CODEX_HOME) becomes a symlink at the local tree whose shared
# directories link back to the seed. Pass V=1 to see the full doctor report.
fix-codex:
	@set -e; \
	report=$$(mktemp); trap 'rm -f $$report' EXIT; \
	pkill -u $(USER) -x codex 2>/dev/null || true; \
	mkdir -p $(CODEX_LOCAL) $(CODEX_SEED); \
	if [ ! -L $(CODEX_HOME) ] && [ -d $(CODEX_HOME) ]; then \
		cp -a $(CODEX_HOME)/. $(CODEX_SEED)/; \
		rm -rf $(CODEX_HOME); \
	fi; \
	if [ ! -d $(CODEX_SEED)/packages ]; then \
		echo "fix-codex: no installation under $(CODEX_SEED)/packages, reinstall codex first" >&2; \
		exit 1; \
	fi; \
	for name in $(CODEX_SHARED); do \
		if [ ! -L $(CODEX_LOCAL)/$$name ] && [ -d $(CODEX_LOCAL)/$$name ]; then \
			mkdir -p $(CODEX_SEED)/$$name; \
			cp -a $(CODEX_LOCAL)/$$name/. $(CODEX_SEED)/$$name/; \
			rm -rf $(CODEX_LOCAL)/$$name; \
		fi; \
		if [ -d $(CODEX_SEED)/$$name ]; then \
			ln -sfn $(CODEX_SEED)/$$name $(CODEX_LOCAL)/$$name; \
		fi; \
	done; \
	for name in $(CODEX_MIRRORED); do \
		cp -au $(CODEX_LOCAL)/$$name $(CODEX_SEED)/ 2>/dev/null || true; \
		cp -au $(CODEX_SEED)/$$name $(CODEX_LOCAL)/ 2>/dev/null || true; \
	done; \
	for db in $(CODEX_SEED)/*.sqlite; do \
		if [ ! -f $$db ]; then continue; fi; \
		if [ -f $(CODEX_LOCAL)/$$(basename $$db) ]; then rm -f $$db; else mv $$db $(CODEX_LOCAL)/; fi; \
	done; \
	rm -f $(CODEX_SEED)/*.sqlite-wal $(CODEX_SEED)/*.sqlite-shm; \
	ln -sfn $(CODEX_LOCAL) $(CODEX_HOME); \
	rm -f $(CODEX_LOCAL)/*.sqlite-wal $(CODEX_LOCAL)/*.sqlite-shm; \
	if [ ! -x $(CODEX) ]; then \
		echo "fix-codex: $(CODEX) is not executable, the relinked tree is incomplete" >&2; \
		exit 1; \
	fi; \
	if $(CODEX) doctor </dev/null >$$report 2>&1; then \
		echo "codex ok: state in $(CODEX_LOCAL), install in $(CODEX_SEED)"; \
		[ -z "$(V)" ] || cat $$report; \
	else \
		echo "codex doctor still failing:"; cat $$report; exit 1; \
	fi
