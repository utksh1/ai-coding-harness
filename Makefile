# AI Coding Harness - standardised evaluation interface.
# The evaluator will run: export AI_API_KEY=... && make setup && make run

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup run gui dashboard test bench-tokens clean lint typecheck gateway tui-go compose-up compose-down

setup:
        @echo ">> Setting up environment..."
        python3 -m venv $(VENV)
        $(PIP) install --upgrade pip
        $(PIP) install -e ".[dev,platform]"
        @echo ">> Verifying environment..."
        @$(PYTHON) -m harness doctor || true
        @echo ">> Setup complete."

run:
        @echo ">> Launching AI Harness..."
        $(PYTHON) -m harness run

gui:
        @echo ">> Launching Foreman Web Dashboard..."
        $(PYTHON) -m harness gui

dashboard: gui

test:
        $(PYTHON) -m pytest

bench-tokens:
        $(PYTHON) -m harness.cli bench

lint:
        $(VENV)/bin/ruff check src tests
        $(VENV)/bin/ruff format --check src tests

typecheck:
        $(VENV)/bin/mypy

# --- Platform layer (issue #69: Go gateway + Go TUI + compose) ---------------
gateway:
        cd gateway && go build -o ../bin/foreman-gateway . && ./../bin/foreman-gateway

tui-go:
        cd gateway && go build -o ../bin/foreman-tui ./cmd/tui && ./../bin/foreman-tui

go-test:
        cd gateway && go test ./... && go vet ./...

compose-up:
        docker compose -f platform/docker-compose.yml up --build

compose-down:
        docker compose -f platform/docker-compose.yml down

clean:
        rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build *.egg-info src/*.egg-info
        @find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        @echo ">> Cleaned."
