.DEFAULT_GOAL := help

COMPOSE := docker compose
PYTHON := .venv/bin/python
DEMO_URL := http://localhost:8080

.PHONY: help up down logs demo unittest testruns clean

help:
	@printf '%s\n' \
		'up        start the demo containers' \
		'down      stop the demo containers' \
		'logs      follow demo container logs' \
		'demo      rebuild the demo and open $(DEMO_URL)' \
		'unittest  run the unit tests' \
		'testruns  run the assignment stage scripts' \
		'clean     stop the demo and remove generated files'

up:
	$(COMPOSE) up -d --wait

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f --tail 200

demo:
	$(COMPOSE) up --build -d --wait
	python3 -c "import webbrowser; webbrowser.open('$(DEMO_URL)')"

unittest:
	$(PYTHON) -m pytest

testruns:
	$(PYTHON) scripts/run_rubric.py
	$(PYTHON) scripts/run_evidence.py
	$(PYTHON) scripts/run_nodes.py
	$(PYTHON) scripts/run_traversal.py
	$(PYTHON) scripts/run_score.py
	$(PYTHON) scripts/run_result.py

clean:
	-$(COMPOSE) down --remove-orphans --rmi local
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info demo/web/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
