PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install data backtest train forecast monitor serve test lint
install:
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[research,dev]"
data:
	$(PY) -m epf.cli data
backtest:
	$(PY) -m epf.cli backtest
train:
	$(PY) -m epf.cli train
forecast:
	$(PY) -m epf.cli forecast
monitor:
	$(PY) -m epf.cli monitor
serve:
	$(PY) -m epf.cli serve
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
