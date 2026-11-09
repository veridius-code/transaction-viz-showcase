# Full pipeline: make build
# CI (or any existing environment) can override the interpreter: make build PY=python
PY ?= .venv/bin/python

.PHONY: setup build data robustness test charts notebook site clean-outputs

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt; \
	else \
		python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; \
	fi

build: data robustness test charts notebook site
	@echo "Built. Open docs/index.html in a browser."

data:
	$(PY) -m src.generate_data

robustness:
	$(PY) -m src.robustness

test:
	$(PY) -m pytest -q

charts:
	$(PY) -m src.charts_static

notebook:
	$(PY) notebook/build_notebook.py
	$(PY) -m nbconvert --to notebook --execute --inplace notebook/analysis.ipynb

site:
	$(PY) -m src.build_site

clean-outputs:
	rm -rf outputs/png/* outputs/pdf/* docs/assets docs/downloads docs/index.html
