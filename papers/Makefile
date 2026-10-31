RUN ?= ../experiments/hl_btc_2026-10
PY  ?= ../.venv/bin/python

.PHONY: all data papers clean

all: data papers

# Rebuild every number, table and figure from the run directory (offline).
data:
	cd .. && $(PY) scripts/hl_report.py --run $(subst ../,,$(RUN)) --papers papers

papers:
	cd methods && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd empirical && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	cd methods && latexmk -C
	cd empirical && latexmk -C
