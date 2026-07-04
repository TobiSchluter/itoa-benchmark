# Convenience wrapper around the CMake build + plotting pipeline.
#
#   make               configure+build+run+plot with g++-16 (default)
#   make PRESET=clang21 ...   use clang++-21 instead
#   make build         just build
#   make run           run the benchmark (writes result/<preset>.csv)
#   make plots         (re)generate PNGs from the CSV
#   make venv          create the Python venv used for plotting
#   make clean         remove the current build tree
#
# Extra benchmark flags go through ARGS, e.g.:
#   make run ARGS="--modes=admixture --admix=4,5;9,10 --types=u64,i64"

PRESET ?= gcc16
BUILDDIR := build/$(PRESET)
CSV := result/$(PRESET).csv
VENV := .venv
PY := $(VENV)/bin/python

ARGS ?=

.PHONY: all build run plots venv configure clean clean-all

all: run plots

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build $(BUILDDIR) -j

run: build
	cd $(BUILDDIR) && ./itoa --out=../../$(CSV) $(ARGS)

$(PY):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q matplotlib numpy

venv: $(PY)

plots: $(PY)
	$(PY) plot_results.py $(CSV) --outdir result/plots

clean:
	rm -rf build/$(PRESET)

clean-all:
	rm -rf build/gcc16 build/clang21
