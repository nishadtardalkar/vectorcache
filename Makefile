# VectorCache HPC workflow — split internet work (login) from offline build/test/bench (compute).
# Requires bash and a shared project path between login and compute nodes.

SHELL := /bin/bash
.ONESHELL:

BUILD_DIR     ?= build
BUILD_TYPE    ?= Release
JOBS          ?= $(shell nproc 2>/dev/null || echo 4)
DATA_DIR      ?= data
DATASETS      ?= all
DATASET       ?=
BENCH_EXTRA_ARGS ?=
CMAKE_OPTS    ?=
ENV_SCRIPT    := scripts/envs.sh
LOGIN_READY   := $(BUILD_DIR)/.login-ready

# OpenAI datasets are converted from parquet shards with Apache Arrow.
ifneq (,$(filter all openai-1536 openai-3072,$(DATASETS)))
LOGIN_FETCH_OPENAI := ON
else
LOGIN_FETCH_OPENAI := OFF
endif

CMAKE_COMMON_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DVECTORCACHE_BUILD_GLOVE=ON \
	-DVECTORCACHE_BUILD_TOOLS=ON \
	-DVECTORCACHE_BUILD_TESTS=ON \
	$(CMAKE_OPTS)

CMAKE_LOGIN_FLAGS := $(CMAKE_COMMON_FLAGS) -DVECTORCACHE_FETCH_OPENAI=$(LOGIN_FETCH_OPENAI)
CMAKE_COMPUTE_FLAGS := $(CMAKE_COMMON_FLAGS) -DVECTORCACHE_FETCH_OPENAI=OFF

.PHONY: help login compute clean

.DEFAULT_GOAL := help

help:
	@echo "VectorCache HPC targets:"
	@echo ""
	@echo "  make login    Configure CMake, fetch dependencies, download datasets (login node)"
	@echo "  make compute  Build, test, and benchmark offline (compute node; DATASET required)"
	@echo "  make clean    Remove build directory"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASETS=$(DATASETS)  DATASET=$(DATASET)"
	@echo "  BENCH_EXTRA_ARGS=$(BENCH_EXTRA_ARGS)"
	@echo "  CMAKE_OPTS=$(CMAKE_OPTS)"
	@echo ""
	@echo "Example: make login DATASETS=glove"
	@echo "Example: make compute DATASET=glove"

login: $(LOGIN_READY)

$(LOGIN_READY):
	set -euo pipefail
	source $(ENV_SCRIPT)
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_LOGIN_FLAGS)
	cmake --build $(BUILD_DIR) --target fetch-datasets -j$(JOBS)
	$(BUILD_DIR)/fetch-datasets --data-dir $(DATA_DIR) $(DATASETS)
	touch $(LOGIN_READY)

compute:
	@if [ ! -f $(LOGIN_READY) ]; then \
		echo "Run 'make login' on a login node first."; \
		exit 1; \
	fi
	@if [ -z "$(DATASET)" ]; then \
		echo "DATASET is required, e.g. make compute DATASET=glove"; \
		exit 1; \
	fi
	set -euo pipefail
	source $(ENV_SCRIPT)
	cmake -S . -B $(BUILD_DIR) \
		-DFETCHCONTENT_FULLY_DISCONNECTED=ON \
		$(CMAKE_COMPUTE_FLAGS)
	cmake --build $(BUILD_DIR) -j$(JOBS)
	cd $(BUILD_DIR) && ctest --output-on-failure
	$(BUILD_DIR)/ingest-bench --dataset $(DATASET) --data-dir $(DATA_DIR) $(BENCH_EXTRA_ARGS)

clean:
	rm -rf $(BUILD_DIR)
