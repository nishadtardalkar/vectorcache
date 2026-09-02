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
CMAKE_COMPUTE_FLAGS := $(CMAKE_COMMON_FLAGS) \
	-DVECTORCACHE_FETCH_DATASETS=OFF \
	-DVECTORCACHE_FETCH_OPENAI=OFF

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
	if ! cmake -L -B $(BUILD_DIR) 2>/dev/null | grep -q '^VECTORCACHE_BUILD_TOOLS:BOOL=ON'; then
		echo "VECTORCACHE_BUILD_TOOLS is OFF; ingest-bench will not be built."
		echo "Reconfigure with -DVECTORCACHE_BUILD_TOOLS=ON (the default for make compute)."
		exit 1
	fi
	cmake --build $(BUILD_DIR) --target vectorcache_tests ingest-bench -j$(JOBS)
	cd $(BUILD_DIR) && ctest --output-on-failure
	INGEST_BENCH=""
	for candidate in \
		"$(BUILD_DIR)/ingest-bench" \
		"$(BUILD_DIR)/$(BUILD_TYPE)/ingest-bench"; do
		if [ -x "$$candidate" ]; then
			INGEST_BENCH="$$candidate"
			break
		fi
	done
	if [ -z "$$INGEST_BENCH" ]; then
		INGEST_BENCH="$$(find "$(BUILD_DIR)" -maxdepth 3 \
			\( -name 'ingest-bench' -o -name 'ingest-bench.exe' \) -type f -print -quit 2>/dev/null || true)"
	fi
	if [ -z "$$INGEST_BENCH" ] || [ ! -f "$$INGEST_BENCH" ]; then
		echo "ingest-bench not found under $(BUILD_DIR)."
		echo "Inspect the build log above for ingest-bench compile/link errors."
		exit 1
	fi
	"$$INGEST_BENCH" --dataset $(DATASET) --data-dir $(DATA_DIR) $(BENCH_EXTRA_ARGS)

clean:
	rm -rf $(BUILD_DIR)
