# VectorCache HPC workflow — split internet work (login) from offline build/test/bench (compute).
# Requires bash and a shared project path between login and compute nodes.

SHELL := /bin/bash
.ONESHELL:

BUILD_DIR     ?= build
BUILD_DIR_ABS := $(abspath $(BUILD_DIR))
BUILD_TYPE    ?= Release
JOBS          ?= $(shell nproc 2>/dev/null || echo 4)
DATA_DIR      ?= data
DATASETS      ?= all
DATASET       ?=
NPY           ?=
SPLIT         ?=
LIMIT         ?=
SEED          ?=
TOP_D         ?=
QUERY_SPLIT   ?=
QUERY_LIMIT   ?=
K             ?=
MAX_HD        ?=
CALIBRATE     ?=
RECALL        ?=
FORCE         ?=
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

CMAKE_COMPILER_FLAGS := \
	-DCMAKE_C_COMPILER="$${CC:-gcc}" \
	-DCMAKE_CXX_COMPILER="$${CXX:-g++}"

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

# Optional CLI flags: only emit when the Make variable is non-empty.
opt_arg = $(if $(strip $($(1))),--$(2) $($(1)),)
flag_arg = $(if $(filter 1 ON on true TRUE yes YES,$($(1))),--$(2),)

# Shared ingest-bench / query-bench args (DATASET|NPY + DATA_DIR always set by compute).
BENCH_COMMON_ARGS = \
	$(call opt_arg,SPLIT,split) \
	$(call opt_arg,LIMIT,limit) \
	$(call opt_arg,SEED,seed) \
	$(call opt_arg,TOP_D,top-d) \
	$(BENCH_EXTRA_ARGS)

INGEST_BENCH_ARGS = $(BENCH_COMMON_ARGS)

QUERY_BENCH_ARGS = \
	$(BENCH_COMMON_ARGS) \
	$(call opt_arg,QUERY_SPLIT,query-split) \
	$(call opt_arg,QUERY_LIMIT,query-limit) \
	$(call opt_arg,K,k) \
	$(call opt_arg,MAX_HD,max-hd) \
	$(call flag_arg,CALIBRATE,calibrate) \
	$(call flag_arg,RECALL,recall)

.PHONY: help login compute clean

.DEFAULT_GOAL := help

help:
	@echo "VectorCache HPC targets:"
	@echo ""
	@echo "  make login    Configure CMake, fetch dependencies, download datasets (login node)"
	@echo "  make compute  Build, test, ingest-bench, and query-bench offline (compute node)"
	@echo "  make clean    Remove build directory"
	@echo ""
	@echo "Build / login variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASETS=$(DATASETS)  FORCE=$(FORCE)"
	@echo "  CMAKE_OPTS=$(CMAKE_OPTS)"
	@echo "  VECTORCACHE_SRHT_ROUNDS (cmake cache, default 1): set to 2 or 3 for multi-round SRHT"
	@echo ""
	@echo "Bench variables (map to ingest-bench / query-bench CLI):"
	@echo "  DATASET / NPY   --dataset or --npy (one required for compute)"
	@echo "  DATA_DIR        --data-dir"
	@echo "  SPLIT           --split"
	@echo "  LIMIT           --limit"
	@echo "  SEED            --seed"
	@echo "  TOP_D           --top-d"
	@echo "  QUERY_SPLIT     --query-split (query-bench only)"
	@echo "  QUERY_LIMIT     --query-limit (query-bench only)"
	@echo "  K               --k (query-bench only)"
	@echo "  MAX_HD          --max-hd (query-bench only)"
	@echo "  CALIBRATE=1     --calibrate (query-bench only)"
	@echo "  RECALL=1        --recall (query-bench only)"
	@echo "  BENCH_EXTRA_ARGS  appended to both benches as-is"
	@echo ""
	@echo "For native SIMD on compute nodes: make compute DATASET=glove CMAKE_OPTS='-DCMAKE_CXX_FLAGS=-march=native'"
	@echo "For 3-round SRHT at compile time: make compute DATASET=glove CMAKE_OPTS='-DVECTORCACHE_SRHT_ROUNDS=3'"
	@echo ""
	@echo "Example: make login DATASETS=glove"
	@echo "Example: make compute DATASET=glove TOP_D=8 RECALL=1 MAX_HD=4"

login: $(LOGIN_READY)

$(LOGIN_READY):
	set -euo pipefail
	source $(ENV_SCRIPT)
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_COMPILER_FLAGS) $(CMAKE_LOGIN_FLAGS)
	cmake --build $(BUILD_DIR) --target fetch-datasets -j$(JOBS)
	$(BUILD_DIR)/fetch-datasets --data-dir $(DATA_DIR) $(call flag_arg,FORCE,force) $(DATASETS)
	touch $(LOGIN_READY)

compute:
	@if [ ! -f $(LOGIN_READY) ]; then \
		echo "Run 'make login' on a login node first."; \
		exit 1; \
	fi
	@if [ -z "$(strip $(DATASET)$(NPY))" ]; then \
		echo "DATASET or NPY is required, e.g. make compute DATASET=glove"; \
		exit 1; \
	fi
	@if [ -n "$(strip $(DATASET))" ] && [ -n "$(strip $(NPY))" ]; then \
		echo "Pass only one of DATASET or NPY, not both."; \
		exit 1; \
	fi
	set -euo pipefail
	source $(ENV_SCRIPT)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_COMPILER_FLAGS) \
		-DFETCHCONTENT_FULLY_DISCONNECTED=ON \
		$(CMAKE_COMPUTE_FLAGS)
	TOOLS_STATUS="$$(grep '^VECTORCACHE_BUILD_TOOLS:BOOL=' "$(BUILD_DIR)/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || true)"
	if [ "$$TOOLS_STATUS" != ON ]; then
		echo "VECTORCACHE_BUILD_TOOLS is $${TOOLS_STATUS:-unset}; ingest-bench and query-bench will not be built."
		if [ -n "$(CMAKE_OPTS)" ]; then
			echo "CMAKE_OPTS is set to '$(CMAKE_OPTS)' and overrides the Makefile default (-DVECTORCACHE_BUILD_TOOLS=ON)."
		fi
		echo "Fix: make clean && make login, then make compute without CMAKE_OPTS=-DVECTORCACHE_BUILD_TOOLS=OFF"
		exit 1
	fi
	cmake --build $(BUILD_DIR) --target vectorcache_tests ingest-bench query-bench -j$(JOBS)
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	INGEST_BENCH=""
	for candidate in \
		"$(BUILD_DIR_ABS)/ingest-bench" \
		"$(BUILD_DIR_ABS)/$(BUILD_TYPE)/ingest-bench"; do
		if [ -x "$$candidate" ]; then
			INGEST_BENCH="$$candidate"
			break
		fi
	done
	if [ -z "$$INGEST_BENCH" ]; then
		INGEST_BENCH="$$(find "$(BUILD_DIR_ABS)" -maxdepth 3 \
			\( -name 'ingest-bench' -o -name 'ingest-bench.exe' \) -type f -print -quit 2>/dev/null || true)"
	fi
	if [ -z "$$INGEST_BENCH" ] || [ ! -f "$$INGEST_BENCH" ]; then
		echo "ingest-bench not found under $(BUILD_DIR_ABS)."
		echo "Inspect the build log above for ingest-bench compile/link errors."
		exit 1
	fi
	if [ -n "$(strip $(NPY))" ]; then
		"$$INGEST_BENCH" --npy $(NPY) --data-dir $(DATA_DIR) $(INGEST_BENCH_ARGS)
	else
		"$$INGEST_BENCH" --dataset $(DATASET) --data-dir $(DATA_DIR) $(INGEST_BENCH_ARGS)
	fi
	QUERY_BENCH=""
	for candidate in \
		"$(BUILD_DIR_ABS)/query-bench" \
		"$(BUILD_DIR_ABS)/$(BUILD_TYPE)/query-bench"; do
		if [ -x "$$candidate" ]; then
			QUERY_BENCH="$$candidate"
			break
		fi
	done
	if [ -z "$$QUERY_BENCH" ]; then
		QUERY_BENCH="$$(find "$(BUILD_DIR_ABS)" -maxdepth 3 \
			\( -name 'query-bench' -o -name 'query-bench.exe' \) -type f -print -quit 2>/dev/null || true)"
	fi
	if [ -z "$$QUERY_BENCH" ] || [ ! -f "$$QUERY_BENCH" ]; then
		echo "query-bench not found under $(BUILD_DIR_ABS)."
		echo "Inspect the build log above for query-bench compile/link errors."
		exit 1
	fi
	if [ -n "$(strip $(NPY))" ]; then
		"$$QUERY_BENCH" --npy $(NPY) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	else
		"$$QUERY_BENCH" --dataset $(DATASET) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	fi

clean:
	rm -rf $(BUILD_DIR)
