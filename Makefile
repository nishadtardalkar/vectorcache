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
DATASET       ?= glove
NPY           ?=
SPLIT         ?=
LIMIT         ?= 100000
SEED          ?=
BITS          ?= 1
BLOCK_DIMS    ?= 1
NUM_PAIR_DIRS ?= 8
BIN_WIDTH     ?= 0.1
PROBE_RADIUS  ?= 1
BUCKET_SEED   ?=
NUM_PAIR_DIRS_LIST ?=
BIN_WIDTHS    ?=
PROBE_RADII   ?=
QUERY_SPLIT   ?=
QUERY_LIMIT   ?=
K             ?= 10
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
	$(call opt_arg,BITS,bits) \
	$(call opt_arg,BLOCK_DIMS,block-dims) \
	$(BENCH_EXTRA_ARGS)

INGEST_BENCH_ARGS = $(BENCH_COMMON_ARGS)

QUERY_BENCH_ARGS = \
	$(BENCH_COMMON_ARGS) \
	$(call opt_arg,QUERY_SPLIT,query-split) \
	$(call opt_arg,QUERY_LIMIT,query-limit) \
	$(call opt_arg,K,k) \
	$(call opt_arg,NUM_PAIR_DIRS,num-pair-dirs) \
	$(call opt_arg,BIN_WIDTH,bin-width) \
	$(call opt_arg,PROBE_RADIUS,probe-radius) \
	$(call opt_arg,BUCKET_SEED,bucket-seed) \
	$(call flag_arg,CALIBRATE,calibrate) \
	$(call flag_arg,RECALL,recall)

QUERY_TUNE_ARGS = \
	$(BENCH_COMMON_ARGS) \
	$(call opt_arg,QUERY_SPLIT,query-split) \
	$(call opt_arg,QUERY_LIMIT,query-limit) \
	$(call opt_arg,K,k) \
	$(call opt_arg,BUCKET_SEED,bucket-seed) \
	$(call opt_arg,NUM_PAIR_DIRS_LIST,num-pair-dirs-list) \
	$(call opt_arg,BIN_WIDTHS,bin-widths) \
	$(call opt_arg,PROBE_RADII,probe-radii)

.PHONY: help login compute tune clean

.DEFAULT_GOAL := help

help:
	@echo "VectorCache HPC targets:"
	@echo ""
	@echo "  make login    Configure CMake, fetch dependencies, download datasets (login node)"
	@echo "  make compute  Build, test, ingest-bench, and query-bench offline (compute node)"
	@echo "  make tune     Build and run query-bench-tune Pareto grid (compute node)"
	@echo "  make clean    Remove build directory"
	@echo ""
	@echo "Build / login variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASETS=$(DATASETS)  FORCE=$(FORCE)"
	@echo "  CMAKE_OPTS=$(CMAKE_OPTS)"
	@echo "  VECTORCACHE_SRHT_ROUNDS (cmake cache, default 2): set to 1 or 3 for multi-round SRHT"
	@echo ""
	@echo "Bench variables (map to ingest-bench / query-bench CLI):"
	@echo "  DATASET / NPY   --dataset or --npy (default DATASET=$(DATASET); pass NPY= to use a file)"
	@echo "  DATA_DIR        --data-dir (default $(DATA_DIR))"
	@echo "  SPLIT           --split"
	@echo "  LIMIT           --limit"
	@echo "  SEED            --seed"
	@echo "  BITS            --bits (default $(BITS); TurboQuantMSE bits/block, 1-8)"
	@echo "  BLOCK_DIMS      --block-dims (default $(BLOCK_DIMS); dims per codebook block, 1-16)"
	@echo "  NUM_PAIR_DIRS   --num-pair-dirs (default $(NUM_PAIR_DIRS); pair-hash 2D dirs L; query-bench)"
	@echo "  BIN_WIDTH       --bin-width (default $(BIN_WIDTH); bin width w on arcsine-CDF u; query-bench)"
	@echo "  PROBE_RADIUS    --probe-radius (default $(PROBE_RADIUS); 1D multi-probe P; query-bench)"
	@echo "  BUCKET_SEED     --bucket-seed (pair-hash seed; query-bench / query-bench-tune)"
	@echo "  NUM_PAIR_DIRS_LIST --num-pair-dirs-list (comma L values; query-bench-tune)"
	@echo "  BIN_WIDTHS      --bin-widths (comma w values; query-bench-tune)"
	@echo "  PROBE_RADII     --probe-radii (comma P values; query-bench-tune)"
	@echo "  QUERY_SPLIT     --query-split (query-bench / query-bench-tune)"
	@echo "  QUERY_LIMIT     --query-limit (query-bench / query-bench-tune)"
	@echo "  K               --k (default $(K); query-bench / query-bench-tune)"
	@echo "  CALIBRATE=1     --calibrate (query-bench only)"
	@echo "  RECALL=1        --recall (query-bench: Recall@1@k + Recall@k; caches exact top-k under .cache/exact_topk/)"
	@echo "  BENCH_EXTRA_ARGS  appended to benches as-is"
	@echo ""
	@echo "For native SIMD on compute nodes: make compute CMAKE_OPTS='-DCMAKE_CXX_FLAGS=-march=native'"
	@echo "For 3-round SRHT at compile time: make compute CMAKE_OPTS='-DVECTORCACHE_SRHT_ROUNDS=3'"
	@echo ""
	@echo "Example: make login DATASETS=glove"
	@echo "Example: make compute"
	@echo "Example: make compute BITS=2 RECALL=1"
	@echo "Example: make compute BITS=1 BLOCK_DIMS=2"
	@echo "Example: make compute NUM_PAIR_DIRS=8 BIN_WIDTH=0.1 PROBE_RADIUS=1"
	@echo "Example: make tune NUM_PAIR_DIRS_LIST=4,8,16 BIN_WIDTHS=0.1,0.2 PROBE_RADII=0,1,2"

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
		echo "DATASET or NPY is required, e.g. make compute"; \
		exit 1; \
	fi
	@if [ -n "$(strip $(NPY))" ] && [ -n "$(strip $(DATASET))" ] && [ "$(origin DATASET)" = "command line" ]; then \
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

tune:
	@if [ ! -f $(LOGIN_READY) ]; then \
		echo "Run 'make login' on a login node first."; \
		exit 1; \
	fi
	@if [ -z "$(strip $(DATASET)$(NPY))" ]; then \
		echo "DATASET or NPY is required, e.g. make tune"; \
		exit 1; \
	fi
	@if [ -n "$(strip $(NPY))" ] && [ -n "$(strip $(DATASET))" ] && [ "$(origin DATASET)" = "command line" ]; then \
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
		echo "VECTORCACHE_BUILD_TOOLS is $${TOOLS_STATUS:-unset}; query-bench-tune will not be built."
		exit 1
	fi
	cmake --build $(BUILD_DIR) --target query-bench-tune -j$(JOBS)
	QUERY_TUNE=""
	for candidate in \
		"$(BUILD_DIR_ABS)/query-bench-tune" \
		"$(BUILD_DIR_ABS)/$(BUILD_TYPE)/query-bench-tune"; do
		if [ -x "$$candidate" ]; then
			QUERY_TUNE="$$candidate"
			break
		fi
	done
	if [ -z "$$QUERY_TUNE" ]; then
		QUERY_TUNE="$$(find "$(BUILD_DIR_ABS)" -maxdepth 3 \
			\( -name 'query-bench-tune' -o -name 'query-bench-tune.exe' \) -type f -print -quit 2>/dev/null || true)"
	fi
	if [ -z "$$QUERY_TUNE" ] || [ ! -f "$$QUERY_TUNE" ]; then
		echo "query-bench-tune not found under $(BUILD_DIR_ABS)."
		exit 1
	fi
	if [ -n "$(strip $(NPY))" ]; then
		"$$QUERY_TUNE" --npy $(NPY) --data-dir $(DATA_DIR) $(QUERY_TUNE_ARGS)
	else
		"$$QUERY_TUNE" --dataset $(DATASET) --data-dir $(DATA_DIR) $(QUERY_TUNE_ARGS)
	fi

clean:
	rm -rf $(BUILD_DIR)
