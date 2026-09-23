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
BITS          ?= 4
QUERY_SPLIT   ?=
QUERY_LIMIT   ?=
K             ?= 10
CALIBRATE     ?=
RECALL        ?=
BUCKETED      ?=
SCAN_FRACTION ?=
VAR_THRESHOLD ?=
FORCE         ?=
BENCH_EXTRA_ARGS ?=
CMAKE_OPTS    ?=
ENV_SCRIPT    := scripts/envs.sh
LOGIN_READY   := $(BUILD_DIR)/.login-ready

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

opt_arg = $(if $(strip $($(1))),--$(2) $($(1)),)
flag_arg = $(if $(filter 1 ON on true TRUE yes YES,$($(1))),--$(2),)

BENCH_COMMON_ARGS = \
	$(call opt_arg,SPLIT,split) \
	$(call opt_arg,LIMIT,limit) \
	$(call opt_arg,BITS,bits) \
	$(BENCH_EXTRA_ARGS)

QUERY_BENCH_ARGS = \
	$(BENCH_COMMON_ARGS) \
	$(call opt_arg,QUERY_SPLIT,query-split) \
	$(call opt_arg,QUERY_LIMIT,query-limit) \
	$(call opt_arg,K,k) \
	$(call flag_arg,CALIBRATE,calibrate) \
	$(call flag_arg,RECALL,recall) \
	$(call flag_arg,BUCKETED,bucketed) \
	$(call opt_arg,SCAN_FRACTION,scan-fraction) \
	$(call opt_arg,VAR_THRESHOLD,var-threshold)

.PHONY: help login compute clean

.DEFAULT_GOAL := help

help:
	@echo "VectorCache HPC targets:"
	@echo ""
	@echo "  make login    Configure CMake, fetch dependencies, download datasets (login node)"
	@echo "  make compute  Build, test, and query-bench offline (compute node)"
	@echo "  make clean    Remove build directory"
	@echo ""
	@echo "Build / login variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASETS=$(DATASETS)  FORCE=$(FORCE)"
	@echo "  CMAKE_OPTS=$(CMAKE_OPTS)"
	@echo ""
	@echo "Bench variables (map to query-bench CLI):"
	@echo "  DATASET / NPY   --dataset or --npy (default DATASET=$(DATASET))"
	@echo "  LIMIT           --limit (default $(LIMIT))"
	@echo "  BITS            --bits (default $(BITS); 2-4)"
	@echo "  K               --k (default $(K))"
	@echo "  QUERY_LIMIT     --query-limit"
	@echo "  QUERY_SPLIT     --query-split (test for glove; holdout for openai)"
	@echo "  CALIBRATE=1     --calibrate (TQ+)"
	@echo "  RECALL=1        --recall (Recall@1@k + Recall@k; caches exact top-k under .cache/exact_topk/)"
	@echo "  BUCKETED=1      --bucketed (streaming cosine k-means IVF)"
	@echo "  SCAN_FRACTION   --scan-fraction (bucketed; default 0.1)"
	@echo "  VAR_THRESHOLD   --var-threshold (bucketed; default 0.5)"
	@echo "  BENCH_EXTRA_ARGS  appended to query-bench as-is"
	@echo ""
	@echo "Example: make login DATASETS=glove"
	@echo "Example: make compute"
	@echo "Example: make compute DATASET=openai-1536 BITS=2 K=64"
	@echo "Example: make compute BITS=4 RECALL=1 CALIBRATE=1"
	@echo "Example: make compute BUCKETED=1 SCAN_FRACTION=0.1"
	@echo "For native SIMD: make compute CMAKE_OPTS='-DCMAKE_CXX_FLAGS=-march=native'"

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
		echo "VECTORCACHE_BUILD_TOOLS is $${TOOLS_STATUS:-unset}; query-bench will not be built."
		exit 1
	fi
	cmake --build $(BUILD_DIR) --target vectorcache_tests query-bench -j$(JOBS)
	ctest --test-dir $(BUILD_DIR) --output-on-failure
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
		exit 1
	fi
	if [ -n "$(strip $(NPY))" ]; then
		"$$QUERY_BENCH" --npy $(NPY) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	else
		"$$QUERY_BENCH" --dataset $(DATASET) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	fi

clean:
	rm -rf $(BUILD_DIR)
