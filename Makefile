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
DATASET       ?= openai-1536
NPY           ?=
SPLIT         ?=
LIMIT         ?= 100000
BITS          ?= 4
RERANK_BITS   ?=
LANDMARKS     ?= 1024
QUERY_SPLIT   ?=
QUERY_LIMIT   ?=
K             ?= 64
RECALL        ?= 1
FORCE         ?=
BENCH_EXTRA_ARGS ?=
CMAKE_OPTS    ?=
# Bench parallelism: match published turbovec 8-vCPU cells.
# Ignore ambient cluster OMP_NUM_THREADS so `make compute` always gets 8 unless
# the user overrides on the make CLI (e.g. make compute OMP_NUM_THREADS=16).
ifeq ($(origin OMP_NUM_THREADS),environment)
OMP_NUM_THREADS := 8
endif
OMP_NUM_THREADS ?= 8
ENV_SCRIPT    := scripts/envs.sh
LOGIN_READY   := $(BUILD_DIR)/.login-ready

TURBOVEC_DIR      ?= .ref-turbovec
TURBOVEC_REPO     ?= https://github.com/RyanCodrai/turbovec.git
TURBOVEC_DATA_DIR ?= $(HOME)/data/py-turboquant
TURBOVEC_SUITE    ?= speed
TURBOVEC_THREADS  ?= mt
TURBOVEC_ARCH     ?=
# Published openai-1536 4-bit staged mt cell (~0.12 ms/q) used 8 vCPUs.
# Ignore ambient TURBOVEC_RAYON_THREADS / RAYON_NUM_THREADS so `make turbovec`
# always gets 8 unless overridden on the make CLI.
ifeq ($(origin TURBOVEC_RAYON_THREADS),environment)
TURBOVEC_RAYON_THREADS := 8
endif
TURBOVEC_RAYON_THREADS ?= 8
# Set to 1 to force whole-index 4-bit scan (TURBOVEC_4BIT_PLANES=0).
TURBOVEC_FULLSCAN ?=
SKIP_FAISS        ?=
TURBOVEC_STOCK    ?=
TURBOVEC_SCRIPT   := scripts/run_turbovec.sh
PYTHON            ?=

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
	$(call opt_arg,RERANK_BITS,rerank-bits) \
	$(call opt_arg,LANDMARKS,landmarks) \
	$(BENCH_EXTRA_ARGS)

QUERY_BENCH_ARGS = \
	$(BENCH_COMMON_ARGS) \
	$(call opt_arg,QUERY_SPLIT,query-split) \
	$(call opt_arg,QUERY_LIMIT,query-limit) \
	$(call opt_arg,K,k) \
	$(call flag_arg,RECALL,recall)

.PHONY: help login compute turbovec turbovec-setup clean

.DEFAULT_GOAL := help

help:
	@echo "VectorCache HPC targets:"
	@echo ""
	@echo "  make login          Configure CMake, fetch dependencies, download datasets (login node)"
	@echo "  make compute        Build, test, and query-bench offline (compute node)"
	@echo "  make turbovec-setup Login node: bootstrap Rust/maturin/deps, clone+build turbovec (needs net)"
	@echo "  make turbovec       Compute node: run official suite offline (needs prior turbovec-setup)"
	@echo "  make clean          Remove build directory"
	@echo ""
	@echo "Build / login variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  DATA_DIR=$(DATA_DIR)  DATASETS=$(DATASETS)  FORCE=$(FORCE)"
	@echo "  CMAKE_OPTS=$(CMAKE_OPTS)"
	@echo ""
	@echo "Bench variables (map to query-bench CLI):"
	@echo "  DATASET / NPY   --dataset or --npy (default DATASET=$(DATASET))"
	@echo "  LIMIT           --limit (default $(LIMIT); 0 = all)"
	@echo "  BITS            --bits / --scan-bits landmark FastScan (default $(BITS); 2-4)"
	@echo "  RERANK_BITS     --rerank-bits full-dim re-rank (default = BITS; 2-4)"
	@echo "  LANDMARKS       --landmarks (default $(LANDMARKS))"
	@echo "  K               --k top-k heap size (default $(K))"
	@echo "  QUERY_LIMIT     --query-limit"
	@echo "  QUERY_SPLIT     --query-split (test for glove/sift1m; holdout for openai)"
	@echo "  RECALL=1        --recall (Recall@1 + Recall@1@k + Recall@k; caches exact top-k under .cache/exact_topk/)"
	@echo "  OMP_NUM_THREADS=$(OMP_NUM_THREADS)  OpenMP pool for query-bench (default 8)"
	@echo "  BENCH_EXTRA_ARGS  appended to query-bench as-is"
	@echo ""
	@echo "Turbovec variables (make turbovec):"
	@echo "  Default = published openai-1536 4-bit staged mt cell (~0.12 ms/query)"
	@echo "  DATASET=$(DATASET)  BITS=$(BITS)  TURBOVEC_SUITE=$(TURBOVEC_SUITE)  (speed|recall)"
	@echo "  TURBOVEC_THREADS=$(TURBOVEC_THREADS)  (mt|st)  TURBOVEC_ARCH=$(TURBOVEC_ARCH)  (x86|arm|auto)"
	@echo "  TURBOVEC_RAYON_THREADS=$(TURBOVEC_RAYON_THREADS)  (mt Rayon pool; default 8)"
	@echo "  TURBOVEC_FULLSCAN=1  whole-index 4-bit scan (default: staged planes)"
	@echo "  TURBOVEC_DIR=$(TURBOVEC_DIR)  TURBOVEC_DATA_DIR=$(TURBOVEC_DATA_DIR)"
	@echo "  SKIP_FAISS=1    speed: turbovec-only (skip memory-heavy FAISS PQFastScan)"
	@echo "  TURBOVEC_STOCK=1  use upstream suite/*.py as-is (often OOMs)"
	@echo "  FORCE=1         rebuild turbovec wheel (login / turbovec-setup)"
	@echo ""
	@echo "Example: make login DATASETS=glove"
	@echo "Example: make login DATASETS=sift1m"
	@echo "Example: make compute"
	@echo "Example: make compute DATASET=openai-1536 BITS=2 K=64"
	@echo "Example: make compute DATASET=sift1m QUERY_LIMIT=10000"
	@echo "Example: make compute LANDMARKS=512 K=64 RECALL=1"
	@echo "Example: make compute BITS=2 LANDMARKS=1024 K=64"
	@echo "Example: make turbovec-setup                    # login node (internet)"
	@echo "Example: make turbovec                          # 4-bit staged mt (~0.12 ms cell)"
	@echo "Example: make turbovec ... SKIP_FAISS=1         # recommended on tight-RAM nodes"
	@echo "Example: make turbovec TURBOVEC_SUITE=recall BITS=4"
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
	# Pin after envs.sh so modules cannot leave a different pool size.
	export OMP_NUM_THREADS="$(OMP_NUM_THREADS)"
	export RAYON_NUM_THREADS="$(TURBOVEC_RAYON_THREADS)"
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
	echo "query-bench OMP_NUM_THREADS=$$OMP_NUM_THREADS"
	if [ -n "$(strip $(NPY))" ]; then
		"$$QUERY_BENCH" --npy $(NPY) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	else
		"$$QUERY_BENCH" --dataset $(DATASET) --data-dir $(DATA_DIR) $(QUERY_BENCH_ARGS)
	fi

# Login node (internet): clone + maturin wheel + wire DATA_DIR into TURBOVEC_DATA_DIR.
turbovec-setup:
	set -euo pipefail
	DATASET="$(DATASET)" BITS="$(BITS)" \
		TURBOVEC_DIR="$(TURBOVEC_DIR)" TURBOVEC_REPO="$(TURBOVEC_REPO)" \
		TURBOVEC_DATA_DIR="$(TURBOVEC_DATA_DIR)" DATA_DIR="$(DATA_DIR)" \
		FORCE="$(FORCE)" SETUP_ONLY=1 TURBOVEC_OFFLINE= \
		$(if $(strip $(PYTHON)),PYTHON="$(PYTHON)",) \
		bash "$(TURBOVEC_SCRIPT)"

# Compute node (offline): run official suite. Requires prior `make turbovec-setup`
# on a login node sharing this project path / Python env.
# Default: openai-1536 4-bit staged mt speed (~0.12 ms/query published cell).
turbovec:
	set -euo pipefail
	# Pin Rayon/OpenMP after any ambient cluster vars so the published 8-vCPU cell wins.
	export TURBOVEC_RAYON_THREADS="$(TURBOVEC_RAYON_THREADS)"
	export RAYON_NUM_THREADS="$(TURBOVEC_RAYON_THREADS)"
	export OMP_NUM_THREADS="$(TURBOVEC_RAYON_THREADS)"
	echo "turbovec RAYON_NUM_THREADS=$$RAYON_NUM_THREADS"
	DATASET="$(DATASET)" BITS="$(BITS)" \
		TURBOVEC_DIR="$(TURBOVEC_DIR)" TURBOVEC_REPO="$(TURBOVEC_REPO)" \
		TURBOVEC_DATA_DIR="$(TURBOVEC_DATA_DIR)" DATA_DIR="$(DATA_DIR)" \
		TURBOVEC_SUITE="$(TURBOVEC_SUITE)" TURBOVEC_THREADS="$(TURBOVEC_THREADS)" \
		TURBOVEC_ARCH="$(TURBOVEC_ARCH)" \
		TURBOVEC_RAYON_THREADS="$(TURBOVEC_RAYON_THREADS)" \
		RAYON_NUM_THREADS="$(TURBOVEC_RAYON_THREADS)" \
		OMP_NUM_THREADS="$(TURBOVEC_RAYON_THREADS)" \
		TURBOVEC_FULLSCAN="$(TURBOVEC_FULLSCAN)" \
		FORCE="$(FORCE)" SKIP_FAISS="$(SKIP_FAISS)" \
		TURBOVEC_STOCK="$(TURBOVEC_STOCK)" TURBOVEC_OFFLINE=1 \
		$(if $(strip $(PYTHON)),PYTHON="$(PYTHON)",) \
		bash "$(TURBOVEC_SCRIPT)"

clean:
	rm -rf $(BUILD_DIR)
