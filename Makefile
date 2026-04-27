# Makefile for CUDA kernel compilation and testing
# Usage:
#   make test_attention    — compile attention kernel test
#   make test_mlp          — compile MLP kernel test
#   make test_int8         — compile INT8 kernel test
#   make all               — compile all
#   make clean             — remove compiled binaries

NVCC       = nvcc
NVCC_FLAGS = -arch=sm_80 --std=c++17 -O3
INCLUDES   = -I kernels -I tests/cuda

.PHONY: all clean

all: test_attention test_mlp test_int8

# ── Attention (Jonathan) ────────────────────────────────────────────────
test_attention: kernels/attention.cu tests/cuda/test_attention.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
		kernels/attention.cu \
		tests/cuda/test_attention.cu \
		-o test_attention

# ── MLP (Shengjing) ────────────────────────────────────────────────────
test_mlp: kernels/mlp.cu tests/cuda/test_mlp.cu
	$(NVCC) $(NVCC_FLAGS) --use_fast_math $(INCLUDES) \
		kernels/mlp.cu \
		tests/cuda/test_mlp.cu \
		-o test_mlp

# ── INT8 (Heling) ──────────────────────────────────────────────────────
test_int8: kernels/int8_attention.cu kernels/int8_mlp.cu kernels/quant_utils.cu tests/cuda/test_int8.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
		kernels/int8_attention.cu \
		kernels/int8_mlp.cu \
		kernels/quant_utils.cu \
		tests/cuda/test_int8.cu \
		-o test_int8

clean:
	rm -f test_attention test_mlp test_int8
