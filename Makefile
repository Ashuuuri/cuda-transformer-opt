# Makefile for CUDA kernel compilation and testing
# Usage:
#   make test_attention    — compile attention kernel test (all optimizations on)
#   make test_mlp          — compile MLP kernel test
#   make test_int8         — compile INT8 kernel test
#   make all               — compile all
#   make clean             — remove compiled binaries
#
# Ablation flags (attention kernel):
#   DFLAGS="-DATTN_WMMA=0"         use scalar fused kernel (no Tensor Cores)
#   DFLAGS="-DATTN_FAST_MATH=0"    use expf instead of __expf
#   Example: make test_attention DFLAGS="-DATTN_WMMA=0 -DATTN_FAST_MATH=0"

NVCC       = nvcc
NVCC_FLAGS = -arch=sm_80 --std=c++17 -O3
INCLUDES   = -I kernels -I tests/cuda
DFLAGS     ?=

.PHONY: all clean

all: test_attention test_mlp test_int8

# ── Attention (Jonathan) ────────────────────────────────────────────────
test_attention: kernels/attention.cu tests/cuda/test_attention.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(DFLAGS) \
		kernels/attention.cu \
		tests/cuda/test_attention.cu \
		-o test_attention

# Convenience: build ablation variants into separate binaries
ablation_attention: kernels/attention.cu tests/cuda/test_attention.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
		kernels/attention.cu tests/cuda/test_attention.cu \
		-o test_attention_wmma
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DATTN_WMMA=0 \
		kernels/attention.cu tests/cuda/test_attention.cu \
		-o test_attention_fused

# ── MLP (Shengjing) ────────────────────────────────────────────────────
test_mlp: kernels/mlp.cu tests/cuda/test_mlp.cu
	$(NVCC) $(NVCC_FLAGS) --use_fast_math -DMLP_STAGE_K=32 $(INCLUDES) \
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
	rm -f test_attention test_mlp test_int8 \
	      test_attention_wmma test_attention_fused
