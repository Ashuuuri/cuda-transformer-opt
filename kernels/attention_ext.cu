// attention_ext.cu — PyTorch/pybind11 binding for the FP16 attention kernel.
//
// Exposes attention_forward to Python via torch.utils.cpp_extension.
// Usage in Python:
//
//   from torch.utils.cpp_extension import load
//   attn_ext = load(
//       name="attention_ext",
//       sources=["kernels/attention.cu", "kernels/attention_ext.cu"],
//       extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O2"],
//       verbose=False,
//   )
//   out = attn_ext.attention_forward(Q, K, V)  # Q/K/V: (batch, heads, seq, head_dim) fp16 CUDA

#include <torch/extension.h>

// Forward-declare the kernel interface defined in attention.cu.
void attention_forward(
    const half* Q, const half* K, const half* V, half* out,
    int batch, int heads, int seq_len, int head_dim
);

torch::Tensor attention_forward_ext(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.device().is_cuda(),             "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16,     "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16,     "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16,     "V must be float16");
    TORCH_CHECK(Q.is_contiguous(),                "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(),                "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(),                "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4,                     "expected shape (batch, heads, seq_len, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V must have the same shape");

    const int batch    = Q.size(0);
    const int heads    = Q.size(1);
    const int seq_len  = Q.size(2);
    const int head_dim = Q.size(3);

    auto out = torch::empty_like(Q);

    attention_forward(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<      half*>(out.data_ptr<at::Half>()),
        batch, heads, seq_len, head_dim
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward_ext,
          "FP16 fused attention (CUDA) — shape (batch, heads, seq_len, head_dim)");
}
