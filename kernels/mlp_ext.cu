// mlp_ext.cu — PyTorch / pybind11 binding for mlp_forward.
//
// This file is compiled only when loading via torch.utils.cpp_extension.load().
// The pure-CUDA test (make test_mlp) uses mlp.cu directly and never touches this.
//
// Exposes a single Python-callable function:
//   mlp_ext.mlp_forward(x, W1, W2) -> Tensor
//   All tensors must be float16, contiguous, on CUDA.

#include <torch/extension.h>
#include <cuda_fp16.h>

// ── Forward declaration (implemented in kernels/mlp.cu) ────────────────
void mlp_forward(
    const half* x, const half* W1, const half* W2, half* out,
    int batch, int seq_len, int d_model, int d_ff
);

// ── Python-facing wrapper ──────────────────────────────────────────────
torch::Tensor mlp_forward_torch(
    torch::Tensor x,    // (batch, seq_len, d_model)  float16  CUDA
    torch::Tensor W1,   // (d_model, d_ff)             float16  CUDA
    torch::Tensor W2    // (d_ff, d_model)              float16  CUDA
) {
    TORCH_CHECK(x.is_cuda(),  "x must be a CUDA tensor");
    TORCH_CHECK(W1.is_cuda(), "W1 must be a CUDA tensor");
    TORCH_CHECK(W2.is_cuda(), "W2 must be a CUDA tensor");
    TORCH_CHECK(x.dtype()  == torch::kFloat16, "x must be float16");
    TORCH_CHECK(W1.dtype() == torch::kFloat16, "W1 must be float16");
    TORCH_CHECK(W2.dtype() == torch::kFloat16, "W2 must be float16");
    TORCH_CHECK(x.is_contiguous(),  "x must be contiguous");
    TORCH_CHECK(W1.is_contiguous(), "W1 must be contiguous");
    TORCH_CHECK(W2.is_contiguous(), "W2 must be contiguous");
    TORCH_CHECK(x.dim()  == 3, "x must be 3-D  (batch, seq_len, d_model)");
    TORCH_CHECK(W1.dim() == 2, "W1 must be 2-D (d_model, d_ff)");
    TORCH_CHECK(W2.dim() == 2, "W2 must be 2-D (d_ff, d_model)");

    const int batch   = x.size(0);
    const int seq_len = x.size(1);
    const int d_model = x.size(2);
    const int d_ff    = W1.size(1);

    TORCH_CHECK(W1.size(0) == d_model, "W1 shape mismatch");
    TORCH_CHECK(W2.size(0) == d_ff,    "W2 shape mismatch");
    TORCH_CHECK(W2.size(1) == d_model, "W2 shape mismatch");

    torch::Tensor out = torch::empty_like(x);

    mlp_forward(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W2.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        batch, seq_len, d_model, d_ff
    );

    return out;
}

// ── Module registration ────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "mlp_forward",
        &mlp_forward_torch,
        "Fused FP16 MLP forward: out = Linear(GELU(Linear(x, W1)), W2)"
    );
}
