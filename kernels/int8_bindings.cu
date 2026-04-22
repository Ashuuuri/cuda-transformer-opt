// int8_bindings.cu — PyTorch Python bindings for INT8 kernels.
// Compiled together with int8_attention.cu, int8_mlp.cu, quant_utils.cu
// via torch.utils.cpp_extension.load() in tests/test_int8.py.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// Forward declarations (implementations in sibling .cu files)
extern "C" void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    float scale_Q, float scale_K, float scale_V,
    int batch, int heads, int seq_len, int head_dim
);

extern "C" void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, half* out,
    float scale_x, float scale_W1, float scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);

// ── Wrappers ───────────────────────────────────────────────────────────

torch::Tensor int8_attention_py(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale_Q, double scale_K, double scale_V
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kInt8, "Q must be int8");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must be (batch, heads, seq_len, head_dim)");

    int batch = Q.size(0);
    int heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(Q.device());
    torch::Tensor out = torch::empty({batch, heads, seq_len, head_dim}, opts);

    int8_attention_forward(
        Q.data_ptr<int8_t>(), K.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (float)scale_Q, (float)scale_K, (float)scale_V,
        batch, heads, seq_len, head_dim
    );
    return out;
}

torch::Tensor int8_mlp_py(
    torch::Tensor x, torch::Tensor W1, torch::Tensor W2,
    double scale_x, double scale_W1, double scale_W2
) {
    TORCH_CHECK(x.is_cuda() && W1.is_cuda() && W2.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8");
    TORCH_CHECK(x.is_contiguous() && W1.is_contiguous() && W2.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be (batch, seq_len, d_model)");

    int batch = x.size(0);
    int seq_len = x.size(1);
    int d_model = x.size(2);
    int d_ff = W1.size(1);

    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
    torch::Tensor out = torch::empty({batch, seq_len, d_model}, opts);

    int8_mlp_forward(
        x.data_ptr<int8_t>(), W1.data_ptr<int8_t>(), W2.data_ptr<int8_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (float)scale_x, (float)scale_W1, (float)scale_W2,
        batch, seq_len, d_model, d_ff
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward", &int8_attention_py, "INT8 attention forward (CUDA)");
    m.def("int8_mlp_forward", &int8_mlp_py, "INT8 MLP forward (CUDA)");
}
