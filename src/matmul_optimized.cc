#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/tensor.h"

#include <stdexcept>

namespace mini_llama {

// 获取默认模式
MatmulMode DefaultMatmulMode() {
    return MatmulMode::kThreadedSimd;
}


static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
    int m = a.shape[0];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum_val = 0.0f;
            for (int k = 0; k < a.shape[1]; ++k) {
                sum_val += a.data[i * a.shape[1] + k] * b.data[k * n + j];
            }
            c.data[i * n + j] = sum_val;
        }
    }
    return c;
}


static Tensor MatmulThreaded(const Tensor&a, const Tensor& b) {

}


static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.num_dims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}, 0.0f);

    for (int j = 0; j < out_features; ++j) {
        float sum_val = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum_val += x.data[k] * weight.data[j * in_features + k];
        }
        result.data[j] = sum_val;
    }
    return result;
}


static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {

}



// matmul的分发模式
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
    if (a.num_dims() != 2 || b.num_dims() != 2) {
        throw std::runtime_error("MatmulDispatch: expected 2D tensors");
    }

    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("MatmulDispatch: dimension mismatch");
    }

    switch(mode) {
        case MatmulMode::kNaive : 
            return MatmulNaive(a, b);
        case MatmulMode::kThreaded:  
            return MatmulNaive(a, b);
        case MatmulMode::kSimd : 
        case MatmulMode::kThreadedSimd : 
            return MatmulNaive(a, b);
    }
    return MatmulNaive(a, b);
}

// linear的分发模式
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
    if (weight.num_dims() != 2) {
        throw std::runtime_error("LinearDispatch: expected weight shape [out_features, in_features]");
    }
    int in_features;
    if (x.num_dims() == 1) {
        in_features = x.shape[0];
    } else if (x.num_dims() == 2 && x.shape[0] == 1) {
        in_features = x.shape[1];
    } else {
        throw std::runtime_error("LinearDispatch: expected x shape [in_features] or [1, in_features]");
    }

    if (weight.shape[1] != in_features) {
        throw std::runtime_error("LinearDispatch: dimension mismatch");
    }

    switch(mode) {
        case MatmulMode::kNaive : 
            return LinearNaive(x, weight);
        case MatmulMode::kThreaded :
            return LinearNaive(x, weight); 
        case MatmulMode::kSimd :
            return LinearNaive(x, weight);
        case MatmulMode::kThreadedSimd :
            return LinearNaive(x, weight);
    }
    return LinearNaive(x, weight);
}


}