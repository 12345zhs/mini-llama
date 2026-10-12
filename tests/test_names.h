#pragma once

#include "mini_llama/tensor.h"
#include "mini_llama/ops.h"


// tensor
using mini_llama::Tensor;
using mini_llama::MakeTensor1D;
using mini_llama::MakeTensor2D;
using mini_llama::MakeTensor3D;
using mini_llama::MakeTensor4D;

// ops
using mini_llama::Matmul;
using mini_llama::Linear;
using mini_llama::RmsNorm;
using mini_llama::Softmax;
using mini_llama::Silu;
using mini_llama::ElementwiseMul;
using mini_llama::SwiGlu;
using mini_llama::Rope;
using mini_llama::RopeType;
using mini_llama::ArgMax;
