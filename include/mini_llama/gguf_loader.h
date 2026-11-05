#pragma once

#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/quant.h"

namespace mini_llama {

MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}