#pragma once

#include "mini_llama/model.h"

namespace mini_llama {

MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}