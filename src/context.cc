#include "mini_llama/context.h" 

namespace mini_llama {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model_ptr) :  model(model_ptr), pos(0){

}

}