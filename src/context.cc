#include "mini_llama/context.h" 

namespace mini_llama {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model_ptr) :  model(model_ptr), pos(0){
    if (model) {
        const auto& c = model->config;
        kv_cache = KvCache(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim);

        // GPU kvcache
        
    }
}

}