#include "mini_llama/batch.h"
#include "mini_llama/model.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"

namespace mini_llama {


int MiniBatch::num_tokens() const {
    return static_cast<int>(tokens.size());
}


// 构造只含有一个token的Batch，给单decode使用
MiniBatch MiniBatch::Single(int token, int pos) {
    MiniBatch batch;
    batch.tokens.push_back(token);
    batch.positions.push_back(pos);
    return batch;
}


// 构造多token的Batch，给prefill使用
MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
    MiniBatch batch;
    batch.tokens = toks;
    batch.positions.reserve(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.positions.push_back(start_pos + static_cast<int>(i));
    }
    return batch;
}


Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch) {
    if (batch.num_tokens() == 0) {
        throw std::runtime_error("ForwardBatch: empty batch");
    }
    if (batch.tokens.size() != batch.positions.size()) {
        throw std::runtime_error("ForwardBatch: tokens and positions size mismatch");
    }

    Tensor logits;
    for (int i = 0; i < batch.num_tokens(); ++i) {
        ctx.pos = batch.positions[i];
        logits = ForwardToken(ctx, model, batch.tokens[i]);
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

}