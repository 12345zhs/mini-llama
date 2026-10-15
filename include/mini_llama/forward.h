#pragma once

#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/batch.h"

namespace mini_llama {

// 单token推理接口，处理当前位置的一个token，并输出logits
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);


// 批量处理接口，顺序调用ForwardToken处理batch数据，更新状态
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch);

// prefill与docode共用同一个接口，prefill传入多个token，而decode之传入刚采样的单个token

}