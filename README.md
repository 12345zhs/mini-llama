# mini-llama
A concise C++ inference kernel

## 推理链路_cuda

本章节详述大模型推理框架在执行 GPU（CUDA）硬件加速时的完整数据流转链路。整个推理过程遵循“数据在显存内驻留（Device-Resident）”的原则，最大限度减少跨总线（PCIe）的数据搬运。

### 1. 执行路径分流

推理的入口位于 `ForwardToken` 函数。在完成基础校验后，系统首先按需初始化 GPU 侧的 KV Cache 状态：

```cpp
if (model.cuda_weights && ctx.cuda_kv_cache.empty()) {
  ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim, model.cuda_weights->device_id);
}
```

如果模型权重已成功加载至 GPU（即 `model.cuda_weights` 不为空），控制流将直接切入专门的 CUDA 运算分支：

```cpp
if (model.cuda_weights) {
  CudaTensor x_dev = EmbedTokenDevice(model, token);
  for (int layer = 0; layer < n_layers; ++layer) {
    x_dev = ForwardLayerDevice(ctx, model, x_dev, layer, model.layers[layer], c);
  }
  return ComputeLogitsDevice(x_dev, model, c);
}
```

未加载 GPU 权重时，系统将回退至传统的 CPU 备用逻辑（`EmbedToken` 与 `ForwardLayer`）。`model.cuda_weights` 指针是否有效，是判断是否启用 CUDA 主链路的唯一开关。

### 2. 链路起点：Embedding Lookup 执行

进入 CUDA 分支后，第一步是对输入的 token 进行词嵌入（Embedding）查表：

```cpp
CudaTensor x_dev = EmbedTokenDevice(model, token);
```

`EmbedTokenDevice` 的执行逻辑是从已上传至 GPU 的 `token_embedding` 权重中，直接按索引拉取对应 token id 的那一行数据：

```cpp
const CudaUploadedWeight* cuda_weight = FindCudaWeight(model, "token_embedding", QuantType::kF32);
return CudaEmbeddingLookupDeviceWeight(cuda_weight->buffer.data(), cuda_weight->shape, token_id, CudaDeviceId(model));
```

获取到的 `x_dev` 作为当前 token 的隐藏状态（hidden states），将持续留在显存中，作为后续 Transformer 层的输入。

### 3. 核心循环：单层 Device Forward 的数据流

接下来，遍历模型的每一层 Transformer Block，调用 `ForwardLayerDevice`。其内部的数据流向依次为：Attention RMSNorm -> QKV Projection -> RoPE -> KV Cache 写入 -> CUDA Attention -> WO Projection -> 残差连接 -> FFN RMSNorm -> FFN 投影 (Gate/Up/Down) -> SiLU + Mul -> 残差连接。

对应的代码实现如下：

```cpp
CudaTensor h = ForwardRmsNormDevice(...);

CudaTensor q;
CudaTensor k;
CudaTensor v;
ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q, k, v);
ForwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos, c.rope_theta, c.rope_type);

CudaTensor attn_out = AttentionForwardDevice(...);
CudaTensor attn_proj = ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);

CudaTensor h2 = ForwardRmsNormDevice(...);
CudaTensor ff = FfnForwardDevice(model, layer_prefix, h2, lw);
return ForwardAddDevice(model, x_attn, ff);
```

在此链路中，所有关键算子的输入和输出均为 `CudaTensor`，数据完全在显存中流转，无中间状态回传至 CPU。

#### 3.1 线性投射：QKV Projection 的分发机制

在执行 Q/K/V 投射时，系统调用支持 device input 的 linear 接口。根据当前权重具体的量化类型，函数内部将分发到对应的计算内核：

```cpp
// 视权重类型分发至以下算子之一：
CudaLinearDeviceInput(...);      // 浮点权重
CudaQ80LinearDeviceInput(...);   // Q8_0 量化权重
CudaQ40LinearDeviceInput(...);   // Q4_0 量化权重
CudaQ41LinearDeviceInput(...);   // Q4_1 量化权重
```

对于带有 attention bias 的模型（如 Qwen2），在得到线性投影结果后，系统会进一步调用逐元素加法算子，确保计算链路完全驻留 GPU：

```cpp
CudaElementwiseAddDeviceWeight(...);
```

#### 3.2 位置编码与缓存：RoPE 与 KV Cache 写入

投影完成后，对 Q 和 K 执行原地的 RoPE 位置编码旋转：

```cpp
CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta, rope_type, CudaDeviceId(model));
```

旋转完毕的 K 和 V 需要写入 GPU 侧的 KV Cache 对应位置：

```cpp
ctx.cuda_kv_cache.WriteDevice(layer, pos, k, v);
```

该过程是纯粹的 Device-to-Device 内部数据拷贝，效率极高。同时，拷贝的数据字节数会被累加到运行时统计项 `cuda kv write bytes` 中。

#### 3.3 注意力计算：CUDA Attention 与 WO 集成

计算 Attention 时，读取当前层已有的 KV 历史状态，并执行并行计算：

```cpp
CudaTensor attn_out = CudaAttentionDecodeDeviceInput(q, ctx.cuda_kv_cache, layer, pos, n_heads, n_kv_heads, head_dim, CudaDeviceId(model));
```

计算输出的 attention 结果被送入 `wo` 线性投射层进行降维：

```cpp
CudaTensor attn_proj = ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
```

最后，通过残差连接与输入 `x` 进行叠加：

```cpp
CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);
```

#### 3.4 前馈网络：FFN 的 Device 驻留路径

随后的 FFN 模块同样遵循严格的内存驻留原则，全程在显存内完成：

```cpp
CudaTensor gate = ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
CudaTensor silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
CudaTensor ff = CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
```

执行过程中的 `linear calls` 与 `activation calls` 会同步记录至运行时计数器，便于性能监控。

### 4. 数据回传：Logits 与 CPU 侧的交接边界

经过所有 Transformer 层后，系统执行最终的 Logits 计算：

```cpp
static Tensor ComputeLogitsDevice(const CudaTensor& x, const MiniLlamaModel& model, const ModelConfig& c) {
  CudaTensor normed = ForwardRmsNormDevice(model, "final_norm", x, model.final_norm, c.rms_norm_eps);
  CudaTensor logits_dev = ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
  Tensor logits = DownloadCudaTensor(model, logits_dev);
  return logits.ReshapeChecked({c.vocab_size}, "compute_logits_device");
}
```

在此步骤中，`final_norm` 和 `lm_head` 依然在 GPU 上完成。只有在得到最终的 `logits_dev` 之后，系统才调用 `DownloadCudaTensor` 将其拉回 CPU，交由 CPU 侧的 Sampler 进行下一步采样。

由此形成了明晰的计算边界：

*   **GPU 区块**：Embedding -> Transformer Layers -> Final Norm -> LM Head
*   **CPU 区块**：Sampler -> Terminal I/O

## 构建

项目使用 CMake 和 C++17。以下命令均从项目根目录执行。

### CPU 版本

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
```

终端结果：

```text
-- Configuring done
-- Generating done
-- Build files have been written to: /home/dministrator/mini_llama/mini_llama_cpp/build
```

```bash
cmake --build build --target mini-llama -j2
```

终端结果：

```text
[100%] Built target mini-llama
```

```bash
cmake --build build --target mini-llama-tests -j2
```

终端结果：

```text
[100%] Built target mini-llama-tests
```

```bash
build/mini-llama-tests
```

终端结果：

```text
Running223 test...
=======================================
Results: 223 passed, 0 failed
```

### CUDA 版本

需要 NVIDIA GPU 和 CUDA Toolkit。RTX 4060 的计算能力为 8.9；下面的 nvcc 路径对应本文记录的 WSL 环境，可按本机 Toolkit 安装路径调整。

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

终端结果：

```text
-- The CUDA compiler identification is NVIDIA 13.1.115
-- Found CUDAToolkit: /usr/local/cuda-13.1/targets/x86_64-linux/include (found version "13.1.115")
-- Configuring done (9.9s)
-- Generating done (0.0s)
-- Build files have been written to: /home/dministrator/mini_llama/mini_llama_cpp/build
```

```bash
cmake --build build --target mini-llama -j2
```

终端结果：

```text
[100%] Built target mini-llama
```

CUDA 构建后，可用 `--backend cuda --device 0` 选择第 0 个设备。省略 `--backend` 时默认使用 CPU。

## 命令行模式

以下均为本次终端记录的完整输出，包括逐 token trace 行。命令按帮助菜单顺序排列。

### 帮助

```bash
build/mini-llama -h
```

终端结果：

```text
Usage: build/mini-llama <command> [options]
Commands:
  generate      One-shot text generation (default)
  run           Interactive chat mode
  inspect       Inspect model metadata
  inspect-gguf  Inspect GGUF file
  bench         Benchmark inference performance
  --cuda-info   Print CUDA device information for the selected CUDA build

Run "build/mini-llama generate --help", "build/mini-llama run --help", or "build/mini-llama bench --help" for details.
```

### generate：单次生成

`models/tiny` 是用于测试执行链路的教学模型，权重为随机值，可能生成空文本或没有语义的 token；它不适合评估回答质量。

```bash
build/mini-llama generate --model models/tiny/model.bin --config models/tiny/model.json --prompt "hello" -n 64 --seed 1 --backend cpu
```

终端结果：

```text
mini-llama.cpp
==============

backend: cpu
compute: cpu
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
prompt: hello
tokens: [1 104 101 108 108 111]
sampling: temperature=0, top_k=0, seed=1
quant: model-native
threads: 16

prefill...
decode loop...

generated tokens: [23, 58, 36, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59]
generated text: ""
trace event: trace_id=req_20261105161748 _ 000001 stage=model_load tokens=0 elapsed_ms=1.003 detail="models/tiny/model.bin"
trace event: trace_id=req_20261105161748 _ 000001 stage=quantize tokens=0 elapsed_ms=0.000 detail="model-naive"
trace event: trace_id=req_20261105161748 _ 000001 stage=tokenize tokens=0 elapsed_ms=0.002 detail="prompt"
trace event: trace_id=req_20261105161748 _ 000001 stage=prefill tokens=6 elapsed_ms=1.382 detail="batch"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.217 detail="pos = 6"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.217 detail="pos = 7"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.219 detail="pos = 8"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.228 detail="pos = 9"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.227 detail="pos = 10"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.225 detail="pos = 11"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.242 detail="pos = 12"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.275 detail="pos = 13"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.253 detail="pos = 14"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.229 detail="pos = 15"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.231 detail="pos = 16"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.238 detail="pos = 17"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.231 detail="pos = 18"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.241 detail="pos = 19"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.253 detail="pos = 20"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.248 detail="pos = 21"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.238 detail="pos = 22"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.372 detail="pos = 23"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.250 detail="pos = 24"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.260 detail="pos = 25"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.254 detail="pos = 26"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.271 detail="pos = 27"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.285 detail="pos = 28"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.393 detail="pos = 29"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.335 detail="pos = 30"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.277 detail="pos = 31"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.267 detail="pos = 32"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.264 detail="pos = 33"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.257 detail="pos = 34"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.270 detail="pos = 35"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.265 detail="pos = 36"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.271 detail="pos = 37"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.285 detail="pos = 38"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.278 detail="pos = 39"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.275 detail="pos = 40"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.264 detail="pos = 41"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.360 detail="pos = 42"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.277 detail="pos = 43"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.274 detail="pos = 44"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.277 detail="pos = 45"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.272 detail="pos = 46"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.268 detail="pos = 47"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.278 detail="pos = 48"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.297 detail="pos = 49"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.287 detail="pos = 50"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.279 detail="pos = 51"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.303 detail="pos = 52"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.286 detail="pos = 53"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.289 detail="pos = 54"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.286 detail="pos = 55"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.311 detail="pos = 56"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.296 detail="pos = 57"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.294 detail="pos = 58"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=2.724 detail="pos = 59"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.314 detail="pos = 60"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.685 detail="pos = 61"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.293 detail="pos = 62"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.305 detail="pos = 63"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.314 detail="pos = 64"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.306 detail="pos = 65"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.299 detail="pos = 66"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.318 detail="pos = 67"
trace event: trace_id=req_20261105161748 _ 000001 stage=decode tokens=1 elapsed_ms=0.288 detail="pos = 68"
trace event: trace_id=req_20261105161748 _ 000001 stage=sample tokens=64 elapsed_ms=0.192 detail="generated_tokens"
trace summary: trace_id=req_20261105161748 _ 000001 mode=generate status=ok backend=cpu model=models/tiny/model.bin prompt_tokens=0 prefill_tokens=6 decode_tokens=63 generated_tokens=64 model_load_ms=1.003 tokenizer_ms=0.002 prefill_ms=1.382 decode_ms=20.185 sample_ms=0.192 total_ms=23.138
```

```bash
build/mini-llama generate --model models/tiny/model.bin --config models/tiny/model.json --prompt "hello" -n 64 --seed 1 --backend cuda --device 0
```

终端结果：

```text
mini-llama.cpp
==============

backend: cuda
cuda: device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, total memory 7.99548 GB, cuda runtime 13010, driver 13010
compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA attention over GPU KV cache + device-resident Decode path; CPU sampler
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
prompt: hello
tokens: [1 104 101 108 108 111]
sampling: temperature=0, top_k=0, seed=1
quant: model-native
threads: 16

uploaded weights: 21
gpu memory used: 132224 bytes (0.13MB)
cuda f32 Linear weights: 15/15

prefill...
decode loop...

generated tokens: [23, 58, 36, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59]
generated text: ""

cuda Linear calls: 1035
cuda activation calls: 1035
cuda attention calls: 138
cpu attention fallback calls: 0
cuda kv write bytes: 9043968
cuda kv read bytes: 1236480
host->device copies: 0 (0 bytes)
device->host copies: 69 (35328 bytes)
trace event: trace_id=req_20261105161801 _ 000001 stage=model_load tokens=0 elapsed_ms=0.561 detail="models/tiny/model.bin"
trace event: trace_id=req_20261105161801 _ 000001 stage=quantize tokens=0 elapsed_ms=0.000 detail="model-naive"
trace event: trace_id=req_20261105161801 _ 000001 stage=tokenize tokens=0 elapsed_ms=0.005 detail="prompt"
trace event: trace_id=req_20261105161801 _ 000001 stage=prefill tokens=6 elapsed_ms=164.532 detail="batch"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.276 detail="pos = 6"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=14.825 detail="pos = 7"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.225 detail="pos = 8"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.000 detail="pos = 9"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.378 detail="pos = 10"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.882 detail="pos = 11"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.082 detail="pos = 12"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=10.704 detail="pos = 13"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.625 detail="pos = 14"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.145 detail="pos = 15"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.132 detail="pos = 16"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=10.834 detail="pos = 17"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.159 detail="pos = 18"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.225 detail="pos = 19"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.438 detail="pos = 20"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.885 detail="pos = 21"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.784 detail="pos = 22"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.449 detail="pos = 23"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.623 detail="pos = 24"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.324 detail="pos = 25"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.176 detail="pos = 26"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.819 detail="pos = 27"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.454 detail="pos = 28"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.706 detail="pos = 29"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.121 detail="pos = 30"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.240 detail="pos = 31"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.115 detail="pos = 32"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.825 detail="pos = 33"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.326 detail="pos = 34"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.276 detail="pos = 35"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.538 detail="pos = 36"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.004 detail="pos = 37"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.489 detail="pos = 38"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.833 detail="pos = 39"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.024 detail="pos = 40"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.267 detail="pos = 41"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.384 detail="pos = 42"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.917 detail="pos = 43"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.706 detail="pos = 44"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.309 detail="pos = 45"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.238 detail="pos = 46"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=10.801 detail="pos = 47"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.498 detail="pos = 48"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.731 detail="pos = 49"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.628 detail="pos = 50"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.477 detail="pos = 51"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=13.437 detail="pos = 52"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.406 detail="pos = 53"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=13.234 detail="pos = 54"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.480 detail="pos = 55"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.840 detail="pos = 56"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.422 detail="pos = 57"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.376 detail="pos = 58"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.183 detail="pos = 59"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=13.858 detail="pos = 60"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.071 detail="pos = 61"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.478 detail="pos = 62"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.733 detail="pos = 63"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.154 detail="pos = 64"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.731 detail="pos = 65"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.857 detail="pos = 66"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=11.144 detail="pos = 67"
trace event: trace_id=req_20261105161801 _ 000001 stage=decode tokens=1 elapsed_ms=12.363 detail="pos = 68"
trace event: trace_id=req_20261105161801 _ 000001 stage=sample tokens=64 elapsed_ms=0.164 detail="generated_tokens"
trace summary: trace_id=req_20261105161801 _ 000001 mode=generate status=ok backend=cuda model=models/tiny/model.bin prompt_tokens=0 prefill_tokens=6 decode_tokens=63 generated_tokens=64 model_load_ms=0.561 tokenizer_ms=0.005 prefill_ms=164.532 decode_ms=750.664 sample_ms=0.164 total_ms=1036.080
```

### run：交互聊天

此示例使用 `models/chat` 中的 Qwen2-0.5B-Instruct Q8_0 GGUF 模型。命令的输出包括终端启动信息、两轮交互结果和逐 token trace。

```bash
build/mini-llama run models/chat --backend cpu -n 64 --seed 1
```

终端结果：

```text
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768 rope_theta=1e+06 rms_norm_eps=1e-06
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768
GGUF model loaded successfully from models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
GGUF tokenizer loaded: vocab=151936 merges=151387 special=3 bos=151643 eos=151645 unk=-1
mini-llama.cpp chat
backend: cpu
compute: cpu
Type /help for commands, /exit to quit.

mini_llama > 你好，介绍一下你自己
assistant >我叫李明，今年12岁，是你们学校的一名学生。我是一个非常热情、开朗、积极向上的人。我热爱运动，经常和朋友们一起去公园或者公园跑步，这不仅让我保持了青春的活力，也让我结交了许多好朋友。此外，我也喜欢阅读，我

trace event: trace_id=req_20261105161829 _ 000001 stage=prompt_build tokens=0 elapsed_ms=0.132 detail="messages=1"
trace event: trace_id=req_20261105161829 _ 000001 stage=tokenizer tokens=23 elapsed_ms=0.794 detail="prompt"
trace event: trace_id=req_20261105161829 _ 000001 stage=prefill tokens=23 elapsed_ms=17964.002 detail="radix_hit=0, prefix_reuse=0"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=851.266 detail="pos= 23"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=846.109 detail="pos= 24"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=751.763 detail="pos= 25"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=920.971 detail="pos= 26"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=797.619 detail="pos= 27"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=730.892 detail="pos= 28"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=880.267 detail="pos= 29"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=801.490 detail="pos= 30"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=911.120 detail="pos= 31"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=777.948 detail="pos= 32"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=807.967 detail="pos= 33"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=874.950 detail="pos= 34"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=869.911 detail="pos= 35"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=1002.695 detail="pos= 36"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=908.619 detail="pos= 37"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=955.170 detail="pos= 38"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=758.288 detail="pos= 39"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=743.445 detail="pos= 40"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=898.955 detail="pos= 41"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=750.252 detail="pos= 42"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=862.377 detail="pos= 43"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=779.764 detail="pos= 44"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=797.443 detail="pos= 45"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=808.808 detail="pos= 46"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=829.457 detail="pos= 47"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=841.655 detail="pos= 48"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=811.593 detail="pos= 49"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=798.605 detail="pos= 50"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=913.707 detail="pos= 51"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=764.781 detail="pos= 52"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=840.453 detail="pos= 53"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=769.725 detail="pos= 54"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=747.351 detail="pos= 55"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=854.170 detail="pos= 56"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=797.350 detail="pos= 57"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=818.469 detail="pos= 58"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=754.998 detail="pos= 59"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=749.694 detail="pos= 60"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=818.154 detail="pos= 61"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=783.499 detail="pos= 62"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=767.872 detail="pos= 63"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=748.147 detail="pos= 64"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=745.065 detail="pos= 65"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=832.799 detail="pos= 66"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=822.523 detail="pos= 67"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=775.706 detail="pos= 68"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=828.674 detail="pos= 69"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=749.313 detail="pos= 70"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=753.698 detail="pos= 71"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=760.942 detail="pos= 72"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=764.997 detail="pos= 73"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=841.621 detail="pos= 74"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=763.320 detail="pos= 75"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=739.041 detail="pos= 76"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=803.476 detail="pos= 77"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=737.671 detail="pos= 78"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=728.816 detail="pos= 79"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=742.536 detail="pos= 80"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=727.449 detail="pos= 81"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=797.208 detail="pos= 82"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=741.084 detail="pos= 83"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=724.751 detail="pos= 84"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=788.646 detail="pos= 85"
trace event: trace_id=req_20261105161829 _ 000001 stage=decode tokens=1 elapsed_ms=874.779 detail="pos= 86"
trace event: trace_id=req_20261105161829 _ 000001 stage=sample tokens=64 elapsed_ms=102.896 detail="generated_tokens"
trace summary: trace_id=req_20261105161829 _ 000001 mode=run status=ok backend=cpu model=models/chat prompt_tokens=23 prefill_tokens=23 decode_tokens=64 generated_tokens=64 model_load_ms=0.000 tokenizer_ms=0.794 prefill_ms=17964.002 decode_ms=51541.884 sample_ms=102.896 total_ms=70124.449
mini_llama > /exit
Goodbye.
```

```bash
build/mini-llama run models/chat --backend cuda --device 0 -n 64 --seed 1
```

终端结果：

```text
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768 rope_theta=1e+06 rms_norm_eps=1e-06
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768
GGUF model loaded successfully from models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
GGUF tokenizer loaded: vocab=151936 merges=151387 special=3 bos=151643 eos=151645 unk=-1
mini-llama.cpp chat
backend: cuda
cuda: device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, total memory 7.99548 GB, cuda runtime 13010, driver 13010
uploaded weights: 291
gpu memory used: 1469554176 bytes (1401.48MB)
cuda f32 Linear weights: 1/169
cuda q8_0 Linear weights: 168/168
compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA attention over GPU KV cache + device-resident Decode path; CPU sampler
Type /help for commands, /exit to quit.

mini_llama > 你好，你是什么模型？
assistant >你好！我是阿里云开发的超大规模语言模型，我叫通义千问。

trace event: trace_id=req_20261105162009 _ 000001 stage=prompt_build tokens=0 elapsed_ms=0.042 detail="messages=1"
trace event: trace_id=req_20261105162009 _ 000001 stage=tokenizer tokens=24 elapsed_ms=0.569 detail="prompt"
trace event: trace_id=req_20261105162009 _ 000001 stage=prefill tokens=24 elapsed_ms=1048.557 detail="radix_hit=0, prefix_reuse=0"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=29.586 detail="pos= 24"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.706 detail="pos= 25"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.211 detail="pos= 26"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.609 detail="pos= 27"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.753 detail="pos= 28"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.891 detail="pos= 29"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=29.400 detail="pos= 30"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.445 detail="pos= 31"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.893 detail="pos= 32"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.738 detail="pos= 33"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.343 detail="pos= 34"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.943 detail="pos= 35"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=26.239 detail="pos= 36"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=26.940 detail="pos= 37"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.207 detail="pos= 38"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.615 detail="pos= 39"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=27.804 detail="pos= 40"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.484 detail="pos= 41"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=28.513 detail="pos= 42"
trace event: trace_id=req_20261105162009 _ 000001 stage=decode tokens=1 elapsed_ms=25.877 detail="pos= 43"
trace event: trace_id=req_20261105162009 _ 000001 stage=sample tokens=20 elapsed_ms=30.712 detail="generated_tokens"
trace summary: trace_id=req_20261105162009 _ 000001 mode=run status=ok backend=cuda model=models/chat prompt_tokens=24 prefill_tokens=24 decode_tokens=20 generated_tokens=20 model_load_ms=0.000 tokenizer_ms=0.569 prefill_ms=1048.557 decode_ms=559.198 sample_ms=30.712 total_ms=2128.377
mini_llama > 一句话介绍西安电子科技大学
assistant >西安电子科技大学是一所位于中国陕西省西安市的著名高等学府，是国家“211工程”和“985工程”重点建设高校。学校以电子科学与技术、计算机科学与技术、信息与通信工程等学科为主，培养了大量具有创新精神和实践能力的

trace event: trace_id=req_20261105162028 _ 000002 stage=prompt_build tokens=0 elapsed_ms=0.048 detail="messages=3"
trace event: trace_id=req_20261105162028 _ 000002 stage=tokenizer tokens=58 elapsed_ms=2.914 detail="prompt"
trace event: trace_id=req_20261105162028 _ 000002 stage=prefill tokens=14 elapsed_ms=598.190 detail="radix_hit=44, prefix_reuse=44"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.745 detail="pos= 58"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.341 detail="pos= 59"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.603 detail="pos= 60"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=26.529 detail="pos= 61"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.264 detail="pos= 62"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.497 detail="pos= 63"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.990 detail="pos= 64"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.641 detail="pos= 65"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.668 detail="pos= 66"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.260 detail="pos= 67"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=33.174 detail="pos= 68"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.962 detail="pos= 69"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.475 detail="pos= 70"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.059 detail="pos= 71"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.032 detail="pos= 72"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.114 detail="pos= 73"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.262 detail="pos= 74"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=34.642 detail="pos= 75"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.481 detail="pos= 76"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.758 detail="pos= 77"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.400 detail="pos= 78"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=26.712 detail="pos= 79"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.848 detail="pos= 80"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.689 detail="pos= 81"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.806 detail="pos= 82"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.234 detail="pos= 83"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.477 detail="pos= 84"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.953 detail="pos= 85"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.154 detail="pos= 86"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.149 detail="pos= 87"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.186 detail="pos= 88"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.263 detail="pos= 89"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=30.925 detail="pos= 90"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.550 detail="pos= 91"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.495 detail="pos= 92"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=30.044 detail="pos= 93"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.279 detail="pos= 94"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.792 detail="pos= 95"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=30.835 detail="pos= 96"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.194 detail="pos= 97"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.220 detail="pos= 98"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.264 detail="pos= 99"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=32.719 detail="pos= 100"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.064 detail="pos= 101"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.087 detail="pos= 102"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.489 detail="pos= 103"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.660 detail="pos= 104"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=31.166 detail="pos= 105"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.145 detail="pos= 106"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.209 detail="pos= 107"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.772 detail="pos= 108"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=32.241 detail="pos= 109"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.908 detail="pos= 110"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.133 detail="pos= 111"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.500 detail="pos= 112"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.005 detail="pos= 113"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.934 detail="pos= 114"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.295 detail="pos= 115"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.171 detail="pos= 116"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=30.748 detail="pos= 117"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=29.269 detail="pos= 118"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=27.958 detail="pos= 119"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=30.531 detail="pos= 120"
trace event: trace_id=req_20261105162028 _ 000002 stage=decode tokens=1 elapsed_ms=28.437 detail="pos= 121"
trace event: trace_id=req_20261105162028 _ 000002 stage=sample tokens=64 elapsed_ms=97.704 detail="generated_tokens"
trace summary: trace_id=req_20261105162028 _ 000002 mode=run status=ok backend=cuda model=models/chat prompt_tokens=58 prefill_tokens=14 decode_tokens=64 generated_tokens=64 model_load_ms=0.000 tokenizer_ms=2.914 prefill_ms=598.190 decode_ms=1842.408 sample_ms=97.704 total_ms=2545.593
mini_llama > /exit
Goodbye.
```

`-n` 是每轮回复的最大 token 数；模型遇到 EOS 时会提前结束。固定 `--seed` 有助于重复测试，但 CPU 和 CUDA 的浮点计算差异可能导致生成内容不同。

### inspect：检查模型

```bash
build/mini-llama inspect models/tiny --backend cpu
```

终端结果：

```text
=== Model Config ===
  vocab_size:    128
  dim:           32
  hidden_dim:    86
  n_layers:      2
  n_heads:       4
  n_kv_heads:    4
  head_dim:      8
  max_seq_len:   128
  rope_theta:    10000
  rms_norm_eps:  1e-05

=== Tensors (21) ===
  token_embedding
    shape:     [128, 32]
    dtype:     float32
    offset:    0
    byte_size: 16384
  layers.0.attention_norm
    shape:     [32]
    dtype:     float32
    offset:    16384
    byte_size: 128
  layers.0.wq
    shape:     [32, 32]
    dtype:     float32
    offset:    16512
    byte_size: 4096
  layers.0.wk
    shape:     [32, 32]
    dtype:     float32
    offset:    20608
    byte_size: 4096
  layers.0.wv
    shape:     [32, 32]
    dtype:     float32
    offset:    24704
    byte_size: 4096
  layers.0.wo
    shape:     [32, 32]
    dtype:     float32
    offset:    28800
    byte_size: 4096
  layers.0.ffn_norm
    shape:     [32]
    dtype:     float32
    offset:    32896
    byte_size: 128
  layers.0.w_gate
    shape:     [86, 32]
    dtype:     float32
    offset:    33024
    byte_size: 11008
  layers.0.w_up
    shape:     [86, 32]
    dtype:     float32
    offset:    44032
    byte_size: 11008
  layers.0.w_down
    shape:     [32, 86]
    dtype:     float32
    offset:    55040
    byte_size: 11008
  layers.1.attention_norm
    shape:     [32]
    dtype:     float32
    offset:    66048
    byte_size: 128
  layers.1.wq
    shape:     [32, 32]
    dtype:     float32
    offset:    66176
    byte_size: 4096
  layers.1.wk
    shape:     [32, 32]
    dtype:     float32
    offset:    70272
    byte_size: 4096
  layers.1.wv
    shape:     [32, 32]
    dtype:     float32
    offset:    74368
    byte_size: 4096
  layers.1.wo
    shape:     [32, 32]
    dtype:     float32
    offset:    78464
    byte_size: 4096
  layers.1.ffn_norm
    shape:     [32]
    dtype:     float32
    offset:    82560
    byte_size: 128
  layers.1.w_gate
    shape:     [86, 32]
    dtype:     float32
    offset:    82688
    byte_size: 11008
  layers.1.w_up
    shape:     [86, 32]
    dtype:     float32
    offset:    93696
    byte_size: 11008
  layers.1.w_down
    shape:     [32, 86]
    dtype:     float32
    offset:    104704
    byte_size: 11008
  final_norm
    shape:     [32]
    dtype:     float32
    offset:    115712
    byte_size: 128
  lm_head
    shape:     [128, 32]
    dtype:     float32
    offset:    115840
    byte_size: 16384

=== Tokenizer ===
  type: json_vocab
  path: vocab.json
  bos_id: 1
  eos_id: 2
  unk_id: 0

Total tensor bytes: 132224
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
backend: cpu
```

```bash
build/mini-llama inspect models/tiny --backend cuda --device 0
```

终端结果：

```text
=== Model Config ===
  vocab_size:    128
  dim:           32
  hidden_dim:    86
  n_layers:      2
  n_heads:       4
  n_kv_heads:    4
  head_dim:      8
  max_seq_len:   128
  rope_theta:    10000
  rms_norm_eps:  1e-05

=== Tensors (21) ===
  token_embedding
    shape:     [128, 32]
    dtype:     float32
    offset:    0
    byte_size: 16384
  layers.0.attention_norm
    shape:     [32]
    dtype:     float32
    offset:    16384
    byte_size: 128
  layers.0.wq
    shape:     [32, 32]
    dtype:     float32
    offset:    16512
    byte_size: 4096
  layers.0.wk
    shape:     [32, 32]
    dtype:     float32
    offset:    20608
    byte_size: 4096
  layers.0.wv
    shape:     [32, 32]
    dtype:     float32
    offset:    24704
    byte_size: 4096
  layers.0.wo
    shape:     [32, 32]
    dtype:     float32
    offset:    28800
    byte_size: 4096
  layers.0.ffn_norm
    shape:     [32]
    dtype:     float32
    offset:    32896
    byte_size: 128
  layers.0.w_gate
    shape:     [86, 32]
    dtype:     float32
    offset:    33024
    byte_size: 11008
  layers.0.w_up
    shape:     [86, 32]
    dtype:     float32
    offset:    44032
    byte_size: 11008
  layers.0.w_down
    shape:     [32, 86]
    dtype:     float32
    offset:    55040
    byte_size: 11008
  layers.1.attention_norm
    shape:     [32]
    dtype:     float32
    offset:    66048
    byte_size: 128
  layers.1.wq
    shape:     [32, 32]
    dtype:     float32
    offset:    66176
    byte_size: 4096
  layers.1.wk
    shape:     [32, 32]
    dtype:     float32
    offset:    70272
    byte_size: 4096
  layers.1.wv
    shape:     [32, 32]
    dtype:     float32
    offset:    74368
    byte_size: 4096
  layers.1.wo
    shape:     [32, 32]
    dtype:     float32
    offset:    78464
    byte_size: 4096
  layers.1.ffn_norm
    shape:     [32]
    dtype:     float32
    offset:    82560
    byte_size: 128
  layers.1.w_gate
    shape:     [86, 32]
    dtype:     float32
    offset:    82688
    byte_size: 11008
  layers.1.w_up
    shape:     [86, 32]
    dtype:     float32
    offset:    93696
    byte_size: 11008
  layers.1.w_down
    shape:     [32, 86]
    dtype:     float32
    offset:    104704
    byte_size: 11008
  final_norm
    shape:     [32]
    dtype:     float32
    offset:    115712
    byte_size: 128
  lm_head
    shape:     [128, 32]
    dtype:     float32
    offset:    115840
    byte_size: 16384

=== Tokenizer ===
  type: json_vocab
  path: vocab.json
  bos_id: 1
  eos_id: 2
  unk_id: 0

Total tensor bytes: 132224
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
backend: cuda
cuda: device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, total memory 7.99548 GB, cuda runtime 13010, driver 13010
uploaded weights: 21
gpu memory used: 132224 bytes (0.13MB)
cuda f32 Linear weights: 15/15
```

### inspect-gguf：检查 GGUF 元数据

该模式只读取 GGUF 文件信息，不执行推理，也不接受 CPU/CUDA 后端参数。

```bash
build/mini-llama inspect-gguf models/tiny/model.gguf
```

终端结果：

```text
=== GGUF Info ===
  version:    3
  n_tensors:  21
  n_metadata: 12
  data_offset: 1728

=== Metadata ===
  general.architecture (string): llama
  general.name (string): mini-llama-tiny
  general.alignment (uint32): 32
  llama.vocab_size (uint32): 128
  llama.block_count (uint32): 2
  llama.context_length (uint32): 128
  llama.embedding_length (uint32): 32
  llama.feed_forward_length (uint32): 86
  llama.attention.head_count (uint32): 4
  llama.attention.head_count_kv (uint32): 4
  llama.rope.freq_base (float32): 10000
  llama.attention.layer_norm_rms_epsilon (float32): 1e-05

=== Tensors ===
  token_embd.weight
    shape:  [32, 128]
    dtype:  F32 (0)
    offset: 0
  blk.0.attn_norm.weight
    shape:  [32]
    dtype:  F32 (0)
    offset: 16384
  blk.0.attn_q.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 16512
  blk.0.attn_k.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 20608
  blk.0.attn_v.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 24704
  blk.0.attn_output.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 28800
  blk.0.ffn_norm.weight
    shape:  [32]
    dtype:  F32 (0)
    offset: 32896
  blk.0.ffn_gate.weight
    shape:  [32, 86]
    dtype:  F32 (0)
    offset: 33024
  blk.0.ffn_up.weight
    shape:  [32, 86]
    dtype:  F32 (0)
    offset: 44032
  blk.0.ffn_down.weight
    shape:  [86, 32]
    dtype:  F32 (0)
    offset: 55040
  blk.1.attn_norm.weight
    shape:  [32]
    dtype:  F32 (0)
    offset: 66048
  blk.1.attn_q.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 66176
  blk.1.attn_k.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 70272
  blk.1.attn_v.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 74368
  blk.1.attn_output.weight
    shape:  [32, 32]
    dtype:  F32 (0)
    offset: 78464
  blk.1.ffn_norm.weight
    shape:  [32]
    dtype:  F32 (0)
    offset: 82560
  blk.1.ffn_gate.weight
    shape:  [32, 86]
    dtype:  F32 (0)
    offset: 82688
  blk.1.ffn_up.weight
    shape:  [32, 86]
    dtype:  F32 (0)
    offset: 93696
  blk.1.ffn_down.weight
    shape:  [86, 32]
    dtype:  F32 (0)
    offset: 104704
  output_norm.weight
    shape:  [32]
    dtype:  F32 (0)
    offset: 115712
  output.weight
    shape:  [32, 128]
    dtype:  F32 (0)
    offset: 115840
```

### bench：推理基准

```bash
build/mini-llama bench models/tiny --prompt "hello" -n 16 --seed 1 --backend cpu
```

终端结果：

```text
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
Benchmark: models/tiny
  backend: cpu
  compute: cpu
  prompt: "hello" (6 tokens)
  n_predict: 16
  seed: 1
  quant: model-native
  threads: 16
  verbose: false

Results:
  prompt tokens:     6
  generated tokens:  16
  Decode tokens:     15
  prefill time:      1.44 ms
  Decode time:       3.58 ms
  total time:        5.02 ms
  tokens/s (total):  3190.06
  tokens/s (Decode): 4194.77

  weight memory:
    actual:    132224 bytes (0.13 MB)
    f32 equiv: 132224 bytes (0.13 MB)
    savings:   1.00x compression
```

```bash
build/mini-llama bench models/tiny --prompt "hello" -n 16 --seed 1 --backend cuda --device 0
```

终端结果：

```text
Loaded config: vacab=128 dim=32 layers=2 heads=4 kv_heads=4 head_dim=8 hidden_dim=86
Model loaded successfully from models/tiny/model.bin
Benchmark: models/tiny
  backend: cuda
  cuda: device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, total memory 7.99548 GB, cuda runtime 13010, driver 13010
  compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA attention over GPU KV cache + device-resident Decode path; CPU sampler
  prompt: "hello" (6 tokens)
  n_predict: 16
  seed: 1
  quant: model-native
  threads: 16
  verbose: false

  uploaded weights: 21
  gpu memory used: 132224 bytes (0.13MB)
  cuda f32 Linear weights: 15/15

Results:
  prompt tokens:     6
  generated tokens:  16
  Decode tokens:     15
  prefill time:      125.76 ms
  Decode time:       173.31 ms
  total time:        299.07 ms
  tokens/s (total):  53.50
  tokens/s (Decode): 86.55

  weight memory:
    actual:    132224 bytes (0.13 MB)
    f32 equiv: 132224 bytes (0.13 MB)
    savings:   1.00x compression

  cuda runtime:
    cuda Linear calls: 315
    cuda activation calls: 315
    cuda attention calls: 42
    cpu attention fallback calls: 0
    cuda kv write bytes: 2752512
    cuda kv read bytes: 118272
    host->device copies: 0 (0 bytes)
    device->host copies: 21 (10752 bytes)
```

### --cuda-info：查询 CUDA 设备

该命令要求程序使用 CUDA 构建。

```bash
build/mini-llama --cuda-info
```

终端结果：

```text
CUDA devices: 1
device 0: NVIDIA GeForce RTX 4060
  device name: NVIDIA GeForce RTX 4060
  compute capability: 8.9
  total memory: 8585084928 bytes (8.00 GB)
  driver version: 13010
  runtime version: 13010
```

## 性能测试

本次使用 `models/chat` 中的 Qwen2-0.5B-Instruct Q8_0 GGUF 模型，在 CPU 与 RTX 4060 CUDA 后端上使用相同提示词、生成长度和随机种子进行基准测试。CPU 使用 16 个线程；CUDA 设备为 RTX 4060（Compute Capability 8.9，8 GB）。

### CPU

```bash
build/mini-llama bench models/chat --prompt "你好，请用一句话介绍你自己。" -n 32 --seed 1 --backend cpu
```

终端结果：

```text
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768 rope_theta=1e+06 rms_norm_eps=1e-06
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768
GGUF model loaded successfully from models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
GGUF tokenizer loaded: vocab=151936 merges=151387 special=3 bos=151643 eos=151645 unk=-1
Benchmark: models/chat
  backend: cpu
  compute: cpu
  prompt: "你好，请用一句话介绍你自己。" (7 tokens)
  n_predict: 32
  seed: 1
  quant: model-native
  threads: 16
  verbose: false

Results:
  prompt tokens:     7
  generated tokens:  32
  Decode tokens:     31
  prefill time:      5704.88 ms
  Decode time:       25038.30 ms
  total time:        30743.18 ms
  tokens/s (total):  1.04
  tokens/s (Decode): 1.24

  weight memory:
    actual:    1469554176 bytes (1401.48 MB)
    f32 equiv: 2520669696 bytes (2403.90 MB)
    savings:   1.72x compression
```

### CUDA

```bash
build/mini-llama bench models/chat --prompt "你好，请用一句话介绍你自己。" -n 32 --seed 1 --backend cuda --device 0
```

终端结果：

```text
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768 rope_theta=1e+06 rms_norm_eps=1e-06
GGUF config: vocab=151936 dim=896 layers=24 heads=14 kv_heads=2 head_dim=64 hidden_dim=4864 max_seq_len=32768
GGUF model loaded successfully from models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
GGUF tokenizer loaded: vocab=151936 merges=151387 special=3 bos=151643 eos=151645 unk=-1
Benchmark: models/chat
  backend: cuda
  cuda: device 0: NVIDIA GeForce RTX 4060, compute capability 8.9, total memory 7.99548 GB, cuda runtime 13010, driver 13010
  compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA attention over GPU KV cache + device-resident Decode path; CPU sampler
  prompt: "你好，请用一句话介绍你自己。" (7 tokens)
  n_predict: 32
  seed: 1
  quant: model-native
  threads: 16
  verbose: false

  uploaded weights: 291
  gpu memory used: 1469554176 bytes (1401.48MB)
  cuda f32 Linear weights: 1/169
  cuda q8_0 Linear weights: 168/168

Results:
  prompt tokens:     7
  generated tokens:  32
  Decode tokens:     31
  prefill time:      617.07 ms
  Decode time:       905.28 ms
  total time:        1522.35 ms
  tokens/s (total):  21.02
  tokens/s (Decode): 34.24

  weight memory:
    actual:    1469554176 bytes (1401.48 MB)
    f32 equiv: 2520669696 bytes (2403.90 MB)
    savings:   1.72x compression

  cuda runtime:
    cuda Linear calls: 6422
    cuda activation calls: 9158
    cuda attention calls: 912
    cpu attention fallback calls: 0
    cuda kv write bytes: 734439407616
    cuda kv read bytes: 127475712
    host->device copies: 0 (0 bytes)
    device->host copies: 38 (23094272 bytes)
```

### 对比

| 指标 | CPU | CUDA | CUDA 相对表现 |
| --- | ---: | ---: | ---: |
| Prefill | 5704.88 ms | 617.07 ms | 约 9.2 倍快 |
| Decode（31 tokens） | 25038.30 ms | 905.28 ms | 约 27.7 倍快 |
| 总耗时 | 30743.18 ms | 1522.35 ms | 约 20.2 倍快 |
| 总吞吐量 | 1.04 tokens/s | 21.02 tokens/s | 约 20.2 倍 |
| Decode 吞吐量 | 1.24 tokens/s | 34.24 tokens/s | 约 27.6 倍 |

本次 Qwen 模型基准中，CUDA 显著快于 CPU。Qwen 模型的计算量更大，GPU 的并行计算收益足以覆盖 kernel 启动等开销。实测速度会受硬件、驱动和系统负载影响。

