#include "mini_llama/quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace mini_llama {

// 将Fp32转化为Fp16
static uint16_t FloatToFp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent);
        if ((shifted & 0x00001000u) != 0) {
            shifted += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (shifted >> 13));
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if ((mantissa & 0x00001000u) != 0) {
        mantissa += 0x00002000u;
        if ((mantissa & 0x00800000u) != 0) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

// 将Fp16转化为Fp32
static float Fp16ToFloat(uint16_t value) {
    // 拆解FP16的符号位、指数位、尾数位
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;  // 符号位
    uint32_t exponent = (value >> 10) & 0x1fu;                     // 指数位
    uint32_t mantissa = value & 0x03ffu;                           // 尾数位
    uint32_t bits = 0;


    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            uint32_t exp32 = exponent + (127 - 15);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        uint32_t exp32 = exponent + (127 - 15);
        bits = sign | (exp32 << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// 计算校队tensor的元素数量
static size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int dim = shape[axis];
        if (dim < 0) {
            throw std::runtime_error(std::string(caller) + ": dimension at axis " + std::to_string(axis) + " must be positive, got " + std::to_string(dim));
        }

        size_t dim_size = static_cast<size_t>(dim);
        if (dim_size * total > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(std::string(caller) + ": shape element count overflow");
        }
        total *= dim_size;
    }

    return total;
} 

// Tensor 转化为 Q80
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
    if (src.size() == 0) {
        return {};
    }
    int row_size = src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.num_elements());
    int n_rows = src.num_dims() >= 2 ? static_cast<int>(src.num_elements()) / row_size : 1;
    int row_blocks = (row_size + kQ80BlockSize -1) / kQ80BlockSize;
    size_t total_blocks = static_cast<size_t>(row_blocks) * n_rows;

    std::vector<BlockQ80> blocks;
    blocks.reserve(total_blocks);

    // 双循环，逐行逐列
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ80BlockSize;        // 当前32个元素的起始索引
            int k_end = std::min(base + kQ80BlockSize, row_offset + row_size); // 防止最后一行不是一整块的大小
            int block_len = k_end - base;

            // 找最大值
            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ80 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d = max_abs / 127.0f;
                block.d = FloatToFp16(d);
                float stored_d = Fp16ToFloat(block.d);   // fp16有精度损失，所以要用block内的d来做量化
                float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

                // 处理block内的数据
                for (int i = 0; i < block_len; ++i) {
                    float q = src.data[base + i] * id;
                    int qi = static_cast<int>(std::round(q));   // 四舍五入，保留整数
                    if (qi > 127) {
                        qi = 127;
                    } else if (qi < -127) {
                        qi = -127;
                    }
                    block.qs[i] = static_cast<int8_t>(qi);
                }
            } else {
                block.d = 0;
            }

            // 边缘补齐
            for (int i = block_len; i < kQ80BlockSize; ++i) {
                block.qs[i] = 0;
            }

            blocks.push_back(block);
        }
    } 
    return blocks;
}

// Q80 转化为tensor
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape) {
    size_t total = CheckedNumel(shape, "DequantizeFromQ80");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_row = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_block = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_row) * row_block;

    if (expected_blocks != blocks.size()) {
        throw std::runtime_error( "DequantizeFromQ80: block count mismatch: expected " + std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size()));
    }


    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_row; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_block; ++rb) {
            int base = row_offset + rb * kQ80BlockSize;
            int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
            const BlockQ80& block = blocks[block_idx++];
            float d = Fp16ToFloat(block.d);
            for (int k = base; k < k_end; ++k) {
                dst.data[k] = d * static_cast<float>(block.qs[k - base]);
            }
        }
    }
    return dst;
}

// tensor 转化为 Q40
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
    if (src.num_elements() == 0) {
        return {};
    }

    int row_size = src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.num_dims());
    int n_row =  src.num_dims() >= 2 ? static_cast<int>(src.num_elements()) / row_size : 1;
    int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t total_block = static_cast<size_t>(n_row) * row_blocks;

    std::vector<BlockQ40> blocks;
    blocks.reserve(total_block);

    for (int row = 0; row <  n_row; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ40BlockSize;
            int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
            int block_len = k_end - base;

            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ40 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d = max_abs / 7;        // 半个字节，范围是[-7, 7];
                block.d = FloatToFp16(d);
                float store_d = Fp16ToFloat(block.d);
                float id = store_d == 0.0f ? 0.0f : store_d;

                for (int j = 0; j < kQ40BlockSize / 2; ++j) {
                    float x0 = 0.0f;
                    float x1 = 0.0f;
                    int idx0 = base + j;                     // 前半段的索引
                    int idx1 = base + j + kQ40BlockSize / 2; // 后半段的索引

                    if (idx0 < k_end) {
                        x0 = src.data[idx0] * id;
                    }
                    if (idx1 < k_end) {
                        x1 = src.data[idx1] * id;
                    }

                    int qi0 = static_cast<int>(std::round(x0 + 8.0f));
                    int qi1 = static_cast<int>(std::round(x1 + 8.0f));
                    if (qi0 < 0) {
                        qi0 = 0;
                    }
                    if (qi1 < 0) {
                        qi1 = 0;
                    }
                    if (qi0 > 15) {
                        qi0 = 15;
                    }
                    if (qi1 > 15) {
                        qi1 = 15;
                    }

                    // 把两个4-bit合成一个byte
                    block.qs[j] = static_cast<uint8_t>(qi0 | (qi1 << 4));
                }
            } else {
                block.d = 0;
            }

            blocks.push_back(block);
        }
    }
    return blocks;
}


// Q40 转化为 tensor
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape) {
    size_t total = CheckedNumel(shape, "DequantizeFromQ40");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_row = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_row) * row_blocks;

    if (expected_blocks != blocks.size()) {
        throw std::runtime_error("DequantizeFromQ40: block count mismatch: expected " + std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size()));
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_row; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ40BlockSize;
            int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
            const BlockQ40& block = blocks[block_idx++];
            float d = Fp16ToFloat(block.d);

            for (int j = 0; j < kQ40BlockSize / 2; ++j) {
                int idx0 = base + j;                        // 前半段索引
                int idx1 = base + j + kQ40BlockSize / 2;    // 后半段索引
                int q0 = static_cast<int>(block.qs[j] & 0x0f) - 8;
                int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
                if (idx0 < k_end) {
                    dst.data[idx0] = static_cast<float>(q0) * d;
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = static_cast<float>(q1) * d;
                }
            }
        }
    }

    return dst;
}


// Q41 转化为 tensor
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape) {
    size_t total = CheckedNumel(shape, "DequantizeFromQ41");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_row = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_row) * row_blocks;

    if (expected_blocks != blocks.size()) {
        throw std::runtime_error("DequantizeFromQ41: block count mismatch: expected " + std::to_string(expected_blocks) + ", got " + std::to_string(blocks.size()));
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_row; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ41BlockSize;
            int k_end = std::min(base + kQ41BlockSize, row_offset + row_size);
            const BlockQ41& block = blocks[block_idx++];
            float d = Fp16ToFloat(block.d);
            float m = Fp16ToFloat(block.m);

            for (int j = 0; j < kQ40BlockSize / 2; ++j) {
                int idx0 = base + j;                        // 前半段索引
                int idx1 = base + j + kQ40BlockSize / 2;    // 后半段索引
                int q0 = static_cast<int>(block.qs[j] & 0x0f);
                int q1 = static_cast<int>(block.qs[j] >> 4);
                if (idx0 < k_end) {
                    dst.data[idx0] = static_cast<float>(q0) * d + m;
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = static_cast<float>(q1) * d + m;
                }
            }
        }
    }

    return dst;
}


// linear                               输入                       权重                                    权重形状
template <typename BlockType, int kBlockSize> 
static Tensor LinearQuantizedImpl(const Tensor& x, const std::vector<BlockType>& blocks, const std::vector<int>& weight_shape, float (*dequant_fn)(const BlockType&, int)) {
    if (weight_shape.size() != 2) {
        throw std::runtime_error("linear_quantized: expected 2D weight shape");
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.num_dims() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.num_dims() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        throw std::runtime_error("linear_quantized: expected x shape [in_features] or [batch, " + "in_features], got " + x.ShapeStringShort());
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        throw std::runtime_error("linear_quantized: dimension mismatch x=" + x.ShapeStringShort() + " weight=" + QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
    }

    // 权重矩阵有多少个block
    int n_blocks_per_row = (in_features + kBlockSize - 1) / kBlockSize;
    size_t expected_blocks = static_cast<size_t>(out_features) * n_blocks_per_row;
    if (expected_blocks != blocks.size()) {
        throw std::runtime_error("linear_quantized: dimension mismatch x=" + x.ShapeStringShort() + " weight=" + QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
    }

    Tensor result(is_1d ? std::vector<int>({out_features}) : std::vector<int>({rows, out_features}), 0.0f);



    // 多线程并行计算
    ParallelFor(rows * out_features, [&](int begin, int end){
        for (int flat_index = begin; flat_index < end; ++flat_index) {
            int row = flat_index / out_features;            // 计算的行
            int j = flat_index % out_features;              // 当前行的第几个输出特征 , weight的第几行
            const float* x_row = x.data.data() + static_cast<size_t>(row) * in_features;  // 输入x的起始位置

            float sum = 0.0f;
            int block_base = j * n_blocks_per_row;                                          // 当前weight所属的那个block
            // 边解压边计算
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockType& block = blocks[block_base + b];
                int base_k = b * kBlockSize;
                int k_end = std::min(base_k + kBlockSize, in_features);
                for (int k = base_k; k < k_end; ++k) {
                    float w = dequant_fn(block, k - base_k);
                    sum += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = sum;
        }
    });


    return result;
}



// 用来拿出一个float
static float DequantQ80(const BlockQ80& block, int idx) {
    float d = Fp16ToFloat(block.d);
    return d * static_cast<float>(block.qs[idx]);
}

static float DequantQ40(const BlockQ40& block, int idx) {
    float d = Fp16ToFloat(block.d);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0f) - 8;      // 低4位
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;   // 后16个数的高4位
    }
    return d * static_cast<float>(q);
}

static float DequantQ41(const BlockQ41& block, int idx) {
    float d = Fp16ToFloat(block.d);
    float m = Fp16ToFloat(block.m);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0f);
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4);
    }
    return d * static_cast<float>(q) + m;
}



// 量化线性层Linear算子
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape, DequantQ80);
}

Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape, DequantQ40);
}

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape) {
    return LinearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape, DequantQ41);
}



// 先解压，后计算
Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = DequantizeFromQ80(weight, weight_shape);
    return Matmul(weight_f32, input);
}
Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input, const std::vector<int>& weight_shape) {
    Tensor weight_f32 = DequantizeFromQ40(weight, weight_shape);
    return Matmul(weight_f32, input);
}



float CompareMatmulError(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = Linear(weight, input);
    std::vector<BlockQ80> qweight = QuantizeToQ80(weight);
    Tensor q8_result = MatmulQ80(qweight, input, weight.shape);

    if (f32_result.shape != q8_result.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.num_elements(); ++i) {
        float err = std::abs(f32_result[i] - q8_result[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}


float CompareQ40Error(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = Linear(weight, input);
    std::vector<BlockQ40> qweight = QuantizeToQ40(weight);
    Tensor q4_result = LinearQ40(input, qweight, weight.shape);
    
    if (f32_result.shape != q4_result.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.num_elements(); ++i) {
        float err = std::abs(f32_result[i] - q4_result[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

}