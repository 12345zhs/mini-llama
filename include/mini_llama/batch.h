#pragma once

#include <vector>

namespace mini_llama {

struct MiniBatch {
    std::vector<int> tokens;          // token ids
    std::vector<int> positions;       // position for each token

    int num_tokens() const;

    static MiniBatch Single(int token, int pos);

    static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);

};

}