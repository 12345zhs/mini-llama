#include "mini_llama/radix_tree.h" 

namespace mini_llama {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;

RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;


void RadixTree::Clear() {
    root_.children.clear();
    root_.key.clear();
    root_.terminal = false;
    terminal_count_ = 0;
}


bool RadixTree::empty() const {
    return terminal_count_ == 0;
}


size_t RadixTree::size() const {
    return terminal_count_;
}


void RadixTree::Insert(const std::vector<int>& tokens) {
    // 空序列
    if (tokens.empty()) {
        if (!root_.terminal) {
            root_.terminal = true;
            ++terminal_count_;
        }
        return;
    }
    InsertInto(&root_, tokens);
}

// 最长前缀查找函数
size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
    // 空序列
    if (tokens.empty()) {
        return 0;
    }

    // 从根节点开始
    const Node* node = &root_;
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto child_it = node->children.find(tokens[pos]);
        if (child_it == node->children.end()) {
            return pos;
        }

        const Node* child = child_it->second.get();
        size_t common = CommonPrefixLength(tokens, pos, child->key);
        pos += common;
        if (common < child->key.size()) {
            return pos;
        }
        node = child;
    }
    return pos;
}


// 公共前缀的长度
size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset, const std::vector<int>& b) {
    size_t i = 0;
    while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
        ++i;
    }
    return i;
}

// 插入函数
void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
    auto child_it = parent->children.find(suffix[0]);
    // 没有这个前缀，在parent下新建
    if (child_it == parent->children.end()) {
        auto child = std::make_unique<Node>();
        child->key = std::move(suffix);
        child->terminal = true;
        parent->children[child->key[0]] = std::move(child);
        ++terminal_count_;
        return; 
    }

    // 找到子节点，比较公共前缀
    std::unique_ptr<Node>& child = child_it->second;
    size_t common = CommonPrefixLength(suffix, 0, child->key);
    // 整条边都匹配(已经存在)
    if (common == child->key.size()) {
        if (common == suffix.size()) {
            if (!child->terminal) {
                child->terminal = true;
                ++terminal_count_;
            }
            return;
        }

        // suffix中还存在一点没有匹配
        std::vector<int> remaining(suffix.begin() + static_cast<size_t>(common), suffix.end());
        InsertInto(child.get(), std::move(remaining));   // 递归调用
        return;
    }


    // 只匹配了一点，此时要分裂节点
    auto split = std::make_unique<Node>();
    split->key.assign(child->key.begin(), child->key.begin() + static_cast<long long>(common));   // 前缀的公共部分
    child->key.erase(child->key.begin(), child->key.begin() + static_cast<long long>(common));    // child的后半部分
    // 把child作为map中的key-value，挂载到split
    int old_child_first = child->key[0];
    split->children[old_child_first] = std::move(child);

    // suffix正好是前缀
    if (common == suffix.size()) {
        split->terminal = true;
        ++terminal_count_;
    } else {
        auto new_child = std::make_unique<Node>();
        new_child->key.assign(suffix.begin() + static_cast<long long>(common), suffix.end());
        new_child->terminal = true;
        split->children[new_child->key[0]] = std::move(new_child);
        ++terminal_count_;
    }

    child_it->second = std::move(split);
}


}