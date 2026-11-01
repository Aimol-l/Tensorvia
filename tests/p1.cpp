// P1 — 推理算子覆盖。
//
// FR-9：N-D 批处理 matmul + 广播，以及 bmm/addmm/linear。参考值由本文件里
// 一份不依赖被测代码的朴素实现给出（按输出多索引直接求内积），所以 CPU lane
// 与各设备 lane 校验的是同一套语义。
#include "test_support.hpp"

namespace {

std::vector<int64_t> shape_of(const Tensor& t) {
    return {t.shape().begin(), t.shape().end()};
}

std::vector<float> ramp(int64_t n, float step = 0.1f, float start = -1.0f) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = start + step * static_cast<float>(i);
    return v;
}

// Tensor 的宿主构造要求源的具名左值；本文件多处只需要"一份临时数据 + 形状"。
template <typename T>
Tensor from_values(std::vector<T> values, std::initializer_list<int64_t> shape) {
    return Tensor(values, shape);
}

// 同一个算子先在 CPU 上算一遍作为参考，再在设备上算一遍：形状、dtype、设备
// 与数值都必须一致（设备路径是用已有算子组装的，参考的是同一套语义）。
void expect_same_as_cpu(const Tensor& device_result, const Tensor& cpu_result,
                        Device dev, std::string_view what) {
    EXPECT_EQ(shape_of(device_result), shape_of(cpu_result)) << what;
    EXPECT_EQ(device_result.dtype(), cpu_result.dtype()) << what;
    EXPECT_EQ(device_result.device(), dev) << what;
    expect_close(host_values<float>(device_result), host_values<float>(cpu_result), what);
}

int64_t flat_index(const std::vector<int64_t>& idx, const std::vector<int64_t>& shape) {
    int64_t off = 0;
    for (size_t i = 0; i < shape.size(); ++i) off = off * shape[i] + idx[i];
    return off;
}

// PyTorch 的完整输出形状（含被降级掉的向量维）：batch + {rows, cols}。
// 参考实现的索引空间就是这个形状，1-D 操作数在前面补 1。
std::vector<int64_t> mm_full(const std::vector<int64_t>& as, const std::vector<int64_t>& bs) {
    const std::vector<int64_t> a = as.size() == 1 ? std::vector<int64_t>{1, as[0]} : as;
    const std::vector<int64_t> b = bs.size() == 1 ? std::vector<int64_t>{bs[0], 1} : bs;
    const std::vector<int64_t> ab(a.begin(), a.end() - 2);
    const std::vector<int64_t> bb(b.begin(), b.end() - 2);
    const size_t nb = std::max(ab.size(), bb.size());
    std::vector<int64_t> full(nb + 2, 1);
    full[nb] = a[a.size() - 2];
    full[nb + 1] = b[b.size() - 1];
    for (size_t i = 0; i < ab.size(); ++i) full[nb - ab.size() + i] = ab[i];
    for (size_t i = 0; i < bb.size(); ++i) {
        int64_t& slot = full[nb - bb.size() + i];
        slot = slot == 1 ? bb[i] : slot;
    }
    return full;
}

// 朴素参考：out[i,j] = Σ_k a[i,k]·b[k,j]，批次维右对齐、尺寸 1 的维度取下标 0。
std::vector<float> ref_matmul(const std::vector<float>& a, const std::vector<int64_t>& as_in,
                              const std::vector<float>& b, const std::vector<int64_t>& bs_in,
                              const std::vector<int64_t>& full) {
    const std::vector<int64_t> as = as_in.size() == 1 ? std::vector<int64_t>{1, as_in[0]} : as_in;
    const std::vector<int64_t> bs = bs_in.size() == 1 ? std::vector<int64_t>{bs_in[0], 1} : bs_in;
    const int64_t inner = as.back();

    std::vector<int64_t> afull(full.size(), 1), bfull(full.size(), 1);
    std::copy(as.begin(), as.end(), afull.end() - static_cast<int64_t>(as.size()));
    std::copy(bs.begin(), bs.end(), bfull.end() - static_cast<int64_t>(bs.size()));

    std::vector<float> out(static_cast<size_t>(calc_numel(full)), 0.0f);
    std::vector<int64_t> idx(full.size(), 0), ai(afull.size(), 0), bi(bfull.size(), 0);
    const size_t rows_at = full.size() - 2;
    const size_t cols_at = full.size() - 1;
    for (size_t o = 0; o < out.size(); ++o) {
        size_t rem = o;
        for (int i = static_cast<int>(full.size()) - 1; i >= 0; --i) {
            const size_t d = static_cast<size_t>(i);
            idx[d] = static_cast<int64_t>(rem % static_cast<size_t>(full[d]));
            rem /= static_cast<size_t>(full[d]);
        }
        float sum = 0.0f;
        for (int64_t k = 0; k < inner; ++k) {
            for (size_t i = 0; i < afull.size(); ++i) {
                int64_t v = idx[full.size() - afull.size() + i];
                if (i == afull.size() - 1) v = k;
                else if (i == afull.size() - 2) v = idx[rows_at];
                ai[i] = afull[i] == 1 ? 0 : v;
            }
            for (size_t i = 0; i < bfull.size(); ++i) {
                int64_t v = idx[full.size() - bfull.size() + i];
                if (i == bfull.size() - 2) v = k;
                else if (i == bfull.size() - 1) v = idx[cols_at];
                bi[i] = bfull[i] == 1 ? 0 : v;
            }
            sum += a[static_cast<size_t>(flat_index(ai, afull))] *
                   b[static_cast<size_t>(flat_index(bi, bfull))];
        }
        out[o] = sum;
    }
    return out;
}

// 一组形状跑通「建张量 → 上设备 → matmul → 形状/数值对参考」。
void check_matmul(const std::vector<int64_t>& as, const std::vector<int64_t>& bs) {
    const std::vector<int64_t> full = mm_full(as, bs);
    const size_t batch_rank = full.size() - 2;

    std::vector<int64_t> want(full.begin(), full.end() - 2);
    if (as.size() != 1) want.push_back(full[batch_rank]);
    if (bs.size() != 1) want.push_back(full[batch_rank + 1]);

    std::vector<float> a = ramp(static_cast<int64_t>(calc_numel(as)));
    std::vector<float> b = ramp(static_cast<int64_t>(calc_numel(bs)), 0.07f, 0.5f);
    const Tensor A = moved_to(Tensor(a, as), kTestDevice);
    const Tensor B = moved_to(Tensor(b, bs), kTestDevice);

    const Tensor C = ops::MatMul(A, B);
    EXPECT_EQ(shape_of(C), want) << "matmul(" << as.size() << "D, " << bs.size() << "D)";
    EXPECT_EQ(C.device(), kTestDevice);
    expect_close(host_values<float>(C), ref_matmul(a, as, b, bs, full), "matmul");
}

}  // namespace

TEST_F(DeviceOps, MatMul2DMatchesReference) {
    check_matmul({3, 4}, {4, 5});
}

TEST_F(DeviceOps, MatMulPromotesVectorOperands) {
    check_matmul({4}, {4});     // 0-D：点积
    check_matmul({3, 4}, {4});  // 矩阵 × 向量
    check_matmul({4}, {4, 3});  // 向量 × 矩阵
}

TEST_F(DeviceOps, MatMulBatchesAndBroadcasts) {
    check_matmul({2, 3, 4}, {2, 4, 5});        // 批次相同
    check_matmul({2, 1, 3, 4}, {1, 5, 4, 2});  // 4-D 双向广播
    check_matmul({2, 1, 3, 4}, {5, 4, 2});     // (2,1) 批 vs (5) 批
    check_matmul({2, 3, 4}, {4, 5});           // 批 × 普通矩阵
    check_matmul({1, 1, 3, 4}, {3, 4, 5});     // 前导 1 维被扩开
}

TEST_F(DeviceOps, MatMulAcceptsNonContiguousOperand) {
    std::vector<float> a = ramp(12);
    std::vector<float> b = ramp(6, 0.07f, 0.5f);
    std::vector<int64_t> as{3, 4};
    std::vector<float> a_t = a;  // 参考用：同一份数据按 {4,3} 解释的转置
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 4; ++j)
            a_t[static_cast<size_t>(j * 3 + i)] = a[static_cast<size_t>(i * 4 + j)];
    const Tensor A0 = moved_to(Tensor(a, as), kTestDevice);
    const Tensor B = moved_to(Tensor(b, std::vector<int64_t>{3, 2}), kTestDevice);
    const Tensor At = ops::Transpose(A0, {1, 0});  // 4×3 的转置视图

    const Tensor C = ops::MatMul(At, B);
    EXPECT_EQ(shape_of(C), (std::vector<int64_t>{4, 2}));
    expect_close(host_values<float>(C), ref_matmul(a_t, {4, 3}, b, {3, 2}, {4, 2}),
                 "matmul(transposed)");
}

TEST_F(DeviceOps, MatMulEmptyAndZeroInner) {
    // 非收缩维为 0：结果是空张量，形状照旧是契约的一部分。
    const Tensor A0 = moved_to(Tensor(std::vector<int64_t>{0, 3}, DataType::FLOAT32), kTestDevice);
    std::vector<float> b = ramp(12);
    const Tensor B0 = moved_to(Tensor(b, std::vector<int64_t>{3, 4}), kTestDevice);
    const Tensor C0 = ops::MatMul(A0, B0);
    EXPECT_EQ(shape_of(C0), (std::vector<int64_t>{0, 4}));
    EXPECT_EQ(C0.numel(), 0u);

    // 收缩维为 0：空点积是 0，不是 NaN。
    const Tensor Az = moved_to(Tensor(std::vector<int64_t>{2, 0}, DataType::FLOAT32), kTestDevice);
    const Tensor Bz = moved_to(Tensor(std::vector<int64_t>{0, 3}, DataType::FLOAT32), kTestDevice);
    const Tensor Cz = ops::MatMul(Az, Bz);
    EXPECT_EQ(shape_of(Cz), (std::vector<int64_t>{2, 3}));
    expect_close(host_values<float>(Cz), std::vector<float>(6, 0.0f), "matmul(k=0)");
}

TEST_F(DeviceOps, MatMulRejectsBadShapes) {
    std::vector<float> v6 = ramp(6), v12 = ramp(12), v24 = ramp(24);
    std::vector<float> v3 = ramp(3), v4 = ramp(4);
    EXPECT_THROW(ops::MatMul(moved_to(Tensor(v6, std::vector<int64_t>{2, 3}), kTestDevice),
                        moved_to(Tensor(v6, std::vector<int64_t>{2, 3}), kTestDevice)),
                 std::runtime_error);  // 内维不匹配
    EXPECT_THROW(ops::MatMul(moved_to(Tensor(v12, std::vector<int64_t>{2, 3, 2}), kTestDevice),
                        moved_to(Tensor(v24, std::vector<int64_t>{4, 2, 3}), kTestDevice)),
                 std::runtime_error);  // 批次 {2,3} 对 {4}：最右批维 3≠4 且都不是 1
    EXPECT_THROW(ops::MatMul(moved_to(Tensor(v3, std::vector<int64_t>{3}), kTestDevice),
                        moved_to(Tensor(v4, std::vector<int64_t>{4}), kTestDevice)),
                 std::runtime_error);  // 内维不匹配（向量的唯一维就是内维）
}

TEST_F(DeviceOps, BmmRequiresMatching3D) {
    std::vector<float> a = ramp(24);
    std::vector<float> b = ramp(24, 0.07f, 0.5f);
    std::vector<float> b3 = ramp(36);
    std::vector<float> m6 = ramp(6);
    const Tensor A = moved_to(Tensor(a, std::vector<int64_t>{2, 3, 4}), kTestDevice);
    const Tensor B = moved_to(Tensor(b, std::vector<int64_t>{2, 4, 3}), kTestDevice);
    const Tensor C = ops::Bmm(A, B);
    EXPECT_EQ(shape_of(C), (std::vector<int64_t>{2, 3, 3}));
    expect_close(host_values<float>(C), ref_matmul(a, {2, 3, 4}, b, {2, 4, 3}, {2, 3, 3}), "bmm");

    EXPECT_THROW(ops::Bmm(moved_to(Tensor(m6, std::vector<int64_t>{2, 3}), kTestDevice), B),
                 std::runtime_error);
    EXPECT_THROW(ops::Bmm(A, moved_to(Tensor(b3, std::vector<int64_t>{3, 4, 3}), kTestDevice)),
                 std::runtime_error);
}

TEST_F(DeviceOps, AddmmScalesAndAccumulates) {
    std::vector<float> in = ramp(6, 0.05f, 1.0f);
    std::vector<float> a = ramp(8);
    std::vector<float> b = ramp(12, 0.07f, 0.5f);
    const Tensor In = moved_to(Tensor(in, std::vector<int64_t>{2, 3}), kTestDevice);
    const Tensor A = moved_to(Tensor(a, std::vector<int64_t>{2, 4}), kTestDevice);
    const Tensor B = moved_to(Tensor(b, std::vector<int64_t>{4, 3}), kTestDevice);

    const std::vector<float> ref = ref_matmul(a, {2, 4}, b, {4, 3}, {2, 3});
    std::vector<float> want = ref;
    for (size_t i = 0; i < want.size(); ++i) want[i] = 2.0f * in[i] + 0.5f * ref[i];

    const Tensor C = ops::Addmm(In, A, B, 2.0f, 0.5f);
    EXPECT_EQ(shape_of(C), (std::vector<int64_t>{2, 3}));
    expect_close(host_values<float>(C), want, "addmm");

    // beta=0 时加数完全不参与；alpha=1 时等价于纯 matmul。
    expect_close(host_values<float>(ops::Addmm(In, A, B, 0.0f)), ref, "addmm(beta=0)");
}

TEST_F(DeviceOps, LinearMatchesBatchedAffine) {
    const int64_t tokens = 5, in_features = 4, out_features = 3;
    std::vector<float> x = ramp(tokens * in_features);
    std::vector<float> w = ramp(out_features * in_features, 0.07f, 0.25f);
    std::vector<float> bias = ramp(out_features, 0.3f, -0.5f);
    std::vector<float> x3 = ramp(2 * tokens * in_features);
    const Tensor X = moved_to(Tensor(x, std::vector<int64_t>{tokens, in_features}), kTestDevice);
    const Tensor W = moved_to(Tensor(w, std::vector<int64_t>{out_features, in_features}),
                              kTestDevice);
    const Tensor B = moved_to(Tensor(bias, std::vector<int64_t>{out_features}), kTestDevice);

    // out[i,j] = Σ_k x[i,k]·w[j,k] + bias[j]
    std::vector<float> want(static_cast<size_t>(tokens * out_features));
    for (int64_t i = 0; i < tokens; ++i)
        for (int64_t j = 0; j < out_features; ++j) {
            float s = 0.0f;
            for (int64_t k = 0; k < in_features; ++k)
                s += x[static_cast<size_t>(i * in_features + k)] *
                     w[static_cast<size_t>(j * in_features + k)];
            want[static_cast<size_t>(i * out_features + j)] = s + bias[static_cast<size_t>(j)];
        }

    const Tensor Y = ops::Linear(X, W, B);
    EXPECT_EQ(shape_of(Y), (std::vector<int64_t>{tokens, out_features}));
    expect_close(host_values<float>(Y), want, "linear");

    // 任意前导批维；无 bias 的重载就是 x @ wᵀ。
    const Tensor X3 = moved_to(Tensor(x3, std::vector<int64_t>{2, tokens, in_features}), kTestDevice);
    EXPECT_EQ(shape_of(ops::Linear(X3, W)), (std::vector<int64_t>{2, tokens, out_features}));
}

// ===== FR-11：形状/数据搬运算子 =====

TEST_F(DeviceOps, SplitReturnsViews) {
    std::vector<float> v = ramp(12, 1.0f, 0.0f);  // 0..11
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{6, 2}), kTestDevice);

    const auto parts = ops::Split(T, 2, 0);
    ASSERT_EQ(parts.size(), 3u);
    for (size_t i = 0; i < parts.size(); ++i) {
        EXPECT_EQ(shape_of(parts[i]), (std::vector<int64_t>{2, 2}));
        EXPECT_EQ(parts[i].device(), kTestDevice);
        std::vector<float> want(v.begin() + static_cast<long>(i * 4), v.begin() + static_cast<long>(i * 4 + 4));
        expect_close(host_values<float>(parts[i]), want, "split part");
    }

    // 不能整除时最后一块更短。
    const auto uneven = ops::Split(T, 4, 0);
    ASSERT_EQ(uneven.size(), 2u);
    EXPECT_EQ(shape_of(uneven[0]), (std::vector<int64_t>{4, 2}));
    EXPECT_EQ(shape_of(uneven[1]), (std::vector<int64_t>{2, 2}));

    EXPECT_THROW(ops::Split(T, 0, 0), std::runtime_error);
}

TEST_F(DeviceOps, ChunkUsesCeilBlockSize) {
    std::vector<float> v = ramp(5, 1.0f, 0.0f);  // 0..4
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{5}), kTestDevice);
    const auto parts = ops::Chunk(T, 2, 0);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(shape_of(parts[0]), (std::vector<int64_t>{3}));
    EXPECT_EQ(shape_of(parts[1]), (std::vector<int64_t>{2}));
    expect_close(host_values<float>(parts[0]), {0, 1, 2}, "chunk 0");
    expect_close(host_values<float>(parts[1]), {3, 4}, "chunk 1");
    EXPECT_THROW(ops::Chunk(T, 0, 0), std::runtime_error);
}

TEST_F(DeviceOps, StackConcatenatesAlongNewAxis) {
    std::vector<float> a = ramp(4, 1.0f, 1.0f);  // 1..4
    std::vector<float> b = ramp(4, 1.0f, 5.0f);  // 5..8
    const Tensor A = moved_to(Tensor(a, std::vector<int64_t>{2, 2}), kTestDevice);
    const Tensor B = moved_to(Tensor(b, std::vector<int64_t>{2, 2}), kTestDevice);

    const Tensor S0 = ops::Stack({A, B}, 0);
    EXPECT_EQ(shape_of(S0), (std::vector<int64_t>{2, 2, 2}));
    expect_close(host_values<float>(S0), {1, 2, 3, 4, 5, 6, 7, 8}, "stack dim 0");

    const Tensor S1 = ops::Stack({A, B}, 1);
    EXPECT_EQ(shape_of(S1), (std::vector<int64_t>{2, 2, 2}));
    expect_close(host_values<float>(S1), {1, 2, 5, 6, 3, 4, 7, 8}, "stack dim 1");

    EXPECT_EQ(shape_of(ops::Stack({A, B}, -1)), (std::vector<int64_t>{2, 2, 2}));
    EXPECT_THROW(ops::Stack({A, B}, 3), std::runtime_error);
}

TEST_F(DeviceOps, PadAddsConstantBorder) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{2, 3}), kTestDevice);

    // 最后一维左右各补 1/2，第 0 维末尾补 1。
    const Tensor P = ops::Pad(T, {1, 2, 0, 1}, 9.0f);
    EXPECT_EQ(shape_of(P), (std::vector<int64_t>{3, 6}));
    expect_close(host_values<float>(P),
                 {9, 1, 2, 3, 9, 9,
                  9, 4, 5, 6, 9, 9,
                  9, 9, 9, 9, 9, 9},
                 "pad");

    // 负 padding 是裁剪。
    const Tensor C = ops::Pad(T, {-1, -1});
    EXPECT_EQ(shape_of(C), (std::vector<int64_t>{2, 1}));
    expect_close(host_values<float>(C), {2, 5}, "pad crop");
    EXPECT_THROW(ops::Pad(T, {1}), std::runtime_error);
}

TEST_F(DeviceOps, RepeatTilesLikeNumpy) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{2, 3}), kTestDevice);

    const Tensor R = ops::Repeat(T, {2, 1});
    EXPECT_EQ(shape_of(R), (std::vector<int64_t>{4, 3}));
    // tile 语义：整块重复，不是逐元素重复。
    expect_close(host_values<float>(R), {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, "repeat");

    // repeats 比秩长时，前面按 1 维补齐。
    std::vector<float> w{7, 8};
    const Tensor V = moved_to(Tensor(w, std::vector<int64_t>{2}), kTestDevice);
    const Tensor R2 = ops::Repeat(V, {2, 3});
    EXPECT_EQ(shape_of(R2), (std::vector<int64_t>{2, 6}));
    std::vector<float> want2;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 6; ++j) want2.push_back(j % 2 == 0 ? 7.0f : 8.0f);
    expect_close(host_values<float>(R2), want2, "repeat 1-D");
}

TEST_F(DeviceOps, FlipAndRollReorderData) {
    std::vector<float> v{1, 2, 3, 4, 5, 6};
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{2, 3}), kTestDevice);

    const Tensor F = ops::Flip(T, {1});
    expect_close(host_values<float>(F), {3, 2, 1, 6, 5, 4}, "flip dim 1");
    const Tensor F2 = ops::Flip(T, {0, 1});
    expect_close(host_values<float>(F2), {6, 5, 4, 3, 2, 1}, "flip both");

    std::vector<float> r{0, 1, 2, 3, 4};
    const Tensor R = moved_to(Tensor(r, std::vector<int64_t>{5}), kTestDevice);
    expect_close(host_values<float>(ops::Roll(R, 2, 0)), {3, 4, 0, 1, 2}, "roll +2");
    expect_close(host_values<float>(ops::Roll(R, -1, 0)), {1, 2, 3, 4, 0}, "roll -1");
    expect_close(host_values<float>(ops::Roll(R, 0, 0)), r, "roll 0");
}

TEST_F(DeviceOps, LeadingViewsTransferTheirOwnElements) {
    // 视图可以连续、offset 为 0，却只覆盖底层 buffer 的一段（split/narrow 前导维）。
    // 搬运算子按 impl 整块搬运，所以 to_host/clone/item 必须先物化；写回这类
    // 视图则必须明确报错，而不是越界写。
    std::vector<float> v = ramp(12, 1.0f, 0.0f);  // 0..11
    const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{6, 2}), kTestDevice);

    const std::vector<Tensor> parts = ops::Split(T, 2, 0);
    const Tensor head = parts[0];  // {2,2}，offset 0，impl 仍是 12 个元素
    EXPECT_EQ(head.numel(), 4);
    expect_close(host_values<float>(head), {0, 1, 2, 3}, "to_host of a leading view");

    // clone 的 copy_to 也是整块拷贝：不物化时 Vulkan 直接抛错、CPU 越界写。
    const Tensor copied = head.clone();
    EXPECT_EQ(copied.device(), kTestDevice);
    EXPECT_EQ(shape_of(copied), (std::vector<int64_t>{2, 2}));
    expect_close(host_values<float>(copied), {0, 1, 2, 3}, "clone of a leading view");

    expect_close(host_values<float>(head.narrow(0, 1, 1)), {2, 3}, "narrow of a view");

    // item()：单元素视图不能按 impl 的 12 个元素往 16 字节栈缓冲写。
    const Tensor flat = moved_to(Tensor(v, std::vector<int64_t>{12}), kTestDevice);
    EXPECT_FLOAT_EQ(flat.narrow(0, 0, 1).item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(flat.narrow(0, 5, 1).item<float>(), 5.0f);

    // 写回：目标视图不覆盖整个 buffer → 各后端一致地拒绝。
    std::vector<float> w = ramp(4, 1.0f, 100.0f);  // 100..103
    const Tensor src = moved_to(Tensor(w, std::vector<int64_t>{2, 2}), kTestDevice);
    Tensor dst_view = ops::Split(T, 2, 0)[1];
    EXPECT_THROW(dst_view.copy_(src), std::runtime_error);

    // 覆盖整个 buffer 的目标仍然可写。
    std::vector<float> u = ramp(12, 1.0f, 50.0f);  // 50..61
    Tensor whole = moved_to(Tensor(v, std::vector<int64_t>{6, 2}), kTestDevice);
    const Tensor from = moved_to(Tensor(u, std::vector<int64_t>{6, 2}), kTestDevice);
    whole.copy_(from);
    expect_close(host_values<float>(whole), u, "copy_ into a whole tensor");
}

// offset 视图进内核：连续但 offset 非 0 的视图（narrow 前导维）在每个后端都要
// 给出与 CPU 相同的值。Vulkan 的内核从 buffer 基址寻址（描述符没有 offset），
// 靠 core 层的线性化/物化兜住 —— 这条用例把那个契约钉死。
TEST_F(DeviceOps, OffsetViewsFeedKernelsCorrectly) {
    std::vector<float> v = ramp(12, 1.0f, 0.0f);  // 0..11
    std::vector<int64_t> s{4, 3};
    const Tensor cpu = Tensor(v, s).narrow(0, 2, 2).clone();  // 6..11，CPU 参考
    const Tensor dev = moved_to(Tensor(v, s), kTestDevice);
    const Tensor row = dev.narrow(0, 2, 2);  // {2,3}，offset 6，仍然连续
    EXPECT_EQ(row.numel(), 6);
    EXPECT_TRUE(row.is_contiguous());

    // exp 跨三个数量级，绝对误差放不住：按相对误差比。
    {
        const std::vector<float> got = host_values<float>(ops::Exp(row));
        const std::vector<float> want = host_values<float>(ops::Exp(cpu));
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], want[i], 1e-5f * std::max(1.0f, std::abs(want[i])))
                << "exp on an offset view at " << i;
    }
    expect_same_as_cpu(ops::Sum(row), ops::Sum(cpu), kTestDevice, "sum on an offset view");
    expect_same_as_cpu(ops::Sum(row, 0), ops::Sum(cpu, 0), kTestDevice, "sum axis on an offset view");
    expect_same_as_cpu(ops::Softmax(row, 1), ops::Softmax(cpu, 1), kTestDevice,
                       "softmax on an offset view");
    expect_same_as_cpu(ops::Transpose(row, {1, 0}), ops::Transpose(cpu, {1, 0}), kTestDevice,
                       "transpose on an offset view");
    expect_same_as_cpu(ops::MatMul(row, ops::Transpose(row, {1, 0})),
                       ops::MatMul(cpu, ops::Transpose(cpu, {1, 0})), kTestDevice,
                       "matmul on an offset view");
    expect_same_as_cpu(ops::Concat({row, row}, 0), ops::Concat({cpu, cpu}, 0), kTestDevice,
                       "concat of offset views");
    expect_same_as_cpu(ops::Flip(row, {0}), ops::Flip(cpu, {0}), kTestDevice,
                       "flip on an offset view");

    std::vector<float> wv = ramp(3, 0.5f, 0.25f);
    std::vector<float> bv = ramp(3, 0.1f, -0.1f);
    std::vector<int64_t> ws{3};
    const Tensor w = Tensor(wv, ws);
    const Tensor b = Tensor(bv, ws);
    expect_same_as_cpu(ops::LayerNorm(row, moved_to(Tensor(wv, ws), kTestDevice),
                                      moved_to(Tensor(bv, ws), kTestDevice), 1),
                       ops::LayerNorm(cpu, w, b, 1), kTestDevice,
                       "layer_norm on an offset view");
}

TEST_F(DeviceOps, SortAndTopkOnCpu) {
    // 这两个算子目前只有 CPU 内核：设备上走 FR-8 的显式错误路径。
    if (kTestDevice != Device::CPU) {
        std::vector<float> v{3, 1, 2};
        const Tensor T = moved_to(Tensor(v, std::vector<int64_t>{3}), kTestDevice);
        EXPECT_THROW(ops::Sort(T), std::runtime_error);
        EXPECT_THROW(ops::Topk(T, 2), std::runtime_error);
        GTEST_SKIP() << "sort/topk are CPU-only";
    }

    std::vector<float> v{3, 1, 2};
    const Tensor T(v, std::vector<int64_t>{3});
    const ops::SortResult asc = ops::Sort(T);
    expect_close(host_values<float>(asc.values), {1, 2, 3}, "sort values");
    EXPECT_EQ(host_values<int64_t>(asc.indices), (std::vector<int64_t>{1, 2, 0}));
    const ops::SortResult desc = ops::Sort(T, 0, true);
    expect_close(host_values<float>(desc.values), {3, 2, 1}, "sort desc");
    EXPECT_EQ(host_values<int64_t>(desc.indices), (std::vector<int64_t>{0, 2, 1}));

    // 2-D、沿 dim 1 排序，各行列独立。
    std::vector<float> m{3, 1, 2, 6, 4, 5};
    const Tensor M(m, std::vector<int64_t>{2, 3});
    const ops::SortResult ms = ops::Sort(M, 1);
    expect_close(host_values<float>(ms.values), {1, 2, 3, 4, 5, 6}, "sort 2-D");
    EXPECT_EQ(host_values<int64_t>(ms.indices), (std::vector<int64_t>{1, 2, 0, 1, 2, 0}));

    const ops::TopkResult tk = ops::Topk(M, 2, 1);
    EXPECT_EQ(shape_of(tk.values), (std::vector<int64_t>{2, 2}));
    expect_close(host_values<float>(tk.values), {3, 2, 6, 5}, "topk values");
    EXPECT_EQ(host_values<int64_t>(tk.indices), (std::vector<int64_t>{0, 2, 0, 2}));
    const ops::TopkResult small = ops::Topk(M, 2, 1, false);
    expect_close(host_values<float>(small.values), {1, 2, 4, 5}, "topk smallest");
    EXPECT_THROW(ops::Topk(M, 4, 1), std::runtime_error);
}

// ===== FR-10 / FR-12：推理关键算子的跨后端一致性 =====

TEST_F(DeviceOps, NormAndSoftmaxFamilyMatchesCpu) {
    std::vector<float> v = ramp(24, 0.1f, -1.0f);
    std::vector<int64_t> s{4, 6};
    const Tensor cpu = Tensor(v, s);
    const Tensor dev = moved_to(Tensor(v, s), kTestDevice);

    expect_same_as_cpu(ops::LogSoftmax(dev, 1), ops::LogSoftmax(cpu, 1), kTestDevice, "log_softmax");
    expect_same_as_cpu(ops::Var(dev, 0), ops::Var(cpu, 0), kTestDevice, "var axis 0");
    expect_same_as_cpu(ops::Std(dev, 1, true), ops::Std(cpu, 1, true), kTestDevice, "std keepdim");
    expect_same_as_cpu(ops::Norm(dev, 2.0f, 1, false), ops::Norm(cpu, 2.0f, 1, false),
                       kTestDevice, "norm p=2");
    expect_same_as_cpu(ops::Norm(dev, 1.0f, 0, true), ops::Norm(cpu, 1.0f, 0, true),
                       kTestDevice, "norm p=1");
    expect_same_as_cpu(ops::Var(dev), ops::Var(cpu), kTestDevice, "var full");
    expect_same_as_cpu(ops::Std(dev), ops::Std(cpu), kTestDevice, "std full");
    expect_same_as_cpu(ops::Norm(dev, 2.0f), ops::Norm(cpu, 2.0f), kTestDevice, "norm full");

    // 层归一化：weight/bias 覆盖尾部块（axis=1 → 每个 6 元素行归一化）。
    std::vector<float> wv = ramp(6, 0.05f, 0.5f);
    std::vector<float> bv = ramp(6, 0.02f, -0.2f);
    std::vector<int64_t> ws{6};
    const Tensor w = Tensor(wv, ws);
    const Tensor b = Tensor(bv, ws);
    const Tensor dw = moved_to(Tensor(wv, ws), kTestDevice);
    const Tensor db = moved_to(Tensor(bv, ws), kTestDevice);

    expect_same_as_cpu(ops::LayerNorm(dev, dw, db, 1), ops::LayerNorm(cpu, w, b, 1),
                       kTestDevice, "layer_norm");
    expect_same_as_cpu(ops::RmsNorm(dev, dw, 1), ops::RmsNorm(cpu, w, 1), kTestDevice, "rms_norm");
}

// Prod 的语义基线（与 PyTorch 的 torch.prod 对照）：全归约返回 0-dim、单/多轴、
// keepdim、负轴、空张量（空乘积为 1）、整型提升到 INT64、dtype 限定、BOOL 拒绝。
// 参考值是手算的确定值，不经过被测代码；设备结果与 CPU 参考逐项比对。
TEST_F(DeviceOps, ProdReductionMatchesReference) {
    const Tensor cpu = from_values<float>({1, 2, 3, 4}, {2, 2});
    const Tensor dev = moved_to(from_values<float>({1, 2, 3, 4}, {2, 2}), kTestDevice);

    // 全归约：0-dim，空集合之外的最小情形 1*2*3*4 = 24。
    const Tensor full = ops::Prod(dev);
    EXPECT_EQ(full.device(), kTestDevice);
    EXPECT_EQ(full.dims(), 0);
    EXPECT_FLOAT_EQ(full.item<float>(), 24.0f);
    expect_same_as_cpu(ops::Prod(dev), ops::Prod(cpu), kTestDevice, "prod full");

    // 单轴与负轴：axis 0 → [3, 8]；axis 1 → [2, 12]；-1 与 1 等价。
    expect_same_as_cpu(ops::Prod(dev, 0), ops::Prod(cpu, 0), kTestDevice, "prod axis 0");
    expect_same_as_cpu(ops::Prod(dev, 1), ops::Prod(cpu, 1), kTestDevice, "prod axis 1");
    expect_same_as_cpu(ops::Prod(dev, -1), ops::Prod(cpu, -1), kTestDevice, "prod axis -1");

    // keepdim：被归约的轴保留为长度 1。
    const Tensor kd = ops::Prod(dev, 1, true);
    EXPECT_EQ(shape_of(kd), (std::vector<int64_t>{2, 1}));
    expect_close(host_values<float>(kd), {2.0f, 12.0f}, "prod keepdim");

    // 多轴：initializer_list 与 vector 两条重载；keepdim 的多轴结果是 {1,1}。
    const Tensor both = ops::Prod(dev, {0, 1});
    EXPECT_EQ(both.dims(), 0);
    EXPECT_FLOAT_EQ(both.item<float>(), 24.0f);
    const Tensor both_kd = ops::Prod(dev, std::vector<int64_t>{0, 1}, true);
    EXPECT_EQ(shape_of(both_kd), (std::vector<int64_t>{1, 1}));
    expect_same_as_cpu(ops::Prod(dev, {0, 1}), ops::Prod(cpu, {0, 1}), kTestDevice,
                       "prod multi-axis");

    // dtype 限定：先按归约规则求值再 cast，FLOAT64 保持 FLOAT64。
    const Tensor as_f64 = ops::Prod(dev, DataType::FLOAT64);
    EXPECT_EQ(as_f64.dtype(), DataType::FLOAT64);
    EXPECT_FLOAT_EQ(as_f64.item<double>(), 24.0);

    // 3-D 双轴归约：{2,2,3} 归约 {1,2} → {2}；1..6 与 7..12 两段各自连乘。
    const Tensor cpu3 = from_values<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    const Tensor dev3 = moved_to(
        from_values<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3}), kTestDevice);
    expect_same_as_cpu(ops::Prod(dev3, {1, 2}), ops::Prod(cpu3, {1, 2}), kTestDevice,
                       "prod 3d two axes");
    expect_close(host_values<float>(ops::Prod(dev3, {1, 2})), {720.0f, 665280.0f},
                 "prod 3d two axes values");

    // 非连续输入：切末维的 narrow 是跨步视图，按 strided 路径物化，结果与 CPU 一致。
    const Tensor cpu_s = from_values<float>({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor dev_s = moved_to(from_values<float>({1, 2, 3, 4, 5, 6}, {2, 3}), kTestDevice);
    const Tensor cpu_t = cpu_s.narrow(1, 1, 2);   // [[2,3],[5,6]]，跨步视图
    const Tensor dev_t = dev_s.narrow(1, 1, 2);
    EXPECT_FALSE(dev_t.is_contiguous());
    expect_same_as_cpu(ops::Prod(dev_t, 1), ops::Prod(cpu_t, 1), kTestDevice,
                       "prod on a strided view");
    expect_same_as_cpu(ops::Prod(dev_t), ops::Prod(cpu_t), kTestDevice,
                       "prod full on a strided view");

    // 空张量：空乘积为 1（全归约 0-dim；空轴归约出全 1）。
    const Tensor empty = moved_to(Tensor(std::vector<int64_t>{0}, DataType::FLOAT32),
                                  kTestDevice);
    const Tensor empty_full = ops::Prod(empty);
    EXPECT_EQ(empty_full.dims(), 0);
    EXPECT_FLOAT_EQ(empty_full.item<float>(), 1.0f);
    const Tensor empty_axis = ops::Prod(
        moved_to(Tensor(std::vector<int64_t>{2, 0}, DataType::FLOAT32), kTestDevice), 1);
    expect_close(host_values<float>(empty_axis), {1.0f, 1.0f}, "prod over an empty axis");

    // 整型提升：INT32/INT64 输入的归约结果是 INT64（reduction_type 规则）。
    const Tensor ints = moved_to(from_values<int32_t>({2, 3, 4}, {3}), kTestDevice);
    const Tensor int_full = ops::Prod(ints);
    EXPECT_EQ(int_full.dtype(), DataType::INT64);
    EXPECT_EQ(int_full.item<int64_t>(), 24);
    const Tensor int_axis = ops::Prod(
        moved_to(from_values<int64_t>({2, 3, 4, 5}, {2, 2}), kTestDevice), 0);
    EXPECT_EQ(int_axis.dtype(), DataType::INT64);
    expect_close(host_values<int64_t>(int_axis), {8, 15}, "prod int64 axis");

    // BOOL 拒绝：归约结果是数值，谓词形式请用 All/Any。
    const Tensor flag = moved_to(Tensor::Zeros({3}, DataType::BOOL), kTestDevice);
    EXPECT_THROW(ops::Prod(flag), std::runtime_error);
}

TEST_F(DeviceOps, GeneratorsLandOnTheDevice) {
    TensorOptions opts(DataType::FLOAT32);
    opts.with_device(kTestDevice, 0);

    const Tensor ar = ops::Arange(0.0f, 6.0f, 1.0f, opts);
    EXPECT_EQ(ar.device(), kTestDevice);
    expect_close(host_values<float>(ar), {0, 1, 2, 3, 4, 5}, "arange");

    const Tensor ls = ops::Linspace(0.0f, 1.0f, 5, opts);
    EXPECT_EQ(ls.device(), kTestDevice);
    expect_close(host_values<float>(ls), {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}, "linspace");

    const Tensor eye = ops::Eye(3, opts);
    EXPECT_EQ(eye.device(), kTestDevice);
    expect_close(host_values<float>(eye), {1, 0, 0, 0, 1, 0, 0, 0, 1}, "eye");

    const Tensor rn = ops::Randn(std::vector<int64_t>{64}, 0.0f, 1.0f, opts);
    EXPECT_EQ(rn.device(), kTestDevice);
    EXPECT_EQ(rn.numel(), 64u);
}

// where / index_select / gather / cumsum 是注意力与 embedding 的骨架：三路广播、
// 沿轴查表、沿轴前缀和。参考值直接写死，不经由被测代码。
TEST_F(DeviceOps, SelectIndexAndScanMatchReference) {
    // where：cond(2,1) 广播到 (2,3)，值来自 a 或 b。
    const Tensor cond = moved_to(from_values<float>({1, 0}, {2, 1}), kTestDevice);
    const Tensor wa = moved_to(from_values<float>({1, 2, 3, 4, 5, 6}, {2, 3}), kTestDevice);
    const Tensor wb = moved_to(from_values<float>({-1, -2, -3, -4, -5, -6}, {2, 3}),
                               kTestDevice);
    const Tensor w = ops::Where(cond, wa, wb);
    EXPECT_EQ(w.device(), kTestDevice);
    EXPECT_EQ(w.dtype(), DataType::FLOAT32);
    expect_close(host_values<float>(w), {1, 2, 3, -4, -5, -6}, "where");

    // 标量版本两个方向。
    expect_close(host_values<float>(ops::Where(cond, wa, 7.0f)),
                 {1, 2, 3, 7, 7, 7}, "where scalar b");
    expect_close(host_values<float>(ops::Where(cond, 7.0f, wb)),
                 {7, 7, 7, -4, -5, -6}, "where scalar a");

    // 非 BOOL 的 cond：非零即真，与 CPU 一致。
    const Tensor num_cond = moved_to(from_values<std::int32_t>({0, 2}, {2, 1}), kTestDevice);
    expect_close(host_values<float>(ops::Where(num_cond, wa, wb)),
                 {-1, -2, -3, 4, 5, 6}, "where int cond");

    // index_select：dim=0 的 embedding 查表与 dim=1 的列选取。
    const Tensor emb = moved_to(from_values<float>(ramp(12, 1.0f, 0.0f), {4, 3}), kTestDevice);
    const Tensor row_idx = moved_to(from_values<std::int32_t>({2, 0, 2}, {3}), kTestDevice);
    const Tensor sel = ops::IndexSelect(emb, 0, row_idx);
    EXPECT_EQ(sel.device(), kTestDevice);
    expect_close(host_values<float>(sel), {6, 7, 8, 0, 1, 2, 6, 7, 8}, "index_select dim 0");

    const Tensor col_idx = moved_to(from_values<std::int64_t>({2, 0}, {2}), kTestDevice);
    expect_close(host_values<float>(ops::IndexSelect(emb, 1, col_idx)),
                 {2, 0, 5, 3, 8, 6, 11, 9}, "index_select dim 1");

    // 越界索引必须抛异常，而不是给出静默的错误结果。
    const Tensor bad = moved_to(from_values<std::int32_t>({5}, {1}), kTestDevice);
    EXPECT_THROW(ops::IndexSelect(emb, 0, bad), std::runtime_error);

    // gather：PyTorch 语义，index 与 input 同秩。
    const Tensor g = moved_to(from_values<float>({1, 2, 3, 4}, {2, 2}), kTestDevice);
    const Tensor gi = moved_to(from_values<std::int32_t>({0, 0, 1, 0}, {2, 2}), kTestDevice);
    expect_close(host_values<float>(ops::Gather(g, 1, gi)), {1, 1, 4, 3}, "gather dim 1");
    const Tensor gi0 = moved_to(from_values<std::int32_t>({0, 1, 1, 0}, {2, 2}), kTestDevice);
    expect_close(host_values<float>(ops::Gather(g, 0, gi0)), {1, 4, 3, 2}, "gather dim 0");

    // 秩与非 dim 维尺寸是公共入口的契约：设备内核直接拿 index 的形状解码输入
    // 坐标，放行就会越界读。这里确认每个后端都在入口处拒绝。
    const Tensor gi_1d = moved_to(from_values<std::int32_t>({0, 1}, {2}), kTestDevice);
    EXPECT_THROW(ops::Gather(g, 0, gi_1d), std::runtime_error);
    const Tensor gi_wide = moved_to(from_values<std::int32_t>({0, 0, 0, 0, 0, 0}, {3, 2}),
                                    kTestDevice);
    EXPECT_THROW(ops::Gather(g, 1, gi_wide), std::runtime_error);
    // 越界索引在 gather 上同样必须抛异常。
    const Tensor gi_bad = moved_to(from_values<std::int32_t>({0, 2, 1, 0}, {2, 2}), kTestDevice);
    EXPECT_THROW(ops::Gather(g, 1, gi_bad), std::runtime_error);

    // cumsum：两个轴各一条，覆盖 outer/inner 都非 1 的情形。
    const Tensor m = moved_to(from_values<float>(ramp(6, 1.0f, 1.0f), {2, 3}), kTestDevice);
    const Tensor cs1 = ops::Cumsum(m, 1);
    EXPECT_EQ(cs1.device(), kTestDevice);
    expect_close(host_values<float>(cs1), {1, 3, 6, 4, 9, 15}, "cumsum axis 1");
    expect_close(host_values<float>(ops::Cumsum(m, 0)), {1, 2, 3, 5, 7, 9},
                 "cumsum axis 0");

    // 更高秩：cumsum 沿中间轴，验证 outer/axis/inner 三段几何。
    const Tensor cube = moved_to(from_values<float>(ramp(12, 1.0f, 0.0f), {2, 2, 3}),
                                 kTestDevice);
    expect_close(host_values<float>(ops::Cumsum(cube, 1)),
                 {0, 1, 2, 3, 5, 7, 6, 7, 8, 15, 17, 19}, "cumsum axis 1 on 3-D");
}

namespace {

// 卷积的参考实现：按定义逐个输出位置求和，不经过被测算子的任何一部分。
// 输入 [N, C, H, W]、权重 [Cout, C, kh, kw]、bias 可空，输出 [N, Cout, OH, OW]。
std::vector<float> conv_reference(const std::vector<float>& in, int64_t n, int64_t c,
                                  int64_t h, int64_t w, const std::vector<float>& weight,
                                  const std::vector<float>* bias, int64_t cout, int64_t kh,
                                  int64_t kw, int64_t stride, int64_t pad) {
    const int64_t oh = (h + 2 * pad - kh) / stride + 1;
    const int64_t ow = (w + 2 * pad - kw) / stride + 1;
    std::vector<float> out(static_cast<size_t>(n * cout * oh * ow));
    for (int64_t b = 0; b < n; ++b)
        for (int64_t oc = 0; oc < cout; ++oc)
            for (int64_t oy = 0; oy < oh; ++oy)
                for (int64_t ox = 0; ox < ow; ++ox) {
                    double acc = bias ? (*bias)[static_cast<size_t>(oc)] : 0.0;
                    for (int64_t ic = 0; ic < c; ++ic)
                        for (int64_t i = 0; i < kh; ++i)
                            for (int64_t j = 0; j < kw; ++j) {
                                const int64_t y = oy * stride + i - pad;
                                const int64_t x = ox * stride + j - pad;
                                if (y < 0 || y >= h || x < 0 || x >= w) continue;
                                acc += static_cast<double>(in[static_cast<size_t>(
                                           ((b * c + ic) * h + y) * w + x)]) *
                                       static_cast<double>(weight[static_cast<size_t>(
                                           ((oc * c + ic) * kh + i) * kw + j)]);
                            }
                    out[static_cast<size_t>(((b * cout + oc) * oh + oy) * ow + ox)] =
                        static_cast<float>(acc);
                }
    return out;
}

}  // namespace

// 带 batch 的 3x3 卷积 + same padding：覆盖 padding 的四个边角与中间位置。
TEST_F(DeviceOps, Conv2dBatchedMatchesReference) {
    constexpr int64_t n = 2, c = 2, h = 5, w = 6, cout = 3, kh = 3, kw = 3;
    // 取值压在一个小量级：卷积输出是几十个乘积之和，绝对误差随量级线性增长，
    // 量级大了 kEps 比较的就变成浮点累加顺序，而不是语义。
    std::vector<float> iv = ramp(n * c * h * w, 0.02f, -1.0f);
    std::vector<float> wv = ramp(cout * c * kh * kw, 0.01f, -0.5f);
    std::vector<float> bv{0.05f, -0.025f, 0.1f};

    const Tensor input  = moved_to(from_values<float>(iv, {n, c, h, w}), kTestDevice);
    const Tensor weight = moved_to(from_values<float>(wv, {cout, c, kh, kw}), kTestDevice);
    const Tensor bias   = moved_to(from_values<float>(bv, {cout}), kTestDevice);

    const Tensor out = ops::Conv2d(input, weight, bias, /*stride=*/1, /*padding=*/1);
    EXPECT_EQ(out.device(), kTestDevice);
    EXPECT_EQ(out.dtype(), DataType::FLOAT32);
    EXPECT_EQ(shape_of(out), (std::vector<int64_t>{n, cout, h, w}));

    expect_close(host_values<float>(out),
                 conv_reference(iv, n, c, h, w, wv, &bv, cout, kh, kw, 1, 1), "conv2d same pad");
}

// 不带 batch 的 [C, H, W] 输入与 stride > 1：输出按 (H - kh)/stride + 1 收缩，
// 且不再带 batch 维。
TEST_F(DeviceOps, Conv2dUnbatchedStridedMatchesReference) {
    constexpr int64_t c = 3, h = 7, w = 7, cout = 2, kh = 2, kw = 2;
    std::vector<float> iv = ramp(c * h * w, 0.02f, -1.0f);
    std::vector<float> wv = ramp(cout * c * kh * kw, 0.01f, -0.5f);

    const Tensor input  = moved_to(from_values<float>(iv, {c, h, w}), kTestDevice);
    const Tensor weight = moved_to(from_values<float>(wv, {cout, c, kh, kw}), kTestDevice);

    const Tensor out = ops::Conv2d(input, weight, /*stride=*/3, /*padding=*/0);
    EXPECT_EQ(out.device(), kTestDevice);
    EXPECT_EQ(shape_of(out), (std::vector<int64_t>{cout, 2, 2}));

    const std::vector<float> want =
        conv_reference(iv, 1, c, h, w, wv, nullptr, cout, kh, kw, 3, 0);
    expect_close(host_values<float>(out), want, "conv2d unbatched strided");
}

// 没有 bias 与给一份全零 bias 必须同值：两条路径在设备上的绑定方式不同，
// 结果不能因此分叉。
TEST_F(DeviceOps, Conv2dWithoutBiasMatchesZeroBias) {
    constexpr int64_t n = 1, c = 2, h = 4, w = 4, cout = 2, kh = 3, kw = 3;
    std::vector<float> iv = ramp(n * c * h * w, 0.02f, -1.0f);
    std::vector<float> wv = ramp(cout * c * kh * kw, 0.01f, -0.5f);
    std::vector<float> zv(static_cast<size_t>(cout), 0.0f);

    const Tensor input  = moved_to(from_values<float>(iv, {n, c, h, w}), kTestDevice);
    const Tensor weight = moved_to(from_values<float>(wv, {cout, c, kh, kw}), kTestDevice);
    const Tensor zero   = moved_to(from_values<float>(zv, {cout}), kTestDevice);

    const Tensor no_bias  = ops::Conv2d(input, weight, 1, 1);
    const Tensor with_bias = ops::Conv2d(input, weight, zero, 1, 1);
    expect_close(host_values<float>(no_bias), host_values<float>(with_bias),
                 "conv2d: no bias vs zero bias");
}

// 形状与取值域的错误必须在公共入口被拒绝：设备内核不做这些检查。
TEST_F(DeviceOps, Conv2dRejectsBadShapes) {
    std::vector<float> iv = ramp(1 * 2 * 4 * 4, 0.1f, 0.0f);
    std::vector<float> wv = ramp(2 * 2 * 3 * 3, 0.1f, 0.0f);
    const Tensor input  = moved_to(from_values<float>(iv, {1, 2, 4, 4}), kTestDevice);
    const Tensor weight = moved_to(from_values<float>(wv, {2, 2, 3, 3}), kTestDevice);

    // 通道数对不上
    std::vector<float> wv_bad = ramp(2 * 3 * 3 * 3, 0.1f, 0.0f);
    const Tensor weight_bad = moved_to(from_values<float>(wv_bad, {2, 3, 3, 3}), kTestDevice);
    EXPECT_THROW(ops::Conv2d(input, weight_bad), std::runtime_error);

    // 输入秩不合法
    std::vector<float> iv_2d = ramp(2 * 4, 0.1f, 0.0f);
    const Tensor input_2d = moved_to(from_values<float>(iv_2d, {2, 4}), kTestDevice);
    EXPECT_THROW(ops::Conv2d(input_2d, weight), std::runtime_error);

    // 核比输入（含 padding）还大 -> 输出尺寸非正
    std::vector<float> wv_big = ramp(2 * 2 * 5 * 5, 0.1f, 0.0f);
    const Tensor weight_big = moved_to(from_values<float>(wv_big, {2, 2, 5, 5}), kTestDevice);
    EXPECT_THROW(ops::Conv2d(input, weight_big, /*stride=*/1, /*padding=*/0),
                 std::runtime_error);

    // stride 至少为 1、padding 不能为负
    EXPECT_THROW(ops::Conv2d(input, weight, 0, 1), std::runtime_error);
    EXPECT_THROW(ops::Conv2d(input, weight, 1, -1), std::runtime_error);

    // bias 元素数必须等于 Cout
    std::vector<float> b_bad(3, 0.0f);
    const Tensor bias_bad = moved_to(from_values<float>(b_bad, {3}), kTestDevice);
    EXPECT_THROW(ops::Conv2d(input, weight, bias_bad, 1, 1), std::runtime_error);

    // 整型输入：卷积只对浮点定义
    std::vector<std::int32_t> ivm = {1, 2, 3, 4};
    std::vector<std::int32_t> wvm(4, 1);
    const Tensor input_i = moved_to(from_values<std::int32_t>(ivm, {1, 1, 2, 2}), kTestDevice);
    const Tensor weight_i = moved_to(from_values<std::int32_t>(wvm, {1, 1, 2, 2}), kTestDevice);
    EXPECT_THROW(ops::Conv2d(input_i, weight_i), std::runtime_error);
}

TEST_F(DeviceOps, UnsupportedOpsFailExplicitlyOnDevices) {
    std::vector<float> tv = ramp(6);
    const Tensor T = moved_to(Tensor(tv, std::vector<int64_t>{2, 3}), kTestDevice);
    std::vector<int64_t> iv{0, 1};
    const Tensor I = moved_to(Tensor(iv, std::vector<int64_t>{2}), kTestDevice);
    const Tensor pred = ops::Greater(T, 0.0f);

    // 条件选择、索引与累积和已经落到每个后端上，不再属于这张清单。
    EXPECT_NO_THROW(ops::Cumsum(T, 1));
    EXPECT_NO_THROW(ops::Where(pred, T, T));
    EXPECT_NO_THROW(ops::IndexSelect(T, 0, I));
    // gather 的索引与输入同秩：dim 0 上形状 {2,3} 的索引取到的是一份重排。
    std::vector<int64_t> gv{0, 1, 1, 0, 1, 0};
    const Tensor I2 = moved_to(Tensor(gv, std::vector<int64_t>{2, 3}), kTestDevice);
    EXPECT_NO_THROW(ops::Gather(T, 0, I2));

    if (kTestDevice == Device::CPU) {
        // CPU 都实现了，这里只是确认它们不是被意外禁用。
        EXPECT_NO_THROW(ops::Erf(T));
        GTEST_SKIP() << "the CPU backend implements every op in this list";
    }
    EXPECT_THROW(ops::Erf(T), std::runtime_error);
    EXPECT_THROW(ops::Sort(T), std::runtime_error);
    EXPECT_THROW(ops::Topk(T, 1), std::runtime_error);

    // 错误必须点名后端，而不是悄悄退回 CPU 或先算出错误结果。
    try {
        (void)ops::Erf(T);
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(device_to_string(kTestDevice)), std::string::npos)
            << e.what();
    }
}
