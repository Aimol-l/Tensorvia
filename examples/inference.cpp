import std;
import tensorvia;

// LLM 推理模拟，按真实推理引擎的四个阶段走：
//
//   text -> tokens -> embedding -> prefill -> decode(自回归) -> logits
//
//   text      字节级分词器（真实模型换成 BPE / SentencePiece，接口一样是 encode/decode）
//   tokens    每个字节一个 id，词表 256
//   embedding 词表查表 + 位置查表，都是 IndexSelect
//   prefill   整段 prompt 一次前向：因果掩码的注意力、整段 K/V 写进 cache，
//             得到 [L, V] logits（每个位置一个分布）
//   decode    一次只喂一个新 token：Q 只有一行，K/V 追加到 cache，注意力读全量 cache，
//             得到 [1, V] logits。这就是“自回归”那一步，也是 KV cache 存在的理由
//
// 权重全部是随机数：形状是对的，数值没有意义——所以生成的字节串不会像人话，
// 这段程序验证的是流水线本身能不能跑通、prefill 与 decode 两条路径能不能对上。
//
// 用法：inference [cpu|cuda|vulkan|sycl] [prompt]

namespace {

constexpr std::int64_t kVocab = 256;  // 字节级分词：id 就是字节
constexpr std::int64_t kDModel = 32;
constexpr std::int64_t kHeads = 2;
constexpr std::int64_t kHeadDim = kDModel / kHeads;
constexpr std::int64_t kFF = 4 * kDModel;
constexpr std::int64_t kLayers = 2;
constexpr std::int64_t kNewTokens = 8;  // decode 阶段生成几个 token
constexpr std::int64_t kMaxPos = 64;    // 位置表容量：prompt + 生成长度不能超过它

// ---------------------------------------------------------------------------
// 分词器：每个 UTF-8 字节就是一个 token。真实模型在这里换成 BPE，模型侧不用改。
// ---------------------------------------------------------------------------
std::vector<std::int64_t> encode(std::string_view text) {
    std::vector<std::int64_t> ids;
    ids.reserve(text.size());
    for (const unsigned char c : text) ids.push_back(static_cast<std::int64_t>(c));
    return ids;
}

std::string decode(const std::vector<std::int64_t>& ids) {
    std::string bytes;
    bytes.reserve(ids.size());
    for (const std::int64_t id : ids) bytes.push_back(static_cast<char>(id));
    return bytes;
}

// 随机权重生成的字节大多不可打印，转义一下才看得清是“一串字节”而不是乱码。
std::string escaped(std::string_view bytes) {
    std::string out;
    for (const unsigned char c : bytes) {
        if (c >= 0x20 && c < 0x7f && c != '\\') {
            out.push_back(static_cast<char>(c));
        } else {
            out += std::format("\\x{:02x}", c);
        }
    }
    return out;
}

// 推理只要求形状对得上，参数取值无所谓；这里统一用零均值的小方差正态。
Tensor rand_param(std::vector<std::int64_t> shape, Device dev, float spread = 0.05f) {
    return ops::Randn(shape, DataType::FLOAT32, 0.0f, spread, dev);
}

// 因果掩码：第 i 行只能看到 0..i。只在 prefill 用——decode 时 Q 只有一行，
// 它要看的全在 cache 里，天然没有“偷看未来”的问题。
Tensor causal_mask(std::int64_t len, Device dev) {
    std::vector<std::int32_t> flags(static_cast<std::size_t>(len * len));
    for (std::int64_t i = 0; i < len; ++i)
        for (std::int64_t j = 0; j < len; ++j)
            flags[static_cast<std::size_t>(i * len + j)] = (j <= i) ? 1 : 0;
    Tensor mask(flags, {len, len});
    mask.to_device(dev);
    return mask;
}

// 每层一份 K/V cache，形状 [H, 已算过多少个位置, hd]。
struct KVCache {
    Tensor k, v;
};

struct DecoderLayer {
    Tensor ln1_w, ln1_b;
    Tensor wq, wk, wv, wo;
    Tensor ln2_w, ln2_b;
    Tensor w1, b1, w2, b2;

    explicit DecoderLayer(Device dev)
        : ln1_w(rand_param({kDModel}, dev, 1.0f)),
          ln1_b(rand_param({kDModel}, dev)),
          wq(rand_param({kDModel, kDModel}, dev)),
          wk(rand_param({kDModel, kDModel}, dev)),
          wv(rand_param({kDModel, kDModel}, dev)),
          wo(rand_param({kDModel, kDModel}, dev)),
          ln2_w(rand_param({kDModel}, dev, 1.0f)),
          ln2_b(rand_param({kDModel}, dev)),
          w1(rand_param({kFF, kDModel}, dev)),
          b1(rand_param({kFF}, dev)),
          w2(rand_param({kDModel, kFF}, dev)),
          b2(rand_param({kDModel}, dev)) {}

    // 注意力：x 是本次要算的 len 行（prefill 时是整段，decode 时是 1 行），
    // cache 里是之前所有位置。K/V 拼上新的行后写回 cache，注意力读全量 K/V。
    Tensor attend(const Tensor& x, std::int64_t len, KVCache& cache, bool causal) const {
        const auto split_heads = [len](const Tensor& t) {
            return t.reshape({len, kHeads, kHeadDim}).permute({1, 0, 2});  // [H, len, hd]
        };
        const Tensor q = split_heads(ops::Linear(x, wq));
        const Tensor k = split_heads(ops::Linear(x, wk));
        const Tensor v = split_heads(ops::Linear(x, wv));

        // 库目前没有跨步原地写，所以 cache 用 Concat 追加；真实实现会预分配
        // [H, max_len, hd] 并按位置写进去，省掉这次拷贝。
        const Tensor k_all = cache.k.numel() == 0 ? k : ops::Concat({cache.k, k}, 1);
        const Tensor v_all = cache.v.numel() == 0 ? v : ops::Concat({cache.v, v}, 1);
        cache.k = k_all;
        cache.v = v_all;

        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        Tensor scores = ops::MatMul(q, ops::Transpose(k_all, {0, 2, 1})) * scale;  // [H, len, t+len]
        if (causal) {
            // 用 -1e9 而不是 -inf：softmax 先减行最大值，exp(-1e9) 下溢成 0；
            // -inf 在整行都没有可见位置时会变成 nan。cond 是 [len, len]，
            // 沿头维广播到 [H, len, len]。
            scores = ops::Where(causal_mask(len, x.device()), scores, -1e9f);
        }
        const Tensor attn = ops::Softmax(scores, 2);
        return ops::MatMul(attn, v_all).permute({1, 0, 2}).reshape({len, kDModel});
    }

    // 注意力的结果送回残差流：residual + out_proj，再走 MLP。
    Tensor tail(const Tensor& x_in, const Tensor& attn_out) const {
        const Tensor x = ops::LayerNorm(x_in, ln1_w, ln1_b, -1);
        Tensor h = x + ops::Linear(attn_out, wo);
        h = ops::LayerNorm(h, ln2_w, ln2_b, -1);
        return h + ops::Linear(ops::Silu(ops::Linear(h, w1, b1)), w2, b2);
    }

    // prefill：整段输入，[L, D] -> [L, D]，K/V 一次写满 cache
    Tensor prefill(const Tensor& x_in, KVCache& cache) const {
        return tail(x_in, attend(x_in, x_in.shape()[0], cache, /*causal=*/true));
    }

    // decode：新 token 一行，[1, D] -> [1, D]，只往 cache 里追加一行
    Tensor decode(const Tensor& x_in, KVCache& cache) const {
        return tail(x_in, attend(x_in, 1, cache, /*causal=*/false));
    }
};

struct TinyGpt {
    Tensor tok_emb, pos_emb, head_w;
    std::vector<DecoderLayer> layers;

    explicit TinyGpt(Device dev)
        : tok_emb(rand_param({kVocab, kDModel}, dev, 0.1f)),
          pos_emb(rand_param({kMaxPos, kDModel}, dev)),
          head_w(rand_param({kVocab, kDModel}, dev)) {
        layers.reserve(kLayers);
        for (std::int64_t i = 0; i < kLayers; ++i) layers.emplace_back(dev);
    }

    // 每次推理会话一份新 cache；prefill 填它，decode 往里追加。
    std::vector<KVCache> new_cache() const { return std::vector<KVCache>(layers.size()); }

    // 词表查表 + 位置查表。`start` 是这批 token 的第一个位置：prefill 从 0 开始，
    // decode 时从 cache 已有的长度开始。
    Tensor embed(const Tensor& ids, std::int64_t start) const {
        const std::int64_t len = ids.shape()[0];
        std::vector<std::int64_t> pos_v(static_cast<std::size_t>(len));
        std::iota(pos_v.begin(), pos_v.end(), start);
        Tensor pos(pos_v, {len});
        pos.to_device(ids.device());
        return ops::IndexSelect(tok_emb, 0, ids) + ops::IndexSelect(pos_emb, 0, pos);
    }

    // prefill：整段 prompt 一次前向 -> logits [L, V]，同时把每层 K/V 写进 cache。
    Tensor prefill(const Tensor& ids, std::vector<KVCache>& caches) const {
        Tensor x = embed(ids, /*start=*/0);
        for (std::size_t i = 0; i < layers.size(); ++i) x = layers[i].prefill(x, caches[i]);
        return ops::Linear(x, head_w);
    }

    // decode：一个新 token -> logits [1, V]。位置由 cache 长度决定。
    Tensor decode_step(std::int64_t id, std::vector<KVCache>& caches) const {
        const std::int64_t start = caches.front().k.shape()[1];
        std::vector<std::int64_t> id_v{id};
        Tensor ids(id_v, {1});
        ids.to_device(caches.front().k.device());

        Tensor x = embed(ids, start);
        for (std::size_t i = 0; i < layers.size(); ++i) x = layers[i].decode(x, caches[i]);
        return ops::Linear(x, head_w);
    }

    std::size_t param_count() const {
        std::size_t n = tok_emb.numel() + pos_emb.numel() + head_w.numel();
        for (const DecoderLayer& l : layers) {
            n += l.ln1_w.numel() + l.ln1_b.numel() + l.wq.numel() + l.wk.numel() +
                 l.wv.numel() + l.wo.numel() + l.ln2_w.numel() + l.ln2_b.numel() +
                 l.w1.numel() + l.b1.numel() + l.w2.numel() + l.b2.numel();
        }
        return n;
    }
};

constexpr std::string_view kDeviceNames[] = {"cpu", "cuda", "vulkan", "sycl"};
constexpr Device kDeviceKinds[] = {Device::CPU, Device::CUDA, Device::VULKAN, Device::SYCL};

// 请求的加速器没编进这个构建（或机器上没有）时明确退回 CPU，不静默降级。
Device resolve_device(std::string_view name) {
    for (std::size_t i = 0; i < std::size(kDeviceNames); ++i) {
        if (name != kDeviceNames[i]) continue;
        if (kDeviceKinds[i] == Device::CPU) return Device::CPU;
        if (ops::device_count(kDeviceKinds[i]) > 0) return kDeviceKinds[i];
        std::println("{} is not available in this build; falling back to CPU",
                     device_to_string(kDeviceKinds[i]));
        return Device::CPU;
    }
    std::println("unknown device '{}'; falling back to CPU", name);
    return Device::CPU;
}

}  // namespace

int main(int argc, char** argv) {
    const Device dev = argc > 1 ? resolve_device(argv[1]) : Device::CPU;
    const std::string prompt_text = argc > 2 ? argv[2] : "Tensorvia runs inference";

    const TinyGpt model(dev);
    std::println("device {} | {} layers | d_model {} | heads {} | vocab {} | {:.1f}K params (random)",
                 device_to_string(dev), kLayers, kDModel, kHeads, kVocab,
                 static_cast<double>(model.param_count()) / 1000.0);

    // ---- 1. text -> tokens ----
    std::vector<std::int64_t> prompt_ids = encode(prompt_text);
    const std::int64_t prefill_len = static_cast<std::int64_t>(prompt_ids.size());
    if (prefill_len + kNewTokens > kMaxPos) {
        std::println("prompt is {} tokens; this demo caps the sequence at {}", prefill_len, kMaxPos);
        return 1;
    }
    std::println("text   {:?}", prompt_text);
    std::println("tokens {} ids (byte-level)", prefill_len);

    // ---- 2. prefill：整段 prompt 一次过 ----
    auto caches = model.new_cache();
    Tensor prompt_tensor(prompt_ids, {prefill_len});
    prompt_tensor.to_device(dev);

    const auto t_prefill = std::chrono::steady_clock::now();
    Tensor logits = model.prefill(prompt_tensor, caches);
    // SYCL 的 CPU 设备在这里还包含内核 JIT 的时间，第一次调用会明显偏慢。
    const auto prefill_us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t_prefill)
                                .count();
    std::println("embedding [{}, {}] -> prefill {} positions in {:.2f} ms -> logits [{}, {}]",
                 prefill_len, kDModel, prefill_len, prefill_us / 1000.0, logits.shape()[0],
                 logits.shape()[1]);
    std::println("kv cache {} layers x [{}, {}, {}] (heads, positions, head_dim)",
                 caches.size(), caches.front().k.shape()[0], caches.front().k.shape()[1],
                 caches.front().k.shape()[2]);

    // ---- 3. decode：自回归，一次一个 token ----
    // 只有最后一个位置的分布是“下一个 token”的预测——前面那些位置在 prefill 里
    // 已经算完了，decode 不会重算它们。
    std::vector<std::int64_t> generated = prompt_ids;
    Tensor last = logits.select(0, prefill_len - 1);  // [V]

    const auto t_decode = std::chrono::steady_clock::now();
    for (std::int64_t step = 0; step < kNewTokens; ++step) {
        Tensor probs = ops::Softmax(last, 0);
        const std::int64_t next = ops::Argmax(probs, 0).item<std::int64_t>();

        // 只有这一步是跨设备读：取被选中 token 的概率
        probs.to_host();
        std::println("decode step {}: pos {:>2} -> token {:>3} {:?}  p = {:.3f}", step + 1,
                     prefill_len + step, next,
                     escaped(std::string(1, static_cast<char>(next))), probs.at<float>({next}));
        generated.push_back(next);

        // 把新 token 塞回模型，得到它这个位置的分布——下一步的输入。
        // 最后一轮算出来的分布预测的是第 9 个 token，生成时用不上，但下面拿它做自检。
        last = model.decode_step(next, caches).select(0, 0);  // [1, V] -> [V]
    }
    const auto decode_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t_decode)
                               .count();
    std::println("decode: {} tokens in {:.2f} ms ({:.1f} tokens/s)", kNewTokens, decode_ms,
                 static_cast<double>(kNewTokens) / (decode_ms / 1000.0));

    // ---- 4. 结果 ----
    std::println("output bytes {:?} (random weights: no meaning, the pipeline is the point)",
                 escaped(decode(generated)));

    // 自检：用全新 cache 把整段重跑一遍 prefill，最后一个位置的 logits 应当与
    // decode 走 cache 得到的同一位置一致——不一致就说明 cache 或掩码写错了。
    auto fresh = model.new_cache();
    Tensor all_ids(generated, {static_cast<std::int64_t>(generated.size())});
    all_ids.to_device(dev);
    const Tensor replay = model.prefill(all_ids, fresh);
    const Tensor replayed_last = replay.select(0, replay.shape()[0] - 1);  // [V]
    const float max_delta =
        ops::Max(ops::Abs(ops::Sub(replayed_last, last))).item<float>();
    const bool same_argmax =
        ops::Argmax(replayed_last, 0).item<std::int64_t>() == ops::Argmax(last, 0).item<std::int64_t>();
    std::println("prefill(replay) vs decode last position: argmax {}, max |delta| = {:.2e}",
                 same_argmax ? "match" : "MISMATCH",
                 static_cast<double>(std::abs(max_delta)));

    ops::synchronize();
    return same_argmax ? 0 : 1;
}
