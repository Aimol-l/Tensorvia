import std;
import tensorvia;

// CNN 推理模拟：图像 -> [conv -> relu -> maxpool] x2 -> fc -> logits
//
//   image      程序在宿主画一张 32x32 的图（演示不依赖图片文件与解码库）
//   ConvBlock  一个卷积块：ops::Conv2d -> Relu -> 2x2 maxpool
//   Cnn        两层卷积块加一层全连接；权重随机，形状与数据通路是真的
//
// ops::Conv2d 的语义与 torch.nn.functional.conv2d 一致（NCHW、same padding、
// stride 1）。各后端有自己的实现：CPU 直接卷积，CUDA/SYCL 一个线程一个输出元素，
// Vulkan 一份 GLSL。maxpool 用现有算子拼出来：reshape + permute 把 2x2 的四个
// 候选挪到最后一维，再 Max(axis=-1)。
//
// 权重布局沿用 PyTorch 的卷积约定 [Cout, C, kh, kw]。
//
// 用法：cnn_inference [cpu|cuda|vulkan|sycl]

namespace {

constexpr std::int64_t kH = 32;  // 输入 [1, 32, 32]
constexpr std::int64_t kW = 32;
constexpr std::int64_t kC1 = 16;  // conv1: 1 -> 16, 3x3
constexpr std::int64_t kC2 = 32;  // conv2: 16 -> 32, 3x3
constexpr std::int64_t kK = 3;
constexpr std::int64_t kPad = 1;
constexpr std::int64_t kClasses = 10;

Tensor rand_param(std::vector<std::int64_t> shape, Device dev, float spread) {
    return ops::Randn(shape, DataType::FLOAT32, 0.0f, spread, dev);
}

std::string shape_str(const Tensor& t) {
    std::string s = "[";
    for (std::size_t i = 0; i < t.dims(); ++i) {
        if (i) s += ", ";
        s += std::to_string(t.shape()[i]);
    }
    return s + "]";
}

// 逐层看一眼激活：均值与峰值能直接看出 relu 把负半轴截掉了、pool 把统计量收紧。
void print_stats(std::string_view name, const Tensor& t) {
    std::println("  {:<18} {:>18}  mean {:+.3f}  max {:+.3f}", name, shape_str(t),
                 ops::Mean(t).item<float>(), ops::Max(t).item<float>());
}

// 把 [C, H, W] 的一层激活画成 ASCII，采样到 16 行，看个大概。
// 按值拿张量：to_host 会改写这个句柄的标记。
void print_ascii(Tensor t, std::int64_t channel) {
    const std::int64_t h = t.shape()[1];
    const std::int64_t w = t.shape()[2];
    static constexpr std::string_view kRamp = " .:-=+*#%@";
    t.to_host();
    Tensor view = t.contiguous();
    const float* p = static_cast<const float*>(view.data());
    const std::int64_t plane = h * w;
    for (std::int64_t y = 0; y < h; y += 2) {
        std::string row;
        row.reserve(static_cast<std::size_t>(w) + 2);
        row += "    ";
        for (std::int64_t x = 0; x < w; ++x) {
            const float v = p[channel * plane + y * w + x];
            const float clamped = std::clamp(v, 0.0f, 1.0f);
            const auto level = static_cast<std::size_t>(clamped * (kRamp.size() - 1) + 0.5f);
            row += kRamp[level];
        }
        std::println("{}", row);
    }
}

// 2x2 最大池化，空间维减半。库没有 pool 算子，这里用 reshape/permute/Max 拼：
// [C, H, W] -> [C, H/2, 2, W/2, 2] -> [C, H/2, W/2, 2, 2] -> [C, H/2, W/2, 4] -> Max。
Tensor maxpool2(const Tensor& x) {
    const std::int64_t c = x.shape()[0];
    const std::int64_t h = x.shape()[1];
    const std::int64_t w = x.shape()[2];
    return ops::Max(x.reshape({c, h / 2, 2, w / 2, 2}).permute({0, 1, 3, 2, 4})
                        .reshape({c, h / 2, w / 2, 4}),
                    3);
}

// 一个卷积块：conv(3x3, same padding) -> relu -> 2x2 maxpool。
// 张量全程 NCHW：卷积的输出通道在前，池化只在空间两维上减半。
class ConvBlock {
public:
    ConvBlock(Device dev, std::int64_t in_ch, std::int64_t out_ch, std::int64_t kernel,
              std::int64_t pad)
        : w_(rand_param({out_ch, in_ch, kernel, kernel}, dev,
                        1.0f / std::sqrt(static_cast<float>(in_ch * kernel * kernel)))),
          b_(rand_param({out_ch}, dev, 0.01f)),
          in_ch_(in_ch),
          kernel_(kernel),
          pad_(pad) {}

    // 单层卷积：[C, H, W] -> [Cout, H, W]。自检拿它和宿主上按定义重算的结果比对。
    Tensor convolve(const Tensor& x) const {
        return ops::Conv2d(x, w_, b_, /*stride=*/1, pad_);
    }

    // 卷积之后的激活与下采样：[Cout, H, W] -> [Cout, H/2, W/2]
    Tensor activate(const Tensor& conv) const { return maxpool2(ops::Relu(conv)); }

    std::int64_t out_ch() const { return w_.shape()[0]; }

    std::size_t param_count() const { return w_.numel() + b_.numel(); }

    // 自检：挑几个输出位置，在宿主上按卷积的定义重算一遍再比对。输出下标错一位、
    // 权重的通道顺序对调，输出形状都照样对，只有数值会全错——所以这个检查看的是
    // 数值，不是形状；它同时把设备上的算子与"按定义在宿主算"对齐。
    float max_delta(const Tensor& conv, const Tensor& in) const {
        Tensor ch = conv, xh = in, wh = w_, bh = b_;
        for (Tensor* t : {&ch, &xh, &wh, &bh}) t->to_host();
        const Tensor cc = ch.contiguous(), xc = xh.contiguous();
        const Tensor wc = wh.contiguous(), bc = bh.contiguous();
        const float* a = static_cast<const float*>(cc.data());
        const float* src = static_cast<const float*>(xc.data());
        const float* z = static_cast<const float*>(wc.data());
        const float* zb = static_cast<const float*>(bc.data());

        const std::int64_t cout = conv.shape()[0];
        const std::int64_t h = conv.shape()[1];
        const std::int64_t wd = conv.shape()[2];
        const std::int64_t in_h = in.shape()[1];
        const std::int64_t in_w = in.shape()[2];
        const auto px = [&](std::int64_t c, std::int64_t y, std::int64_t x) -> float {
            const std::int64_t yy = y - pad_;
            const std::int64_t xx = x - pad_;
            if (yy < 0 || yy >= in_h || xx < 0 || xx >= in_w) return 0.0f;
            return src[(c * in_h + yy) * in_w + xx];
        };

        const std::int64_t samples[4][3] = {{0, 0, 0},
                                            {cout / 3, h / 2, wd / 2},
                                            {cout - 1, h - 1, wd - 1},
                                            {cout / 2, 0, wd - 1}};
        float worst = 0.0f;
        for (const auto& s : samples) {
            const std::int64_t oc = s[0];
            const std::int64_t oy = s[1];
            const std::int64_t ox = s[2];
            float want = zb[oc];
            for (std::int64_t c = 0; c < in_ch_; ++c)
                for (std::int64_t i = 0; i < kernel_; ++i)
                    for (std::int64_t j = 0; j < kernel_; ++j)
                        want += px(c, oy + i, ox + j) *
                                z[((oc * in_ch_ + c) * kernel_ + i) * kernel_ + j];
            worst = std::max(worst, std::abs(a[(oc * h + oy) * wd + ox] - want));
        }
        return worst;
    }

private:
    Tensor w_, b_;  // [Cout, C, k, k], [Cout]
    std::int64_t in_ch_;
    std::int64_t kernel_;
    std::int64_t pad_;
};

// 整网：两层卷积块 + 全连接。image [C, H, W] -> logits [classes]。
class Cnn {
public:
    // 每层算完调一次，拿到这一层的输入、卷积输出与激活输出。推理引擎里的 layer
    // hook 就是这个意思：例子用它逐层打印、顺带做数值自检；纯推理传空。
    using LayerHook = std::function<void(const ConvBlock&, const Tensor& in,
                                         const Tensor& conv, const Tensor& out)>;

    explicit Cnn(Device dev) {
        blocks_.reserve(2);
        blocks_.emplace_back(dev, 1, kC1, kK, kPad);
        blocks_.emplace_back(dev, kC1, kC2, kK, kPad);

        // 每个卷积块把 H/W 各减半一次，全连接的输入长度由此确定。
        std::int64_t h = kH, w = kW;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            h /= 2;
            w /= 2;
        }
        features_ = blocks_.back().out_ch() * h * w;
        fc_w_ = rand_param({kClasses, features_}, dev,
                           1.0f / std::sqrt(static_cast<float>(features_)));
        fc_b_ = rand_param({kClasses}, dev, 0.01f);
    }

    Tensor forward(const Tensor& image, const LayerHook& on_layer = {}) const {
        Tensor x = image;
        for (const ConvBlock& block : blocks_) {
            const Tensor in = x;
            const Tensor conv = block.convolve(in);
            x = block.activate(conv);
            if (on_layer) on_layer(block, in, conv, x);
        }
        return ops::Linear(x.reshape({1, features_}), fc_w_, fc_b_).select(0, 0);
    }

    std::size_t param_count() const {
        std::size_t n = fc_w_.numel() + fc_b_.numel();
        for (const ConvBlock& block : blocks_) n += block.param_count();
        return n;
    }

private:
    std::vector<ConvBlock> blocks_;
    Tensor fc_w_, fc_b_;  // [classes, features], [classes]
    std::int64_t features_ = 0;
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

    // ---- 1. 输入图：程序画一张，背景是渐变，前景一个圆块与一个方块 ----
    std::vector<float> pixels(static_cast<std::size_t>(kH * kW));
    for (std::int64_t y = 0; y < kH; ++y) {
        for (std::int64_t x = 0; x < kW; ++x) {
            float v = 0.4f * static_cast<float>(y) / static_cast<float>(kH - 1);  // 上暗下亮
            if ((x - 10) * (x - 10) + (y - 22) * (y - 22) <= 36) v = 1.0f;    // 圆块
            if (x >= 19 && x <= 26 && y >= 5 && y <= 12) v = 0.8f;             // 方块
            pixels[static_cast<std::size_t>(y * kW + x)] = v;
        }
    }
    Tensor image(pixels, {1, kH, kW});
    image.to_device(dev);

    // ---- 2. 模型：权重随机，形状按 PyTorch 的卷积约定 ----
    const Cnn model(dev);
    std::println("device {} | conv {}-{}-{} -> conv {}-{}-{} | fc {} -> {}",
                 device_to_string(dev), kC1, kK, kK, kC2, kK, kK,
                 kC2 * (kH / 4) * (kW / 4), kClasses);
    std::println("parameters: {}K (random)", model.param_count() / 1000);
    std::println("input image [1, {}, {}]:", kH, kW);
    print_ascii(image, 0);

    // ---- 3. 前向：钩子逐层打印，并挑位置在宿主上重算做自检 ----
    std::vector<float> deltas;
    const auto t0 = std::chrono::steady_clock::now();
    const Tensor logits = model.forward(
        image, [&](const ConvBlock& block, const Tensor& in, const Tensor& conv,
                   const Tensor& out) {
            const std::size_t layer = deltas.size() + 1;
            deltas.push_back(block.max_delta(conv, in));
            print_stats(std::format("conv{}", layer), conv);
            print_stats(std::format("relu+pool{}", layer), out);
            if (layer == 1) std::println("conv1 activation (channel 0):");
            if (layer == 1) print_ascii(out, 0);
        });
    print_stats("fc", logits);  // [classes]

    Tensor probs = ops::Softmax(logits, 0);  // [classes]
    const std::int64_t best = ops::Argmax(probs, 0).item<std::int64_t>();
    probs.to_host();
    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

    // ---- 4. 结果 ----
    float worst = 0.0f;
    for (const float d : deltas) worst = std::max(worst, d);
    std::println("conv self-check: max |delta| = {:.2e}", static_cast<double>(worst));
    std::println("forward: {:.2f} ms", ms);
    std::println("logits {}", shape_str(logits));
    for (std::int64_t c = 0; c < kClasses; ++c)
        std::println("  class {}: p = {:.4f}{}", c, probs.at<float>({c}),
                     c == best ? "   <- argmax" : "");
    std::println("prediction: class {} (random weights: the label carries no meaning)", best);

    ops::synchronize();
    return worst < 1e-4f ? 0 : 1;
}
