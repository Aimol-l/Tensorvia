import std;
import tensorvia;

// CNN 推理模拟：图像 -> [conv -> relu -> maxpool] x2 -> fc -> logits
//
//   image      程序在宿主画一张 32x32 的图（演示不依赖图片文件与解码库）
//   ConvBlock  ops::Conv2d -> Relu -> 2x2 MaxPool2d
//   Cnn        两层卷积块加一层全连接；权重随机，形状与数据通路是真的
//
// ops::Conv2d 的语义与 torch.nn.functional.conv2d 一致（NCHW、same padding、
// stride 1）；maxpool 走 ops::MaxPool2d。两者四后端同语义，各后端有自己的
// 实现（CPU 直接卷积，CUDA/SYCL 一个线程一个输出元素，Vulkan 一份 GLSL）。
// 权重布局沿用 PyTorch 的卷积约定 [Cout, C, kh, kw]。
//
// 每层卷积挑几个输出位置，在宿主上按定义重算做数值自检（self-check）——
// 输出下标错一位、权重通道顺序对调，输出形状都照样对，只有数值会全错。
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

// 拷回宿主并按连续内存取出元素（非连续视图先物化）。自检的参考计算在宿主上做。
std::vector<float> host_values(Tensor t) {
    t.to_host();
    t = t.contiguous();
    const float* p = static_cast<const float*>(t.data());
    return {p, p + t.numel()};
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

    Tensor convolve(const Tensor& x) const { return ops::Conv2d(x, w_, b_, /*stride=*/1, pad_); }

    Tensor activate(const Tensor& conv) const {
        return ops::MaxPool2d(ops::Relu(conv), 2, 2);  // stride 缺省取 kernel 尺寸
    }

    std::int64_t out_ch() const { return w_.shape()[0]; }

    std::size_t param_count() const { return w_.numel() + b_.numel(); }

    // 挑几个输出位置，在宿主上按卷积的定义重算一遍再比对（见文件头）。
    float max_delta(const Tensor& conv, const Tensor& in) const {
        const std::vector<float> a   = host_values(conv);
        const std::vector<float> src = host_values(in);
        const std::vector<float> z   = host_values(w_);
        const std::vector<float> zb  = host_values(b_);

        const std::int64_t cout = conv.shape()[0];
        const std::int64_t h    = conv.shape()[1];
        const std::int64_t wd   = conv.shape()[2];
        const std::int64_t in_h = in.shape()[1];
        const std::int64_t in_w = in.shape()[2];
        const auto px = [&](std::int64_t c, std::int64_t y, std::int64_t x) -> float {
            const std::int64_t yy = y - pad_;
            const std::int64_t xx = x - pad_;
            if (yy < 0 || yy >= in_h || xx < 0 || xx >= in_w) return 0.0f;
            return src[static_cast<std::size_t>((c * in_h + yy) * in_w + xx)];
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
            float want = zb[static_cast<std::size_t>(oc)];
            for (std::int64_t c = 0; c < in_ch_; ++c)
                for (std::int64_t i = 0; i < kernel_; ++i)
                    for (std::int64_t j = 0; j < kernel_; ++j)
                        want += px(c, oy + i, ox + j) *
                                z[static_cast<std::size_t>(((oc * in_ch_ + c) * kernel_ + i) *
                                                           kernel_ + j)];
            worst = std::max(
                worst, std::abs(a[static_cast<std::size_t>((oc * h + oy) * wd + ox)] - want));
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
    explicit Cnn(Device dev) {
        blocks_.reserve(2);
        blocks_.emplace_back(dev, 1, kC1, kK, kPad);
        blocks_.emplace_back(dev, kC1, kC2, kK, kPad);

        // 每个卷积块把 H/W 各减半一次，全连接的输入长度由此确定。
        features_ = blocks_.back().out_ch() * (kH / 4) * (kW / 4);
        fc_w_ = rand_param({kClasses, features_}, dev,
                           1.0f / std::sqrt(static_cast<float>(features_)));
        fc_b_ = rand_param({kClasses}, dev, 0.01f);
    }

    const std::vector<ConvBlock>& blocks() const { return blocks_; }

    Tensor logits(const Tensor& features) const {
        return ops::Linear(features.reshape({1, features_}), fc_w_, fc_b_).select(0, 0);
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

// 请求的加速器没编进这个构建（或机器上没有）时明确退回 CPU，不静默降级。
Device resolve_device(std::string_view name) {
    constexpr std::string_view kNames[] = {"cpu", "cuda", "vulkan", "sycl"};
    constexpr Device kKinds[] = {Device::CPU, Device::CUDA, Device::VULKAN, Device::SYCL};
    for (std::size_t i = 0; i < std::size(kNames); ++i) {
        if (name != kNames[i]) continue;
        if (kKinds[i] == Device::CPU || ops::device_count(kKinds[i]) > 0) return kKinds[i];
        std::println("{} is not available in this build; falling back to CPU",
                     device_to_string(kKinds[i]));
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

    // ---- 3. 前向：逐层打印激活统计，并按定义在宿主上重算做数值自检 ----
    const auto t0 = std::chrono::steady_clock::now();
    Tensor x = image;
    float worst = 0.0f;
    std::int64_t layer = 0;
    for (const ConvBlock& block : model.blocks()) {
        ++layer;
        const Tensor conv = block.convolve(x);
        print_stats(std::format("conv{}", layer), conv);
        worst = std::max(worst, block.max_delta(conv, x));
        x = block.activate(conv);
        print_stats(std::format("relu+pool{}", layer), x);
    }
    const Tensor logits = model.logits(x);
    print_stats("fc", logits);  // [classes]

    // ---- 4. 结果 ----
    Tensor probs = ops::Softmax(logits, 0);  // [classes]
    const std::int64_t best = ops::Argmax(probs, 0).item<std::int64_t>();
    probs.to_host();
    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
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
