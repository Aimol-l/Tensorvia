import std;
import tensorvia;

// CNN 推理模拟：随机灰度图 -> [conv -> relu -> maxpool] x2 -> fc -> softmax。
//
//   ConvBlock  一层：ops::Conv2d(3x3, same padding) -> Relu -> 2x2 MaxPool2d
//   ConvModel  两层 ConvBlock 加一层全连接，[1, H, W] -> logits [classes]
//
// 两个算子都是 NCHW、四后端同语义（CPU 直接卷积，CUDA/SYCL 一个线程一个
// 输出元素，Vulkan 一份 GLSL）；权重布局沿用 PyTorch 的卷积约定
// [Cout, C, kh, kw]，全部随机初始化。张量落在库的默认设备（CPU）；换后端
// 用 TensorOptions 或 set_default_device。

namespace {

constexpr std::int64_t kH = 32;  // 输入 [1, 32, 32]
constexpr std::int64_t kW = 32;
constexpr std::int64_t kC1 = 16;  // conv1: 1 -> 16
constexpr std::int64_t kC2 = 32;  // conv2: 16 -> 32
constexpr std::int64_t kK = 3;
constexpr std::int64_t kPad = 1;
constexpr std::int64_t kClasses = 10;

class ConvBlock {
public:
    ConvBlock(std::int64_t in_ch, std::int64_t out_ch)
        : w_(ops::Randn({out_ch, in_ch, kK, kK}, DataType::FLOAT32, 0.0f,
                        1.0f / std::sqrt(static_cast<float>(in_ch * kK * kK)))),
          b_(ops::Randn({out_ch}, DataType::FLOAT32, 0.0f, 0.01f)) {}

    // [C, H, W] -> [Cout, H/2, W/2]：卷积 same padding 保尺寸，池化空间减半。
    Tensor forward(const Tensor& x) const {
        return ops::MaxPool2d(ops::Relu(ops::Conv2d(x, w_, b_, /*stride=*/1, kPad)), 2, 2);
    }

private:
    Tensor w_, b_;  // [Cout, C, k, k], [Cout]
};

class ConvModel {
public:
    ConvModel()
        : block1_(1, kC1),
          block2_(kC1, kC2),
          fc_w_(ops::Randn({kClasses, features()}, DataType::FLOAT32, 0.0f,
                           1.0f / std::sqrt(static_cast<float>(features())))),
          fc_b_(ops::Randn({kClasses}, DataType::FLOAT32, 0.0f, 0.01f)) {}

    // image [1, H, W] -> logits [classes]。
    Tensor forward(const Tensor& image) const {
        const Tensor x = block2_.forward(block1_.forward(image));
        return ops::Linear(x.reshape({1, features()}), fc_w_, fc_b_).select(0, 0);
    }

private:
    // 每个卷积块把 H/W 各减半一次，全连接的输入长度由此确定。
    static std::int64_t features() { return kC2 * (kH / 4) * (kW / 4); }

    ConvBlock block1_, block2_;
    Tensor fc_w_, fc_b_;  // [classes, features], [classes]
};

}  // namespace

int main() {
    const Tensor image = Tensor::Random({1,kH,kW},0.0f,1.0f,DataType::FLOAT32);
    const ConvModel model;
    const Tensor logits = model.forward(image);
    const std::int64_t best = ops::Argmax(ops::Softmax(logits, 0), 0).item<std::int64_t>();
    std::println("device {} | prediction: class {} (random weights: the label carries "
                 "no meaning)",
                 device_to_string(image.device()), best);
    return 0;
}
