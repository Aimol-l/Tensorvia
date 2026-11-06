import std;
import tensorvia;

// CNN 推理模拟：随机灰度图 -> [conv -> relu -> maxpool] x2 -> fc -> softmax。

namespace {

constexpr std::int64_t kH = 640;  // 输入 [1, 640, 640]
constexpr std::int64_t kW = 640;
constexpr std::int64_t kInCh = 3;  // 输入通道：RGB 图片
constexpr std::int64_t kC1 = 16;  // conv1: 1 -> 16
constexpr std::int64_t kC2 = 32;  // conv2: 16 -> 32
constexpr std::int64_t kK = 3;
constexpr std::int64_t kPad = 1;
constexpr std::int64_t kClasses = 10;

class ConvBlock {
private:
    Tensor w_, b_;  // [Cout, C, k, k], [Cout]
public:
    ConvBlock() = default;  // ConvModel 的成员先默认构造、再在函数体里赋值

    ConvBlock(std::int64_t in_ch, std::int64_t out_ch) {
        w_ = ops::Randn({out_ch, in_ch, kK, kK}, DataType::FLOAT32, 0.0f,
                        1.0f / std::sqrt(static_cast<float>(in_ch * kK * kK)));
        b_ = ops::Randn({out_ch}, DataType::FLOAT32, 0.0f, 0.01f);
    }

    Tensor forward(const Tensor& x) const {
        const Tensor conv = ops::Conv2d(x, w_, b_, /*stride=*/1, kPad);  // [Cout, H, W]
        const Tensor act  = ops::Relu(conv);
        return ops::MaxPool2d(act, 2, 2);                                // [Cout, H/2, W/2]
    }
};

class ConvModel {
private:
    Tensor fc_w_, fc_b_;
    ConvBlock block1_, block2_;

    static std::int64_t features() { return kC2 * (kH / 4) * (kW / 4); }
public:
    ConvModel() {
        block1_ = ConvBlock(kInCh, kC1);
        block2_ = ConvBlock(kC1, kC2);
        fc_w_ = ops::Randn({kClasses, features()}, DataType::FLOAT32, 0.0f,
                           1.0f / std::sqrt(static_cast<float>(features())));
        fc_b_ = ops::Randn({kClasses}, DataType::FLOAT32, 0.0f, 0.01f);
    }

    Tensor forward(const Tensor& image) const {  // [N, C, H, W] -> [classes]
        Tensor x = block1_.forward(image);
        x = block2_.forward(x);
        const Tensor flat = x.reshape({1, features()});
        const Tensor out  = ops::Linear(flat, fc_w_, fc_b_);  // [1, classes]
        return out.select(0, 0);                              // -> [classes]
    }
};

}  // namespace

int main() {
    const Tensor image = Tensor::Random({1,kInCh,kH,kW},0.0f,1.0f,DataType::FLOAT32);

    const ConvModel model;

    const Tensor logits = model.forward(image);

    const std::int64_t best = ops::Argmax(ops::Softmax(logits, 0), 0).item<std::int64_t>();

    std::println("prediction: class {}",best);
    
    return 0;
}
