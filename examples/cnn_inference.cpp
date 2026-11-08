import std;
import tensorvia;

// CNN 推理模拟：随机 RGB 图 -> [conv -> relu -> maxpool] x2 -> fc -> softmax，
// 顺带计量前向耗时（预热一次后多次取平均，每次含同步）。
//
//   ConvBlock  一层：ops::Conv2d(3x3, same padding) -> Relu -> 2x2 MaxPool2d
//   ConvModel  两层 ConvBlock 加一层全连接，[N, C, H, W] -> logits [classes]
//
// 两个算子都是 NCHW、四后端同语义（CPU 直接卷积，CUDA/SYCL 一个线程一个
// 输出元素，Vulkan 一份 GLSL）；权重布局沿用 PyTorch 的卷积约定
// [Cout, C, kh, kw]，全部随机初始化。
//
// 设备按编译进来的后端选择（与测试套件的 lane 选择同一机制）：mcpp 带
// --features vulkan 构建，这个二进制就跑 VULKAN。因此四种后端各构建一次，
// 打印的耗时才有对比意义。

#if defined(BACKEND_VULKAN)
    constexpr Device kDevice = Device::VULKAN;
#elif defined(BACKEND_CUDA)
    constexpr Device kDevice = Device::CUDA;
#elif defined(BACKEND_SYCL)
    constexpr Device kDevice = Device::SYCL;
#else
    constexpr Device kDevice = Device::CPU;
#endif

namespace {

constexpr std::int64_t kH = 640;  // 输入 [1, 3, 640, 640]
constexpr std::int64_t kW = 640;
constexpr std::int64_t kInCh = 3;  // 输入通道：RGB 图片
constexpr std::int64_t kC1 = 16;  // conv1: 3 -> 16
constexpr std::int64_t kC2 = 32;  // conv2: 16 -> 32
constexpr std::int64_t kK = 3;
constexpr std::int64_t kPad = 1;
constexpr std::int64_t kClasses = 10;
constexpr int kRuns = 10;  // 计时轮数（另有一次预热）

// 权重与输入都显式落到 kDevice：不跟 default_device()（那是 CPU）。
TensorOptions weight_opts() { return TensorOptions{DataType::FLOAT32, kDevice}; }

class ConvBlock {
private:
    Tensor w_, b_;  // [Cout, C, k, k], [Cout]
public:
    ConvBlock() = default;  // ConvModel 的成员先默认构造、再在函数体里赋值

    ConvBlock(std::int64_t in_ch, std::int64_t out_ch) {
        const TensorOptions opt = weight_opts();
        w_ = ops::Randn({out_ch, in_ch, kK, kK}, 0.0f,
                        1.0f / std::sqrt(static_cast<float>(in_ch * kK * kK)), opt);
        b_ = ops::Randn({out_ch}, 0.0f, 0.01f, opt);
    }

    // [C, H, W] -> [Cout, H/2, W/2]：融合块一次算完 conv+relu+pool，省掉两个
    // 算子的提交与同步往返（Vulkan lane 上这是前向的主要成本）。
    Tensor forward(const Tensor& x) const {
        return ops::Conv2dReluMaxPool2d(x, w_, b_, /*stride=*/1, kPad);
    }
};

class ConvModel {
private:
    Tensor fc_w_, fc_b_;
    ConvBlock block1_, block2_;

    // 每个卷积块把 H/W 各减半一次，全连接的输入长度由此确定。
    static std::int64_t features() { return kC2 * (kH / 4) * (kW / 4); }
public:
    ConvModel() {
        const TensorOptions opt = weight_opts();
        block1_ = ConvBlock(kInCh, kC1);
        block2_ = ConvBlock(kC1, kC2);
        fc_w_ = ops::Randn({kClasses, features()}, 0.0f,
                           1.0f / std::sqrt(static_cast<float>(features())), opt);
        fc_b_ = ops::Randn({kClasses}, 0.0f, 0.01f, opt);
    }

    // image [N, C, H, W] -> logits [classes]。
    Tensor forward(const Tensor& image) const {
        Tensor x = block1_.forward(image);
        x = block2_.forward(x);
        const Tensor flat = x.reshape({1, features()});
        const Tensor out  = ops::Linear(flat, fc_w_, fc_b_);  // [1, classes]
        return out.select(0, 0);                              // -> [classes]
    }
};

}  // namespace

int main() {
    const Tensor image =
        Tensor::Random({1, kInCh, kH, kW}, 0.0f, 1.0f,
                       TensorOptions{DataType::FLOAT32, kDevice});
    const ConvModel model;

    Tensor logits = model.forward(image);

    ops::synchronize();

    const auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < kRuns; ++i) 
        logits = model.forward(image);

    ops::synchronize();  // 提交即返回：不等就只量到宿主提交开销

    const auto stop = std::chrono::steady_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(stop - start).count() / kRuns;

    ops::println(logits);

    const std::int64_t best = ops::Argmax(ops::Softmax(logits, 0), 0).item<std::int64_t>();
    std::println("device {} | forward {:.2f} ms/run ({} runs, {}x{}x{}) | prediction: "
                 "class {}",
                 device_to_string(kDevice), avg_ms, kRuns, kInCh, kH, kW, best);
    return 0;
}
