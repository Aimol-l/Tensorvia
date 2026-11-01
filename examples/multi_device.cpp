import std;
import tensorvia;

// 多设备：同一个后端下的第几张卡由运行期序号决定，和设备种类合称 DeviceId。
// 这段程序在任何机器上都跑：只有一张卡时前面几段照常输出，双卡用例打印提示。
int main() {
    // 1. 每张张量都带一个 DeviceId，CPU 恒为 {CPU, 0}
    auto host = Tensor::Ones({2, 3}, DataType::FLOAT32);
    std::println("host tensor: {}", device_id_to_string(host.device_id()));

    // 2. 各后端的设备数。没编进这个构建的后端报 0
    for (const Device kind : {Device::CPU, Device::CUDA, Device::VULKAN, Device::SYCL}) {
        std::println("{:<7} {} device(s)", device_to_string(kind), ops::device_count(kind));
    }

    // 3. 挑一个可用加速器（优先 CUDA，其次 Vulkan / SYCL）
    Device accel = Device::CPU;
    for (const Device kind : {Device::CUDA, Device::VULKAN, Device::SYCL}) {
        if (ops::device_count(kind) > 0) {
            accel = kind;
            break;
        }
    }
    if (accel == Device::CPU) {
        std::println("no accelerator in this build; stopping here");
        return 0;
    }

    // 4. current device 是线程私有的：不设置时就是这个后端的默认卡
    std::println("{} default current device: {}",
                 device_to_string(accel), ops::current_device(accel));

    // 5. 不写序号的构造落在当前卡上；DeviceGuard 在作用域内改，出来恢复
    auto a = Tensor::Ones({2, 3}, DataType::FLOAT32, accel);
    std::println("a lives on {}", device_id_to_string(a.device_id()));
    {
        ops::DeviceGuard guard(accel, 0);
        auto b = Tensor::Ones({2, 3}, DataType::FLOAT32, accel);
        std::println("inside guard, b lives on {} (current = {})",
                     device_id_to_string(b.device_id()), ops::current_device(accel));
    }
    std::println("after guard, current = {}", ops::current_device(accel));

    // 6. 算子的输出跟随输入所在的实例
    auto c = a + a;
    std::println("a + a lives on {}", device_id_to_string(c.device_id()));
    auto col_sum = ops::Sum(c, 0);            // 仍在这张卡上
    col_sum.to_host();                        // at<> 只对 host 张量可用
    std::println("sum(a, dim=0)[0] = {}", col_sum.at<float>({0}));
    ops::println(c);

    // 7. 换卡：同序号直接返回，跨序号经 host 中转
    if (ops::device_count(accel) >= 2) {
        auto moved = a.clone();
        moved.to_device(DeviceId{accel, 1});
        std::println("moved to {}", device_id_to_string(moved.device_id()));
        // 不同卡上的张量不能直接当操作数：拿 A 卡的 context 读 B 卡的指针是
        // 静默的数据错误，所以这里会抛。
        try {
            (void)(a + moved);
        } catch (const std::exception& e) {
            std::println("cross-device add rejected: {}", e.what());
        }
        ops::println(moved);
    } else {
        std::println("only one {} device here; cross-index copy not exercised", device_to_string(accel));
    }

    // 8. 同步与回收覆盖所有已经起起来的实例（多卡时每个 context 各一份）
    ops::synchronize();
    ops::empty_cache();
    std::println("synchronize() + empty_cache() ok");
    return 0;
}
