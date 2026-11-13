import std;
import tensorvia;

int main() {
    // 构造 → 归约 → 取值：摸过门面导出的 Tensor、ops::Sum 与 item<T>()。
    const Tensor t = Tensor::Fill({4, 4}, 2.0f, DataType::FLOAT32);
    const Tensor s = ops::Sum(t);
    const float sum  = s.item<float>();
    if (sum != 32.0f) {
        std::println(std::cerr, "import tensorvia: sum = {}, expected 32", sum);
        return 1;
    }
    ops::println("import tensorvia: OK (sum = {})", sum);
    return 0;
}
