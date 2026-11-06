// `import tensorvia;` 冒烟：C++23 模块消费路径的最薄探针。
//
// 不 include 任何 tensorvia 头，只 import 模块——模块门面
// src/interface/tensorvia.ixx 缺失、facade_exports.def 漂移、或安装面断链，
// 这里在构建期/运行期最先响。examples/ 下有各路径的消费者样例，但 tests/ 里
// 没有一个被构建系统自动跑的模块导入探针；本文件补上这一格。
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
