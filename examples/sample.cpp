import std;
import tensorvia;

int main() {
    // 1. 创建张量
    auto a = Tensor::Ones({2, 3}, DataType::FLOAT32);
    auto b = Tensor::Fill({2, 3}, 2.0f, DataType::FLOAT32);
    auto r = Tensor::Random({3}, 0.0f, 10.0f, DataType::FLOAT32);

    // 2. 逐元素算术（运算符重载 + 显式 ops:: 调用都行）
    auto c = a + b;
    ops::println("a + b =");
    ops::println(c);

    // 3. 矩阵乘法（ops::MatMul 即 matmul）
    auto d = ops::MatMul(a.view({3, 2}), b.view({2, 3}));
    ops::println("matmul(a(3x2), b(2x3)) =");
    ops::println(d);

    // 4. 归约：返回 0-dim 张量，item<T>() 才把值取回 host
    ops::println("sum(c) = {}", ops::Sum(c).item<float>());
    ops::println("mean(c) = {}", ops::Mean(c).item<float>());

    // 5. 激活函数
    auto sm = ops::Softmax(r, 0);
    ops::println("softmax(r) =");
    ops::println(sm);

    // 6. 索引读写
    ops::println("c[0][1] = {}", c.at<float>({0, 1}));

    // 7. 切片 / 转置 / 拼接
    auto sliced = c.slice({{0, 2}, {0, 2}});
    auto t = sliced.t();
    auto cat = ops::Concat({t, t}, 1);
    ops::println("concat(t, t, dim=1) =");
    ops::println(cat);

    return 0;
}
