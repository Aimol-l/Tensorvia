#include <print>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include "ops.h"

using namespace via;

// ---------------------------------------------------------------------------
// Timing utility
// ---------------------------------------------------------------------------

struct BenchResult {
    std::string name;
    double avg_ms;
    double min_ms;
    double max_ms;
    int iterations;
};

#define BENCH(name_str, warmup, iters, ...) do {                    \
    for (int _i = 0; _i < (warmup); _i++) { __VA_ARGS__; }         \
    double _mn = 1e18, _mx = 0, _sum = 0;                          \
    for (int _i = 0; _i < (iters); _i++) {                          \
        auto _t0 = std::chrono::steady_clock::now();                \
        __VA_ARGS__;                                                 \
        auto _t1 = std::chrono::steady_clock::now();                \
        double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count(); \
        _sum += _ms; if (_ms < _mn) _mn = _ms; if (_ms > _mx) _mx = _ms; \
    }                                                                \
    results.push_back({name_str, _sum / (iters), _mn, _mx, (iters)}); \
} while(0)

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

static void print_header() {
    std::println("{:<35s} {:>10s} {:>10s} {:>10s} {:>6s}",
                 "Operation", "Avg (ms)", "Min (ms)", "Max (ms)", "Iters");
    std::println("{:-<75s}", "");
}

static void print_result(const BenchResult& r) {
    std::println("{:<35s} {:>10.2f} {:>10.2f} {:>10.2f} {:>6d}",
                 r.name, r.avg_ms, r.min_ms, r.max_ms, r.iterations);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::vector<BenchResult> results;

    std::println("Tensorvia Benchmark Suite");
    std::println("========================\n");

    // --- Matmul 2D: 2592x2048 @ 2048x4096 (matches README) ---
    {
        Tensor a = Tensor::Random({2592, 2048}, -1.0, 1.0, DataType::FLOAT32);
        Tensor b = Tensor::Random({2048, 4096}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Matmul 2592x2048@2048x4096", 2, 5, ops::Mul(a, b));
    }

    // --- Matmul 2D: 1024x1024 ---
    {
        Tensor a = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        Tensor b = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Matmul 1024x1024", 3, 20, ops::Mul(a, b));
    }

    // --- Matmul 3D batched: 32x512x512 ---
    {
        Tensor a = Tensor::Random({32, 512, 512}, -1.0, 1.0, DataType::FLOAT32);
        Tensor b = Tensor::Random({32, 512, 512}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Matmul 32x512x512 (batched)", 3, 20, ops::Mul(a, b));
    }

    // --- Element-wise Add ---
    {
        Tensor a = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        Tensor b = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Add 1024x1024", 5, 50, ops::Add(a, b));
    }

    // --- Sum reduction (global) ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Sum 2048x2048 (global)", 5, 50, (void)ops::Sum(a));
    }

    // --- Sum reduction (axis) ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Sum 2048x2048 (axis=1)", 5, 50, ops::Sum(a, 1));
    }

    // --- Argmax ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Argmax 2048x2048 (axis=1)", 5, 50, ops::Argmax(a, 1));
    }

    // --- Softmax ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Softmax 2048x2048", 5, 50, ops::Softmax(a, 1));
    }

    // --- ReLU ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("ReLU 2048x2048", 5, 50, ops::Relu(a));
    }

    // --- Transpose 2D ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Transpose 2048x2048", 5, 50, ops::Transpose(a));
    }

    // --- Transpose ND ---
    {
        Tensor a = Tensor::Random({128, 256, 512}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Transpose 128x256x512 (ND)", 5, 50, ops::Transpose(a, {2, 0, 1}));
    }

    // --- Concat ---
    {
        Tensor a = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        Tensor b = Tensor::Random({1024, 1024}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Concat 2x1024x1024 (dim=1)", 5, 50, ops::Concat({a, b}, 1));
    }

    // --- Slice ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Slice 2048x2048", 5, 50, ops::Slice(a, {{0, 1024}, {0, 1024}}));
    }

    // --- Typecast ---
    {
        Tensor a = Tensor::Random({2048, 2048}, -1.0, 1.0, DataType::FLOAT32);
        BENCH("Typecast FP32->FP64", 5, 50, ops::Typecast(a, DataType::FLOAT64));
    }

    // --- Print results ---
    print_header();
    for (auto& r : results) {
        print_result(r);
    }

    return 0;
}
