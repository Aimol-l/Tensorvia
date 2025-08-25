#include <print>
#include <random>
#include <thread>
#include <iostream>
#include "ops.h"


void test(int iter){
    for(int i=0;i<iter;i++){
        Tensor a = Tensor::Random({200,300,400},0,10,DataType::FLOAT32);
        auto c = ops::Argmax(a,0);
    }
}

int main() {
    constexpr int iter_num = 100;
    constexpr int threads_num = 1;
    std::vector<std::thread> threads;
    for(int i=0;i<threads_num;i++){
        threads.push_back(std::thread(test,iter_num));
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(auto& t:threads) t.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout<<std::format("total time = {}ms", duration);
    return 0;
}