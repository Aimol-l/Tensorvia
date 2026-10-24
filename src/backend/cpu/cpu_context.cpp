#if defined(_WIN32)
    // MinGW-w64 下不引入 <windows.h>：它会级联包含 C++ 标准头，与 GCC 模块冲突。
    // 改用 GCC/Clang 自带的 <cpuid.h> 读 CPU 品牌串。
    #if defined(__x86_64__) || defined(__i386__)
        #include <cpuid.h>
    #endif
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#elif defined(__linux__) || defined(__unix__)
    #include <unistd.h>
    #include <fstream>
#endif

#include <cstdint>
#include <cstring>
#include <format>
#include <print>
#include <string>
#include <thread>

#include "interface/tensorvia.hpp"
#include "interface/context.hpp"
#include "interface/cpu_context.hpp"

namespace {

#if defined(_WIN32)

std::string read_cpu_model_name() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int regs[4] = {};
    if (!__get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]) ||
        regs[0] < 0x80000004) {
        return "unknown";
    }
    char brand[49] = {};
    char* p = brand;
    for (unsigned int leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
        __get_cpuid(leaf, &regs[0], &regs[1], &regs[2], &regs[3]);
        std::memcpy(p, regs, 16);
        p += 16;
    }
    std::string s(brand);
    auto b = s.find_first_not_of(' ');
    auto e = s.find_last_not_of(' ');
    return (b == std::string::npos) ? "unknown" : s.substr(b, e - b + 1);
#else
    return "unknown";
#endif
}

size_t read_mem_total_bytes() { return 0; }

#elif defined(__APPLE__)

std::string read_cpu_model_name() {
    char   buf[128] = {};
    size_t len      = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
        return std::string(buf);
    }
    return "unknown";
}

size_t read_mem_total_bytes() {
    int64_t mem = 0;
    size_t  len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
        return static_cast<size_t>(mem);
    }
    return 0;
}

#elif defined(__linux__) || defined(__unix__)

std::string read_cpu_model_name() {
#if defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.starts_with("model name")) {
            auto pos = line.find(':');
            if (pos != std::string::npos) return line.substr(pos + 2);
        }
    }
#endif
    return "unknown";
}

size_t read_mem_total_bytes() {
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 0;
}

#else

std::string read_cpu_model_name() { return "unknown"; }
size_t      read_mem_total_bytes() { return 0; }

#endif

[[maybe_unused]] std::string detect_isa_features() {
    auto yes_no = [](bool b) { return b ? "yes" : "no"; };

#if defined(__x86_64__) || defined(__i386__)
    #if defined(__GNUC__) || defined(__clang__)
        return std::format("AVX2: {} | FMA: {} | F16C: {} | AVX512F: {} | SSE4.2: {}",
                           yes_no(__builtin_cpu_supports("avx2")),
                           yes_no(__builtin_cpu_supports("fma")),
                           yes_no(__builtin_cpu_supports("f16c")),
                           yes_no(__builtin_cpu_supports("avx512f")),
                           yes_no(__builtin_cpu_supports("sse4.2")));
    #else
        return "x86 ISA detection not supported on this toolchain";
    #endif
#elif defined(__aarch64__) || defined(__arm__)
    #if defined(__GNUC__) || defined(__clang__)
        return std::format("NEON: {} | SVE: {}",
                           yes_no(__builtin_cpu_supports("neon")),
                           yes_no(__builtin_cpu_supports("sve")));
    #else
        return "ARM ISA detection not supported on this toolchain";
    #endif
#else
    return "ISA detection not supported on this platform";
#endif
}

}  // namespace

CPUContext::CPUContext()
    : m_num_cores(static_cast<int>(std::thread::hardware_concurrency())),
      m_model_name(read_cpu_model_name()),
      m_memory_total_bytes(read_mem_total_bytes()) {
#ifndef NDEBUG
    std::println("***************************CPU Device Info******************");
    std::println("CPU Model:        {}", m_model_name);
    std::println("Logical cores:    {}", m_num_cores);
    std::println("Total memory:     {} MB", m_memory_total_bytes / (1024 * 1024));
    std::println("{}", detect_isa_features());
    std::println("************************************************************");
#endif
}
