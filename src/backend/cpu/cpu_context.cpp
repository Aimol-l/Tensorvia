#if defined(_WIN32)
    // MinGW-w64 下不引入 <windows.h>：它会级联包含 C++ 标准头，与 GCC 模块冲突。
    #if defined(__x86_64__) || defined(__i386__)
        // 读 CPUID 的两条路互斥，只能选一条：
        //   MSVC / clang-cl（_MSC_VER）用 <intrin.h> 的 __cpuid()。GCC 和 Clang 自带的
        //   <cpuid.h> 不能用——它把 __cpuid 定义成函数宏，而 MSVC 的 <wchar.h>（被
        //   <format> 拉进来）会包含 <intrin.h>，其中的 `void __cpuid(int[4], int);`
        //   正好被那个宏吃掉，报 "too few arguments provided to function-like macro
        //   invocation"。<intrin.h> 反过来没有这个问题：标准头本来就会包含它。
        //   MinGW 的 GCC/Clang 没有 <intrin.h>，用 <cpuid.h> 的 __get_cpuid()。
        #if defined(_MSC_VER)
            #include <intrin.h>
        #else
            #include <cpuid.h>
        #endif
    #endif
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#elif defined(__linux__) || defined(__unix__)
    #include <unistd.h>
    #include <fstream>
    #if defined(__x86_64__) || defined(__i386__)
        // Linux 上没有 <wchar.h> → <intrin.h> 那条级联（那是 MSVC UCRT 的行为），
        // <cpuid.h> 的 __cpuid 宏不会撞车，可以放心用。
        #include <cpuid.h>
    #endif
#endif

#include <cstdint>
#include <cstring>
#include <format>
#include <print>
#include <string>
#include <thread>

#include "tensorvia.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/cpu/cpu_context.hpp"

namespace {

#if defined(__x86_64__) || defined(__i386__)
// 读一个 CPUID leaf 到 out[4]（eax/ebx/ecx/edx）。这个 leaf 不被支持时返回 false，
// 此时 out 的内容无意义。
bool read_cpuid(unsigned int leaf, unsigned int out[4]) {
  #if defined(_MSC_VER)
    int r[4] = {};
    __cpuid(r, static_cast<int>(leaf));
    std::memcpy(out, r, sizeof(r));
    return true;
  #else
    return __get_cpuid(leaf, &out[0], &out[1], &out[2], &out[3]) != 0;
  #endif
}
#endif

#if defined(_WIN32)

std::string read_cpu_model_name() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int regs[4] = {};
    if (!read_cpuid(0x80000000u, regs) || regs[0] < 0x80000004u) {
        return "unknown";
    }
    char brand[49] = {};
    for (unsigned int leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
        read_cpuid(leaf, regs);
        std::memcpy(brand + (leaf - 0x80000002u) * 16, regs, 16);
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
    // 不用 __builtin_cpu_supports：clang 打 MSVC ABI 时它引用 libgcc 的
    // __cpu_model / __cpu_features2，而 -fms-runtime-lib=dll -fuse-ld=lld 的链接
    // 线上没有 libgcc，直接 undefined symbol。编译期一点事没有，链接才炸，所以
    // 这个问题一直藏着。改成自己问 CPUID，任何编译器、任何 ABI 都能用。
    unsigned int leaf1[4] = {};
    unsigned int leaf7[4] = {};
    read_cpuid(1u, leaf1);
    read_cpuid(7u, leaf7);
    // CPUID.1:ECX[27]：OS 开了 XSAVE。AVX 一族按理还得看 XGETBV，但那要 -mxsave
    // 的内部函数，不带 -march 的构建会直接编译不过，不值得；这一位已经能挡住
    // "CPU 支持、系统没开" 的大多数情况。读失败的 leaf 缓冲区是零，位自然是 0。
    const bool osxsave = (leaf1[2] & (1u << 27)) != 0;

    return std::format("AVX2: {} | FMA: {} | F16C: {} | AVX512F: {} | SSE4.2: {}",
                       yes_no(osxsave && (leaf7[1] & (1u << 5))  != 0),  // CPUID.7.0:EBX[5]
                       yes_no(osxsave && (leaf1[2] & (1u << 12)) != 0),  // CPUID.1:ECX[12]
                       yes_no(osxsave && (leaf1[2] & (1u << 29)) != 0),  // CPUID.1:ECX[29]
                       yes_no(osxsave && (leaf7[1] & (1u << 16)) != 0),  // CPUID.7.0:EBX[16]
                       yes_no(          (leaf1[2] & (1u << 20)) != 0)); // CPUID.1:ECX[20]
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
