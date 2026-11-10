// Host/ARM64-compatible benchmark, not evidence of Android performance.
// From repository root:
// g++ -std=c++17 -O2 -DNDEBUG -ffp-contract=off -Wall -Wextra -Werror
//   (continue command with) -Ilanguage/runtime research/mobile_kernel_003/bench_int8.cpp -o /tmp/nl-int8-bench
// /tmp/nl-int8-bench [samples=2048] [trials=9]
// Replace g++ with the intended compiler; use identical flags for comparisons.
// Setup, correctness, model construction, and printing are outside timing.
// Both runtime variants use observe_full_reuse + infer(false). The independent
// baseline likewise validates and copies input, retains z/a, and checks every z.
// Report every trial and its latency distribution; no pass threshold is inferred.
#define NEURAL_MOBILE_PLAN
#include "neural_runtime.hpp"
#include "mobile_plan.hpp"
#include <array>

// Exact mobile_function_source output for sequential INT8 ReLU/linear .nir
// brains with these shapes; only the callback names differ. No handwritten
// replacement for the compiler kernel is used in the specialized runtime.
static bool nl_mobile_32_64_16(neural::Model& m){
if(m.graph||m.layers.size()!=2||m.ninput!=32)return false;
if(m.layers[0].in!=32||m.layers[0].out!=64||m.layers[0].type!=1||m.layers[0].storage!=1||m.layers[0].disabled!=0)return false;
if(m.layers[1].in!=64||m.layers[1].out!=16||m.layers[1].type!=0||m.layers[1].storage!=1||m.layers[1].disabled!=0)return false;
if((int)m.z[0].size()!=64){m.z[0].assign(64,0);m.a[0].assign(64,0);}
neural::mobile_dense_int8_ilp4<32,64,1>(m.layers[0].q.data(),m.layers[0].b.data(),m.input.data(),m.z[0].data(),m.a[0].data(),m.layers[0].scale);
if((int)m.z[1].size()!=16){m.z[1].assign(16,0);m.a[1].assign(16,0);}
neural::mobile_dense_int8_ilp4<64,16,0>(m.layers[1].q.data(),m.layers[1].b.data(),m.a[0].data(),m.z[1].data(),m.a[1].data(),m.layers[1].scale);
return true;
}
static bool nl_mobile_128_128_32(neural::Model& m){
if(m.graph||m.layers.size()!=2||m.ninput!=128)return false;
if(m.layers[0].in!=128||m.layers[0].out!=128||m.layers[0].type!=1||m.layers[0].storage!=1||m.layers[0].disabled!=0)return false;
if(m.layers[1].in!=128||m.layers[1].out!=32||m.layers[1].type!=0||m.layers[1].storage!=1||m.layers[1].disabled!=0)return false;
if((int)m.z[0].size()!=128){m.z[0].assign(128,0);m.a[0].assign(128,0);}
neural::mobile_dense_int8_ilp4<128,128,1>(m.layers[0].q.data(),m.layers[0].b.data(),m.input.data(),m.z[0].data(),m.a[0].data(),m.layers[0].scale);
if((int)m.z[1].size()!=32){m.z[1].assign(32,0);m.a[1].assign(32,0);}
neural::mobile_dense_int8_ilp4<128,32,0>(m.layers[1].q.data(),m.layers[1].b.data(),m.a[0].data(),m.z[1].data(),m.a[1].data(),m.layers[1].scale);
return true;
}

namespace {
void ensure(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// Independent numeric checks and scalar arithmetic, without runtime helpers.
void numeric(double value) {
    if (!std::isfinite(value) || std::abs(value) >= 1e9)
        throw std::runtime_error("independent numeric bound");
}

template<int In, int Out, bool Relu>
struct PlainLayer {
    std::array<int8_t, In * Out> q{};
    std::array<uint8_t, In * Out> enabled{};
    std::array<double, Out> b{}, z{}, a{};
    double scale;
    explicit PlainLayer(const neural::Layer& layer): scale(layer.scale) {
        ensure(layer.w.empty() && layer.q.size() == q.size(), "packed weight shape");
        std::copy(layer.q.begin(), layer.q.end(), q.begin());
        std::copy(layer.b.begin(), layer.b.end(), b.begin());
        std::copy(layer.enabled.begin(), layer.enabled.end(), enabled.begin());
    }
    template<bool Masked = false>
    void run(const double* x) {
        for (int row = 0; row < Out; ++row) {
            double sum = b[row];
            for (int col = 0; col < In; ++col) {
                const size_t index = static_cast<size_t>(row) * In + col;
                double weight = double(q[index]) * scale;
                if constexpr (Masked) { if (!enabled[index]) weight = 0.0; }
                sum += weight * x[col];
            }
            numeric(sum);
            z[row] = sum;
            if constexpr (Relu) a[row] = std::max(0.0, sum);
            else a[row] = sum;
        }
    }
};

template<int In, int Hidden, int Out>
struct Plain {
    std::array<double, In> input{};
    PlainLayer<In, Hidden, true> first;
    PlainLayer<Hidden, Out, false> second;
    explicit Plain(const neural::Model& model): first(model.layers[0]), second(model.layers[1]) {}
    template<bool Masked = false>
    void predict(const std::vector<double>& row) {
        ensure(row.size() == In, "independent input shape");
        for (double value : row) numeric(value);
        std::copy(row.begin(), row.end(), input.begin());
        first.template run<Masked>(input.data());
        second.run(first.a.data());
    }
};

template<size_t N>
bool equal(const std::vector<double>& actual, const std::array<double, N>& expected) {
    return actual.size() == N && std::equal(actual.begin(), actual.end(), expected.begin());
}

template<int In, int Hidden, int Out>
void equivalence(const neural::Model& generic, const neural::Model& mobile,
                 const Plain<In, Hidden, Out>& plain) {
    ensure(generic.z == mobile.z && generic.a == mobile.a, "runtime z/a mismatch");
    ensure(equal(generic.z[0], plain.first.z) && equal(generic.a[0], plain.first.a)
        && equal(generic.z[1], plain.second.z) && equal(generic.a[1], plain.second.a),
        "independent z/a mismatch");
}

struct Measured { double ns, checksum; };
template<typename Predict>
Measured measure(const std::vector<std::vector<double>>& rows, Predict predict) {
    volatile double checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& row : rows) {
        const auto& result = predict(row);
        for (double value : result) checksum += value;
    }
    const auto stop = std::chrono::steady_clock::now();
    return {std::chrono::duration<double, std::nano>(stop - start).count() / rows.size(), checksum};
}

template<int In, int Hidden, int Out>
void workload(size_t samples, int trials, bool (*plan)(neural::Model&)) {
    const std::string shape = std::to_string(In) + "x" + std::to_string(Hidden) + "x" + std::to_string(Out);
    neural::Model generic("PackedBench", In);
    generic.layer("hidden", Hidden, 1, 71u, 1);
    generic.layer("out", Out, 0, 73u, 1);
    // Deterministic full-range packed weights, no checkpoint reads or writes.
    for (size_t k = 0; k < generic.layers.size(); ++k) {
        auto& layer = generic.layers[k];
        for (size_t i = 0; i < layer.q.size(); ++i)
            layer.q[i] = static_cast<int8_t>(static_cast<int>((i * 73 + k * 19) % 255) - 127);
        for (int i = 0; i < layer.out; ++i) layer.b[i] = (i % 7 - 3) * 0.01;
    }
    generic.mode = neural::Model::FULL;
    auto mobile = generic.isolated_copy(); mobile.mobile_forward = plan;
    Plain<In, Hidden, Out> plain(generic);
    std::vector<std::vector<double>> rows(samples, std::vector<double>(In));
    uint32_t state = 20261110u;
    for (auto& row : rows) for (double& value : row) {
        state = state * 1664525u + 1013904223u;
        value = double((state >> 8) % 1000000u) / 500000.0 - 1.0;
    }
    for (const auto& row : rows) {
        generic.observe_full_reuse(row); mobile.observe_full_reuse(row);
        generic.infer(false); mobile.infer(false); plain.predict(row);
        equivalence(generic, mobile, plain);
    }
    generic.gate("hidden", 0, 0, false); mobile.gate("hidden", 0, 0, false);
    plain.first.enabled[0] = 0;
    const size_t calls = mobile.counters.mobile_plan;
    generic.infer(false); mobile.infer(false); plain.template predict<true>(rows.back());
    equivalence(generic, mobile, plain);
    ensure(mobile.counters.mobile_plan == calls, "disabled edge failed to fall back");
    generic.gate("hidden", 0, 0, true); mobile.gate("hidden", 0, 0, true);
    plain.first.enabled[0] = 1;
    std::cout << "EQUIVALENCE_PASS shape=" << shape << " vectors=" << samples
              << " exact_za=true disabled_fallback=true\n";

    auto run_generic = [&](const auto& row) -> const auto& {
        generic.observe_full_reuse(row); return generic.infer(false);
    };
    auto run_mobile = [&](const auto& row) -> const auto& {
        mobile.observe_full_reuse(row); return mobile.infer(false);
    };
    auto run_plain = [&](const auto& row) -> const auto& {
        plain.predict(row); return plain.second.a;
    };
    // One complete untimed pass warms buffers/code for each engine.
    const auto warm_generic = measure(rows, run_generic);
    const auto warm_mobile = measure(rows, run_mobile);
    const auto warm_plain = measure(rows, run_plain);
    ensure(warm_generic.checksum == warm_mobile.checksum && warm_generic.checksum == warm_plain.checksum,
           "warmup checksum mismatch");
    const char* names[] = {"generic_full_reuse", "mobile_int8_full_reuse", "independent_cpp_fp64_dequant"};
    std::array<std::vector<double>, 3> timings;
    std::cout << "BENCH_META shape=" << shape << " samples=" << samples << " trials=" << trials
              << " numeric=fp64_strict_row_order input_validation=true input_copy=true"
              << " native_integer_dot_product=false device=execution_host\n";
    // Six permutations counterbalance positions and predecessor engines.
    constexpr int orders[6][3] = {{0,1,2}, {2,1,0}, {1,2,0}, {0,2,1}, {2,0,1}, {1,0,2}};
    for (int trial = 0; trial < trials; ++trial) {
        std::array<Measured, 3> result{};
        for (int position = 0; position < 3; ++position) {
            const int engine = orders[trial % 6][position];
            result[engine] = engine == 0 ? measure(rows, run_generic)
                : engine == 1 ? measure(rows, run_mobile) : measure(rows, run_plain);
            timings[engine].push_back(result[engine].ns);
            std::cout << "TRIAL shape=" << shape << " trial=" << trial + 1
                      << " position=" << position + 1 << " engine=" << names[engine]
                      << " ns_per_infer=" << result[engine].ns
                      << " checksum=" << result[engine].checksum << '\n';
        }
        ensure(result[0].checksum == result[1].checksum && result[0].checksum == result[2].checksum,
               "timed checksum mismatch");
    }
    for (int engine = 0; engine < 3; ++engine) {
        auto values = timings[engine]; std::sort(values.begin(), values.end());
        const size_t p95 = (values.size() * 95 + 99) / 100 - 1;
        std::cout << "SUMMARY shape=" << shape << " engine=" << names[engine]
                  << " min_ns=" << values.front() << " p50_ns=" << values[values.size()/2]
                  << " p95_ns=" << values[p95] << " max_ns=" << values.back()
                  << " distribution=per_trial_mean_ns_per_inference\n";
    }
}
size_t argument(const char* value, size_t low, size_t high) {
    size_t end = 0; const std::string text(value);
    const auto result = std::stoul(text, &end);
    ensure(end == text.size() && result >= low && result <= high, "argument out of range");
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        ensure(argc <= 3, "usage: bench_int8 [samples=2048] [trials=9]");
        const size_t samples = argc > 1 ? argument(argv[1], 32, 65536) : 2048;
        const int trials = argc > 2 ? static_cast<int>(argument(argv[2], 3, 31)) : 9;
        std::cout << std::setprecision(17);
        workload<32,64,16>(samples, trials, &nl_mobile_32_64_16);
        workload<128,128,32>(samples, trials, &nl_mobile_128_128_32);
        std::cout << "INT8_BENCH_PASS performance_gate=not_defined android_claim=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "INT8_BENCH_FAIL " << error.what() << '\n';
        return 1;
    }
}
