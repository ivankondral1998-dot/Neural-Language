// Host-only cost of canonical scalar recomputation. Build without BLAS.
#include "neural_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int width = 512;
constexpr int column = 17;
constexpr int warmup = 32;
constexpr int samples = 240;

struct Result {
    std::vector<double> us;
    std::vector<double> outputs;
    size_t partial;
};

neural::Model make_model(const std::string& workload, neural::Model::Mode mode) {
    neural::Model m("Cost", width);
    m.layer("out", width, 0, 53);
    if (workload == "sparse_input") {
        for (int row = 0; row < width; ++row) {
            if (row % 16) m.layers[0].w[static_cast<size_t>(row) * width + column] = 0.;
        }
    }
    m.mode = mode;
    m.observe(std::vector<double>(width, 0.125));
    m.infer(false);
    return m;
}

Result measure(const std::string& workload, neural::Model::Mode mode) {
    auto m = make_model(workload, mode);
    Result result{};
    result.us.reserve(samples);
    result.outputs.reserve(samples * width);
    for (int i = -warmup; i < samples; ++i) {
        const double d = (i % 2 ? -1. : 1.) / 1024.;
        const auto start = Clock::now();
        if (workload == "weight") m.change("out", 17, column, d);
        else m.change_input(column, d);
        const auto& values = m.infer(false);
        const auto stop = Clock::now();
        if (i == -1) m.counters = {};
        if (i >= 0) {
            result.us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
            result.outputs.insert(result.outputs.end(), values.begin(), values.end());
        }
    }
    result.partial = m.counters.delta;
    return result;
}

double quantile(std::vector<double> x, size_t percent) {
    std::sort(x.begin(), x.end());
    return x[(x.size() - 1) * percent / 100];
}
} // namespace

int main() {
    if (neural::has_blas) throw std::runtime_error("this benchmark requires the scalar build");
    std::cout << "workload,mode,trial,full_p50_us,other_p50_us,full_p95_us,other_p95_us,partial_steps,unequal,max_abs\n";
    for (const std::string workload : {"dense_input", "sparse_input", "weight"}) {
        for (const auto mode : {neural::Model::AUTO, neural::Model::DELTA}) {
            for (int trial = 0; trial < 5; ++trial) {
                Result f, other;
                if (trial % 2) {
                    other = measure(workload, mode);
                    f = measure(workload, neural::Model::FULL);
                } else {
                    f = measure(workload, neural::Model::FULL);
                    other = measure(workload, mode);
                }
                if (f.outputs.size() != other.outputs.size()) throw std::runtime_error("shape mismatch");
                size_t unequal = 0;
                double max_abs = 0.;
                for (size_t i = 0; i < f.outputs.size(); ++i) {
                    const double diff = std::abs(f.outputs[i] - other.outputs[i]);
                    if (!std::isfinite(diff) || diff > 1e-9) throw std::runtime_error("numeric mismatch");
                    unequal += diff != 0.;
                    max_abs = std::max(max_abs, diff);
                }
                std::cout << std::setprecision(12) << workload << ','
                          << (mode == neural::Model::AUTO ? "AUTO" : "DELTA") << ','
                          << trial << ',' << quantile(f.us, 50) << ',' << quantile(other.us, 50)
                          << ',' << quantile(f.us, 95) << ',' << quantile(other.us, 95)
                          << ',' << other.partial << ',' << unequal << ',' << max_abs << '\n';
            }
        }
    }
}
