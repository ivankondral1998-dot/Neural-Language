// NL-MOBILE-OPT-002: FULL input reuse must be semantically identical to
// observe(by-value), including cache behavior and invalid input rejection.
#include "neural_runtime.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    try {
        neural::Model baseline("InputReuse", 3), reused("InputReuse", 3);
        baseline.layer("hidden", 4, 3, 17u);
        reused.layer("hidden", 4, 3, 17u);
        baseline.layer("output", 2, 0, 23u);
        reused.layer("output", 2, 0, 23u);
        baseline.mode = neural::Model::FULL;
        reused.mode = neural::Model::FULL;

        const std::vector<std::vector<double>> inputs = {
            {0.0, 0.0, 0.0}, {0.2, -0.1, 0.5}, {0.2, -0.1, 0.5},
            {-0.6, 0.1, 0.2}, {0.0, 0.0, 0.0}, {0.7, -0.3, -0.2}
        };
        const double* stable_buffer = nullptr;
        for (int repeat = 0; repeat < 50; ++repeat) {
            for (const auto& row : inputs) {
                baseline.observe(row);
                reused.observe_full_reuse(row);
                const auto expected = baseline.infer(false);
                const auto actual = reused.infer(false);
                if (actual != expected || reused.input != baseline.input)
                    throw std::runtime_error("FULL reused input changes numerical results");
                if (!stable_buffer) stable_buffer = reused.input.data();
                if (stable_buffer != reused.input.data())
                    throw std::runtime_error("FULL reused input allocated another buffer");
            }
        }
        const auto saved_input = reused.input;
        const auto saved_predictions = reused.infer(false);
        const auto saved_counts = reused.counters;
        try {
            reused.observe_full_reuse({0.1, std::numeric_limits<double>::quiet_NaN(), 0.3});
            throw std::runtime_error("nonfinite observation accepted");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "nonfinite observation accepted") throw;
        }
        if (reused.input != saved_input || reused.infer(false) != saved_predictions ||
            reused.counters.full != saved_counts.full)
            throw std::runtime_error("failed observation mutated live state");
        reused.mode = neural::Model::AUTO;
        try {
            reused.observe_full_reuse({0.1, 0.2, 0.3});
            throw std::runtime_error("AUTO accepted a FULL-only input path");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "AUTO accepted a FULL-only input path") throw;
        }
        std::cout << "MOBILE_INPUT_REUSE_PASS samples=" << inputs.size() * 50 << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MOBILE_INPUT_REUSE_FAIL " << error.what() << "\n";
        return 1;
    }
}
