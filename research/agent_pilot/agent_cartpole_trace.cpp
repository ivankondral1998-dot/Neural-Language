// Deterministic pretrained CartPole trace from the ACTUAL Neural Language runtime.
// Stdout TRACE rows are consumed by compare_native_cartpole.py.
// Not a benchmark: emitting rows dominates timings. Never use for speed claims.
#include "neural_runtime.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using State = std::array<double, 4>;

struct CartPole {
  State s{};
  explicit CartPole(uint32_t seed) {
    uint32_t a = seed;
    for (auto &v : s) {
      a = 1664525u * a + 1013904223u;
      v = (double(a) / 4294967296.0 - 0.5) * 0.1;
    }
  }
  bool step(int action) {
    const double x = s[0], xd = s[1], th = s[2], thd = s[3];
    const double ct = std::cos(th), st = std::sin(th);
    const double temp = ((action ? 10.0 : -10.0) + 0.05 * thd * thd * st) / 1.1;
    const double tha = (9.8 * st - ct * temp) /
        (0.5 * (4.0 / 3.0 - 0.1 * ct * ct / 1.1));
    const double xa = temp - 0.05 * tha * ct / 1.1;
    s = {x + 0.02 * xd, xd + 0.02 * xa, th + 0.02 * thd, thd + 0.02 * tha};
    return !(std::abs(s[0]) > 2.4 || std::abs(s[2]) > 12.0 * std::acos(-1.0) / 180.0);
  }
};

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: agent_cartpole_trace MODEL.nlw EPISODES");
    const int episodes = std::stoi(argv[2]);
    if (episodes < 1 || episodes > 1000) throw std::runtime_error("episodes out of range");
    neural::Model automatic("CartPolePretrained", 4);
    neural::Model reference("CartPolePretrained", 4);
    for (auto *model : {&automatic, &reference}) {
      model->layer("hidden1", 64, 3, 42);
      model->layer("hidden2", 64, 3, 43);
      model->layer("actions", 2, 0, 44);
      model->load_model(argv[1]);
    }
    automatic.mode = neural::Model::AUTO;
    reference.mode = neural::Model::FULL;
    std::cout << std::setprecision(17);
    size_t emitted = 0;
    for (int episode = 1; episode <= episodes; ++episode) {
      CartPole env(static_cast<uint32_t>(episode));
      for (int step = 0; step < 500; ++step) {
        const auto observation = env.s;
        const std::vector<double> input(observation.begin(), observation.end());
        automatic.observe(input);
        reference.observe(input);
        const auto fast = automatic.infer(false);
        const auto full = reference.infer(false);
        if (fast.size() != 2 || full.size() != 2) throw std::runtime_error("invalid output size");
        for (int j = 0; j < 2; ++j) {
          if (!std::isfinite(fast[j]) || !std::isfinite(full[j]))
            throw std::runtime_error("nonfinite model output");
        }
        const int action = fast[1] > fast[0] ? 1 : 0;
        std::cout << "TRACE " << episode << ' ' << step;
        for (double value : observation) std::cout << ' ' << value;
        std::cout << ' ' << fast[0] << ' ' << fast[1]
                  << ' ' << full[0] << ' ' << full[1]
                  << ' ' << action << '\n';
        ++emitted;
        if (!env.step(action)) break;
      }
    }
    std::cout << "TRACE_SUMMARY " << episodes << ' ' << emitted << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "PILOT_TRACE_ERROR " << e.what() << '\n';
    return 2;
  }
}
