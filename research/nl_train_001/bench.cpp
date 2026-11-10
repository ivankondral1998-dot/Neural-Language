// Independent fixed-shape FP64 C++ baseline. Only the input/checkpoint loaders
// are shared; the dot products, activations, gradients and updates are separate.
#include "neural_dataset.hpp"
#include <array>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int inputs = 561, hidden = 16, classes = 6, epochs = 20;
constexpr int train_count = 7352, test_count = 2947;
constexpr uint32_t seed = 20261110;
constexpr double rate = .01, tolerance = 2e-8;

struct Mlp {
    std::array<double, hidden * inputs> w{};
    std::array<double, hidden> b{};
    std::array<double, classes * hidden> v{};
    std::array<double, classes> c{};

    explicit Mlp(const neural::Model& model) {
        for (const auto& layer : model.layers)
            neural::require(layer.storage == 0 && layer.disabled == 0, "baseline expects dense FP64");
        std::copy(model.layers[0].w.begin(), model.layers[0].w.end(), w.begin());
        std::copy(model.layers[0].b.begin(), model.layers[0].b.end(), b.begin());
        std::copy(model.layers[1].w.begin(), model.layers[1].w.end(), v.begin());
        std::copy(model.layers[1].b.begin(), model.layers[1].b.end(), c.begin());
    }

    std::array<double, classes> forward(const std::vector<double>& x,
                                        std::array<double, hidden>& h) const {
        for (int row = 0; row < hidden; ++row) {
            double sum = b[row];
            for (int col = 0; col < inputs; ++col) sum += w[row * inputs + col] * x[col];
            h[row] = std::tanh(sum);
        }
        auto out = c;
        for (int row = 0; row < classes; ++row)
            for (int col = 0; col < hidden; ++col) out[row] += v[row * hidden + col] * h[col];
        return out;
    }

    void learn(const neural::DatasetRow& sample) {
        std::array<double, hidden> h{}, grad{};
        auto error = forward(sample.features, h);
        for (int k = 0; k < classes; ++k)
            error[k] = 2. * (error[k] - (k == sample.label ? 1. : 0.)) / classes;
        for (int j = 0; j < hidden; ++j) {
            for (int k = 0; k < classes; ++k) grad[j] += v[k * hidden + j] * error[k];
            grad[j] *= 1. - h[j] * h[j];
        }
        for (int j = 0; j < hidden; ++j) {
            for (int i = 0; i < inputs; ++i) w[j * inputs + i] -= rate * grad[j] * sample.features[i];
            b[j] -= rate * grad[j];
        }
        for (int k = 0; k < classes; ++k) {
            for (int j = 0; j < hidden; ++j) v[k * hidden + j] -= rate * error[k] * h[j];
            c[k] -= rate * error[k];
        }
    }
};

neural::Model model(const std::string& file) {
    neural::Model m("Bench", inputs);
    m.layer("hidden", hidden, 3, 1);
    m.layer("activity", classes, 0, 2);
    m.load_model(file);
    return m;
}

void train_baseline(Mlp& m) {
    const auto train = neural::read_dataset("har_train.txt", train_count, inputs, classes);
    std::vector<size_t> order(train.size());
    std::mt19937 rng(seed);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::iota(order.begin(), order.end(), 0);
        for (size_t i = order.size(); i > 1; --i)
            std::swap(order[i - 1], order[(static_cast<uint64_t>(rng()) * i) >> 32]);
        for (size_t i : order) m.learn(train[i]);
    }
}

template <class T>
double difference(const T& x, const T& y) {
    double worst = 0.;
    for (size_t i = 0; i < x.size(); ++i) {
        const double delta = std::abs(x[i] - y[i]);
        neural::require(std::isfinite(delta) && delta <= tolerance, "baseline comparison failed");
        worst = std::max(worst, delta);
    }
    return worst;
}

template <class Predictor>
void measure(const char* mode, const std::vector<neural::DatasetRow>& rows,
             const std::vector<std::array<double, classes>>& expected, Predictor predict) {
    std::vector<double> us;
    us.reserve(rows.size());
    double worst = 0.;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto start = Clock::now();
        const auto got = predict(rows[i].features);
        const auto stop = Clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
        worst = std::max(worst, difference(got, expected[i]));
    }
    std::sort(us.begin(), us.end());
    std::cout << mode << ',' << us[us.size()/2] << ',' << us[(us.size()-1)*95/100] << ','
              << worst;
}
} // namespace

int main() {
    try {
        const Mlp trained(model("har_model.nlw"));
        auto start_model = model("har_initial.nlw");
        Mlp cpp_train(start_model);
        auto start = Clock::now();
        train_baseline(cpp_train);
        std::cout << "CPP_TRAIN_seconds_including_input=" <<
            std::chrono::duration<double>(Clock::now()-start).count() <<
            " max_param_difference=" << std::max({difference(cpp_train.w,trained.w),difference(cpp_train.b,trained.b),
                                                   difference(cpp_train.v,trained.v),difference(cpp_train.c,trained.c)}) << '\n';
        const auto rows = neural::read_dataset("har_test.txt", test_count, inputs);
        std::vector<std::array<double, classes>> expected;
        for (const auto& row : rows) {
            std::array<double, hidden> scratch{};
            expected.push_back(trained.forward(row.features, scratch));
        }
        std::cout << "mode,p50_us,p95_us,max_difference,full_steps,partial_steps\n";
        auto base = model("har_model.nlw");
        for (int trial = 0; trial < 7; ++trial) {
            for (int i = 0; i < 4; ++i) {
                const int mode = (i + trial) % 4;
                if (mode == 3) {
                    measure("C++", rows, expected, [&](const auto& x) {
                        std::array<double, hidden> scratch{};
                        return trained.forward(x, scratch);
                    });
                    std::cout << ',' << rows.size() << ",0\n";
                } else {
                    auto m = base;
                    m.mode = static_cast<neural::Model::Mode>(mode);
                    // Give each model the same warm cache before timing.
                    m.observe(rows.back().features);
                    m.infer(false);
                    m.counters = {};
                    const char* names[] = {"AUTO", "FULL", "DELTA"};
                    measure(names[mode], rows, expected, [&](const auto& x) {
                        m.observe(x);
                        const auto& values = m.infer(false);
                        return std::array<double, classes>{values[0], values[1], values[2], values[3], values[4], values[5]};
                    });
                    std::cout << ',' << m.counters.full << ',' << m.counters.delta << '\n';
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
