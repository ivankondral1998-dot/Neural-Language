// Acceptance bench for NL-MOBILE-OPT-001. Thresholds are fixed in protocol.txt.
// A result from this file is a host result, never a Samsung A52 PASS.
#include "neural_runtime.hpp"
#include "mobile_plan.hpp"
#include "neural_dataset.hpp"

#include <dlfcn.h>
#include <sys/resource.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

namespace {
std::atomic<size_t> alloc_count{0};
bool count_allocs = false;

void* counted_malloc(std::size_t n) {
    if (count_allocs) alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (n == 0) n = 1;
    return std::malloc(n);
}
}  // namespace

void* operator new(std::size_t n) {
    if (void* p = counted_malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    if (void* p = counted_malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return counted_malloc(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return counted_malloc(n); }

namespace {
using Clock = std::chrono::steady_clock;

enum class Kind { Generic, Mobile, Scalar, Ilp, Blas };

const char* kind_name(Kind kind) {
    switch (kind) {
    case Kind::Generic: return "nl_generic_observe_infer";
    case Kind::Mobile: return "nl_mobile_observe_infer";
    case Kind::Scalar: return "cpp_scalar_kernel";
    case Kind::Ilp: return "cpp_ilp_kernel";
    case Kind::Blas: return "openblas_dgemv_kernel";
    }
    return "unknown";
}

using Gemv = void (*)(int, int, int64_t, int64_t, double, const double*, int64_t,
                      const double*, int64_t, double, double*, int64_t);
Gemv blas_gemv = nullptr;

int argmax_of(const double* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

template <int In, int Out, int Act>
void cpp_scalar(const double* w, const double* b, const double* x, double* y) {
    for (int row = 0; row < Out; ++row) {
        double sum = b[row];
        const double* r = w + static_cast<size_t>(row) * In;
        for (int col = 0; col < In; ++col) sum += r[col] * x[col];
        if constexpr (Act == 3) y[row] = std::tanh(sum);
        else if constexpr (Act == 1) y[row] = std::max(0.0, sum);
        else if constexpr (Act == 2)
            y[row] = sum >= 0 ? 1.0 / (1.0 + std::exp(-sum)) : std::exp(sum) / (1.0 + std::exp(sum));
        else y[row] = sum;
    }
}

// Hand-written twin of the plan. It does not call neural::mobile_dense_ilp4.
template <int In, int Out, int Act>
void cpp_ilp_rows(const double* w, const double* b, const double* x, double* y) {
    int row = 0;
    for (; row + 3 < Out; row += 4) {
        double s0 = b[row], s1 = b[row + 1], s2 = b[row + 2], s3 = b[row + 3];
        const double* a = w + static_cast<size_t>(row) * In;
        const double* c = a + In;
        const double* d = c + In;
        const double* e = d + In;
        for (int col = 0; col < In; ++col) {
            const double xv = x[col];
            s0 += a[col] * xv;
            s1 += c[col] * xv;
            s2 += d[col] * xv;
            s3 += e[col] * xv;
        }
        auto store = [](double s) {
            if constexpr (Act == 3) return std::tanh(s);
            else if constexpr (Act == 1) return std::max(0.0, s);
            else if constexpr (Act == 2)
                return s >= 0 ? 1.0 / (1.0 + std::exp(-s)) : std::exp(s) / (1.0 + std::exp(s));
            else return s;
        };
        y[row] = store(s0);
        y[row + 1] = store(s1);
        y[row + 2] = store(s2);
        y[row + 3] = store(s3);
    }
    for (; row < Out; ++row) {
        double sum = b[row];
        const double* r = w + static_cast<size_t>(row) * In;
        for (int col = 0; col < In; ++col) sum += r[col] * x[col];
        if constexpr (Act == 3) y[row] = std::tanh(sum);
        else if constexpr (Act == 1) y[row] = std::max(0.0, sum);
        else if constexpr (Act == 2)
            y[row] = sum >= 0 ? 1.0 / (1.0 + std::exp(-sum)) : std::exp(sum) / (1.0 + std::exp(sum));
        else y[row] = sum;
    }
}

void blas_layer(const double* w, const double* b, const double* x, double* y, int out, int in, int act) {
    blas_gemv(101, 111, out, in, 1.0, w, in, x, 1, 0.0, y, 1);
    for (int row = 0; row < out; ++row) {
        const double sum = y[row] + b[row];
        if (act == 3) y[row] = std::tanh(sum);
        else if (act == 1) y[row] = std::max(0.0, sum);
        else if (act == 2)
            y[row] = sum >= 0 ? 1.0 / (1.0 + std::exp(-sum)) : std::exp(sum) / (1.0 + std::exp(sum));
        else y[row] = sum;
    }
}

bool har_plan(neural::Model& m) {
    if (m.graph || m.layers.size() != 2 || m.ninput != 561) return false;
    const auto& a = m.layers[0];
    const auto& b = m.layers[1];
    if (a.in != 561 || a.out != 16 || a.type != 3 || a.storage != 0 || a.disabled != 0) return false;
    if (b.in != 16 || b.out != 6 || b.type != 0 || b.storage != 0 || b.disabled != 0) return false;
    if (static_cast<int>(m.z[0].size()) != 16) { m.z[0].assign(16, 0); m.a[0].assign(16, 0); }
    if (static_cast<int>(m.z[1].size()) != 6) { m.z[1].assign(6, 0); m.a[1].assign(6, 0); }
    neural::mobile_dense_ilp4<561, 16, 3>(a.w.data(), a.b.data(), m.input.data(), m.z[0].data(), m.a[0].data());
    neural::mobile_dense_ilp4<16, 6, 0>(b.w.data(), b.b.data(), m.a[0].data(), m.z[1].data(), m.a[1].data());
    return true;
}

bool cart_plan(neural::Model& m) {
    if (m.graph || m.layers.size() != 3 || m.ninput != 4) return false;
    const auto& a = m.layers[0];
    const auto& b = m.layers[1];
    const auto& c = m.layers[2];
    if (a.in != 4 || a.out != 64 || a.type != 3 || a.storage != 0 || a.disabled != 0) return false;
    if (b.in != 64 || b.out != 64 || b.type != 3 || b.storage != 0 || b.disabled != 0) return false;
    if (c.in != 64 || c.out != 2 || c.type != 0 || c.storage != 0 || c.disabled != 0) return false;
    if (static_cast<int>(m.z[0].size()) != 64) { m.z[0].assign(64, 0); m.a[0].assign(64, 0); }
    if (static_cast<int>(m.z[1].size()) != 64) { m.z[1].assign(64, 0); m.a[1].assign(64, 0); }
    if (static_cast<int>(m.z[2].size()) != 2) { m.z[2].assign(2, 0); m.a[2].assign(2, 0); }
    neural::mobile_dense_ilp4<4, 64, 3>(a.w.data(), a.b.data(), m.input.data(), m.z[0].data(), m.a[0].data());
    neural::mobile_dense_ilp4<64, 64, 3>(b.w.data(), b.b.data(), m.a[0].data(), m.z[1].data(), m.a[1].data());
    neural::mobile_dense_ilp4<64, 2, 0>(c.w.data(), c.b.data(), m.a[1].data(), m.z[2].data(), m.a[2].data());
    return true;
}

struct Task {
    std::string name;
    std::string model_file;
    std::vector<neural::DatasetRow> rows;
    std::vector<int> labels;
};

neural::Model make_model(const Task& task, bool mobile) {
    if (task.name == "har") {
        neural::Model model("HAR", 561);
        model.layer("hidden", 16, 3, 42u);
        model.layer("activity", 6, 0, 43u);
        model.load_model(task.model_file);
        model.mode = neural::Model::FULL;
        if (mobile) model.mobile_forward = &har_plan;
        return model;
    }
    neural::Model model("CartPolePretrained", 4);
    model.layer("hidden1", 64, 3, 42u);
    model.layer("hidden2", 64, 3, 43u);
    model.layer("actions", 2, 0, 44u);
    model.load_model(task.model_file);
    model.mode = neural::Model::FULL;
    if (mobile) model.mobile_forward = &cart_plan;
    return model;
}

void kernel(const Task& task, Kind kind, const neural::Model& weights, const double* x, double* y) {
    if (task.name == "har") {
        double hidden[16];
        const auto& l0 = weights.layers[0];
        const auto& l1 = weights.layers[1];
        if (kind == Kind::Scalar) {
            cpp_scalar<561, 16, 3>(l0.w.data(), l0.b.data(), x, hidden);
            cpp_scalar<16, 6, 0>(l1.w.data(), l1.b.data(), hidden, y);
        } else if (kind == Kind::Ilp) {
            cpp_ilp_rows<561, 16, 3>(l0.w.data(), l0.b.data(), x, hidden);
            cpp_ilp_rows<16, 6, 0>(l1.w.data(), l1.b.data(), hidden, y);
        } else {
            blas_layer(l0.w.data(), l0.b.data(), x, hidden, 16, 561, 3);
            blas_layer(l1.w.data(), l1.b.data(), hidden, y, 6, 16, 0);
        }
        return;
    }
    double h1[64], h2[64];
    const auto& l0 = weights.layers[0];
    const auto& l1 = weights.layers[1];
    const auto& l2 = weights.layers[2];
    if (kind == Kind::Scalar) {
        cpp_scalar<4, 64, 3>(l0.w.data(), l0.b.data(), x, h1);
        cpp_scalar<64, 64, 3>(l1.w.data(), l1.b.data(), h1, h2);
        cpp_scalar<64, 2, 0>(l2.w.data(), l2.b.data(), h2, y);
    } else if (kind == Kind::Ilp) {
        cpp_ilp_rows<4, 64, 3>(l0.w.data(), l0.b.data(), x, h1);
        cpp_ilp_rows<64, 64, 3>(l1.w.data(), l1.b.data(), h1, h2);
        cpp_ilp_rows<64, 2, 0>(l2.w.data(), l2.b.data(), h2, y);
    } else {
        blas_layer(l0.w.data(), l0.b.data(), x, h1, 64, 4, 3);
        blas_layer(l1.w.data(), l1.b.data(), h1, h2, 64, 64, 3);
        blas_layer(l2.w.data(), l2.b.data(), h2, y, 2, 64, 0);
    }
}

double percentile(std::vector<double> values, int index) {
    std::sort(values.begin(), values.end());
    return values.at(static_cast<size_t>(index));
}

struct Summary {
    double p50 = 0;
    double p95 = 0;
    double allocs_per_sample = 0;
    double max_abs = 0;
    int correct = 0;
    int class_mismatch = 0;
    size_t plan_calls = 0;
};

std::vector<Summary> run_suite(const Task& task, const std::vector<Kind>& kinds) {
    auto generic = make_model(task, false);
    auto mobile = make_model(task, true);
    auto weights = make_model(task, false);
    const int classes = task.name == "har" ? 6 : 2;
    const int n = static_cast<int>(task.rows.size());
    std::vector<double> y(static_cast<size_t>(classes));
    auto invoke = [&](Kind kind, const neural::DatasetRow& row) {
        if (kind == Kind::Generic) {
            generic.observe(row.features);
            const auto& out = generic.infer(false);
            std::copy(out.begin(), out.end(), y.begin());
        } else if (kind == Kind::Mobile) {
            mobile.observe(row.features);
            const auto& out = mobile.infer(false);
            std::copy(out.begin(), out.end(), y.begin());
        } else {
            kernel(task, kind, weights, row.features.data(), y.data());
        }
    };
    for (Kind kind : kinds)
        for (const auto& row : task.rows) invoke(kind, row);

    std::vector<std::vector<double>> p50s(kinds.size()), p95s(kinds.size());
    std::vector<size_t> allocs(kinds.size(), 0);
    for (int pass = 0; pass < 7; ++pass) {
        for (size_t slot = 0; slot < kinds.size(); ++slot) {
            const size_t index = (slot + static_cast<size_t>(pass)) % kinds.size();
            const Kind kind = kinds[index];
            std::vector<double> samples;
            samples.reserve(task.rows.size());
            const bool track = pass == 0;
            if (track) {
                alloc_count.store(0);
                count_allocs = true;
            }
            for (const auto& row : task.rows) {
                const auto t0 = Clock::now();
                invoke(kind, row);
                const auto t1 = Clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            if (track) {
                count_allocs = false;
                allocs[index] = alloc_count.load();
            }
            p50s[index].push_back(percentile(samples, n / 2));
            p95s[index].push_back(percentile(samples, (n - 1) * 95 / 100));
        }
    }
    const size_t plan_calls = mobile.counters.mobile_plan;
    std::vector<Summary> summaries(kinds.size());
    auto oracle = make_model(task, false);
    for (size_t index = 0; index < kinds.size(); ++index) {
        Summary summary;
        summary.p50 = percentile(p50s[index], 3);
        summary.p95 = percentile(p95s[index], 3);
        summary.allocs_per_sample = static_cast<double>(allocs[index]) / static_cast<double>(n);
        summary.plan_calls = kinds[index] == Kind::Mobile ? plan_calls : 0;
        for (int i = 0; i < n; ++i) {
            invoke(kinds[index], task.rows[static_cast<size_t>(i)]);
            oracle.observe(task.rows[static_cast<size_t>(i)].features);
            const auto& ref = oracle.infer(false);
            for (int c = 0; c < classes; ++c)
                summary.max_abs = std::max(summary.max_abs, std::abs(y[static_cast<size_t>(c)] - ref[static_cast<size_t>(c)]));
            const int got = argmax_of(y.data(), classes);
            const int expect = argmax_of(ref.data(), classes);
            if (got != expect) summary.class_mismatch++;
            if (task.name == "har" && got == task.labels[static_cast<size_t>(i)]) summary.correct++;
            if (task.name != "har" && got == expect) summary.correct++;
        }
        summaries[index] = summary;
    }
    return summaries;
}

std::vector<neural::DatasetRow> cart_states() {
    std::vector<neural::DatasetRow> rows;
    rows.reserve(1024);
    uint32_t state = 20261110u;
    for (int i = 0; i < 1024; ++i) {
        neural::DatasetRow row;
        row.features.resize(4);
        for (int c = 0; c < 4; ++c) {
            state = state * 1664525u + 1013904223u;
            row.features[static_cast<size_t>(c)] =
                static_cast<double>((state >> 8) % 1000000u) / 1000000.0 - 0.5;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::vector<int> read_labels(const std::string& path, int n) {
    std::ifstream in(path);
    neural::require(bool(in), "cannot open labels");
    std::vector<int> labels;
    for (int i = 0; i < n; ++i) {
        int label = -1;
        neural::require(bool(in >> label) && label >= 0 && label < 6, "bad label");
        labels.push_back(label);
    }
    return labels;
}

long rss_kb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void load_timing(const Task& task) {
    std::vector<double> samples;
    for (int i = 0; i < 8; ++i) {
        const auto t0 = Clock::now();
        auto model = make_model(task, false);
        const auto t1 = Clock::now();
        neural::require(model.layers.size() >= 2, "load produced an empty model");
        if (i == 0) continue;
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::cout << "load_model_median_us=" << percentile(samples, 3) << "\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string task_name = "har";
        std::string only;
        bool do_load = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--task" && i + 1 < argc) task_name = argv[++i];
            else if (arg == "--only" && i + 1 < argc) only = argv[++i];
            else if (arg == "--load") do_load = true;
            else throw std::runtime_error(std::string("unknown argument ") + arg);
        }
        Task task;
        task.name = task_name;
        if (task_name == "har") {
            task.model_file = "har_model.nlw";
            task.rows = neural::read_dataset("har_test.txt", 2947, 561);
            task.labels = read_labels("har_labels.txt", 2947);
        } else if (task_name == "cartpole") {
            task.model_file = "cartpole_pretrained.nlw";
            task.rows = cart_states();
        } else {
            throw std::runtime_error("unknown task");
        }
        std::cout << std::setprecision(17);
        std::cout << "task=" << task.name << " rows=" << task.rows.size() << "\n";
        if (do_load) {
            load_timing(task);
            std::cout << "rss_kb=" << rss_kb() << "\n";
            return 0;
        }
        if (const char* path = std::getenv("NL_OPENBLAS")) {
            void* handle = dlopen(path, RTLD_NOW);
            neural::require(handle != nullptr, dlerror() ? dlerror() : "dlopen failed");
            blas_gemv = reinterpret_cast<Gemv>(dlsym(handle, "scipy_cblas_dgemv64_"));
            neural::require(blas_gemv != nullptr, "scipy_cblas_dgemv64_ missing");
        }
        std::vector<Kind> kinds;
        auto add = [&](const std::string& name, Kind kind) {
            if (only.empty() || only == name) kinds.push_back(kind);
        };
        add("generic", Kind::Generic);
        add("mobile", Kind::Mobile);
        add("scalar", Kind::Scalar);
        add("ilp", Kind::Ilp);
        if (blas_gemv) add("blas", Kind::Blas);
        if (!only.empty() && kinds.empty()) throw std::runtime_error("requested implementation is unavailable");
        const auto summaries = run_suite(task, kinds);
        for (size_t i = 0; i < kinds.size(); ++i) {
            const auto& s = summaries[i];
            std::cout << "impl=" << kind_name(kinds[i])
                      << " p50_us=" << s.p50
                      << " p95_us=" << s.p95
                      << " allocs_per_sample=" << s.allocs_per_sample
                      << " max_abs_vs_generic=" << s.max_abs
                      << " correct=" << s.correct
                      << " class_mismatch_vs_generic=" << s.class_mismatch
                      << " plan_calls=" << s.plan_calls
                      << "\n";
        }
        std::cout << "rss_kb=" << rss_kb() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "BENCH_FAIL " << e.what() << '\n';
        return 1;
    }
}
