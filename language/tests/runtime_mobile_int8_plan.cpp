// Packed INT8 storage, FP64 arithmetic: exact reduction, mixed plans and guards.
#include "neural_runtime.hpp"
#include "mobile_plan.hpp"
#include "generated_mobile_plan.hpp"
#include <limits>

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<int Out, int Act>
static void kernel_case() {
    constexpr int In = 5;
    std::vector<int8_t> q(In * Out);
    std::vector<double> b(Out), z(Out), a(Out);
    const std::vector<double> x = {0.125, -0.75, 0.3, 0.1, -0.25};
    for (int i = 0; i < In * Out; ++i) q[i] = static_cast<int8_t>((i * 73) % 255 - 127);
    for (int row = 0; row < Out; ++row) b[row] = (row - 2) * 0.07;
    for (double scale : {1.0 / 127.0, 0.03125, 0.00013}) {
        neural::mobile_dense_int8_ilp4<In, Out, Act>(
            q.data(), b.data(), x.data(), z.data(), a.data(), scale);
        for (int row = 0; row < Out; ++row) {
            double expected = b[row];
            for (int col = 0; col < In; ++col) {
                const double weight = double(q[row * In + col]) * scale;
                expected += weight * x[col];
            }
            double activated = expected;
            if constexpr (Act == 1) activated = std::max(0.0, expected);
            if constexpr (Act == 2)
                activated = expected >= 0 ? 1.0 / (1.0 + std::exp(-expected))
                    : std::exp(expected) / (1.0 + std::exp(expected));
            if constexpr (Act == 3) activated = std::tanh(expected);
            check(z[row] == expected && a[row] == activated,
                  "INT8 storage changed FP64 reduction or activation");
        }
    }
}

static neural::Model model() {
    neural::Model m("Mixed", 5);
    m.layer("hidden", 7, 1, 11u, 1);
    m.layer("middle", 3, 2, 12u);
    m.layer("out", 5, 3, 13u, 1);
    m.mode = neural::Model::FULL;
    m.observe({0.25, -0.5, 0.1, 0.3, -0.2});
    return m;
}

int main() {
    try {
        kernel_case<1, 0>(); kernel_case<3, 1>(); kernel_case<4, 2>();
        kernel_case<5, 3>(); kernel_case<7, 0>(); kernel_case<8, 1>();
        // Cancellation distinguishes canonical summation from reassociation.
        const int8_t q[] = {127, 1, -127};
        const double b[] = {0.0}, x[] = {900000.0, 1e-12, 900000.0};
        double z[1], a[1];
        neural::mobile_dense_int8_ilp4<3, 1, 0>(q, b, x, z, a, 1.0 / 127.0);
        double expected = b[0];
        for (int col = 0; col < 3; ++col) expected += (double(q[col]) * (1.0 / 127.0)) * x[col];
        check(z[0] == expected, "INT8 cancellation reduction changed");

        auto generic = model(), mobile = model();
        mobile.mobile_forward = &nl_mobile_try;
        check(mobile.layers[0].w.empty() && mobile.layers[2].w.empty(), "INT8 allocated FP64 weights");
        for (int i = 0; i < 100; ++i) {
            const std::vector<double> row = {i * 0.001, -0.2, 0.125, -0.375, 0.5};
            generic.observe(row); mobile.observe(row);
            generic.infer(false); mobile.infer(false);
            check(generic.z == mobile.z && generic.a == mobile.a, "mixed plan differs from FULL");
        }
        check(mobile.counters.mobile_plan == 100, "mixed plan did not run");
        // All guards must reject before touching outputs, including a late layer.
        const auto before = mobile.z;
        mobile.layers[2].storage = 0;
        check(!nl_mobile_try(mobile) && mobile.z == before, "storage mismatch ran plan");
        mobile.layers[2].storage = 1;
        mobile.layers[2].in += 1;
        check(!nl_mobile_try(mobile) && mobile.z == before, "shape mismatch ran plan");
        mobile.layers[2].in -= 1;
        generic.gate("hidden", 0, 0, false); mobile.gate("hidden", 0, 0, false);
        const auto calls = mobile.counters.mobile_plan;
        generic.infer(false); mobile.infer(false);
        check(generic.z == mobile.z && generic.a == mobile.a, "masked fallback differs");
        check(mobile.counters.mobile_plan == calls, "masked layer bypassed fallback");
        generic.gate("hidden", 0, 0, true); mobile.gate("hidden", 0, 0, true);
        generic.infer(false); mobile.infer(false);
        check(generic.z == mobile.z && generic.a == mobile.a, "reenabled plan differs");
        check(mobile.counters.mobile_plan == calls + 1, "reenabled plan did not resume");

        // Numeric checks remain active in the specialized path.
        auto bad = model(); bad.mobile_forward = &nl_mobile_try;
        bad.layers[0].b[0] = std::numeric_limits<double>::infinity();
        bool rejected = false;
        try { bad.infer(false); }
        catch (const std::runtime_error& e) {
            rejected = std::string(e.what()) == "invalid numeric value: forward activation";
        }
        check(rejected && !bad.cache, "specialized plan accepted nonfinite preactivation");
        bad.layers[0].b[0] = 0;
        bad.infer(false);
        check(bad.cache && bad.cache_canonical, "specialized plan failed to recover");
        std::cout << "MOBILE_INT8_PLAN_PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MOBILE_INT8_PLAN_FAIL " << e.what() << '\n';
        return 1;
    }
}
