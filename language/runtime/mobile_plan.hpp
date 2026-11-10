#pragma once
// Neural Language mobile-cpu plan kernel.
// Include only after neural_runtime.hpp. Not a general tensor compiler.
//
// Numeric contract (fp64-strict-row-order):
//   For each output row, z = bias + ((w[0]*x[0]) + w[1]*x[1]) + ... + w[In-1]*x[In-1]
//   in that exact left-to-right order. This is the same reduction as
//   scalar_preactivation<true> used by generic FULL.
//   Groups of four rows use distinct accumulators. That is instruction-level
//   parallelism, not a reassociation of one row and not an FMA contraction.
//   Activation matches neural::act for linear / relu / sigmoid / tanh.
//   finite() is still applied to every pre-activation.
//   If any edge is disabled or storage is not FP64, the generated plan must
//   return false and Model::forward uses the generic loop.
// Signed zero is outside the value contract, as in NL-NUMERIC-STRICT-1.
#include <algorithm>
#include <cstddef>

namespace neural {

template <int Act>
inline double mobile_activate(double z) {
    if constexpr (Act == 1) return std::max(0.0, z);
    if constexpr (Act == 2)
        return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
    if constexpr (Act == 3) return std::tanh(z);
    else return z;
}

template <int In, int Out, int Act>
inline void mobile_dense_ilp4(const double* __restrict weights, const double* __restrict bias,
                             const double* __restrict input, double* __restrict pre,
                             double* __restrict activation) {
    int row = 0;
    for (; row + 3 < Out; row += 4) {
        double s0 = bias[row];
        double s1 = bias[row + 1];
        double s2 = bias[row + 2];
        double s3 = bias[row + 3];
        const double* r0 = weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(In);
        const double* r1 = r0 + In;
        const double* r2 = r1 + In;
        const double* r3 = r2 + In;
        for (int col = 0; col < In; ++col) {
            const double x = input[col];
            s0 += r0[col] * x;
            s1 += r1[col] * x;
            s2 += r2[col] * x;
            s3 += r3[col] * x;
        }
        finite(s0, "forward activation");
        finite(s1, "forward activation");
        finite(s2, "forward activation");
        finite(s3, "forward activation");
        pre[row] = s0;
        pre[row + 1] = s1;
        pre[row + 2] = s2;
        pre[row + 3] = s3;
        activation[row] = mobile_activate<Act>(s0);
        activation[row + 1] = mobile_activate<Act>(s1);
        activation[row + 2] = mobile_activate<Act>(s2);
        activation[row + 3] = mobile_activate<Act>(s3);
    }
    for (; row < Out; ++row) {
        double sum = bias[row];
        const double* r = weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(In);
        for (int col = 0; col < In; ++col) sum += r[col] * input[col];
        finite(sum, "forward activation");
        pre[row] = sum;
        activation[row] = mobile_activate<Act>(sum);
    }
}

}  // namespace neural
