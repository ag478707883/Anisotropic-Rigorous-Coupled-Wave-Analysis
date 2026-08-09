#include "rcwa/wavevector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rcwa {

namespace {

constexpr Real branchTolerance() {
#ifdef RCWA_SINGLE_PRECISION
    return Real{5e-6f};
#else
    return Real{5e-14};
#endif
}

} // namespace

Complex outgoingSqrt(Complex qSquared) {
    if (!isFinite(qSquared)) {
        throw std::invalid_argument(
            std::string("longitudinal wave-number square must be finite"));
    }
    Complex q = std::sqrt(qSquared);
    const Real tolerance = branchTolerance() * std::max(Real{1}, std::abs(q));
    if (std::imag(q) < -tolerance ||
        (std::abs(std::imag(q)) <= tolerance && std::real(q) < -tolerance)) {
        q = -q;
    }
    return q;
}

} // namespace rcwa
