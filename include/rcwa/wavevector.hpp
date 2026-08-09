#pragma once

#include "rcwa/types.hpp"

namespace rcwa {

// Outgoing square-root branch for exp(-i omega t): Im(q)>=0, with Re(q)>=0
// for the remaining lossless propagating case.
[[nodiscard]] Complex outgoingSqrt(Complex qSquared);

} // namespace rcwa
