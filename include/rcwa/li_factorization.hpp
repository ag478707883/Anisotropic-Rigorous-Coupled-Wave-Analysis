#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

#include "rcwa/fourier.hpp"

namespace rcwa::detail {

enum class LiInterfaceAxis : std::size_t {
    X = 0,
    Y = 1,
};

using LiTensorBlocks = std::array<std::array<Matrix, 3>, 3>;
using LiTensorFunction = std::function<Complex(const Tensor3&)>;
using LiConvolutionBuilder = std::function<Matrix(const LiTensorFunction&)>;

struct LiFactorization {
    LiTensorBlocks q{};
    Matrix zzInverse{};
};

[[nodiscard]] LiFactorization liFactorizeUniform(
    const Tensor3& tensor,
    std::size_t harmonicCount,
    std::string_view tensorName);

[[nodiscard]] LiFactorization liFactorizeDirect(
    const LiTensorBlocks& direct,
    std::string_view tensorName);

[[nodiscard]] LiFactorization liFactorizeScalarLamellar(
    const Matrix& direct,
    const Matrix& reciprocal,
    LiInterfaceAxis normalAxis,
    std::string_view tensorName);

[[nodiscard]] LiFactorization liFactorizeTensorLamellar(
    LiInterfaceAxis normalAxis,
    const LiConvolutionBuilder& convolution,
    std::string_view tensorName);

// RETICOLO V9 res0 default (li=1): apply the local rule along y first,
// Fourier transform/assemble along x second, then project the rectangular
// harmonic envelope onto the requested basis.
[[nodiscard]] LiFactorization liFactorizeTensor2dReticolo(
    const std::vector<Tensor3>& samples,
    int sampleCountX,
    int sampleCountY,
    const HarmonicBasis& basis,
    std::string_view tensorName);

// Exact piecewise-constant counterpart of the sampled route. Boundaries are
// normalized cell coordinates, must span [-0.5, 0.5], and describe x-fast
// samples with (xBoundaryCount-1)*(yBoundaryCount-1) entries.
[[nodiscard]] LiFactorization liFactorizeTensor2dReticoloCells(
    const std::vector<Tensor3>& cells,
    const std::vector<Real>& xBoundaries,
    const std::vector<Real>& yBoundaries,
    const HarmonicBasis& basis,
    std::string_view tensorName);

} // namespace rcwa::detail
