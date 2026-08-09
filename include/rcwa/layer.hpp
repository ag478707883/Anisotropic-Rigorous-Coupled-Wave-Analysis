#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "rcwa/fourier.hpp"
#include "rcwa/matrix.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

enum class LayerComputationKind {
    // Every homogeneous layer is solved by the same per-harmonic 4x4
    // Berreman transfer formulation.  Scalar media use an exact analytic
    // eigenbasis for the same 4x4 state, avoiding numerical degeneracy without
    // selecting a separate 2x2 layer/interface algorithm.  The stack combines
    // the transfer blocks with a stable scattering cascade rather than
    // multiplying an unstable global transfer matrix.
    UniformTmm,
    PatternedRcwa,
};

struct UniformModeBlock {
    // Row-major local modal matrix. Rows are Ex/Ey/Hx/Hy and columns are
    // down-1/down-2/up-1/up-2 for one retained Fourier harmonic.
    std::array<Complex, 16> values{};
    std::array<Complex, 16> inverse{};
    bool inverseReliable{false};
};

// Modes for one z-stacked layer in the S4-style modal formulation.
//
// The state vector is [Ex, Ey, Hx, Hy]^T for every retained Fourier order.
// Columns 0..2N-1 are downward/forward modes; columns 2N..4N-1 are
// upward/backward modes. Propagation phases are applied by the stack
// scattering code between adjacent interfaces.
struct LayerModes {
    // Patterned layers retain the dense 4N x 4N matrix. Uniform layers leave
    // W empty and store one compact 4 x 4 modal block per harmonic below.
    Matrix W;
    std::vector<Complex> phase;  // Diagonal propagation factors, preferred storage
    std::vector<Complex> gamma;  // normalized kz/k0 for each modal column
    LayerComputationKind kind{LayerComputationKind::UniformTmm};
    // Diagnostic only: the 4x4 Berreman state used the exact scalar-material
    // eigenbasis instead of a numerically degenerate eigendecomposition.
    bool usesAnalyticScalarBasis{false};
    // True when a periodic 2N eigenproblem was rigorously separated into the
    // two invariant y-mirror parity sectors. This is diagnostic only.
    bool usedYMirrorSymmetrySplit{false};
    std::vector<UniformModeBlock> uniformBlocks;
};

struct TransverseSymmetryTransform {
    Real kxParity{};
    Real kyParity{};
    std::array<Real, 4> stateParity{};
    std::array<Real, 3> tensorParity{};
};

[[nodiscard]] bool hasCompactUniformModes(const LayerModes& modes) noexcept;
[[nodiscard]] std::size_t layerHarmonicCount(const LayerModes& modes);
[[nodiscard]] std::size_t layerStateSize(const LayerModes& modes);
[[nodiscard]] Complex modeStateValue(const LayerModes& modes,
                                      std::size_t row,
                                      std::size_t column);
[[nodiscard]] Matrix materializeModeMatrix(const LayerModes& modes);
[[nodiscard]] const std::vector<Complex>& propagationPhase(const LayerModes& modes);
[[nodiscard]] Matrix propagationMatrixBlock(const LayerModes& modes,
                                               std::size_t offset,
                                               std::size_t count);
[[nodiscard]] LayerModes layerModesWithThickness(
    const LayerModes& modalBasisSource,
    Real thicknessUm,
    Real wavelengthUm);
[[nodiscard]] Real stateFluxZ(const Matrix& state, std::size_t column);
[[nodiscard]] Real propagatingFluxWeight(const LayerModes& modes, std::size_t column);

[[nodiscard]] std::optional<std::vector<std::size_t>> transverseSymmetryHarmonicMap(
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    Real kxParity,
    Real kyParity);

[[nodiscard]] std::optional<LayerModes> tryMakeTransverseSymmetryPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    const TransverseSymmetryTransform& transform,
    const std::vector<std::size_t>* precomputedMap = nullptr);

[[nodiscard]] std::optional<LayerModes> tryMakeTransverseSymmetryUniformModes(
    const LayerModes& source,
    const Tensor3& eps,
    const Tensor3& mu,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    const TransverseSymmetryTransform& transform,
    const std::vector<std::size_t>* precomputedMap = nullptr);

[[nodiscard]] std::optional<LayerModes> tryMakeXMirroredPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy);

[[nodiscard]] std::optional<LayerModes> tryMakeInPlaneInvertedPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy);

LayerModes computeUniformModes(
    const Tensor3& eps,
    const Tensor3& mu,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm
);

// Dimensionless Berreman state matrix for one homogeneous material and one
// retained transverse harmonic.  The uniform-layer eigensolver constructs the
// canonical Delta matrix in psi=[Ex,Hy,Ey,-Hx]^T, then this public diagnostic
// returns its exactly equivalent [Ex,Ey,Hx,Hy]^T representation.  In either
// state the eigenproblem is A v=gamma v and fields are proportional to
// exp(i*k0*gamma*z).
[[nodiscard]] Matrix berremanStateMatrix(const Tensor3& eps,
                                          const Tensor3& mu,
                                          Complex kx,
                                          Complex ky);

LayerModes computePeriodicModes(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit = true
);

void annotateTensorFourierMetadata(TensorFourierMatrices& tensors,
                                      std::size_t harmonicCount);

[[nodiscard]] std::uint64_t modalFingerprintForUniformTensors(
    const Tensor3& eps,
    const Tensor3& mu) noexcept;

} // namespace rcwa
