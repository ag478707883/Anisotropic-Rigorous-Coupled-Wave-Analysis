#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "rcwa/material.hpp"
#include "rcwa/matrix.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

struct HarmonicBasis {
    std::vector<HarmonicIndex> orders;
    int orderX{0};
    int orderY{0};
    LatticeTruncation truncation{LatticeTruncation::Circular};

    [[nodiscard]] std::size_t size() const noexcept { return orders.size(); }
    [[nodiscard]] bool is1dX() const noexcept;
    [[nodiscard]] int centerIndex() const;
};

// Optional convergence enhancements compatible with the formulations exposed
// by the original S4 API.  The default keeps the existing analytic/Li route.
enum class PolarizationBasis {
    VectorField, // S4 PolBasisVL: smooth, globally normalized tangent field
    Normal,      // S4 PolBasisNV: pointwise unit-normal/tangent field
    Jones,       // S4 PolBasisJones: complex Jones-vector change of basis
};

enum class FourierFormulation {
    Default,
    PolBasisVL,
    PolBasisNV,
    PolBasisJones,
    Kottke,
};

// Applicability of Li's Cartesian tensor factorization to discontinuous
// anisotropic material interfaces.  CoordinateAligned means every relevant
// facet is parallel to x=const or y=const; curved/oblique interfaces require a
// convergence study and are not claimed to be finite-order strict equivalents.
enum class TensorFactorizationApplicability {
    Unknown,
    NotApplicable,
    CoordinateAligned,
    CurvedOrOblique,
};

struct FourierConvergenceOptions {
    bool polarizationDecomposition{false};
    PolarizationBasis polarizationBasis{PolarizationBasis::VectorField};
    bool subpixelSmoothing{false};
    bool lanczosSmoothing{false};
    int lanczosPower{1};
    Real lanczosWidth{1.0};
    int resolution{8};
};

[[nodiscard]] FourierFormulation effectiveFourierFormulation(
    const FourierConvergenceOptions& options) noexcept;
void validateFourierConvergenceOptions(const FourierConvergenceOptions& options);
[[nodiscard]] Real lanczosSmoothingFactor(
    const HarmonicBasis& basis,
    int deltaOrderX,
    int deltaOrderY,
    Real periodXUm,
    Real periodYUm,
    const FourierConvergenceOptions& options);
void applyLanczosSmoothing(
    Matrix& convolution,
    const HarmonicBasis& basis,
    Real periodXUm,
    Real periodYUm,
    const FourierConvergenceOptions& options);

bool fourierUsesFftw();

std::vector<Complex> fourierCoeffsUniform(Complex value, int harmonicCount);

std::vector<Complex> fourierCoeffsBinaryGrating(
    Real fillFactor,
    Complex ridgeValue,
    Complex grooveValue,
    int harmonicCount,
    Real centerOffsetFraction = 0.0
);

std::vector<Complex> fourierCoeffsFromSamples(
    const std::vector<Complex>& samples,
    int harmonicCount
);

std::vector<HarmonicIndex> harmonicIndices2d(int Nx, int Ny);

// Circular selection follows the physical reciprocal-lattice metric
// |m*b1+n*b2|.  The current lattice model is orthogonal, with
// b1=(2*pi/periodX,0) and b2=(0,2*pi/periodY).
HarmonicBasis makeHarmonicBasisCount(
    int harmonicCount,
    LatticeTruncation truncation = LatticeTruncation::Circular,
    Real periodXUm = 1.0,
    Real periodYUm = 1.0
);
// For explicit circular limits, the physical reciprocal-space disk is
// inscribed in the requested |m|<=orderX, |n|<=orderY bounds.
HarmonicBasis makeHarmonicBasisOrders(
    int orderX,
    int orderY,
    LatticeTruncation truncation = LatticeTruncation::Circular,
    Real periodXUm = 1.0,
    Real periodYUm = 1.0
);

std::vector<Complex> fourierCoeffsFromSamples2d(
    const std::vector<Complex>& samples,
    int sampleCountX,
    int sampleCountY,
    int Nx,
    int Ny
);

Matrix toeplitzConvMatrix(const std::vector<Complex>& coeffs);
Matrix toeplitzConvMatrix(const std::vector<Complex>& coeffs, int retainedHarmonicCount);
Matrix convMatrix2d(const std::vector<Complex>& coeffs, int Nx, int Ny);
Matrix convMatrix2d(
    const std::vector<Complex>& coeffs,
    int coeffCountX,
    int coeffCountY,
    int retainedCountX,
    int retainedCountY
);
Matrix convMatrix2d(
    const std::vector<Complex>& coeffs,
    int coeffCountX,
    int coeffCountY,
    const HarmonicBasis& basis
);

struct TensorFourierMatrices {
    std::array<std::array<Matrix, 3>, 3> eps{};
    std::array<std::array<Matrix, 3>, 3> mu{};

    // Fourier-factorized material blocks used by the modal layer solver. The
    // interface-normal block follows Li's inverse rule and tangential blocks
    // follow the direct rule. TE and TM therefore consume different blocks of
    // this one Maxwell operator; incident polarization never selects a second
    // material factorization. Uniform media use uniform blocks, x/y-lamellar
    // gratings use the matching normal, and coordinate-aligned sampled 2D
    // media use RETICOLO li=1 y-then-x factorization on a complete rectangular
    // harmonic envelope before projection. Normal-vector and subpixel methods
    // for curved or oblique interfaces are separate convergence formulations.
    std::array<std::array<Matrix, 3>, 3> liEpsQ{};
    Matrix liEpsQZzInverse{};
    std::array<std::array<Matrix, 3>, 3> liMuQ{};
    Matrix liMuQZzInverse{};

    struct Metadata {
        bool valid{false};
        std::size_t harmonicCount{};
        bool factorizationComplete{false};
        bool scalarEpsilonBlocks{false};
        bool commonScalarMuIdentity{false};
        Complex commonScalarMu{};
        bool diagonalEpsilonBlocks{false};
        bool sequentialLiFactorization2d{false};
        TensorFactorizationApplicability tensorFactorizationApplicability{
            TensorFactorizationApplicability::Unknown};
        bool factorizationUsesSampledGeometry{false};
        bool requiresJointConvergence{false};
        FourierFormulation formulation{FourierFormulation::Default};
        bool lanczosSmoothingApplied{false};
        int formulationResolution{};
        // Exact-content fingerprint for modal cache keys. It is generated
        // when tensor metadata is annotated and excludes layer thickness.
        std::uint64_t fingerprint{};
    };
    Metadata metadata{};
};

TensorFourierMatrices tensorFourierMatricesUniform(
    const Tensor3& eps,
    const Tensor3& mu,
    int harmonicCount
);

TensorFourierMatrices tensorFourierMatricesUniform(
    const Tensor3& eps,
    const Tensor3& mu,
    const HarmonicBasis& basis
);

TensorFourierMatrices tensorFourierMatricesBinaryGrating(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    int harmonicCount,
    Real centerOffsetFraction = 0.0
);

TensorFourierMatrices tensorFourierMatricesBinaryGrating(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    const HarmonicBasis& basis,
    Real centerOffsetFraction,
    Real periodXUm,
    Real periodYUm,
    const FourierConvergenceOptions& options
);

// Analytic y-normal counterpart used when a full-width PatternedLayer2D
// rectangle is exactly a lamellar tensor grating.
TensorFourierMatrices tensorFourierMatricesBinaryGratingY(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    const HarmonicBasis& basis,
    Real centerOffsetFraction = 0.0
);

TensorFourierMatrices tensorFourierMatricesBinaryGrating2d(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    int retainedCountX,
    int retainedCountY,
    Real centerOffsetFraction = 0.0
);

TensorFourierMatrices tensorFourierMatricesBinaryGrating(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    const HarmonicBasis& basis,
    Real centerOffsetFraction = 0.0
);

TensorFourierMatrices tensorFourierMatricesFromSamples2d(
    const std::vector<Tensor3>& epsSamples,
    const std::vector<Tensor3>& muSamples,
    int sampleCountX,
    int sampleCountY,
    int Nx,
    int Ny
);

TensorFourierMatrices tensorFourierMatricesFromSamples2d(
    const std::vector<Tensor3>& epsSamples,
    const std::vector<Tensor3>& muSamples,
    int sampleCountX,
    int sampleCountY,
    const HarmonicBasis& basis
);

[[nodiscard]] std::pair<DiagonalOperator, DiagonalOperator>
makeTransverseWavevectorOperators(
    Real kx0Normalized,
    Real ky0Normalized,
    Real gxNormalized,
    Real gyNormalized,
    const HarmonicBasis& basis);

namespace detail {

inline void assignIsotropicTensorBlocks(
    std::array<std::array<Matrix, 3>, 3>& blocks,
    const Matrix& scalar) {
    const Matrix zero(scalar.rows(), scalar.cols());
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            blocks[row][column] = row == column ? scalar : zero;
        }
    }
}

inline bool matrixIsScalarIdentity(const Matrix& matrix,
                                   Real tolerance,
                                   Complex* value = nullptr) {
    if (matrix.empty() || matrix.rows() != matrix.cols()) {
        return false;
    }
    const Complex diagonal = matrix(0, 0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            const Complex expected = row == column ? diagonal : Complex{};
            if (std::abs(matrix(row, column) - expected) > tolerance) {
                return false;
            }
        }
    }
    if (value != nullptr) {
        *value = diagonal;
    }
    return true;
}

} // namespace detail

} // namespace rcwa
