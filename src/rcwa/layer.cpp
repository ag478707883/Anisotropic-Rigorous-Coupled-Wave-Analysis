#include "rcwa/layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "rcwa/geometry.hpp"
#include "rcwa/physics.hpp"

namespace rcwa {

namespace {

inline constexpr Real matrixScalarCheckTol =
#ifdef RCWA_SINGLE_PRECISION
    Real{1e-5f};
#else
    Real{1e-12};
#endif

inline constexpr Real mirrorSymmetryCheckTol =
#ifdef RCWA_SINGLE_PRECISION
    Real{3e-4f};
#else
    Real{3e-11};
#endif

std::uint64_t tensorFourierFingerprint(const TensorFourierMatrices& tensors,
                                       std::size_t harmonicCount) noexcept {
    detail::Fingerprint64 fingerprint(UINT64_C(0xcbf29ce484222325));
    fingerprint.append(static_cast<std::uint64_t>(harmonicCount));
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            fingerprint.append(tensors.eps[row][column]);
            fingerprint.append(tensors.mu[row][column]);
            fingerprint.append(tensors.liEpsQ[row][column]);
            fingerprint.append(tensors.liMuQ[row][column]);
        }
    }
    fingerprint.append(tensors.liEpsQZzInverse);
    fingerprint.append(tensors.liMuQZzInverse);
    return fingerprint.nonzeroValue();
}

bool matrixIsZero(const Matrix& m, Real tol = matrixScalarCheckTol) {
    for (const auto& v : m.data()) {
        if (std::abs(v) > tol) {
            return false;
        }
    }
    return true;
}

Complex scalarMuFromTensorBlocks(const TensorFourierMatrices& tensors,
                                     const char* scalarError,
                                     const char* commonError,
                                     const char* diagonalError) {
    Complex muX{};
    Complex muY{};
    Complex muZ{};
    if (!detail::matrixIsScalarIdentity(
            tensors.mu[0][0], matrixScalarCheckTol, &muX) ||
        !detail::matrixIsScalarIdentity(
            tensors.mu[1][1], matrixScalarCheckTol, &muY) ||
        !detail::matrixIsScalarIdentity(
            tensors.mu[2][2], matrixScalarCheckTol, &muZ)) {
        throw std::runtime_error(scalarError);
    }
    if (std::abs(muX - muY) > matrixScalarCheckTol ||
        std::abs(muX - muZ) > matrixScalarCheckTol) {
        throw std::runtime_error(commonError);
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (r != c && !matrixIsZero(tensors.mu[r][c])) {
                throw std::runtime_error(diagonalError);
            }
        }
    }
    return muX;
}

Matrix block4(const Matrix& a00, const Matrix& a01, const Matrix& a02, const Matrix& a03,
              const Matrix& a10, const Matrix& a11, const Matrix& a12, const Matrix& a13,
              const Matrix& a20, const Matrix& a21, const Matrix& a22, const Matrix& a23,
              const Matrix& a30, const Matrix& a31, const Matrix& a32, const Matrix& a33) {
    const std::size_t n = a00.rows();
    Matrix out(4 * n, 4 * n);
    out.setBlock(0, 0, a00);         out.setBlock(0, n, a01);
    out.setBlock(0, 2 * n, a02);     out.setBlock(0, 3 * n, a03);
    out.setBlock(n, 0, a10);         out.setBlock(n, n, a11);
    out.setBlock(n, 2 * n, a12);     out.setBlock(n, 3 * n, a13);
    out.setBlock(2 * n, 0, a20);     out.setBlock(2 * n, n, a21);
    out.setBlock(2 * n, 2 * n, a22); out.setBlock(2 * n, 3 * n, a23);
    out.setBlock(3 * n, 0, a30);     out.setBlock(3 * n, n, a31);
    out.setBlock(3 * n, 2 * n, a32); out.setBlock(3 * n, 3 * n, a33);
    return out;
}

Matrix block2(const Matrix& a00, const Matrix& a01, const Matrix& a10, const Matrix& a11) {
    const std::size_t n = a00.rows();
    Matrix out(2 * n, 2 * n);
    out.setBlock(0, 0, a00);
    out.setBlock(0, n, a01);
    out.setBlock(n, 0, a10);
    out.setBlock(n, n, a11);
    return out;
}

void requireDiagonalProductSize(const std::vector<Complex>& a,
                                   const std::vector<Complex>& b,
                                   const char* op) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(std::string(op) + " diagonal product sizes must match");
    }
}

void requireSquareSize(const Matrix& m, std::size_t n, const char* op) {
    if (m.rows() != n || m.cols() != n) {
        throw std::invalid_argument(std::string(op) + " matrix size must match diagonal size");
    }
}

void addDiagonalProductInPlace(Matrix& m,
                                   const std::vector<Complex>& a,
                                   const std::vector<Complex>& b,
                                   Complex scale = {1.0, 0.0}) {
    requireDiagonalProductSize(a, b, "matrix diagonal update");
    requireSquareSize(m, a.size(), "matrix diagonal update");
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < a.size(); ++i) {
        m.data()[i * n + i] += scale * a[i] * b[i];
    }
}

void subtractDiagonalProductInPlace(Matrix& m,
                                        const std::vector<Complex>& a,
                                        const std::vector<Complex>& b,
                                        Complex scale = {1.0, 0.0}) {
    addDiagonalProductInPlace(m, a, b, -scale);
}

Matrix diagonalProductMatrix(const std::vector<Complex>& a,
                               const std::vector<Complex>& b,
                               Complex scale = {1.0, 0.0}) {
    requireDiagonalProductSize(a, b, "diagonal matrix");
    const std::size_t n = a.size();
    Matrix out(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        out.data()[i * n + i] = scale * a[i] * b[i];
    }
    return out;
}

std::vector<Complex> selectValues(const std::vector<Complex>& values,
                                   const std::vector<std::size_t>& idx) {
    std::vector<Complex> out;
    out.reserve(idx.size());
    for (const auto i : idx) {
        if (i >= values.size()) {
            throw std::invalid_argument("selected value index is outside source bounds");
        }
        out.push_back(values[i]);
    }
    return out;
}

Matrix selectColumnsWithFluxNormalization(
    const Matrix& source,
    const std::vector<std::size_t>& cols) {
    if (source.rows() % 4 != 0) {
        throw std::invalid_argument("mode selection expects a 4N state matrix");
    }
    Matrix out(source.rows(), cols.size());
    const auto& src = source.data();
    auto& dst = out.data();
    const std::size_t srcCols = source.cols();
    const std::size_t dstCols = out.cols();
    const std::size_t N = source.rows() / 4;
    std::vector<Real> inverseFluxScale(cols.size(), Real{1});
    for (std::size_t c = 0; c < cols.size(); ++c) {
        const std::size_t sourceCol = cols[c];
        if (sourceCol >= srcCols) {
            throw std::invalid_argument("selected matrix column is outside source bounds");
        }
        Complex flux{};
        for (std::size_t i = 0; i < N; ++i) {
            const Complex Ex = src[i * srcCols + sourceCol];
            const Complex Ey = src[(N + i) * srcCols + sourceCol];
            const Complex Hx = src[(2 * N + i) * srcCols + sourceCol];
            const Complex Hy = src[(3 * N + i) * srcCols + sourceCol];
            flux += Ex * std::conj(Hy) - Ey * std::conj(Hx);
        }
        const Real magnitude = std::abs(std::real(flux));
        if (magnitude > Real{1e-300}) {
            inverseFluxScale[c] = Real{1} / std::sqrt(magnitude);
        }
    }
    for (std::size_t c = 0; c < cols.size(); ++c) {
        const std::size_t sourceCol = cols[c];
        const Real scale = inverseFluxScale[c];
        for (std::size_t r = 0; r < source.rows(); ++r) {
            dst[r * dstCols + c] = scale * src[r * srcCols + sourceCol];
        }
    }
    return out;
}

void normalizeColumns(Matrix& m) {
    auto& data = m.data();
    const std::size_t cols = m.cols();
    for (std::size_t c = 0; c < m.cols(); ++c) {
        Real norm = 0.0;
        for (std::size_t r = 0; r < m.rows(); ++r) {
            norm += std::norm(data[r * cols + c]);
        }
        norm = std::sqrt(norm);
        if (norm > 1e-300) {
            for (std::size_t r = 0; r < m.rows(); ++r) {
                data[r * cols + c] /= norm;
            }
        }
    }
}

Real stateFluxZUnchecked(const Matrix& W, std::size_t c) {
    const std::size_t N = W.rows() / 4;
    const std::size_t cols = W.cols();
    const auto& data = W.data();
    Complex flux{};
    for (std::size_t i = 0; i < N; ++i) {
        const Complex Ex = data[i * cols + c];
        const Complex Ey = data[(N + i) * cols + c];
        const Complex Hx = data[(2 * N + i) * cols + c];
        const Complex Hy = data[(3 * N + i) * cols + c];
        flux += Ex * std::conj(Hy) - Ey * std::conj(Hx);
    }
    return std::real(flux);
}

Real stateFluxZUnchecked(const std::array<Complex, 16>& W, std::size_t c) {
    const Complex Ex = W[c];
    const Complex Ey = W[4 + c];
    const Complex Hx = W[8 + c];
    const Complex Hy = W[12 + c];
    return std::real(Ex * std::conj(Hy) - Ey * std::conj(Hx));
}

template <typename FluxFn>
std::vector<std::size_t> fluxModeOrderImpl(
    const std::vector<Complex>& values,
    std::size_t half,
    FluxFn&& fluxFn) {
    std::vector<std::size_t> idx(values.size());
    std::vector<Real> flux(values.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
        flux[i] = fluxFn(i);
    }
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        auto score = [&](std::size_t i) {
            const Real im = std::imag(values[i]);
            if (std::abs(im) > 1e-10) {
                return im;
            }
            return flux[i];
        };
        const Real sa = score(a);
        const Real sb = score(b);
        if (std::abs(sa - sb) > 1e-12) {
            return sa > sb;
        }
        return std::real(values[a]) > std::real(values[b]);
    });
    if (idx.size() != 2 * half) {
        throw std::runtime_error("mode partition expected exactly 2*half eigenvalues");
    }
    return idx;
}

std::vector<std::size_t> fluxModeOrder(const std::vector<Complex>& values,
                                         const Matrix& W,
                                         std::size_t half) {
    if (W.rows() % 4 != 0 || W.cols() < values.size()) {
        throw std::invalid_argument("mode partition expects a 4N state matrix with enough columns");
    }
    return fluxModeOrderImpl(
        values,
        half,
        [&](std::size_t i) { return stateFluxZUnchecked(W, i); });
}

std::vector<std::size_t> fluxModeOrder(const std::vector<Complex>& values,
                                         const std::array<Complex, 16>& W,
                                         std::size_t half) {
    return fluxModeOrderImpl(
        values,
        half,
        [&](std::size_t i) { return stateFluxZUnchecked(W, i); });
}

std::array<std::size_t, 4> fluxModeOrder4(
    const std::array<Complex, 4>& values,
    const std::array<Complex, 16>& W) {
    std::array<std::size_t, 4> idx{0, 1, 2, 3};
    std::array<Real, 4> flux{};
    for (std::size_t i = 0; i < idx.size(); ++i) {
        flux[i] = stateFluxZUnchecked(W, i);
    }
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        const auto score = [&](std::size_t i) {
            const Real im = std::imag(values[i]);
            return std::abs(im) > Real{1e-10} ? im : flux[i];
        };
        const Real sa = score(a);
        const Real sb = score(b);
        if (std::abs(sa - sb) > Real{1e-12}) {
            return sa > sb;
        }
        return std::real(values[a]) > std::real(values[b]);
    });
    return idx;
}

std::vector<Complex> makePropagationPhases(const std::vector<Complex>& gamma,
                                             std::size_t forwardCount,
                                             Real thicknessUm,
                                             Real wavelengthUm) {
    if (forwardCount > gamma.size()) {
        throw std::invalid_argument("propagation phase forward block exceeds gamma storage");
    }
    const Real k0d = twoPi * thicknessUm / wavelengthUm;
    if (k0d == Real{0}) {
        return std::vector<Complex>(
            gamma.size(),
            Complex{1.0, 0.0});
    }
    std::vector<Complex> phase(gamma.size());
    for (std::size_t i = 0; i < gamma.size(); ++i) {
        const bool down = i < forwardCount;
        phase[i] = boundedExp(iu * (down ? gamma[i] : -gamma[i]) * k0d);
    }
    return phase;
}

bool matricesClose(const Matrix& a, const Matrix& b, Real tol = matrixScalarCheckTol) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    const Real scale = std::max({Real{1.0}, maxAbs(a), maxAbs(b)});
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a.data()[i] - b.data()[i]) > tol * scale) {
            return false;
        }
    }
    return true;
}

bool tensorEpsilonBlocksAreScalarMaterial(const TensorFourierMatrices& tensors) {
    if (tensors.eps[0][0].empty()) {
        return false;
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (r == c) {
                if (!matricesClose(tensors.eps[r][c], tensors.eps[0][0])) {
                    return false;
                }
            } else if (!matrixIsZero(tensors.eps[r][c])) {
                return false;
            }
        }
    }
    return true;
}

bool tensorMuBlocksAreCommonScalarIdentity(const TensorFourierMatrices& tensors,
                                                Complex* commonMu = nullptr) {
    Complex muX{};
    Complex muY{};
    Complex muZ{};
    if (!detail::matrixIsScalarIdentity(
            tensors.mu[0][0], matrixScalarCheckTol, &muX) ||
        !detail::matrixIsScalarIdentity(
            tensors.mu[1][1], matrixScalarCheckTol, &muY) ||
        !detail::matrixIsScalarIdentity(
            tensors.mu[2][2], matrixScalarCheckTol, &muZ)) {
        return false;
    }
    if (std::abs(muX - muY) > matrixScalarCheckTol ||
        std::abs(muX - muZ) > matrixScalarCheckTol) {
        return false;
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (r != c && !matrixIsZero(tensors.mu[r][c])) {
                return false;
            }
        }
    }
    if (commonMu != nullptr) {
        *commonMu = muX;
    }
    return true;
}

bool matrixHasShape(const Matrix& m, std::size_t N) {
    return m.rows() == N && m.cols() == N;
}

bool tensorQBlocksComplete(const std::array<std::array<Matrix, 3>, 3>& blocks,
                              std::size_t N) {
    for (const auto& row : blocks) {
        for (const auto& block : row) {
            if (!matrixHasShape(block, N)) {
                return false;
            }
        }
    }
    return true;
}

void validateTensorBlockSize(const TensorFourierMatrices& tensors, std::size_t N) {
    const auto validate = [&](const Matrix& m, const char* name) {
        if (m.rows() != N || m.cols() != N) {
            throw std::invalid_argument(
                std::string(name) + " tensor Fourier block size must match harmonic count");
        }
    };
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            validate(tensors.eps[r][c], "epsilon");
            validate(tensors.mu[r][c], "mu");
            validate(tensors.liEpsQ[r][c], "factorized epsilon");
            validate(tensors.liMuQ[r][c], "factorized mu");
        }
    }
    validate(tensors.liEpsQZzInverse, "factorized epsilon zz inverse");
    validate(tensors.liMuQZzInverse, "factorized mu zz inverse");
}

bool diagonalEpsilonBlocksSupportReducedModes(const TensorFourierMatrices& tensors) {
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (r != c && !matrixIsZero(tensors.liEpsQ[r][c])) {
                return false;
            }
        }
    }
    return true;
}

bool inPlaneEpsilonBlocksSupportReducedModes(const TensorFourierMatrices& tensors) {
    return matrixIsZero(tensors.liEpsQ[0][2]) &&
           matrixIsZero(tensors.liEpsQ[1][2]) &&
           matrixIsZero(tensors.liEpsQ[2][0]) &&
           matrixIsZero(tensors.liEpsQ[2][1]);
}

bool tensorMetadataValidFor(const TensorFourierMatrices& tensors, std::size_t N) {
    return tensors.metadata.valid && tensors.metadata.harmonicCount == N;
}

struct SparseSymmetryColumn {
    std::array<std::size_t, 2> rows{};
    std::array<Real, 2> coefficients{};
    std::size_t count{};
};

struct YMirrorSplitBasis {
    std::vector<std::size_t> mirror;
    std::vector<SparseSymmetryColumn> evenSector;
    std::vector<SparseSymmetryColumn> oddSector;
};

bool finiteComplex(Complex value) {
    return std::isfinite(std::real(value)) && std::isfinite(std::imag(value));
}

std::optional<std::vector<std::size_t>> yMirrorHarmonicMap(
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky) {
    if (Kx.rows() == 0 || Kx.rows() != Kx.cols() ||
        Ky.rows() != Kx.rows() || Ky.cols() != Kx.cols()) {
        return std::nullopt;
    }

    const std::size_t N = Kx.rows();
    const Real scale = std::max({Real{1}, maxAbs(Kx), maxAbs(Ky)});
    const Real tolerance = mirrorSymmetryCheckTol * scale;
    for (std::size_t row = 0; row < N; ++row) {
        if (!finiteComplex(Kx[row]) || !finiteComplex(Ky[row])) {
            return std::nullopt;
        }
    }

    std::vector<std::size_t> mirror(N, N);
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        std::size_t candidate = N;
        for (std::size_t other = 0; other < N; ++other) {
            if (std::abs(Kx[other] - Kx[harmonic]) <= tolerance &&
                std::abs(Ky[other] + Ky[harmonic]) <= tolerance) {
                if (candidate != N) {
                    // A valid Fourier basis has one unique (kx,-ky) partner.
                    return std::nullopt;
                }
                candidate = other;
            }
        }
        if (candidate == N) {
            return std::nullopt;
        }
        mirror[harmonic] = candidate;
    }
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        if (mirror[mirror[harmonic]] != harmonic) {
            return std::nullopt;
        }
    }
    return mirror;
}

bool matrixHasMirrorParity(const Matrix& matrix,
                           const std::vector<std::size_t>& mirror,
                           Real parity) {
    const std::size_t N = mirror.size();
    if (matrix.rows() != N || matrix.cols() != N) {
        return false;
    }
    const auto& values = matrix.data();
    const bool exactlyZero = std::all_of(
        values.begin(),
        values.end(),
        [](Complex value) { return value == Complex{}; });
    if (exactlyZero) {
        return true;
    }
    const Real tolerance = mirrorSymmetryCheckTol *
        std::max(Real{1}, maxAbs(matrix));
    for (std::size_t row = 0; row < N; ++row) {
        for (std::size_t column = 0; column < N; ++column) {
            const std::size_t index = row * N + column;
            const std::size_t mirroredIndex = mirror[row] * N + mirror[column];
            // The mirrored relation is an involution. Checking its canonical
            // member proves the reverse relation as well.
            if (index > mirroredIndex) {
                continue;
            }
            if (std::abs(values[index] - parity * values[mirroredIndex]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

Real tensorMaxAbs(const Tensor3& tensor) {
    Real maximum = Real{0};
    for (const Complex value : tensor.v) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

bool tensorHasSpatialSymmetry(
    const Tensor3& tensor,
    const std::array<Real, 3>& componentParity) {
    const Real tolerance = mirrorSymmetryCheckTol *
        std::max(Real{1}, tensorMaxAbs(tensor));
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            if (componentParity[row] * componentParity[column] > Real{0}) {
                continue;
            }
            if (std::abs(tensor(row, column)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool uniformTensorsHaveSpatialSymmetry(
    const Tensor3& eps,
    const Tensor3& mu,
    const std::array<Real, 3>& componentParity) {
    return tensorHasSpatialSymmetry(eps, componentParity) &&
           tensorHasSpatialSymmetry(mu, componentParity);
}

struct RealWavevectorEntry {
    Real kx{};
    Real ky{};
    std::size_t source{};
};

std::optional<std::vector<std::size_t>> realTransverseSymmetryHarmonicMap(
    const std::vector<Complex>& sourceKx,
    const std::vector<Complex>& sourceKy,
    const std::vector<Complex>& targetKx,
    const std::vector<Complex>& targetKy,
    Real kxParity,
    Real kyParity,
    Real tolerance) {
    const std::size_t N = sourceKx.size();
    std::vector<RealWavevectorEntry> entries;
    entries.reserve(N);
    for (std::size_t source = 0; source < N; ++source) {
        if (std::abs(std::imag(sourceKx[source])) > tolerance ||
            std::abs(std::imag(sourceKy[source])) > tolerance ||
            std::abs(std::imag(targetKx[source])) > tolerance ||
            std::abs(std::imag(targetKy[source])) > tolerance) {
            return std::nullopt;
        }
        entries.push_back({
            kxParity * std::real(sourceKx[source]),
            kyParity * std::real(sourceKy[source]),
            source});
    }
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        if (left.kx != right.kx) {
            return left.kx < right.kx;
        }
        if (left.ky != right.ky) {
            return left.ky < right.ky;
        }
        return left.source < right.source;
    });

    std::vector<std::size_t> targetToSource(N, N);
    std::vector<unsigned char> sourceUsed(N, 0);
    for (std::size_t target = 0; target < N; ++target) {
        const Real wantedKx = std::real(targetKx[target]);
        const Real wantedKy = std::real(targetKy[target]);
        const auto begin = std::lower_bound(
            entries.begin(),
            entries.end(),
            wantedKx - tolerance,
            [](const RealWavevectorEntry& entry, Real value) {
                return entry.kx < value;
            });
        for (auto it = begin;
             it != entries.end() && it->kx <= wantedKx + tolerance;
             ++it) {
            if (std::abs(it->ky - wantedKy) > tolerance) {
                continue;
            }
            if (targetToSource[target] != N || sourceUsed[it->source]) {
                return std::nullopt;
            }
            targetToSource[target] = it->source;
            sourceUsed[it->source] = 1;
        }
        if (targetToSource[target] == N) {
            return std::nullopt;
        }
    }
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        if (targetToSource[targetToSource[harmonic]] != harmonic) {
            return std::nullopt;
        }
    }
    return targetToSource;
}

std::optional<std::vector<std::size_t>> bruteForceTransverseSymmetryHarmonicMap(
    const std::vector<Complex>& sourceKx,
    const std::vector<Complex>& sourceKy,
    const std::vector<Complex>& targetKx,
    const std::vector<Complex>& targetKy,
    Real kxParity,
    Real kyParity,
    Real tolerance) {
    const std::size_t N = sourceKx.size();
    std::vector<std::size_t> targetToSource(N, N);
    std::vector<unsigned char> sourceUsed(N, 0);
    for (std::size_t target = 0; target < N; ++target) {
        const Complex wantedKx = targetKx[target];
        const Complex wantedKy = targetKy[target];
        for (std::size_t source = 0; source < N; ++source) {
            if (std::abs(wantedKx - kxParity * sourceKx[source]) <= tolerance &&
                std::abs(wantedKy - kyParity * sourceKy[source]) <= tolerance) {
                if (targetToSource[target] != N || sourceUsed[source]) {
                    return std::nullopt;
                }
                targetToSource[target] = source;
                sourceUsed[source] = 1;
            }
        }
        if (targetToSource[target] == N) {
            return std::nullopt;
        }
    }
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        if (targetToSource[targetToSource[harmonic]] != harmonic) {
            return std::nullopt;
        }
    }
    return targetToSource;
}

std::optional<std::vector<std::size_t>> transverseSymmetryHarmonicMapImpl(
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    Real kxParity,
    Real kyParity) {
    if (sourceKx.rows() == 0 || sourceKx.rows() != sourceKx.cols() ||
        sourceKy.rows() != sourceKx.rows() || sourceKy.cols() != sourceKx.cols() ||
        targetKx.rows() != sourceKx.rows() || targetKx.cols() != sourceKx.cols() ||
        targetKy.rows() != sourceKx.rows() || targetKy.cols() != sourceKx.cols()) {
        return std::nullopt;
    }

    const std::size_t N = sourceKx.rows();
    const auto& sourceKxValues = sourceKx.values();
    const auto& sourceKyValues = sourceKy.values();
    const auto& targetKxValues = targetKx.values();
    const auto& targetKyValues = targetKy.values();
    const Real scale = std::max({
        Real{1}, maxAbs(sourceKx), maxAbs(sourceKy),
        maxAbs(targetKx), maxAbs(targetKy)});
    const Real tolerance = mirrorSymmetryCheckTol * scale;
    for (const DiagonalOperator* matrix :
         {&sourceKx, &sourceKy, &targetKx, &targetKy}) {
        for (std::size_t row = 0; row < N; ++row) {
            if (!finiteComplex((*matrix)[row])) {
                return std::nullopt;
            }
        }
    }

    // targetToSource[t] identifies the source harmonic whose transformed
    // transverse wavevector is the target harmonic. Current physical Kx/Ky
    // operators are real-valued, so use an O(N log N) ordered lookup there
    // and retain the full complex O(N^2) matcher as a conservative fallback.
    if (auto realMap = realTransverseSymmetryHarmonicMap(
            sourceKxValues,
            sourceKyValues,
            targetKxValues,
            targetKyValues,
            kxParity,
            kyParity,
            tolerance)) {
        return realMap;
    }
    return bruteForceTransverseSymmetryHarmonicMap(
        sourceKxValues,
        sourceKyValues,
        targetKxValues,
        targetKyValues,
        kxParity,
        kyParity,
        tolerance);
}

bool tensorFourierMatricesHaveSpatialSymmetry(
    const TensorFourierMatrices& tensors,
    const std::vector<std::size_t>& mirror,
    const std::array<Real, 3>& componentParity) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const Real parity = componentParity[row] * componentParity[column];
            if (!matrixHasMirrorParity(tensors.eps[row][column], mirror, parity) ||
                !matrixHasMirrorParity(tensors.mu[row][column], mirror, parity) ||
                !matrixHasMirrorParity(tensors.liEpsQ[row][column], mirror, parity) ||
                !matrixHasMirrorParity(tensors.liMuQ[row][column], mirror, parity)) {
                return false;
            }
        }
    }
    return (tensors.liEpsQZzInverse.empty() ||
            matrixHasMirrorParity(tensors.liEpsQZzInverse, mirror, Real{1})) &&
           (tensors.liMuQZzInverse.empty() ||
            matrixHasMirrorParity(tensors.liMuQZzInverse, mirror, Real{1}));
}

std::uint64_t tensorSymmetryCacheKey(
    const TensorFourierMatrices& tensors,
    const std::vector<std::size_t>& mirror,
    const std::array<Real, 3>& componentParity) noexcept {
    if (!tensors.metadata.valid || tensors.metadata.fingerprint == 0) {
        return 0;
    }
    detail::Fingerprint64 fingerprint(UINT64_C(0x53594d4d45545259));
    fingerprint.append(static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(&tensors)));
    fingerprint.append(tensors.metadata.fingerprint);
    fingerprint.append(static_cast<std::uint64_t>(tensors.metadata.harmonicCount));
    fingerprint.append(static_cast<std::uint64_t>(mirror.size()));
    for (const std::size_t index : mirror) {
        fingerprint.append(static_cast<std::uint64_t>(index));
    }
    for (const Real parity : componentParity) {
        fingerprint.append(parity);
    }
    return fingerprint.nonzeroValue();
}

bool cachedTensorFourierMatricesHaveSpatialSymmetry(
    const TensorFourierMatrices& tensors,
    const std::vector<std::size_t>& mirror,
    const std::array<Real, 3>& componentParity) {
    const std::uint64_t key =
        tensorSymmetryCacheKey(tensors, mirror, componentParity);
    if (key == 0) {
        return tensorFourierMatricesHaveSpatialSymmetry(
            tensors, mirror, componentParity);
    }

    thread_local std::unordered_map<std::uint64_t, bool> cache;
    if (const auto cached = cache.find(key); cached != cache.end()) {
        return cached->second;
    }

    const bool result =
        tensorFourierMatricesHaveSpatialSymmetry(tensors, mirror, componentParity);
    if (cache.size() > 4096) {
        cache.clear();
    }
    cache.emplace(key, result);
    return result;
}

SparseSymmetryColumn singleSymmetryColumn(std::size_t row) {
    SparseSymmetryColumn column;
    column.rows[0] = row;
    column.coefficients[0] = Real{1};
    column.count = 1;
    return column;
}

SparseSymmetryColumn pairedSymmetryColumn(std::size_t first,
                                          std::size_t second,
                                          Real secondSign) {
    constexpr Real inverseSqrtTwo = Real{0.707106781186547524400844362104849039};
    SparseSymmetryColumn column;
    column.rows = {first, second};
    column.coefficients = {inverseSqrtTwo, secondSign * inverseSqrtTwo};
    column.count = 2;
    return column;
}

std::optional<YMirrorSplitBasis> makeYMirrorSplitBasis(
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky) {
    auto mirror = yMirrorHarmonicMap(Kx, Ky);
    if (!mirror) {
        return std::nullopt;
    }

    const std::size_t N = mirror->size();
    YMirrorSplitBasis basis;
    basis.mirror = std::move(*mirror);
    basis.evenSector.reserve(N);
    basis.oddSector.reserve(N);
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        const std::size_t partner = basis.mirror[harmonic];
        if (partner == harmonic) {
            // S = diag(P_y,-P_y): Ex is even and Ey is odd on ky=0.
            basis.evenSector.push_back(singleSymmetryColumn(harmonic));
            basis.oddSector.push_back(singleSymmetryColumn(N + harmonic));
            continue;
        }
        if (partner < harmonic) {
            continue;
        }

        // For each nonzero +/-ky pair the even sector contains Ex-even and
        // Ey-odd combinations; the odd sector contains Ex-odd and Ey-even.
        basis.evenSector.push_back(
            pairedSymmetryColumn(harmonic, partner, Real{1}));
        basis.evenSector.push_back(
            pairedSymmetryColumn(N + harmonic, N + partner, Real{-1}));
        basis.oddSector.push_back(
            pairedSymmetryColumn(harmonic, partner, Real{-1}));
        basis.oddSector.push_back(
            pairedSymmetryColumn(N + harmonic, N + partner, Real{1}));
    }
    if (basis.evenSector.size() != N || basis.oddSector.size() != N) {
        return std::nullopt;
    }
    return basis;
}

Complex projectedOperatorEntry(const Matrix& full,
                               const SparseSymmetryColumn& left,
                               const SparseSymmetryColumn& right) {
    Complex value{};
    for (std::size_t i = 0; i < left.count; ++i) {
        for (std::size_t j = 0; j < right.count; ++j) {
            value += left.coefficients[i] * full(left.rows[i], right.rows[j]) *
                right.coefficients[j];
        }
    }
    return value;
}

Matrix projectSymmetrySector(
    const Matrix& full,
    const std::vector<SparseSymmetryColumn>& sector) {
    Matrix projected(sector.size(), sector.size());
    for (std::size_t row = 0; row < sector.size(); ++row) {
        for (std::size_t column = 0; column < sector.size(); ++column) {
            projected(row, column) =
                projectedOperatorEntry(full, sector[row], sector[column]);
        }
    }
    return projected;
}

Matrix projectSymmetryBlock(
    const Matrix& full,
    const std::vector<SparseSymmetryColumn>& left,
    const std::vector<SparseSymmetryColumn>& right) {
    Matrix projected(left.size(), right.size());
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t column = 0; column < right.size(); ++column) {
            projected(row, column) =
                projectedOperatorEntry(full, left[row], right[column]);
        }
    }
    return projected;
}

Real crossSectorResidual(
    const Matrix& full,
    const std::vector<SparseSymmetryColumn>& left,
    const std::vector<SparseSymmetryColumn>& right) {
    Real residual{};
    for (const auto& leftColumn : left) {
        for (const auto& rightColumn : right) {
            residual = std::max(
                residual,
                std::abs(projectedOperatorEntry(full, leftColumn, rightColumn)));
        }
    }
    return residual;
}

bool eigensystemFinite(const Eigensystem& eigensystem, std::size_t N) {
    if (eigensystem.values.size() != N ||
        eigensystem.vectors.rows() != N || eigensystem.vectors.cols() != N) {
        return false;
    }
    for (std::size_t mode = 0; mode < N; ++mode) {
        if (!finiteComplex(eigensystem.values[mode])) {
            return false;
        }
        for (std::size_t row = 0; row < N; ++row) {
            if (!finiteComplex(eigensystem.vectors(row, mode))) {
                return false;
            }
        }
    }
    return true;
}

void expandSymmetryEigenvectors(
    Matrix& target,
    std::size_t targetColumn,
    const std::vector<SparseSymmetryColumn>& sector,
    const Matrix& sectorVectors) {
    for (std::size_t mode = 0; mode < sectorVectors.cols(); ++mode) {
        for (std::size_t basisColumn = 0; basisColumn < sector.size(); ++basisColumn) {
            const Complex amplitude = sectorVectors(basisColumn, mode);
            const auto& sparse = sector[basisColumn];
            for (std::size_t entry = 0; entry < sparse.count; ++entry) {
                target(sparse.rows[entry], targetColumn + mode) +=
                    sparse.coefficients[entry] * amplitude;
            }
        }
    }
}

struct MirrorSplitReducedModes {
    Eigensystem electric;
    Matrix magneticNumerator;
};

std::optional<MirrorSplitReducedModes> solveYMirrorSectors(
    Matrix evenOperator,
    Matrix oddOperator,
    const Matrix& magneticOperator,
    const YMirrorSplitBasis& basis) {
    const Real magneticScale = std::max(Real{1}, maxAbs(magneticOperator));
    const Real forbiddenMagneticResidual = std::max(
        crossSectorResidual(
            magneticOperator, basis.evenSector, basis.evenSector),
        crossSectorResidual(
            magneticOperator, basis.oddSector, basis.oddSector));
    if (forbiddenMagneticResidual >
        mirrorSymmetryCheckTol * magneticScale) {
        return std::nullopt;
    }

    // H is an axial vector. For a physical even mode its [Hx,Hy] coefficient
    // pattern is the numerical odd-sector basis, and conversely for an odd
    // mode. C therefore maps E-even -> H-even(odd basis) and
    // E-odd -> H-odd(even basis).
    const Matrix evenMagneticOperator = projectSymmetryBlock(
        magneticOperator,
        basis.oddSector,
        basis.evenSector);
    const Matrix oddMagneticOperator = projectSymmetryBlock(
        magneticOperator,
        basis.evenSector,
        basis.oddSector);

    Eigensystem even;
    Eigensystem odd;
    try {
        even = eig(evenOperator);
        odd = eig(oddOperator);
    } catch (const std::runtime_error&) {
        return std::nullopt;
    }
    if (!eigensystemFinite(even, evenOperator.rows()) ||
        !eigensystemFinite(odd, oddOperator.rows())) {
        return std::nullopt;
    }
    normalizeColumns(even.vectors);
    normalizeColumns(odd.vectors);

    const std::size_t N = basis.evenSector.size();
    MirrorSplitReducedModes combined;
    combined.electric.values.reserve(2 * N);
    combined.electric.values.insert(
        combined.electric.values.end(), even.values.begin(), even.values.end());
    combined.electric.values.insert(
        combined.electric.values.end(), odd.values.begin(), odd.values.end());
    combined.electric.vectors = Matrix(2 * N, 2 * N);
    expandSymmetryEigenvectors(
        combined.electric.vectors, 0, basis.evenSector, even.vectors);
    expandSymmetryEigenvectors(
        combined.electric.vectors, N, basis.oddSector, odd.vectors);

    const Matrix evenMagnetic = evenMagneticOperator * even.vectors;
    const Matrix oddMagnetic = oddMagneticOperator * odd.vectors;
    combined.magneticNumerator = Matrix(2 * N, 2 * N);
    expandSymmetryEigenvectors(
        combined.magneticNumerator, 0, basis.oddSector, evenMagnetic);
    expandSymmetryEigenvectors(
        combined.magneticNumerator, N, basis.evenSector, oddMagnetic);
    return combined;
}

std::optional<MirrorSplitReducedModes> tryYMirrorSplitEigensystem(
    const Matrix& reducedOperator,
    const Matrix& magneticOperator,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky) {
    const std::size_t N = Kx.rows();
    if (reducedOperator.rows() != 2 * N || reducedOperator.cols() != 2 * N) {
        return std::nullopt;
    }
    const auto basis = makeYMirrorSplitBasis(Kx, Ky);
    if (!basis) {
        return std::nullopt;
    }

    const Real operatorScale = std::max(Real{1}, maxAbs(reducedOperator));
    const Real crossResidual = std::max(
        crossSectorResidual(
            reducedOperator, basis->evenSector, basis->oddSector),
        crossSectorResidual(
            reducedOperator, basis->oddSector, basis->evenSector));
    if (crossResidual > mirrorSymmetryCheckTol * operatorScale) {
        return std::nullopt;
    }

    Matrix evenOperator =
        projectSymmetrySector(reducedOperator, basis->evenSector);
    Matrix oddOperator =
        projectSymmetrySector(reducedOperator, basis->oddSector);
    return solveYMirrorSectors(
        std::move(evenOperator),
        std::move(oddOperator),
        magneticOperator,
        *basis);
}

std::optional<MirrorSplitReducedModes> tryYMirrorSplitProductEigensystem(
    const Matrix& electricFromMagnetic,
    const Matrix& magneticFromElectric,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky) {
    const std::size_t N = Kx.rows();
    if (electricFromMagnetic.rows() != 2 * N ||
        electricFromMagnetic.cols() != 2 * N ||
        magneticFromElectric.rows() != 2 * N ||
        magneticFromElectric.cols() != 2 * N) {
        return std::nullopt;
    }
    const auto basis = makeYMirrorSplitBasis(Kx, Ky);
    if (!basis) {
        return std::nullopt;
    }

    // Q maps axial H coefficients to polar E coefficients. Same physical
    // parity therefore appears in opposite numerical sector bases.
    const Real electricScale =
        std::max(Real{1}, maxAbs(electricFromMagnetic));
    const Real forbiddenElectricResidual = std::max(
        crossSectorResidual(
            electricFromMagnetic, basis->evenSector, basis->evenSector),
        crossSectorResidual(
            electricFromMagnetic, basis->oddSector, basis->oddSector));
    if (forbiddenElectricResidual >
        mirrorSymmetryCheckTol * electricScale) {
        return std::nullopt;
    }

    const Matrix evenElectricOperator = projectSymmetryBlock(
        electricFromMagnetic,
        basis->evenSector,
        basis->oddSector);
    const Matrix oddElectricOperator = projectSymmetryBlock(
        electricFromMagnetic,
        basis->oddSector,
        basis->evenSector);
    const Matrix evenMagneticOperator = projectSymmetryBlock(
        magneticFromElectric,
        basis->oddSector,
        basis->evenSector);
    const Matrix oddMagneticOperator = projectSymmetryBlock(
        magneticFromElectric,
        basis->evenSector,
        basis->oddSector);
    Matrix evenOperator = evenElectricOperator * evenMagneticOperator;
    Matrix oddOperator = oddElectricOperator * oddMagneticOperator;
    return solveYMirrorSectors(
        std::move(evenOperator),
        std::move(oddOperator),
        magneticFromElectric,
        *basis);
}

Complex periodicScalarMuFromTensorBlocks(const TensorFourierMatrices& tensors,
                                              std::size_t harmonicCount) {
    if (tensorMetadataValidFor(tensors, harmonicCount) &&
        tensors.metadata.factorizationComplete &&
        tensors.metadata.commonScalarMuIdentity) {
        return tensors.metadata.commonScalarMu;
    }
    return scalarMuFromTensorBlocks(
        tensors,
        "reduced periodic RCWA route requires scalar permeability",
        "reduced periodic RCWA route requires one scalar permeability value",
        "reduced periodic RCWA route requires diagonal scalar permeability");
}

Matrix makeInPlaneElectricFromMagnetic(
    const Matrix& inverseEzz,
    const std::vector<Complex>& kx,
    const std::vector<Complex>& ky,
    Complex muR) {
    const Matrix kxInvEzz = leftScaleRows(kx, inverseEzz);
    const Matrix kyInvEzz = leftScaleRows(ky, inverseEzz);
    const Matrix muIdentity = muR * Matrix::identity(kx.size());
    return block2(
        rightScaleColumns(kxInvEzz, ky),
        muIdentity - rightScaleColumns(kxInvEzz, kx),
        rightScaleColumns(kyInvEzz, ky) - muIdentity,
        -rightScaleColumns(kyInvEzz, kx));
}

Matrix makeInPlaneMagneticFromElectric(
    const Matrix& E11,
    const Matrix& E12,
    const Matrix& E21,
    const Matrix& E22,
    const std::vector<Complex>& kx,
    const std::vector<Complex>& ky,
    Complex invMu) {
    return block2(
        diagonalProductMatrix(kx, ky, -invMu) - E21,
        diagonalProductMatrix(kx, kx, invMu) - E22,
        E11 - diagonalProductMatrix(ky, ky, invMu),
        diagonalProductMatrix(ky, kx, invMu) + E12);
}

LayerModes makeModesFromReducedEigensystem(Matrix electricModes,
                                                Matrix magneticNumerator,
                                                std::vector<Complex> gammaSq,
                                               Real thicknessUm,
                                               Real wavelengthUm,
                                               bool usedYMirrorSymmetrySplit) {
    const std::size_t modeCount = electricModes.rows();
    if (modeCount == 0 || modeCount % 2 != 0 ||
        electricModes.cols() != modeCount ||
        magneticNumerator.rows() != modeCount ||
        magneticNumerator.cols() != modeCount ||
        gammaSq.size() != modeCount) {
        throw std::invalid_argument("reduced periodic eigensystem has inconsistent dimensions");
    }
    const std::size_t N = modeCount / 2;
    const std::size_t stateSize = 2 * modeCount;
    Matrix W(stateSize, stateSize);
    std::vector<Complex> gamma(stateSize);
    std::vector<Complex> inverseGamma(modeCount);

    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        const Complex g = outgoingSqrt(gammaSq[mode]);
        if (std::abs(g) < Real{1e-12} ||
            !std::isfinite(std::real(g)) ||
            !std::isfinite(std::imag(g))) {
            throw std::runtime_error("reduced periodic mode solve encountered a singular mode");
        }
        const std::size_t upCol = modeCount + mode;
        gamma[mode] = g;
        gamma[upCol] = -g;
        inverseGamma[mode] = Complex{1.0, 0.0} / g;
    }

    const auto& electric = electricModes.data();
    const auto& magnetic = magneticNumerator.data();
    auto& states = W.data();
    std::vector<Real> inverseFluxScale(modeCount, Real{1});
    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        Complex flux{};
        const Complex invG = inverseGamma[mode];
        for (std::size_t h = 0; h < N; ++h) {
            const Complex ex = electric[h * modeCount + mode];
            const Complex ey = electric[(N + h) * modeCount + mode];
            const Complex hx =
                magnetic[h * modeCount + mode] * invG;
            const Complex hy =
                magnetic[(N + h) * modeCount + mode] * invG;
            flux += ex * std::conj(hy) - ey * std::conj(hx);
        }
        const Real magnitude = std::abs(std::real(flux));
        if (magnitude > Real{1e-300}) {
            inverseFluxScale[mode] = Real{1} / std::sqrt(magnitude);
        }
    }

    for (std::size_t h = 0; h < N; ++h) {
        const Complex* ex = electric.data() + h * modeCount;
        const Complex* ey = electric.data() + (N + h) * modeCount;
        const Complex* hxNumerator = magnetic.data() + h * modeCount;
        const Complex* hyNumerator = magnetic.data() + (N + h) * modeCount;
        Complex* exState = states.data() + h * stateSize;
        Complex* eyState = states.data() + (N + h) * stateSize;
        Complex* hxState = states.data() + (2 * N + h) * stateSize;
        Complex* hyState = states.data() + (3 * N + h) * stateSize;
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            const std::size_t upCol = modeCount + mode;
            const Complex scale{inverseFluxScale[mode], Real{0}};
            const Complex scaledEx = scale * ex[mode];
            const Complex scaledEy = scale * ey[mode];
            const Complex hx = scale * hxNumerator[mode] * inverseGamma[mode];
            const Complex hy = scale * hyNumerator[mode] * inverseGamma[mode];
            exState[mode] = scaledEx;
            eyState[mode] = scaledEy;
            hxState[mode] = hx;
            hyState[mode] = hy;
            exState[upCol] = scaledEx;
            eyState[upCol] = scaledEy;
            hxState[upCol] = -hx;
            hyState[upCol] = -hy;
        }
    }

    std::vector<Complex> phase =
        makePropagationPhases(gamma, modeCount, thicknessUm, wavelengthUm);

    return {
        std::move(W),
        std::move(phase),
        std::move(gamma),
        LayerComputationKind::PatternedRcwa,
        false,
        usedYMirrorSymmetrySplit,
        {},
    };
}

std::optional<LayerModes> computePeriodicModesReduced(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Complex muR,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit) {
    if (!diagonalEpsilonBlocksSupportReducedModes(tensors)) {
        return std::nullopt;
    }

    const std::size_t N = Kx.rows();
    const Complex invMu = Complex{1.0, 0.0} / muR;
    const auto& kx = Kx.values();
    const auto& ky = Ky.values();

    const Matrix& E11 = tensors.liEpsQ[0][0];
    const Matrix& E12 = tensors.liEpsQ[0][1];
    const Matrix& E21 = tensors.liEpsQ[1][0];
    const Matrix& E22 = tensors.liEpsQ[1][1];
    const Matrix& E33 = tensors.liEpsQ[2][2];

    Matrix computedInvE33;
    const Matrix* invE33 = &tensors.liEpsQZzInverse;
    if (invE33->empty()) {
        computedInvE33 = inverse(E33);
        invE33 = &computedInvE33;
    }
    const Matrix kxInvE33 = leftScaleRows(kx, *invE33);
    const Matrix kyInvE33 = leftScaleRows(ky, *invE33);

    const Matrix C = makeInPlaneMagneticFromElectric(
        E11, E12, E21, E22, kx, ky, invMu);

    if (allowYMirrorSymmetrySplit) {
        const Matrix Q = makeInPlaneElectricFromMagnetic(
            *invE33, kx, ky, muR);
        if (auto split = tryYMirrorSplitProductEigensystem(Q, C, Kx, Ky)) {
            return makeModesFromReducedEigensystem(
                std::move(split->electric.vectors),
                std::move(split->magneticNumerator),
                std::move(split->electric.values),
                thicknessUm,
                wavelengthUm,
                true);
        }
    }

    const Matrix kxE11 = leftScaleRows(kx, E11);
    const Matrix kyE22 = leftScaleRows(ky, E22);
    Matrix R00 = muR * E11;
    multiplyAccumulate(R00, kxInvE33, kxE11, {-1.0, 0.0});
    subtractDiagonalProductInPlace(R00, ky, ky);
    Matrix R01(N, N);
    multiplyAccumulate(R01, kxInvE33, kyE22, {-1.0, 0.0});
    addDiagonalProductInPlace(R01, kx, ky);
    Matrix R10(N, N);
    multiplyAccumulate(R10, kyInvE33, kxE11, {-1.0, 0.0});
    addDiagonalProductInPlace(R10, kx, ky);
    Matrix R11 = muR * E22;
    multiplyAccumulate(R11, kyInvE33, kyE22, {-1.0, 0.0});
    subtractDiagonalProductInPlace(R11, kx, kx);
    const Matrix reducedOperator = block2(R00, R01, R10, R11);

    std::optional<MirrorSplitReducedModes> split;
    if (allowYMirrorSymmetrySplit) {
        split = tryYMirrorSplitEigensystem(
            reducedOperator,
            C,
            Kx,
            Ky);
    }
    const bool usedYMirrorSymmetrySplit = split.has_value();
    Eigensystem er;
    Matrix magneticNumerator;
    if (split) {
        er = std::move(split->electric);
        magneticNumerator = std::move(split->magneticNumerator);
    } else {
        er = eig(reducedOperator);
        normalizeColumns(er.vectors);
        magneticNumerator = C * er.vectors;
    }
    return makeModesFromReducedEigensystem(
        std::move(er.vectors),
        std::move(magneticNumerator),
        std::move(er.values),
        thicknessUm,
        wavelengthUm,
        usedYMirrorSymmetrySplit);
}

std::optional<LayerModes> computePeriodicModesInPlaneReduced(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Complex muR,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit) {
    if (!inPlaneEpsilonBlocksSupportReducedModes(tensors)) {
        return std::nullopt;
    }

    const std::size_t N = Kx.rows();
    const Complex invMu = Complex{1.0, 0.0} / muR;
    const auto& kx = Kx.values();
    const auto& ky = Ky.values();

    const Matrix& E11 = tensors.liEpsQ[0][0];
    const Matrix& E12 = tensors.liEpsQ[0][1];
    const Matrix& E21 = tensors.liEpsQ[1][0];
    const Matrix& E22 = tensors.liEpsQ[1][1];
    const Matrix& E33 = tensors.liEpsQ[2][2];

    Matrix computedInvE33;
    const Matrix* invE33 = &tensors.liEpsQZzInverse;
    if (invE33->empty()) {
        computedInvE33 = inverse(E33);
        invE33 = &computedInvE33;
    }

    // In-plane tensor epsilon with scalar mu has the block form
    // gamma * E_t = Q * H_t, gamma * H_t = C * E_t.  Solving
    // (Q*C)E_t = gamma^2 E_t avoids the full 4N first-order eigenproblem.
    const Matrix Q = makeInPlaneElectricFromMagnetic(
        *invE33, kx, ky, muR);
    const Matrix C = makeInPlaneMagneticFromElectric(
        E11, E12, E21, E22, kx, ky, invMu);

    std::optional<MirrorSplitReducedModes> split;
    if (allowYMirrorSymmetrySplit) {
        split = tryYMirrorSplitProductEigensystem(Q, C, Kx, Ky);
    }
    const bool usedYMirrorSymmetrySplit = split.has_value();
    Eigensystem er;
    Matrix magneticNumerator;
    if (split) {
        er = std::move(split->electric);
        magneticNumerator = std::move(split->magneticNumerator);
    } else {
        const Matrix reducedOperator = Q * C;
        er = eig(reducedOperator);
        normalizeColumns(er.vectors);
        magneticNumerator = C * er.vectors;
    }
    return makeModesFromReducedEigensystem(
        std::move(er.vectors),
        std::move(magneticNumerator),
        std::move(er.values),
        thicknessUm,
        wavelengthUm,
        usedYMirrorSymmetrySplit);
}

LayerModes modesFromStateMatrix(const Matrix& A,
                                   Real thicknessUm,
                                   Real wavelengthUm) {
    const std::size_t total = A.rows();
    if (total % 4 != 0 || A.cols() != total) {
        throw std::invalid_argument("layer matrix must be 4N x 4N");
    }
    const std::size_t N = total / 4;
    const std::size_t modeCount = 2 * N;

    Eigensystem er = eig(A);
    normalizeColumns(er.vectors);
    auto order = fluxModeOrder(er.values, er.vectors, modeCount);
    Matrix W = selectColumnsWithFluxNormalization(er.vectors, order);
    std::vector<Complex> gamma = selectValues(er.values, order);

    std::vector<Complex> phase =
        makePropagationPhases(gamma, modeCount, thicknessUm, wavelengthUm);

    return {
        std::move(W),
        std::move(phase),
        std::move(gamma),
        LayerComputationKind::PatternedRcwa,
        false,
        false,
        {},
    };
}

Matrix berremanDeltaMatrixImpl(const Tensor3& eps,
                               const Tensor3& mu,
                               Complex kx,
                               Complex ky) {
    if (std::abs(eps(2, 2)) < 1e-30 || std::abs(mu(2, 2)) < 1e-30) {
        throw std::runtime_error(
            "uniform anisotropic layer has singular epsilon_zz or mu_zz");
    }

    // Canonical Berreman/Passler state psi=[Ex, Hy, Ey, -Hx]^T. For the
    // project-wide exp(-i*omega*t+i*k0*k.r) convention, eliminate Ez and Hz
    // from k x E=mu H and k x H=-eps E, then form Delta psi=gamma psi.
    // See Passler and Paarmann, JOSA B 34 (2017) 2128, Eq. (5); the expression
    // below is derived directly from Maxwell's equations.
    const Complex invE33 = Complex{1.0, 0.0} / eps(2, 2);
    const Complex invM33 = Complex{1.0, 0.0} / mu(2, 2);

    const std::array<Complex, 4> ez{
        -eps(2, 0) * invE33,
        -kx * invE33,
        -eps(2, 1) * invE33,
        -ky * invE33,
    };
    const std::array<Complex, 4> hz{
        -ky * invM33,
        -mu(2, 1) * invM33,
        kx * invM33,
        mu(2, 0) * invM33,
    };

    Matrix delta(4, 4);
    for (std::size_t column = 0; column < 4; ++column) {
        // gamma Ex = kx Ez + mu_yy Hy - mu_yx(-Hx) + mu_yz Hz
        delta(0, column) = kx * ez[column] + mu(1, 2) * hz[column];
        if (column == 1) {
            delta(0, column) += mu(1, 1);
        } else if (column == 3) {
            delta(0, column) -= mu(1, 0);
        }

        // gamma Hy = ky Hz + eps_xx Ex + eps_xy Ey + eps_xz Ez
        delta(1, column) = ky * hz[column] + eps(0, 2) * ez[column];
        if (column == 0) {
            delta(1, column) += eps(0, 0);
        } else if (column == 2) {
            delta(1, column) += eps(0, 1);
        }

        // gamma Ey = ky Ez + mu_xx(-Hx) - mu_xy Hy - mu_xz Hz
        delta(2, column) = ky * ez[column] - mu(0, 2) * hz[column];
        if (column == 1) {
            delta(2, column) -= mu(0, 1);
        } else if (column == 3) {
            delta(2, column) += mu(0, 0);
        }

        // gamma(-Hx) = eps_yx Ex + eps_yy Ey + eps_yz Ez - kx Hz
        delta(3, column) = eps(1, 2) * ez[column] - kx * hz[column];
        if (column == 0) {
            delta(3, column) += eps(1, 0);
        } else if (column == 2) {
            delta(3, column) += eps(1, 1);
        }
    }
    return delta;
}

Matrix electricMagneticStateMatrix(const Matrix& delta) {
    if (delta.rows() != 4 || delta.cols() != 4) {
        throw std::invalid_argument("Berreman delta matrix must be 4x4");
    }
    // [Ex,Ey,Hx,Hy]^T = [psi0,psi2,-psi3,psi1]^T.
    constexpr std::array<std::size_t, 4> component{0, 2, 3, 1};
    constexpr std::array<Real, 4> sign{1.0, 1.0, -1.0, 1.0};
    Matrix state(4, 4);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            state(row, column) = sign[row] *
                delta(component[row], component[column]) * sign[column];
        }
    }
    return state;
}

Real uniformBlockFlux(const UniformModeBlock& block, std::size_t column) {
    const Complex Ex = block.values[column];
    const Complex Ey = block.values[4 + column];
    const Complex Hx = block.values[8 + column];
    const Complex Hy = block.values[12 + column];
    return std::real(Ex * std::conj(Hy) - Ey * std::conj(Hx));
}

void prepareUniformBlockInverse(UniformModeBlock& block) {
    std::array<Complex, 32> augmented{};
    Real matrixNorm = 0.0;
    for (std::size_t row = 0; row < 4; ++row) {
        Real rowNorm = 0.0;
        for (std::size_t column = 0; column < 4; ++column) {
            const Complex value = block.values[row * 4 + column];
            augmented[row * 8 + column] = value;
            rowNorm += std::abs(value);
        }
        augmented[row * 8 + 4 + row] = Complex{1.0, 0.0};
        matrixNorm = std::max(matrixNorm, rowNorm);
    }

    const Real pivotFloor = std::numeric_limits<Real>::epsilon() *
        std::max(matrixNorm, Real{1}) * Real{64};
    for (std::size_t column = 0; column < 4; ++column) {
        std::size_t pivot = column;
        Real pivotMagnitude = std::abs(augmented[column * 8 + column]);
        for (std::size_t row = column + 1; row < 4; ++row) {
            const Real candidate = std::abs(augmented[row * 8 + column]);
            if (candidate > pivotMagnitude) {
                pivot = row;
                pivotMagnitude = candidate;
            }
        }
        if (!(pivotMagnitude > pivotFloor) || !std::isfinite(pivotMagnitude)) {
            block.inverseReliable = false;
            return;
        }
        if (pivot != column) {
            for (std::size_t entry = 0; entry < 8; ++entry) {
                std::swap(
                    augmented[column * 8 + entry],
                    augmented[pivot * 8 + entry]);
            }
        }

        const Complex diagonal = augmented[column * 8 + column];
        for (std::size_t entry = 0; entry < 8; ++entry) {
            augmented[column * 8 + entry] /= diagonal;
        }
        for (std::size_t row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const Complex factor = augmented[row * 8 + column];
            for (std::size_t entry = 0; entry < 8; ++entry) {
                augmented[row * 8 + entry] -=
                    factor * augmented[column * 8 + entry];
            }
        }
    }

    Real inverseNorm = 0.0;
    for (std::size_t row = 0; row < 4; ++row) {
        Real rowNorm = 0.0;
        for (std::size_t column = 0; column < 4; ++column) {
            const Complex value = augmented[row * 8 + 4 + column];
            block.inverse[row * 4 + column] = value;
            rowNorm += std::abs(value);
        }
        inverseNorm = std::max(inverseNorm, rowNorm);
    }

    Real residualNorm = 0.0;
    for (std::size_t row = 0; row < 4; ++row) {
        Real rowResidual = 0.0;
        for (std::size_t column = 0; column < 4; ++column) {
            Complex value{};
            for (std::size_t inner = 0; inner < 4; ++inner) {
                value += block.values[row * 4 + inner] *
                    block.inverse[inner * 4 + column];
            }
            if (row == column) {
                value -= Complex{1.0, 0.0};
            }
            rowResidual += std::abs(value);
        }
        residualNorm = std::max(residualNorm, rowResidual);
    }

    const Real conditionEstimate = matrixNorm * inverseNorm;
    const Real epsilon = std::numeric_limits<Real>::epsilon();
    const Real maximumCondition = Real{1} / std::sqrt(epsilon);
    const Real residualLimit = Real{512} * epsilon *
        std::max(conditionEstimate, Real{1});
    block.inverseReliable = std::isfinite(conditionEstimate) &&
        conditionEstimate <= maximumCondition &&
        std::isfinite(residualNorm) && residualNorm <= residualLimit;
}

void prepareUniformBlockInverses(std::vector<UniformModeBlock>& blocks) {
    for (auto& block : blocks) {
        prepareUniformBlockInverse(block);
    }
}

void setOrderMode(std::vector<UniformModeBlock>& blocks,
                    std::vector<Complex>& gamma,
                    std::size_t harmonic,
                    std::size_t harmonicCount,
                    std::size_t localSlot,
                    const std::array<Complex, 16>& localVectors,
                    std::size_t localCol,
                    Complex localGamma) {
    const Real f = std::abs(stateFluxZUnchecked(localVectors, localCol));
    const Real scale = f > 1e-300 ? std::sqrt(f) : 1.0;
    UniformModeBlock& block = blocks[harmonic];
    for (std::size_t component = 0; component < 4; ++component) {
        block.values[component * 4 + localSlot] =
            localVectors[component * 4 + localCol] / scale;
    }
    const std::size_t globalCol =
        localSlot * harmonicCount + harmonic;
    gamma[globalCol] = localGamma;
}

bool tensorHasExactZDecoupling(const Tensor3& tensor) noexcept {
    for (std::size_t transverse = 0; transverse < 2; ++transverse) {
        if (tensor(transverse, 2) != Complex{} ||
            tensor(2, transverse) != Complex{}) {
            return false;
        }
    }
    return true;
}

std::array<Complex, 4> multiply2x2(
    const std::array<Complex, 4>& left,
    const std::array<Complex, 4>& right) {
    return {
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    };
}

bool finiteArray(const std::array<Complex, 4>& values) noexcept {
    return std::all_of(values.begin(), values.end(), finiteComplex);
}

bool reduced2x2Eigenvector(const std::array<Complex, 4>& matrix,
                           Complex eigenvalue,
                           std::size_t repeatedSlot,
                           std::array<Complex, 2>& vector) {
    const Real matrixScale = std::max({
        Real{1}, std::abs(matrix[0]), std::abs(matrix[1]),
        std::abs(matrix[2]), std::abs(matrix[3]), std::abs(eigenvalue)});
    const Real roundoff = Real{64} * std::numeric_limits<Real>::epsilon() *
        matrixScale;
    const bool numericallyScalar =
        std::abs(matrix[1]) <= roundoff &&
        std::abs(matrix[2]) <= roundoff &&
        std::abs(matrix[0] - matrix[3]) <= roundoff &&
        std::abs(eigenvalue - (matrix[0] + matrix[3]) * Real{0.5}) <= roundoff;
    if (numericallyScalar) {
        vector = repeatedSlot == 0
            ? std::array<Complex, 2>{Complex{1}, Complex{}}
            : std::array<Complex, 2>{Complex{}, Complex{1}};
        return true;
    }

    const std::array<Complex, 2> first{
        matrix[1], eigenvalue - matrix[0]};
    const std::array<Complex, 2> second{
        eigenvalue - matrix[3], matrix[2]};
    const Real firstNormSquared = std::norm(first[0]) + std::norm(first[1]);
    const Real secondNormSquared = std::norm(second[0]) + std::norm(second[1]);
    const auto& candidate = firstNormSquared >= secondNormSquared ? first : second;
    const Real norm = std::sqrt(std::max(firstNormSquared, secondNormSquared));
    if (!(norm > roundoff) || !std::isfinite(norm)) {
        return false;
    }
    vector = {candidate[0] / norm, candidate[1] / norm};

    const Complex residual0 =
        matrix[0] * vector[0] + matrix[1] * vector[1] - eigenvalue * vector[0];
    const Complex residual1 =
        matrix[2] * vector[0] + matrix[3] * vector[1] - eigenvalue * vector[1];
    return std::max(std::abs(residual0), std::abs(residual1)) <=
        Real{256} * std::numeric_limits<Real>::epsilon() * matrixScale;
}

bool tryReducedBerremanEigensystem(const Tensor3& eps,
                                   const Tensor3& mu,
                                   Complex kx,
                                   Complex ky,
                                   std::array<Complex, 4>& eigenvalues,
                                   std::array<Complex, 16>& stateVectors) {
    if (!tensorHasExactZDecoupling(eps) ||
        !tensorHasExactZDecoupling(mu) ||
        std::abs(eps(2, 2)) < Real{1e-30} ||
        std::abs(mu(2, 2)) < Real{1e-30}) {
        return false;
    }

    // With no z/transverse constitutive coupling, the canonical Berreman
    // state separates as gamma*E=A*M and gamma*M=C*E, where
    // E=[Ex,Ey] and M=[Hy,-Hx].  Solving the 2x2 AC problem gives the exact
    // four modes without invoking a general 4x4 eigensolver.
    const Complex invEzz = Complex{1} / eps(2, 2);
    const Complex invMzz = Complex{1} / mu(2, 2);
    const Complex kxKy = kx * ky;
    const std::array<Complex, 4> electricFromMagnetic{
        mu(1, 1) - kx * kx * invEzz,
        -mu(1, 0) - kxKy * invEzz,
        -mu(0, 1) - kxKy * invEzz,
        mu(0, 0) - ky * ky * invEzz,
    };
    const std::array<Complex, 4> magneticFromElectric{
        eps(0, 0) - ky * ky * invMzz,
        eps(0, 1) + kxKy * invMzz,
        eps(1, 0) + kxKy * invMzz,
        eps(1, 1) - kx * kx * invMzz,
    };
    const std::array<Complex, 4> squaredOperator =
        multiply2x2(electricFromMagnetic, magneticFromElectric);
    if (!finiteArray(electricFromMagnetic) ||
        !finiteArray(magneticFromElectric) ||
        !finiteArray(squaredOperator)) {
        return false;
    }

    const Complex trace = squaredOperator[0] + squaredOperator[3];
    const Complex determinant =
        squaredOperator[0] * squaredOperator[3] -
        squaredOperator[1] * squaredOperator[2];
    Complex root = std::sqrt(
        (squaredOperator[0] - squaredOperator[3]) *
            (squaredOperator[0] - squaredOperator[3]) +
        Complex{4} * squaredOperator[1] * squaredOperator[2]);
    if (std::abs(trace - root) > std::abs(trace + root)) {
        root = -root;
    }
    std::array<Complex, 2> squaredEigenvalues{
        (trace + root) * Real{0.5},
        (trace - root) * Real{0.5},
    };
    const Real operatorScale = std::max({
        Real{1}, std::abs(squaredOperator[0]), std::abs(squaredOperator[1]),
        std::abs(squaredOperator[2]), std::abs(squaredOperator[3])});
    if (std::abs(squaredEigenvalues[0]) >
        Real{64} * std::numeric_limits<Real>::epsilon() * operatorScale) {
        squaredEigenvalues[1] = determinant / squaredEigenvalues[0];
    }

    const Real coefficientScale = std::max({
        Real{1},
        std::abs(electricFromMagnetic[0]),
        std::abs(electricFromMagnetic[1]),
        std::abs(electricFromMagnetic[2]),
        std::abs(electricFromMagnetic[3]),
        std::abs(magneticFromElectric[0]),
        std::abs(magneticFromElectric[1]),
        std::abs(magneticFromElectric[2]),
        std::abs(magneticFromElectric[3])});
    for (std::size_t mode = 0; mode < 2; ++mode) {
        std::array<Complex, 2> electric{};
        if (!finiteComplex(squaredEigenvalues[mode]) ||
            !reduced2x2Eigenvector(
                squaredOperator, squaredEigenvalues[mode], mode, electric)) {
            return false;
        }
        const Complex principalGamma = std::sqrt(squaredEigenvalues[mode]);
        const Real gammaFloor = Real{256} *
            std::numeric_limits<Real>::epsilon() *
            std::max(Real{1}, std::sqrt(operatorScale));
        if (!finiteComplex(principalGamma) ||
            std::abs(principalGamma) <= gammaFloor) {
            return false;
        }

        for (std::size_t direction = 0; direction < 2; ++direction) {
            const Complex gamma = direction == 0 ? principalGamma : -principalGamma;
            const std::array<Complex, 2> magnetic{
                (magneticFromElectric[0] * electric[0] +
                 magneticFromElectric[1] * electric[1]) / gamma,
                (magneticFromElectric[2] * electric[0] +
                 magneticFromElectric[3] * electric[1]) / gamma,
            };
            const std::size_t column = mode + 2 * direction;
            eigenvalues[column] = gamma;
            stateVectors[column] = electric[0];
            stateVectors[4 + column] = electric[1];
            stateVectors[8 + column] = -magnetic[1];
            stateVectors[12 + column] = magnetic[0];

            Real normSquared{};
            for (std::size_t component = 0; component < 4; ++component) {
                normSquared += std::norm(stateVectors[component * 4 + column]);
            }
            const Real norm = std::sqrt(normSquared);
            if (!(norm > Real{0}) || !std::isfinite(norm)) {
                return false;
            }
            for (std::size_t component = 0; component < 4; ++component) {
                stateVectors[component * 4 + column] /= norm;
            }

            const Complex Ex = stateVectors[column];
            const Complex Ey = stateVectors[4 + column];
            const Complex Hy = stateVectors[12 + column];
            const Complex minusHx = -stateVectors[8 + column];
            const Real residual = std::max({
                std::abs(electricFromMagnetic[0] * Hy +
                         electricFromMagnetic[1] * minusHx - gamma * Ex),
                std::abs(electricFromMagnetic[2] * Hy +
                         electricFromMagnetic[3] * minusHx - gamma * Ey),
                std::abs(magneticFromElectric[0] * Ex +
                         magneticFromElectric[1] * Ey - gamma * Hy),
                std::abs(magneticFromElectric[2] * Ex +
                         magneticFromElectric[3] * Ey - gamma * minusHx),
            });
            const Real residualLimit =
#ifdef RCWA_SINGLE_PRECISION
                Real{2e-4f} * coefficientScale;
#else
                Real{5e-11} * coefficientScale;
#endif
            if (!std::isfinite(residual) || residual > residualLimit) {
                return false;
            }
        }
    }
    return true;
}

bool exactYGyrotropicEpsilon(const Tensor3& eps,
                             Complex& diagonal,
                             Complex& gyration) noexcept {
    diagonal = eps(0, 0);
    gyration = eps(0, 2);
    return gyration != Complex{} &&
        eps(1, 1) == diagonal && eps(2, 2) == diagonal &&
        eps(2, 0) == -gyration &&
        eps(0, 1) == Complex{} && eps(1, 0) == Complex{} &&
        eps(1, 2) == Complex{} && eps(2, 1) == Complex{};
}

std::array<Complex, 3> cross3(const std::array<Complex, 3>& left,
                              const std::array<Complex, 3>& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

Real normSquared3(const std::array<Complex, 3>& vector) noexcept {
    return std::norm(vector[0]) + std::norm(vector[1]) + std::norm(vector[2]);
}

bool tryYGyrotropicBerremanEigensystem(
    const Tensor3& eps,
    const Tensor3& mu,
    Complex kx,
    Complex ky,
    std::array<Complex, 4>& eigenvalues,
    std::array<Complex, 16>& stateVectors) {
    Complex epsilon{};
    Complex gyration{};
    Complex permeability{};
    if (!exactYGyrotropicEpsilon(eps, epsilon, gyration) ||
        !tensorIsScalar(mu, &permeability, Real{0}) ||
        std::abs(epsilon) < Real{1e-30} ||
        std::abs(permeability) < Real{1e-30}) {
        return false;
    }

    // A y-axis gyrotropic tensor is rotationally invariant in the x-z plane.
    // For q^2=kx^2+gamma^2 and u=mu*epsilon-q^2, Maxwell's determinant is
    // epsilon*(u-ky^2)^2 + mu*gyration^2*u = 0.  Solve this quadratic, then
    // recover the two +/-gamma pairs and their null-space field vectors.
    const Complex kySquared = ky * ky;
    const Complex b = permeability * gyration * gyration -
        Complex{2} * epsilon * kySquared;
    const Complex c = epsilon * kySquared * kySquared;
    Complex discriminantRoot = std::sqrt(
        b * b - Complex{4} * epsilon * c);
    if (std::abs(-b - discriminantRoot) >
        std::abs(-b + discriminantRoot)) {
        discriminantRoot = -discriminantRoot;
    }
    std::array<Complex, 2> u{
        (-b + discriminantRoot) / (Complex{2} * epsilon),
        (-b - discriminantRoot) / (Complex{2} * epsilon),
    };
    const Real quadraticScale = std::max({
        Real{1}, std::abs(b), std::abs(c), std::abs(discriminantRoot)});
    if (std::abs(u[0]) >
        Real{64} * std::numeric_limits<Real>::epsilon() * quadraticScale) {
        u[1] = kySquared * kySquared / u[0];
    }

    const Complex muEpsilon = permeability * epsilon;
    const Complex inverseMu = Complex{1} / permeability;
    for (std::size_t mode = 0; mode < 2; ++mode) {
        const Complex qSquared = muEpsilon - u[mode];
        const Complex gammaSquared = qSquared - kx * kx;
        const Complex principalGamma = std::sqrt(gammaSquared);
        const Real gammaScale = std::max({
            Real{1}, std::sqrt(std::abs(qSquared)), std::abs(kx)});
        if (!finiteComplex(principalGamma) ||
            std::abs(principalGamma) <= Real{256} *
                std::numeric_limits<Real>::epsilon() * gammaScale) {
            return false;
        }

        for (std::size_t direction = 0; direction < 2; ++direction) {
            const Complex gamma = direction == 0 ? principalGamma : -principalGamma;
            const Complex waveNumberSquared =
                kx * kx + kySquared + gamma * gamma;
            const std::array<Complex, 3> row0{
                kx * kx - waveNumberSquared + muEpsilon,
                kx * ky,
                kx * gamma + permeability * gyration,
            };
            const std::array<Complex, 3> row1{
                ky * kx,
                kySquared - waveNumberSquared + muEpsilon,
                ky * gamma,
            };
            const std::array<Complex, 3> row2{
                gamma * kx - permeability * gyration,
                gamma * ky,
                gamma * gamma - waveNumberSquared + muEpsilon,
            };
            const std::array<std::array<Complex, 3>, 3> candidates{
                cross3(row0, row1),
                cross3(row0, row2),
                cross3(row1, row2),
            };
            std::size_t best = 0;
            Real bestNormSquared = normSquared3(candidates[0]);
            for (std::size_t candidate = 1; candidate < candidates.size(); ++candidate) {
                const Real candidateNormSquared = normSquared3(candidates[candidate]);
                if (candidateNormSquared > bestNormSquared) {
                    best = candidate;
                    bestNormSquared = candidateNormSquared;
                }
            }
            const Real equationScale = std::max({
                Real{1},
                std::abs(row0[0]), std::abs(row0[1]), std::abs(row0[2]),
                std::abs(row1[0]), std::abs(row1[1]), std::abs(row1[2]),
                std::abs(row2[0]), std::abs(row2[1]), std::abs(row2[2])});
            const Real electricNorm = std::sqrt(bestNormSquared);
            if (!(electricNorm > Real{256} *
                    std::numeric_limits<Real>::epsilon() * equationScale * equationScale) ||
                !std::isfinite(electricNorm)) {
                return false;
            }
            const std::array<Complex, 3> electric{
                candidates[best][0] / electricNorm,
                candidates[best][1] / electricNorm,
                candidates[best][2] / electricNorm,
            };
            const std::array<Complex, 3> magnetic{
                (ky * electric[2] - gamma * electric[1]) * inverseMu,
                (gamma * electric[0] - kx * electric[2]) * inverseMu,
                (kx * electric[1] - ky * electric[0]) * inverseMu,
            };

            const std::array<Complex, 3> residual{
                row0[0] * electric[0] + row0[1] * electric[1] +
                    row0[2] * electric[2],
                row1[0] * electric[0] + row1[1] * electric[1] +
                    row1[2] * electric[2],
                row2[0] * electric[0] + row2[1] * electric[1] +
                    row2[2] * electric[2],
            };
            const Real residualMagnitude = std::max({
                std::abs(residual[0]),
                std::abs(residual[1]),
                std::abs(residual[2])});
            const Real residualLimit =
#ifdef RCWA_SINGLE_PRECISION
                Real{3e-4f} * equationScale;
#else
                Real{8e-11} * equationScale;
#endif
            if (!std::isfinite(residualMagnitude) ||
                residualMagnitude > residualLimit) {
                return false;
            }

            const std::size_t column = mode + 2 * direction;
            eigenvalues[column] = gamma;
            stateVectors[column] = electric[0];
            stateVectors[4 + column] = electric[1];
            stateVectors[8 + column] = magnetic[0];
            stateVectors[12 + column] = magnetic[1];
            Real tangentialNormSquared{};
            for (std::size_t component = 0; component < 4; ++component) {
                tangentialNormSquared +=
                    std::norm(stateVectors[component * 4 + column]);
            }
            const Real tangentialNorm = std::sqrt(tangentialNormSquared);
            if (!(tangentialNorm > Real{0}) || !std::isfinite(tangentialNorm)) {
                return false;
            }
            for (std::size_t component = 0; component < 4; ++component) {
                stateVectors[component * 4 + column] /= tangentialNorm;
            }
        }
    }
    return true;
}

LayerModes tensorBerremanModes(const Tensor3& eps,
                               const Tensor3& mu,
                               const DiagonalOperator& Kx,
                               const DiagonalOperator& Ky,
                               Real thicknessUm,
                               Real wavelengthUm) {
    const std::size_t N = Kx.rows();
    const std::size_t modeCount = 2 * N;
    std::vector<UniformModeBlock> blocks(N);
    std::vector<Complex> gamma(4 * N);
    const auto& kx = Kx.values();
    const auto& ky = Ky.values();
    const bool canUseReducedBerreman =
        tensorHasExactZDecoupling(eps) && tensorHasExactZDecoupling(mu);
    Complex gyrotropicDiagonal{};
    Complex gyrotropicCoupling{};
    Complex scalarMu{};
    const bool canUseYGyrotropicBerreman =
        exactYGyrotropicEpsilon(
            eps, gyrotropicDiagonal, gyrotropicCoupling) &&
        tensorIsScalar(mu, &scalarMu, Real{0});

    for (std::size_t h = 0; h < N; ++h) {
        std::array<Complex, 16> stateVectors{};
        std::array<Complex, 4> reducedValues{};
        bool usedReduced = canUseReducedBerreman &&
            tryReducedBerremanEigensystem(
                eps, mu, kx[h], ky[h], reducedValues, stateVectors);
        if (!usedReduced && canUseYGyrotropicBerreman) {
            usedReduced = tryYGyrotropicBerremanEigensystem(
                eps, mu, kx[h], ky[h], reducedValues, stateVectors);
        }
        if (usedReduced) {
            const auto order = fluxModeOrder4(reducedValues, stateVectors);
            for (std::size_t slot = 0; slot < 4; ++slot) {
                setOrderMode(
                    blocks, gamma, h, N, slot, stateVectors,
                    order[slot], reducedValues[order[slot]]);
            }
            prepareUniformBlockInverse(blocks[h]);
            usedReduced = blocks[h].inverseReliable;
        }
        if (!usedReduced) {
            blocks[h] = UniformModeBlock{};
            Eigensystem er = eig(berremanDeltaMatrixImpl(eps, mu, kx[h], ky[h]));
            normalizeColumns(er.vectors);
            for (std::size_t column = 0; column < 4; ++column) {
                stateVectors[column] = er.vectors(0, column);
                stateVectors[4 + column] = er.vectors(2, column);
                stateVectors[8 + column] = -er.vectors(3, column);
                stateVectors[12 + column] = er.vectors(1, column);
            }
            const auto order = fluxModeOrder(er.values, stateVectors, 2);
            for (std::size_t slot = 0; slot < 4; ++slot) {
                setOrderMode(
                    blocks, gamma, h, N, slot, stateVectors,
                    order[slot], er.values[order[slot]]);
            }
            prepareUniformBlockInverse(blocks[h]);
        }
    }

    std::vector<Complex> phase =
        makePropagationPhases(gamma, modeCount, thicknessUm, wavelengthUm);
    return {
        Matrix{},
        std::move(phase),
        std::move(gamma),
        LayerComputationKind::UniformTmm,
        false,
        false,
        std::move(blocks),
    };
}

LayerModes scalarBerremanModes(Complex epsR,
                               Complex muR,
                               const DiagonalOperator& Kx,
                               const DiagonalOperator& Ky,
                               Real thicknessUm,
                               Real wavelengthUm) {
    const std::size_t N = Kx.rows();
    std::vector<UniformModeBlock> blocks(N);
    const auto& kx = Kx.values();
    const auto& ky = Ky.values();
    std::vector<Complex> gamma(4 * N);

    auto fillState = [&](std::array<Complex, 16>& local,
                         std::size_t column,
                         std::size_t i,
                         Complex gz,
                         bool te) {
        const Complex ax = kx[i];
        const Complex ay = ky[i];
        const Real rho = std::sqrt(std::norm(ax) + std::norm(ay));
        Complex Ex{}, Ey{}, Hx{}, Hy{};
        if (te) {
            if (rho < 1e-14) {
                Ey = {1.0, 0.0};
            } else {
                Ex = -ay / rho;
                Ey = ax / rho;
            }
            const Complex Ez = std::abs(gz) < Real{1e-14}
                ? Complex{}
                : -(ax * Ex + ay * Ey) / gz;
            Hx = (ay * Ez - gz * Ey) / muR;
            Hy = (gz * Ex - ax * Ez) / muR;
        } else {
            if (rho < 1e-14) {
                Hy = {1.0, 0.0};
            } else {
                Hx = -ay / rho;
                Hy = ax / rho;
            }
            const Complex Hz = std::abs(gz) < Real{1e-14}
                ? Complex{}
                : -(ax * Hx + ay * Hy) / gz;
            Ex = (gz * Hy - ay * Hz) / epsR;
            Ey = (ax * Hz - gz * Hx) / epsR;
        }
        local[column] = Ex;
        local[4 + column] = Ey;
        local[8 + column] = Hx;
        local[12 + column] = Hy;
    };

    for (std::size_t i = 0; i < N; ++i) {
        const Complex gz = isotropicLongitudinalWaveNumber(
            epsR, muR, kx[i], ky[i]);
        std::array<Complex, 16> local{};
        fillState(local, 0, i, gz, true);
        fillState(local, 1, i, gz, false);
        fillState(local, 2, i, -gz, true);
        fillState(local, 3, i, -gz, false);

        const auto plusIsForward = [&](std::size_t plusColumn) {
            const Real eigenTolerance = Real{64} *
                std::numeric_limits<Real>::epsilon() *
                std::max(Real{1}, std::abs(gz));
            if (std::imag(gz) > eigenTolerance) {
                return true;
            }
            if (std::imag(gz) < -eigenTolerance) {
                return false;
            }
            const Real flux = stateFluxZUnchecked(local, plusColumn);
            const Real fluxTolerance = Real{64} *
                std::numeric_limits<Real>::epsilon() *
                std::max(Real{1}, std::abs(flux));
            if (flux > fluxTolerance) {
                return true;
            }
            if (flux < -fluxTolerance) {
                return false;
            }
            // At a cutoff the two directions coalesce.  Keep the principal
            // root in the forward slot; identical-basis interfaces have an
            // exact transparent limiting treatment in the TMM cascade.
            return std::real(gz) >= Real{0};
        };

        const bool plusTeForward = plusIsForward(0);
        const bool plusTmForward = plusIsForward(1);
        const std::size_t downTe = plusTeForward ? 0 : 2;
        const std::size_t upTe = plusTeForward ? 2 : 0;
        const std::size_t downTm = plusTmForward ? 1 : 3;
        const std::size_t upTm = plusTmForward ? 3 : 1;
        setOrderMode(
            blocks, gamma, i, N, 0, local, downTe,
            downTe < 2 ? gz : -gz);
        setOrderMode(
            blocks, gamma, i, N, 1, local, downTm,
            downTm < 2 ? gz : -gz);
        setOrderMode(
            blocks, gamma, i, N, 2, local, upTe,
            upTe < 2 ? gz : -gz);
        setOrderMode(
            blocks, gamma, i, N, 3, local, upTm,
            upTm < 2 ? gz : -gz);
    }
    prepareUniformBlockInverses(blocks);

    std::vector<Complex> phase =
        makePropagationPhases(gamma, 2 * N, thicknessUm, wavelengthUm);
    return {
        Matrix{},
        std::move(phase),
        std::move(gamma),
        LayerComputationKind::UniformTmm,
        true,
        false,
        std::move(blocks),
    };
}

} // namespace

std::optional<std::vector<std::size_t>> transverseSymmetryHarmonicMap(
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    Real kxParity,
    Real kyParity) {
    return transverseSymmetryHarmonicMapImpl(
        sourceKx,
        sourceKy,
        targetKx,
        targetKy,
        kxParity,
        kyParity);
}

std::optional<LayerModes> tryMakeTransverseSymmetryPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    const TransverseSymmetryTransform& transform,
    const std::vector<std::size_t>* precomputedMap) {
    if (source.kind != LayerComputationKind::PatternedRcwa ||
        hasCompactUniformModes(source)) {
        return std::nullopt;
    }
    std::optional<std::vector<std::size_t>> ownedMap;
    const std::vector<std::size_t>* mirror = precomputedMap;
    if (mirror == nullptr) {
        ownedMap = transverseSymmetryHarmonicMapImpl(
            sourceKx,
            sourceKy,
            targetKx,
            targetKy,
            transform.kxParity,
            transform.kyParity);
        if (!ownedMap) {
            return std::nullopt;
        }
        mirror = &*ownedMap;
    } else if (mirror->size() != sourceKx.rows()) {
        return std::nullopt;
    }
    if (!cachedTensorFourierMatricesHaveSpatialSymmetry(
            tensors,
            *mirror,
            transform.tensorParity)) {
        return std::nullopt;
    }

    const std::size_t N = mirror->size();
    const std::size_t stateSize = 4 * N;
    if (source.W.rows() != stateSize || source.W.cols() != stateSize ||
        source.gamma.size() != stateSize ||
        propagationPhase(source).size() != stateSize) {
        return std::nullopt;
    }

    Matrix targetW(stateSize, stateSize);
    const auto& sourceValues = source.W.data();
    auto& targetValues = targetW.data();
    for (std::size_t component = 0; component < 4; ++component) {
        for (std::size_t targetHarmonic = 0; targetHarmonic < N;
             ++targetHarmonic) {
            const std::size_t sourceHarmonic = (*mirror)[targetHarmonic];
            const std::size_t sourceBase =
                (component * N + sourceHarmonic) * stateSize;
            const std::size_t targetBase =
                (component * N + targetHarmonic) * stateSize;
            for (std::size_t column = 0; column < stateSize; ++column) {
                targetValues[targetBase + column] =
                    transform.stateParity[component] *
                    sourceValues[sourceBase + column];
            }
        }
    }
    return LayerModes{
        std::move(targetW),
        source.phase,
        source.gamma,
        LayerComputationKind::PatternedRcwa,
        false,
        source.usedYMirrorSymmetrySplit,
        {},
    };
}

std::optional<LayerModes> tryMakeTransverseSymmetryUniformModes(
    const LayerModes& source,
    const Tensor3& eps,
    const Tensor3& mu,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy,
    const TransverseSymmetryTransform& transform,
    const std::vector<std::size_t>* precomputedMap) {
    if (source.kind != LayerComputationKind::UniformTmm ||
        !hasCompactUniformModes(source) ||
        !uniformTensorsHaveSpatialSymmetry(eps, mu, transform.tensorParity)) {
        return std::nullopt;
    }
    std::optional<std::vector<std::size_t>> ownedMap;
    const std::vector<std::size_t>* mirror = precomputedMap;
    if (mirror == nullptr) {
        ownedMap = transverseSymmetryHarmonicMapImpl(
            sourceKx,
            sourceKy,
            targetKx,
            targetKy,
            transform.kxParity,
            transform.kyParity);
        if (!ownedMap) {
            return std::nullopt;
        }
        mirror = &*ownedMap;
    } else if (mirror->size() != sourceKx.rows()) {
        return std::nullopt;
    }

    const std::size_t N = mirror->size();
    const std::size_t stateSize = 4 * N;
    if (source.uniformBlocks.size() != N ||
        source.gamma.size() != stateSize ||
        source.phase.size() != stateSize) {
        return std::nullopt;
    }

    LayerModes target = source;
    target.W = Matrix{};
    target.gamma.assign(stateSize, Complex{});
    target.phase.assign(stateSize, Complex{});
    target.uniformBlocks.assign(N, UniformModeBlock{});
    for (std::size_t targetHarmonic = 0; targetHarmonic < N; ++targetHarmonic) {
        const std::size_t sourceHarmonic = (*mirror)[targetHarmonic];
        if (sourceHarmonic >= N) {
            return std::nullopt;
        }
        const UniformModeBlock& sourceBlock = source.uniformBlocks[sourceHarmonic];
        UniformModeBlock& targetBlock = target.uniformBlocks[targetHarmonic];
        targetBlock.inverseReliable = sourceBlock.inverseReliable;
        for (std::size_t row = 0; row < 4; ++row) {
            const Complex rowScale{transform.stateParity[row], Real{0}};
            const std::size_t rowBase = row * 4;
            for (std::size_t column = 0; column < 4; ++column) {
                targetBlock.values[rowBase + column] =
                    rowScale * sourceBlock.values[rowBase + column];
            }
        }
        for (std::size_t row = 0; row < 4; ++row) {
            const std::size_t rowBase = row * 4;
            for (std::size_t column = 0; column < 4; ++column) {
                targetBlock.inverse[rowBase + column] =
                    sourceBlock.inverse[rowBase + column] *
                    Complex{transform.stateParity[column], Real{0}};
            }
        }
        for (std::size_t localSlot = 0; localSlot < 4; ++localSlot) {
            const std::size_t sourceColumn = localSlot * N + sourceHarmonic;
            const std::size_t targetColumn = localSlot * N + targetHarmonic;
            target.gamma[targetColumn] = source.gamma[sourceColumn];
            target.phase[targetColumn] = source.phase[sourceColumn];
        }
    }
    return target;
}

std::optional<LayerModes> tryMakeXMirroredPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy) {
    return tryMakeTransverseSymmetryPeriodicModes(
        source,
        tensors,
        sourceKx,
        sourceKy,
        targetKx,
        targetKy,
        TransverseSymmetryTransform{
            Real{-1},
            Real{1},
            std::array<Real, 4>{Real{-1}, Real{1}, Real{1}, Real{-1}},
            std::array<Real, 3>{Real{-1}, Real{1}, Real{1}},
        });
}

std::optional<LayerModes> tryMakeInPlaneInvertedPeriodicModes(
    const LayerModes& source,
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& sourceKx,
    const DiagonalOperator& sourceKy,
    const DiagonalOperator& targetKx,
    const DiagonalOperator& targetKy) {
    return tryMakeTransverseSymmetryPeriodicModes(
        source,
        tensors,
        sourceKx,
        sourceKy,
        targetKx,
        targetKy,
        TransverseSymmetryTransform{
            Real{-1},
            Real{-1},
            std::array<Real, 4>{Real{-1}, Real{-1}, Real{-1}, Real{-1}},
            std::array<Real, 3>{Real{-1}, Real{-1}, Real{1}},
        });
}

void annotateTensorFourierMetadata(TensorFourierMatrices& tensors,
                                      std::size_t harmonicCount) {
    tensors.metadata.valid = true;
    tensors.metadata.harmonicCount = harmonicCount;
    tensors.metadata.factorizationComplete =
        tensorQBlocksComplete(tensors.liEpsQ, harmonicCount) &&
        tensorQBlocksComplete(tensors.liMuQ, harmonicCount) &&
        matrixHasShape(tensors.liEpsQZzInverse, harmonicCount) &&
        matrixHasShape(tensors.liMuQZzInverse, harmonicCount);
    tensors.metadata.scalarEpsilonBlocks =
        tensorEpsilonBlocksAreScalarMaterial(tensors);
    Complex commonMu{};
    tensors.metadata.commonScalarMuIdentity =
        tensorMuBlocksAreCommonScalarIdentity(tensors, &commonMu);
    tensors.metadata.commonScalarMu = commonMu;
    tensors.metadata.diagonalEpsilonBlocks =
        diagonalEpsilonBlocksSupportReducedModes(tensors);
    tensors.metadata.fingerprint = tensorFourierFingerprint(tensors, harmonicCount);
}

std::uint64_t modalFingerprintForUniformTensors(
    const Tensor3& eps,
    const Tensor3& mu) noexcept {
    detail::Fingerprint64 fingerprint(UINT64_C(0x4d4f44414c554e49));
    for (const Tensor3* tensor : {&eps, &mu}) {
        for (const Complex value : tensor->v) {
            fingerprint.append(value);
        }
    }
    return fingerprint.nonzeroValue();
}

const std::vector<Complex>& propagationPhase(const LayerModes& modes) {
    if (!modes.phase.empty()) {
        return modes.phase;
    }
    throw std::invalid_argument("layer modes do not contain propagation phase storage");
}

LayerModes layerModesWithThickness(const LayerModes& modalBasisSource,
                                   Real thicknessUm,
                                   Real wavelengthUm) {
    if (thicknessUm < Real{0} || !std::isfinite(thicknessUm)) {
        throw std::invalid_argument("layer thickness must be finite and non-negative");
    }
    requirePositive(wavelengthUm, "wavelength");
    if (modalBasisSource.gamma.empty() || modalBasisSource.gamma.size() % 2 != 0) {
        throw std::invalid_argument(
            "modal basis gamma storage must contain equal forward and backward blocks");
    }

    LayerModes out = modalBasisSource;
    out.phase = makePropagationPhases(
        out.gamma,
        out.gamma.size() / 2,
        thicknessUm,
        wavelengthUm);
    return out;
}

bool hasCompactUniformModes(const LayerModes& modes) noexcept {
    return !modes.uniformBlocks.empty();
}

std::size_t layerHarmonicCount(const LayerModes& modes) {
    if (hasCompactUniformModes(modes)) {
        return modes.uniformBlocks.size();
    }
    if (modes.W.rows() == 0 || modes.W.rows() % 4 != 0 ||
        modes.W.cols() != modes.W.rows()) {
        throw std::invalid_argument(
            "layer modes must contain compact uniform blocks or a square 4N mode matrix");
    }
    return modes.W.rows() / 4;
}

std::size_t layerStateSize(const LayerModes& modes) {
    return 4 * layerHarmonicCount(modes);
}

Complex modeStateValue(const LayerModes& modes,
                       std::size_t row,
                       std::size_t column) {
    const std::size_t N = layerHarmonicCount(modes);
    const std::size_t stateSize = 4 * N;
    if (row >= stateSize || column >= stateSize) {
        throw std::out_of_range("mode matrix entry is outside layer mode storage");
    }
    if (!hasCompactUniformModes(modes)) {
        return modes.W(row, column);
    }

    const std::size_t rowHarmonic = row % N;
    const std::size_t columnHarmonic = column % N;
    if (rowHarmonic != columnHarmonic) {
        return {};
    }
    const std::size_t component = row / N;
    const std::size_t localColumn = column / N;
    return modes.uniformBlocks[rowHarmonic].values[component * 4 + localColumn];
}

Matrix materializeModeMatrix(const LayerModes& modes) {
    if (!hasCompactUniformModes(modes)) {
        return modes.W;
    }
    const std::size_t N = layerHarmonicCount(modes);
    Matrix dense(4 * N, 4 * N);
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        const auto& block = modes.uniformBlocks[harmonic].values;
        for (std::size_t component = 0; component < 4; ++component) {
            const std::size_t row = component * N + harmonic;
            for (std::size_t localColumn = 0; localColumn < 4; ++localColumn) {
                const std::size_t column = localColumn * N + harmonic;
                dense(row, column) = block[component * 4 + localColumn];
            }
        }
    }
    return dense;
}

Real stateFluxZ(const Matrix& W, std::size_t c) {
    if (W.rows() % 4 != 0 || c >= W.cols()) {
        throw std::invalid_argument("flux calculation expects a 4N state matrix and valid column");
    }
    return stateFluxZUnchecked(W, c);
}

Real propagatingFluxWeight(const LayerModes& modes, std::size_t column) {
    const std::size_t N = layerHarmonicCount(modes);
    if (column >= 4 * N) {
        throw std::invalid_argument("flux calculation column is outside layer mode storage");
    }
    Real rawFlux{};
    if (hasCompactUniformModes(modes)) {
        rawFlux = uniformBlockFlux(
            modes.uniformBlocks[column % N],
            column / N);
    } else {
        rawFlux = stateFluxZ(modes.W, column);
    }
    const Real flux = std::abs(rawFlux);
    return flux > Real{1e-10} ? flux : Real{0.0};
}

Matrix propagationMatrixBlock(const LayerModes& modes, std::size_t offset, std::size_t count) {
    const auto& phase = propagationPhase(modes);
    if (offset + count > phase.size()) {
        throw std::invalid_argument("propagation phase block is outside layer mode storage");
    }
    Matrix out(count, count);
    auto& data = out.data();
    for (std::size_t i = 0; i < count; ++i) {
        data[i * count + i] = phase[offset + i];
    }
    return out;
}

LayerModes computeUniformModes(
    const Tensor3& eps,
    const Tensor3& mu,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm) {
    if (Kx.rows() != Kx.cols() || Ky.rows() != Ky.cols() || Kx.rows() != Ky.rows()) {
        throw std::invalid_argument("Kx and Ky must be square matrices with equal size");
    }

    Complex epsScalar{};
    Complex muScalar{};
    // Backend selection is exact: a nonzero tensor component, however small,
    // is physical input and must not be discarded by an isotropy tolerance.
    if (tensorIsScalar(eps, &epsScalar, Real{0}) &&
        tensorIsScalar(mu, &muScalar, Real{0})) {
        // The scalar case is still the same 4x4 Berreman TMM state.  Its two
        // forward and two backward eigenvalues are exactly degenerate, so use
        // the analytic TE/TM eigenbasis instead of accepting an arbitrary and
        // potentially ill-conditioned numerical basis from a 4x4 eigensolve.
        return scalarBerremanModes(
            epsScalar, muScalar, Kx, Ky, thicknessUm, wavelengthUm);
    }

    return tensorBerremanModes(eps, mu, Kx, Ky, thicknessUm, wavelengthUm);
}

static LayerModes computeIsotropicPeriodicModesImpl(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit);

static LayerModes computeAnisotropicTensorPeriodicModesImpl(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm);

LayerModes computePeriodicModes(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit) {
    const std::size_t N = Kx.rows();
    validateTensorBlockSize(tensors, N);
    const TensorFourierMatrices& effective = tensors;
    const bool useMetadata = tensorMetadataValidFor(effective, N) &&
        effective.metadata.factorizationComplete;
    const bool scalarEpsilon = useMetadata
        ? effective.metadata.scalarEpsilonBlocks
        : tensorEpsilonBlocksAreScalarMaterial(effective);
    const bool commonScalarMu = useMetadata
        ? effective.metadata.commonScalarMuIdentity
        : tensorMuBlocksAreCommonScalarIdentity(effective);
    const bool diagonalEpsilon = useMetadata
        ? effective.metadata.diagonalEpsilonBlocks
        : diagonalEpsilonBlocksSupportReducedModes(effective);
    if (scalarEpsilon && commonScalarMu && diagonalEpsilon) {
        return computeIsotropicPeriodicModesImpl(
            effective,
            Kx,
            Ky,
            thicknessUm,
            wavelengthUm,
            allowYMirrorSymmetrySplit);
    }
    if (commonScalarMu) {
        const Complex muR = periodicScalarMuFromTensorBlocks(effective, N);
        if (auto reduced = computePeriodicModesInPlaneReduced(
                effective,
                Kx,
                Ky,
                muR,
                thicknessUm,
                wavelengthUm,
                allowYMirrorSymmetrySplit)) {
            return std::move(*reduced);
        }
    }
    return computeAnisotropicTensorPeriodicModesImpl(
        effective, Kx, Ky, thicknessUm, wavelengthUm);
}

static LayerModes computeIsotropicPeriodicModesImpl(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm,
    bool allowYMirrorSymmetrySplit) {
    const std::size_t N = Kx.rows();
    if (Kx.cols() != N || Ky.rows() != N || Ky.cols() != N) {
        throw std::invalid_argument("Kx and Ky must be square matrices with equal size");
    }
    validateTensorBlockSize(tensors, N);
    if (!tensorMetadataValidFor(tensors, N) &&
        !tensorEpsilonBlocksAreScalarMaterial(tensors)) {
        throw std::invalid_argument(
            "isotropic periodic RCWA route requires scalar epsilon tensor blocks");
    }
    const Complex muR = periodicScalarMuFromTensorBlocks(tensors, N);
    if (auto reduced = computePeriodicModesReduced(
            tensors,
            Kx,
            Ky,
            muR,
            thicknessUm,
            wavelengthUm,
            allowYMirrorSymmetrySplit)) {
        return std::move(*reduced);
    }
    throw std::runtime_error("isotropic periodic RCWA route requires diagonal epsilon blocks");
}

static LayerModes computeAnisotropicTensorPeriodicModesImpl(
    const TensorFourierMatrices& tensors,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    Real thicknessUm,
    Real wavelengthUm) {
    const std::size_t N = Kx.rows();
    if (Kx.cols() != N || Ky.rows() != N || Ky.cols() != N) {
        throw std::invalid_argument("Kx and Ky must be square matrices with equal size");
    }
    validateTensorBlockSize(tensors, N);
    const auto& kx = Kx.values();
    const auto& ky = Ky.values();

    const Matrix& E11 = tensors.liEpsQ[0][0];
    const Matrix& E12 = tensors.liEpsQ[0][1];
    const Matrix& E13 = tensors.liEpsQ[0][2];
    const Matrix& E21 = tensors.liEpsQ[1][0];
    const Matrix& E22 = tensors.liEpsQ[1][1];
    const Matrix& E23 = tensors.liEpsQ[1][2];
    const Matrix& E31 = tensors.liEpsQ[2][0];
    const Matrix& E32 = tensors.liEpsQ[2][1];
    const Matrix& E33 = tensors.liEpsQ[2][2];
    const Matrix& M11 = tensors.liMuQ[0][0];
    const Matrix& M12 = tensors.liMuQ[0][1];
    const Matrix& M13 = tensors.liMuQ[0][2];
    const Matrix& M21 = tensors.liMuQ[1][0];
    const Matrix& M22 = tensors.liMuQ[1][1];
    const Matrix& M23 = tensors.liMuQ[1][2];
    const Matrix& M31 = tensors.liMuQ[2][0];
    const Matrix& M32 = tensors.liMuQ[2][1];
    const Matrix& M33 = tensors.liMuQ[2][2];

    Matrix computedInvE33;
    const Matrix* invE33 = &tensors.liEpsQZzInverse;
    if (invE33->empty()) {
        computedInvE33 = inverse(E33);
        invE33 = &computedInvE33;
    }
    Matrix computedInvM33;
    const Matrix* invM33 = &tensors.liMuQZzInverse;
    if (invM33->empty()) {
        computedInvM33 = inverse(M33);
        invM33 = &computedInvM33;
    }
    const Matrix kxInvE33 = leftScaleRows(kx, *invE33);
    const Matrix kyInvE33 = leftScaleRows(ky, *invE33);
    const Matrix kxInvM33 = leftScaleRows(kx, *invM33);
    const Matrix kyInvM33 = leftScaleRows(ky, *invM33);
    const Matrix E23InvE33 = E23 * *invE33;
    const Matrix E13InvE33 = E13 * *invE33;
    const Matrix M23InvM33 = M23 * *invM33;
    const Matrix M13InvM33 = M13 * *invM33;
    const Matrix invE33E31 = *invE33 * E31;
    const Matrix invE33E32 = *invE33 * E32;
    const Matrix invM33M31 = *invM33 * M31;
    const Matrix invM33M32 = *invM33 * M32;

    const Matrix A00 = -leftScaleRows(kx, invE33E31) -
        rightScaleColumns(M23InvM33, ky);
    const Matrix A01 = -leftScaleRows(kx, invE33E32) +
        rightScaleColumns(M23InvM33, kx);
    const Matrix A02 = rightScaleColumns(kxInvE33, ky) + M21 - M23InvM33 * M31;
    const Matrix A03 = -rightScaleColumns(kxInvE33, kx) + M22 - M23InvM33 * M32;

    const Matrix A10 = -leftScaleRows(ky, invE33E31) +
        rightScaleColumns(M13InvM33, ky);
    const Matrix A11 = -leftScaleRows(ky, invE33E32) -
        rightScaleColumns(M13InvM33, kx);
    const Matrix A12 = rightScaleColumns(kyInvE33, ky) - M11 + M13InvM33 * M31;
    const Matrix A13 = -rightScaleColumns(kyInvE33, kx) - M12 + M13InvM33 * M32;

    const Matrix A20 = -rightScaleColumns(kxInvM33, ky) - E21 + E23InvE33 * E31;
    const Matrix A21 = rightScaleColumns(kxInvM33, kx) - E22 + E23InvE33 * E32;
    const Matrix A22 = -leftScaleRows(kx, invM33M31) -
        rightScaleColumns(E23InvE33, ky);
    const Matrix A23 = -leftScaleRows(kx, invM33M32) +
        rightScaleColumns(E23InvE33, kx);

    const Matrix A30 = -rightScaleColumns(kyInvM33, ky) + E11 - E13InvE33 * E31;
    const Matrix A31 = rightScaleColumns(kyInvM33, kx) + E12 - E13InvE33 * E32;
    const Matrix A32 = -leftScaleRows(ky, invM33M31) +
        rightScaleColumns(E13InvE33, ky);
    const Matrix A33 = -leftScaleRows(ky, invM33M32) -
        rightScaleColumns(E13InvE33, kx);

    const Matrix A = block4(A00, A01, A02, A03,
                            A10, A11, A12, A13,
                            A20, A21, A22, A23,
                            A30, A31, A32, A33);
    return modesFromStateMatrix(A, thicknessUm, wavelengthUm);
}

Matrix berremanStateMatrix(const Tensor3& eps,
                           const Tensor3& mu,
                           Complex kx,
                           Complex ky) {
    return electricMagneticStateMatrix(
        berremanDeltaMatrixImpl(eps, mu, kx, ky));
}

} // namespace rcwa
