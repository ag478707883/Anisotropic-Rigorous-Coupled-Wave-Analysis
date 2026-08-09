#include "rcwa/smatrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace rcwa {

namespace {

void setMatrixColumns(Matrix& target,
                        std::size_t targetCol,
                        const Matrix& source,
                        std::size_t sourceCol,
                        std::size_t columnCount,
                        Complex scale = {1.0, 0.0}) {
    if (target.rows() != source.rows() ||
        targetCol > target.cols() || target.cols() - targetCol < columnCount ||
        sourceCol > source.cols() || source.cols() - sourceCol < columnCount) {
        throw std::invalid_argument("matrix column copy is outside bounds");
    }
    const std::size_t targetCols = target.cols();
    const std::size_t sourceCols = source.cols();
    const auto& src = source.data();
    auto& dst = target.data();
    for (std::size_t r = 0; r < source.rows(); ++r) {
        const std::size_t targetBase = r * targetCols + targetCol;
        const std::size_t sourceBase = r * sourceCols + sourceCol;
        for (std::size_t c = 0; c < columnCount; ++c) {
            dst[targetBase + c] = scale * src[sourceBase + c];
        }
    }
}

Matrix downModesMultiplyColumns(const LayerModes& modes,
                                   const Matrix& first,
                                   const Matrix& second) {
    const std::size_t stateSize = layerStateSize(modes);
    const std::size_t modeCount = stateSize / 2;
    if (first.rows() != modeCount || second.rows() != modeCount) {
        throw std::invalid_argument("down-mode projection expects matching row counts");
    }

    Matrix rhs(modeCount, first.cols() + second.cols());
    rhs.setBlock(0, 0, first);
    rhs.setBlock(0, first.cols(), second);
    if (hasCompactUniformModes(modes)) {
        const std::size_t N = layerHarmonicCount(modes);
        Matrix out(stateSize, rhs.cols());
        const auto& rhsData = rhs.data();
        auto& outData = out.data();
        const std::size_t rhsCols = rhs.cols();
        for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
            const auto& block = modes.uniformBlocks[harmonic].values;
            for (std::size_t component = 0; component < 4; ++component) {
                const std::size_t outBase =
                    (component * N + harmonic) * rhsCols;
                for (std::size_t c = 0; c < rhsCols; ++c) {
                    outData[outBase + c] =
                        block[component * 4] * rhsData[harmonic * rhsCols + c] +
                        block[component * 4 + 1] *
                            rhsData[(N + harmonic) * rhsCols + c];
                }
            }
        }
        return out;
    }
    return multiplyColumnBlock(modes.W, 0, modeCount, rhs);
}

void setModeBlock(Matrix& target,
                    std::size_t row,
                    std::size_t col,
                    const LayerModes& modes,
                    bool down,
                    Complex scale = {1.0, 0.0}) {
    const std::size_t stateSize = layerStateSize(modes);
    const std::size_t modeCount = stateSize / 2;
    const std::size_t sourceOffset = down ? 0 : modeCount;
    const std::size_t targetCols = target.cols();
    auto& dst = target.data();
    if (hasCompactUniformModes(modes)) {
        const std::size_t N = layerHarmonicCount(modes);
        const std::size_t localOffset = down ? 0 : 2;
        for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
            const auto& block = modes.uniformBlocks[harmonic].values;
            for (std::size_t component = 0; component < 4; ++component) {
                const std::size_t targetBase =
                    (row + component * N + harmonic) * targetCols + col;
                dst[targetBase + harmonic] =
                    scale * block[component * 4 + localOffset];
                dst[targetBase + N + harmonic] =
                    scale * block[component * 4 + localOffset + 1];
            }
        }
        return;
    }
    const std::size_t sourceCols = modes.W.cols();
    const auto& source = modes.W.data();
    for (std::size_t r = 0; r < stateSize; ++r) {
        const std::size_t targetBase = (row + r) * targetCols + col;
        const std::size_t sourceBase = r * sourceCols + sourceOffset;
        for (std::size_t c = 0; c < modeCount; ++c) {
            dst[targetBase + c] = scale * source[sourceBase + c];
        }
    }
}

void addModeBlock(Matrix& target,
                    std::size_t row,
                    std::size_t col,
                    const LayerModes& modes,
                    bool down,
                    Complex scale = {1.0, 0.0}) {
    const std::size_t stateSize = layerStateSize(modes);
    const std::size_t modeCount = stateSize / 2;
    const std::size_t sourceOffset = down ? 0 : modeCount;
    const std::size_t targetCols = target.cols();
    auto& dst = target.data();
    if (hasCompactUniformModes(modes)) {
        const std::size_t N = layerHarmonicCount(modes);
        const std::size_t localOffset = down ? 0 : 2;
        for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
            const auto& block = modes.uniformBlocks[harmonic].values;
            for (std::size_t component = 0; component < 4; ++component) {
                const std::size_t targetBase =
                    (row + component * N + harmonic) * targetCols + col;
                dst[targetBase + harmonic] +=
                    scale * block[component * 4 + localOffset];
                dst[targetBase + N + harmonic] +=
                    scale * block[component * 4 + localOffset + 1];
            }
        }
        return;
    }
    const std::size_t sourceCols = modes.W.cols();
    const auto& source = modes.W.data();
    for (std::size_t r = 0; r < stateSize; ++r) {
        const std::size_t targetBase = (row + r) * targetCols + col;
        const std::size_t sourceBase = r * sourceCols + sourceOffset;
        for (std::size_t c = 0; c < modeCount; ++c) {
            dst[targetBase + c] += scale * source[sourceBase + c];
        }
    }
}

void setSelectedModeColumns(Matrix& target,
                               std::size_t row,
                               std::size_t col,
                               const LayerModes& modes,
                               bool down,
                               const std::vector<std::size_t>& columns,
                               Complex scale = {1.0, 0.0}) {
    const std::size_t stateSize = layerStateSize(modes);
    const std::size_t modeCount = stateSize / 2;
    const std::size_t sourceOffset = down ? 0 : modeCount;
    const std::size_t targetCols = target.cols();
    auto& targetData = target.data();
    if (hasCompactUniformModes(modes)) {
        const std::size_t N = layerHarmonicCount(modes);
        const std::size_t localOffset = down ? 0 : 2;
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const std::size_t source = columns[c];
            if (source >= modeCount) {
                throw std::invalid_argument("selected mode column is out of range");
            }
            const std::size_t harmonic = source % N;
            const std::size_t localColumn = localOffset + source / N;
            const auto& block = modes.uniformBlocks[harmonic].values;
            for (std::size_t component = 0; component < 4; ++component) {
                targetData[(row + component * N + harmonic) * targetCols + col + c] =
                    scale * block[component * 4 + localColumn];
            }
        }
        return;
    }
    const std::size_t sourceCols = modes.W.cols();
    const auto& sourceData = modes.W.data();
    for (std::size_t c = 0; c < columns.size(); ++c) {
        const std::size_t source = columns[c];
        if (source >= modeCount) {
            throw std::invalid_argument("selected mode column is out of range");
        }
        for (std::size_t r = 0; r < stateSize; ++r) {
            targetData[(row + r) * targetCols + col + c] =
                scale * sourceData[r * sourceCols + sourceOffset + source];
        }
    }
}

Matrix modeBlockMatrix(const LayerModes& modes, bool down) {
    const std::size_t stateSize = layerStateSize(modes);
    Matrix out(stateSize, stateSize / 2);
    setModeBlock(out, 0, 0, modes, down);
    return out;
}

std::optional<Matrix> applyCompactUniformInverse(const LayerModes& modes,
                                                   const Matrix& rhs) {
    if (!hasCompactUniformModes(modes) || rhs.rows() != layerStateSize(modes)) {
        return std::nullopt;
    }
    for (const auto& block : modes.uniformBlocks) {
        if (!block.inverseReliable) {
            return std::nullopt;
        }
    }

    const std::size_t N = layerHarmonicCount(modes);
    const std::size_t cols = rhs.cols();
    Matrix out(4 * N, cols);
    const auto& src = rhs.data();
    auto& dst = out.data();
    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        const auto& inverse = modes.uniformBlocks[harmonic].inverse;
        for (std::size_t localMode = 0; localMode < 4; ++localMode) {
            const std::size_t outBase = (localMode * N + harmonic) * cols;
            for (std::size_t c = 0; c < cols; ++c) {
                Complex value{};
                for (std::size_t component = 0; component < 4; ++component) {
                    value += inverse[localMode * 4 + component] *
                        src[(component * N + harmonic) * cols + c];
                }
                dst[outBase + c] = value;
            }
        }
    }
    return out;
}

struct CompactTransformedHalves {
    Matrix down;
    Matrix up;
};

std::optional<CompactTransformedHalves> transformModeBlockByCompactInverse(
    const LayerModes& compact,
    const LayerModes& modes,
    bool downBlock) {
    if (!hasCompactUniformModes(compact)) {
        return std::nullopt;
    }
    for (const auto& block : compact.uniformBlocks) {
        if (!block.inverseReliable) {
            return std::nullopt;
        }
    }

    const std::size_t N = layerHarmonicCount(compact);
    const std::size_t stateSize = 4 * N;
    const std::size_t modeCount = 2 * N;
    if (layerStateSize(modes) != stateSize) {
        return std::nullopt;
    }

    CompactTransformedHalves out{
        Matrix(modeCount, modeCount),
        Matrix(modeCount, modeCount),
    };
    const std::size_t sourceOffset = downBlock ? 0 : modeCount;
    auto& downData = out.down.data();
    auto& upData = out.up.data();

    if (!hasCompactUniformModes(modes)) {
        const auto& source = modes.W.data();
        const std::size_t sourceCols = modes.W.cols();
        for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
            const auto& inverse = compact.uniformBlocks[harmonic].inverse;
            for (std::size_t c = 0; c < modeCount; ++c) {
                std::array<Complex, 4> sourceColumn{};
                for (std::size_t component = 0; component < 4; ++component) {
                    sourceColumn[component] =
                        source[(component * N + harmonic) * sourceCols +
                               sourceOffset + c];
                }
                for (std::size_t localMode = 0; localMode < 4; ++localMode) {
                    Complex value{};
                    for (std::size_t component = 0; component < 4; ++component) {
                        value += inverse[localMode * 4 + component] *
                            sourceColumn[component];
                    }
                    const std::size_t localRow = localMode % 2;
                    const std::size_t row = localRow * N + harmonic;
                    if (localMode < 2) {
                        downData[row * modeCount + c] = value;
                    } else {
                        upData[row * modeCount + c] = value;
                    }
                }
            }
        }
        return out;
    }

    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        const auto& inverse = compact.uniformBlocks[harmonic].inverse;
        for (std::size_t c = 0; c < modeCount; ++c) {
            const std::size_t sourceColumn = sourceOffset + c;
            if (sourceColumn % N != harmonic) {
                continue;
            }
            const std::size_t localColumn = sourceColumn / N;
            const auto& block = modes.uniformBlocks[harmonic].values;
            for (std::size_t localMode = 0; localMode < 4; ++localMode) {
                Complex value{};
                for (std::size_t component = 0; component < 4; ++component) {
                    value += inverse[localMode * 4 + component] *
                        block[component * 4 + localColumn];
                }
                const std::size_t localRow = localMode % 2;
                const std::size_t row = localRow * N + harmonic;
                if (localMode < 2) {
                    downData[row * modeCount + c] = value;
                } else {
                    upData[row * modeCount + c] = value;
                }
            }
        }
    }
    return out;
}

Matrix stackUnknownBlocks(const Matrix& first, const Matrix& second) {
    if (first.rows() != second.rows() || first.cols() != second.cols()) {
        throw std::invalid_argument("coupled interface solution blocks must match");
    }
    Matrix out(first.rows() + second.rows(), first.cols());
    out.setBlock(0, 0, first);
    out.setBlock(first.rows(), 0, second);
    return out;
}

inline constexpr Real schurMinimumReciprocalCondition =
#ifdef RCWA_SINGLE_PRECISION
    Real{1e-6f};
#else
    Real{1e-13};
#endif

std::optional<Matrix> solveInterfaceWithRightCompact(
    const Matrix& leftBlock,
    const LayerModes& right,
    const Matrix& rhs) {
    const auto transformedLeft = applyCompactUniformInverse(right, leftBlock);
    const auto transformedRhs = applyCompactUniformInverse(right, rhs);
    if (!transformedLeft || !transformedRhs) {
        return std::nullopt;
    }
    const std::size_t n = layerStateSize(right) / 2;
    const Matrix downLeft = transformedLeft->block(0, 0, n, n);
    const Matrix upLeft = transformedLeft->block(n, 0, n, n);
    const Matrix downRhs = transformedRhs->block(0, 0, n, rhs.cols());
    const Matrix upRhs = transformedRhs->block(n, 0, n, rhs.cols());
    auto first = trySolveLinearWellConditioned(
        upLeft,
        upRhs,
        schurMinimumReciprocalCondition);
    if (!first) {
        return std::nullopt;
    }
    Matrix second = downRhs;
    for (Complex& value : second.data()) {
        value = -value;
    }
    multiplyAccumulate(second, downLeft, *first);
    return stackUnknownBlocks(*first, second);
}

std::optional<Matrix> solveInterfaceWithLeftCompact(
    const LayerModes& left,
    const Matrix& feedback,
    const Matrix& transmission,
    const LayerModes& right,
    bool includeRightIncidence) {
    const std::size_t n = layerStateSize(left) / 2;
    const bool zeroFeedback = feedback.empty();
    if ((!zeroFeedback && (feedback.rows() != n || feedback.cols() != n)) ||
        transmission.rows() != n || layerStateSize(right) != 2 * n) {
        throw std::invalid_argument("left-compact interface blocks have incompatible shapes");
    }
    const bool useFeedback = !zeroFeedback;

    const auto rightDown =
        transformModeBlockByCompactInverse(left, right, true);
    if (!rightDown) {
        return std::nullopt;
    }
    const Matrix& downCoupling = rightDown->down;
    const Matrix& upCoupling = rightDown->up;
    Matrix schur = downCoupling;
    for (Complex& value : schur.data()) {
        value = -value;
    }
    if (useFeedback) {
        multiplyAccumulate(schur, feedback, upCoupling);
    }

    const std::size_t incidentColumns = transmission.cols();
    const std::size_t totalColumns =
        incidentColumns + (includeRightIncidence ? n : 0);
    Matrix reducedRhs(n, totalColumns);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < incidentColumns; ++column) {
            reducedRhs(row, column) = -transmission(row, column);
        }
    }

    Matrix rightUpUp;
    if (includeRightIncidence) {
        auto transformed =
            transformModeBlockByCompactInverse(left, right, false);
        if (!transformed) {
            return std::nullopt;
        }
        rightUpUp = std::move(transformed->up);
        if (useFeedback) {
            Matrix rightIncident = std::move(transformed->down);
            multiplyAccumulate(
                rightIncident, feedback, rightUpUp, {-1.0, 0.0});
            reducedRhs.setBlock(0, incidentColumns, rightIncident);
        } else {
            reducedRhs.setBlock(0, incidentColumns, transformed->down);
        }
    }

    auto second = trySolveLinearWellConditioned(
        schur,
        reducedRhs,
        schurMinimumReciprocalCondition);
    if (!second) {
        return std::nullopt;
    }
    Matrix first = upCoupling * *second;
    if (includeRightIncidence) {
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < n; ++column) {
                first(row, incidentColumns + column) += rightUpUp(row, column);
            }
        }
    }
    return stackUnknownBlocks(first, *second);
}

bool isUniformTmm(const LayerModes& modes) {
    return modes.kind == LayerComputationKind::UniformTmm;
}

void requirePhaseRange(const LayerModes& layer, std::size_t offset, std::size_t count) {
    if (offset + count > propagationPhase(layer).size()) {
        throw std::invalid_argument("phase block is outside layer mode storage");
    }
}

Matrix leftScaleRowsByPhase(const LayerModes& layer,
                                std::size_t phaseOffset,
                                const Matrix& m) {
    requirePhaseRange(layer, phaseOffset, m.rows());
    const auto& phase = propagationPhase(layer);
    Matrix out(m.rows(), m.cols());
    const auto& src = m.data();
    auto& dst = out.data();
    const std::size_t cols = m.cols();
    for (std::size_t r = 0; r < m.rows(); ++r) {
        const Complex scale = phase[phaseOffset + r];
        const std::size_t base = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            dst[base + c] = scale * src[base + c];
        }
    }
    return out;
}

Matrix rightScaleColumnsByPhase(const LayerModes& layer,
                                    std::size_t phaseOffset,
                                    const Matrix& m) {
    requirePhaseRange(layer, phaseOffset, m.cols());
    const auto& phase = propagationPhase(layer);
    Matrix out(m.rows(), m.cols());
    const auto& src = m.data();
    auto& dst = out.data();
    const std::size_t cols = m.cols();
    for (std::size_t r = 0; r < m.rows(); ++r) {
        const std::size_t base = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            dst[base + c] = src[base + c] * phase[phaseOffset + c];
        }
    }
    return out;
}

Matrix scaleRowsAndColumnsByPhase(const LayerModes& layer,
                                       std::size_t rowPhaseOffset,
                                       std::size_t colPhaseOffset,
                                       const Matrix& m) {
    requirePhaseRange(layer, rowPhaseOffset, m.rows());
    requirePhaseRange(layer, colPhaseOffset, m.cols());
    const auto& phase = propagationPhase(layer);
    Matrix out(m.rows(), m.cols());
    const auto& src = m.data();
    auto& dst = out.data();
    const std::size_t cols = m.cols();
    for (std::size_t r = 0; r < m.rows(); ++r) {
        const Complex rowScale = phase[rowPhaseOffset + r];
        const std::size_t base = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            dst[base + c] = rowScale * src[base + c] * phase[colPhaseOffset + c];
        }
    }
    return out;
}

using Block2 = std::array<Complex, 4>;

struct LocalSMatrix {
    Block2 S11{};
    Block2 S12{};
    Block2 S21{};
    Block2 S22{};
};

Block2 zeroBlock() {
    return {};
}

Block2 identityBlock() {
    return {Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{1.0, 0.0}};
}

Block2 diagonalBlock(Complex a, Complex b) {
    return {a, Complex{}, Complex{}, b};
}

Block2 blockFromMatrix(const Matrix& m) {
    if (m.rows() != 2 || m.cols() != 2) {
        throw std::invalid_argument("2x2 block extraction expects a 2x2 matrix");
    }
    return {m(0, 0), m(0, 1), m(1, 0), m(1, 1)};
}

Block2 blockFromMatrix(const Matrix& m, std::size_t row, std::size_t col) {
    if (row + 2 > m.rows() || col + 2 > m.cols()) {
        throw std::invalid_argument("2x2 block extraction is outside matrix bounds");
    }
    return {m(row, col), m(row, col + 1), m(row + 1, col), m(row + 1, col + 1)};
}

Matrix matrixFromBlock(const Block2& block) {
    Matrix out(2, 2);
    out(0, 0) = block[0];
    out(0, 1) = block[1];
    out(1, 0) = block[2];
    out(1, 1) = block[3];
    return out;
}

Block2 addBlock(const Block2& a, const Block2& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

Block2 multiplyBlock(const Block2& a, const Block2& b) {
    return {
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    };
}

Block2 identityMinusProductBlock(const Block2& a, const Block2& b) {
    const Block2 p = multiplyBlock(a, b);
    return {
        Complex{1.0, 0.0} - p[0],
        -p[1],
        -p[2],
        Complex{1.0, 0.0} - p[3],
    };
}

Block2 solveBlock(const Block2& A, const Block2& B) {
    const Real scale = std::max({
        std::abs(A[0]), std::abs(A[1]), std::abs(A[2]), std::abs(A[3])});
    if (!(scale > Real{0}) || !std::isfinite(scale)) {
        return blockFromMatrix(solveLinear(matrixFromBlock(A), matrixFromBlock(B)));
    }
    const Block2 scaledA{
        A[0] / scale, A[1] / scale, A[2] / scale, A[3] / scale};
    const Block2 scaledB{
        B[0] / scale, B[1] / scale, B[2] / scale, B[3] / scale};
    const Complex det =
        scaledA[0] * scaledA[3] - scaledA[1] * scaledA[2];
    const Real determinantFloor = Real{128} *
        std::numeric_limits<Real>::epsilon();
    if (!(std::abs(det) > determinantFloor) ||
        !std::isfinite(std::abs(det))) {
        return blockFromMatrix(solveLinear(matrixFromBlock(A), matrixFromBlock(B)));
    }
    const Block2 solution{
        (scaledA[3] * scaledB[0] - scaledA[1] * scaledB[2]) / det,
        (scaledA[3] * scaledB[1] - scaledA[1] * scaledB[3]) / det,
        (-scaledA[2] * scaledB[0] + scaledA[0] * scaledB[2]) / det,
        (-scaledA[2] * scaledB[1] + scaledA[0] * scaledB[3]) / det,
    };

    const Block2 residual = addBlock(
        multiplyBlock(A, solution),
        {-B[0], -B[1], -B[2], -B[3]});
    const Real residualNorm = std::max({
        std::abs(residual[0]), std::abs(residual[1]),
        std::abs(residual[2]), std::abs(residual[3])});
    const Real rhsNorm = std::max({
        std::abs(B[0]), std::abs(B[1]), std::abs(B[2]), std::abs(B[3])});
    const Real solutionNorm = std::max({
        std::abs(solution[0]), std::abs(solution[1]),
        std::abs(solution[2]), std::abs(solution[3])});
    const Real residualScale = std::max(
        Real{1}, rhsNorm + scale * solutionNorm);
    if (!std::isfinite(residualNorm) ||
        residualNorm > Real{512} * std::numeric_limits<Real>::epsilon() *
            residualScale) {
        return blockFromMatrix(solveLinear(matrixFromBlock(A), matrixFromBlock(B)));
    }
    return solution;
}

LocalSMatrix identityLocalSmatrix() {
    return {
        zeroBlock(),
        identityBlock(),
        identityBlock(),
        zeroBlock(),
    };
}

LocalSMatrix localStarProduct(const LocalSMatrix& left, const LocalSMatrix& right) {
    const Block2 A1 = identityMinusProductBlock(right.S11, left.S22);
    const Block2 D1_R11_L21 = solveBlock(A1, multiplyBlock(right.S11, left.S21));
    const Block2 D1_R12 = solveBlock(A1, right.S12);
    const Block2 D2_L21 = addBlock(left.S21, multiplyBlock(left.S22, D1_R11_L21));
    const Block2 D2_L22_R12 = multiplyBlock(left.S22, D1_R12);

    return {
        addBlock(left.S11, multiplyBlock(left.S12, D1_R11_L21)),
        multiplyBlock(left.S12, D1_R12),
        multiplyBlock(right.S21, D2_L21),
        addBlock(right.S22, multiplyBlock(right.S21, D2_L22_R12)),
    };
}

LocalSMatrix localFromSmatrix(const SMatrix& S) {
    return {
        blockFromMatrix(S.S11),
        blockFromMatrix(S.S12),
        blockFromMatrix(S.S21),
        blockFromMatrix(S.S22),
    };
}

SMatrix smatrixFromLocal(const LocalSMatrix& local) {
    return {
        matrixFromBlock(local.S11),
        matrixFromBlock(local.S12),
        matrixFromBlock(local.S21),
        matrixFromBlock(local.S22),
    };
}

Matrix identityMinusProduct(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.cols()) {
        throw std::invalid_argument("identity-minus product expects a square result");
    }
    const std::size_t n = a.rows();
    Matrix out = Matrix::identity(n);
    multiplyAccumulate(out, a, b, {-1.0, 0.0});
    return out;
}

Matrix addProduct(const Matrix& base,
                  const Matrix& a,
                  const Matrix& b,
                  Complex alpha = {1.0, 0.0}) {
    Matrix out = base;
    multiplyAccumulate(out, a, b, alpha);
    return out;
}

SMatrix fullStarProduct2x2(const SMatrix& left, const SMatrix& right) {
    return smatrixFromLocal(
        localStarProduct(localFromSmatrix(left), localFromSmatrix(right)));
}

SMatrix fullStarProduct(const SMatrix& left, const SMatrix& right) {
    const std::size_t n = left.S11.rows();
    if (left.S11.cols() != n || left.S12.rows() != n || left.S12.cols() != n ||
        left.S21.rows() != n || left.S21.cols() != n ||
        left.S22.rows() != n || left.S22.cols() != n ||
        right.S11.rows() != n || right.S11.cols() != n ||
        right.S12.rows() != n || right.S12.cols() != n ||
        right.S21.rows() != n || right.S21.cols() != n ||
        right.S22.rows() != n || right.S22.cols() != n) {
        throw std::invalid_argument("full star_product expects matching square S-matrix blocks");
    }

    if (n == 2) {
        return fullStarProduct2x2(left, right);
    }

    const Matrix A1 = identityMinusProduct(right.S11, left.S22);

    Matrix rhs1(n, 2 * n);
    rhs1.setBlock(0, 0, right.S11 * left.S21);
    rhs1.setBlock(0, n, right.S12);
    const Matrix D1 = solveLinear(A1, rhs1);
    const Matrix D1_R11_L21 = D1.block(0, 0, n, n);
    const Matrix D1_R12 = D1.block(0, n, n, n);

    const Matrix D2_L21 = addProduct(left.S21, left.S22, D1_R11_L21);
    const Matrix D2_L22_R12 = left.S22 * D1_R12;

    SMatrix S;
    S.S11 = addProduct(left.S11, left.S12, D1_R11_L21);
    S.S12 = left.S12 * D1_R12;
    S.S21 = right.S21 * D2_L21;
    S.S22 = addProduct(right.S22, right.S21, D2_L22_R12);
    return S;
}

std::array<std::size_t, 4> localStateRows(std::size_t harmonic, std::size_t N) {
    return {harmonic, N + harmonic, 2 * N + harmonic, 3 * N + harmonic};
}

std::array<std::size_t, 2> localDownCols(std::size_t harmonic, std::size_t N) {
    return {harmonic, N + harmonic};
}

std::array<std::size_t, 2> localUpCols(std::size_t harmonic, std::size_t N) {
    return {2 * N + harmonic, 3 * N + harmonic};
}

std::array<std::size_t, 2> localBlockIndices(std::size_t harmonic, std::size_t N) {
    return {harmonic, N + harmonic};
}

bool localModeBasesEquivalent(const LayerModes& left,
                              const LayerModes& right,
                              std::size_t harmonic) {
    const std::size_t N = layerHarmonicCount(left);
    if (layerHarmonicCount(right) != N || harmonic >= N) {
        return false;
    }
    Real scale = Real{1};
    Real difference = Real{0};
    for (std::size_t component = 0; component < 4; ++component) {
        const std::size_t row = component * N + harmonic;
        for (std::size_t localColumn = 0; localColumn < 4; ++localColumn) {
            const std::size_t column = localColumn * N + harmonic;
            const Complex a = modeStateValue(left, row, column);
            const Complex b = modeStateValue(right, row, column);
            scale = std::max({scale, std::abs(a), std::abs(b)});
            difference = std::max(difference, std::abs(a - b));
        }
    }
    return difference <= Real{128} * std::numeric_limits<Real>::epsilon() * scale;
}

std::optional<LocalSMatrix> localInterfaceFromBerremanBasisChange(
    const LayerModes& left,
    const LayerModes& right,
    std::size_t harmonic) {
    if (!hasCompactUniformModes(left) || !hasCompactUniformModes(right) ||
        harmonic >= left.uniformBlocks.size() ||
        harmonic >= right.uniformBlocks.size()) {
        return std::nullopt;
    }
    const auto& rightBlock = right.uniformBlocks[harmonic];
    if (!rightBlock.inverseReliable) {
        return std::nullopt;
    }

    // C = W_right^{-1} W_left is the complete coupled 4x4 Berreman basis
    // change.  Partitioning it into forward/backward 2x2 port blocks is a
    // Schur elimination of that one 4x4 TMM interface; it does not separate
    // TE and TM, and every block may contain cross-polarization coupling.
    const auto& leftValues = left.uniformBlocks[harmonic].values;
    std::array<Complex, 16> basisChange{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            Complex value{};
            for (std::size_t inner = 0; inner < 4; ++inner) {
                value += rightBlock.inverse[row * 4 + inner] *
                    leftValues[inner * 4 + column];
            }
            basisChange[row * 4 + column] = value;
        }
    }
    const auto extract = [&](std::size_t row, std::size_t column) {
        return Block2{
            basisChange[row * 4 + column],
            basisChange[row * 4 + column + 1],
            basisChange[(row + 1) * 4 + column],
            basisChange[(row + 1) * 4 + column + 1],
        };
    };
    const Block2 C11 = extract(0, 0);
    const Block2 C12 = extract(0, 2);
    const Block2 C21 = extract(2, 0);
    const Block2 C22 = extract(2, 2);

    try {
        const Block2 inverseC22 = solveBlock(C22, identityBlock());
        const Block2 minusC21{-C21[0], -C21[1], -C21[2], -C21[3]};
        const Block2 S11 = multiplyBlock(inverseC22, minusC21);
        const Block2 S12 = inverseC22;
        const Block2 S21 = addBlock(C11, multiplyBlock(C12, S11));
        const Block2 S22 = multiplyBlock(C12, inverseC22);
        return LocalSMatrix{S11, S12, S21, S22};
    } catch (const std::runtime_error&) {
        // Fall through to the pivoted full 4x4 continuity solve below.
        return std::nullopt;
    }
}

LocalSMatrix localInterfaceSmatrix(const LayerModes& left,
                                     const LayerModes& right,
    std::size_t harmonic) {
    // At an exact Rayleigh cutoff, forward and backward plane-wave columns
    // coalesce.  An interface between identical modal bases is nevertheless
    // exactly transparent; bypassing the singular amplitude representation
    // gives the continuous limiting S-matrix without adding artificial loss.
    if (localModeBasesEquivalent(left, right, harmonic)) {
        return identityLocalSmatrix();
    }
    if (const auto transformed =
            localInterfaceFromBerremanBasisChange(left, right, harmonic)) {
        return *transformed;
    }
    const std::size_t N = layerHarmonicCount(left);
    const auto rows = localStateRows(harmonic, N);
    const auto downCols = localDownCols(harmonic, N);
    const auto upCols = localUpCols(harmonic, N);

    Matrix A(4, 4);
    Matrix B(4, 4);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t c = 0; c < downCols.size(); ++c) {
            A(r, c) = modeStateValue(left, rows[r], upCols[c]);
            A(r, 2 + c) = -modeStateValue(right, rows[r], downCols[c]);
            B(r, c) = -modeStateValue(left, rows[r], downCols[c]);
            B(r, 2 + c) = modeStateValue(right, rows[r], upCols[c]);
        }
    }

    const Matrix X = solveLinear(A, B);
    return {
        blockFromMatrix(X, 0, 0),
        blockFromMatrix(X, 0, 2),
        blockFromMatrix(X, 2, 0),
        blockFromMatrix(X, 2, 2),
    };
}

LocalSMatrix localPropagationSmatrix(const LayerModes& layer, std::size_t harmonic) {
    const std::size_t N = layerHarmonicCount(layer);
    const auto& phase = propagationPhase(layer);
    if (phase.size() < 4 * N) {
        throw std::invalid_argument("uniform layer propagation phase storage is incomplete");
    }

    return {
        zeroBlock(),
        diagonalBlock(phase[2 * N + harmonic], phase[3 * N + harmonic]),
        diagonalBlock(phase[harmonic], phase[N + harmonic]),
        zeroBlock(),
    };
}

void setGlobalBlock(Matrix& target,
                      const Block2& block,
                      std::size_t harmonic,
                      std::size_t N) {
    const auto idx = localBlockIndices(harmonic, N);
    target(idx[0], idx[0]) = block[0];
    target(idx[0], idx[1]) = block[1];
    target(idx[1], idx[0]) = block[2];
    target(idx[1], idx[1]) = block[3];
}

SMatrix uniformPropagationSmatrix(const LayerModes& layer) {
    const std::size_t stateSize = layerStateSize(layer);
    if (stateSize % 4 != 0) {
        throw std::invalid_argument("uniform propagation state size must be divisible by four");
    }
    const std::size_t N = stateSize / 4;
    SMatrix S;
    S.S11 = Matrix::zeros(2 * N, 2 * N);
    S.S12 = Matrix::zeros(2 * N, 2 * N);
    S.S21 = Matrix::zeros(2 * N, 2 * N);
    S.S22 = Matrix::zeros(2 * N, 2 * N);
    for (std::size_t h = 0; h < N; ++h) {
        const LocalSMatrix local = localPropagationSmatrix(layer, h);
        setGlobalBlock(S.S12, local.S12, h, N);
        setGlobalBlock(S.S21, local.S21, h, N);
    }
    return S;
}

SMatrix propagationSmatrix(const LayerModes& layer) {
    if (isUniformTmm(layer)) {
        return uniformPropagationSmatrix(layer);
    }
    const std::size_t n = layerStateSize(layer) / 2;
    SMatrix S;
    S.S11 = Matrix::zeros(n, n);
    S.S12 = propagationMatrixBlock(layer, n, n);
    S.S21 = propagationMatrixBlock(layer, 0, n);
    S.S22 = Matrix::zeros(n, n);
    return S;
}

SMatrix denseInterfaceSmatrix(const LayerModes& left, const LayerModes& right) {
    const std::size_t stateSize = layerStateSize(left);
    if (layerStateSize(right) != stateSize) {
        throw std::invalid_argument("interface media must have matching modal dimensions");
    }
    const std::size_t modeCount = stateSize / 2;
    Matrix B(stateSize, stateSize);
    setModeBlock(B, 0, 0, left, true, {-1.0, 0.0});
    setModeBlock(B, 0, modeCount, right, false);

    Matrix X;
    if (hasCompactUniformModes(left)) {
        auto reduced = solveInterfaceWithLeftCompact(
            left,
            Matrix{},
            Matrix::identity(modeCount),
            right,
            true);
        if (reduced) {
            X = std::move(*reduced);
        }
    } else if (hasCompactUniformModes(right)) {
        Matrix leftBlock = modeBlockMatrix(left, false);
        auto reducedSolution = solveInterfaceWithRightCompact(leftBlock, right, B);
        if (reducedSolution) {
            X = std::move(*reducedSolution);
        }
    }
    if (X.empty()) {
        Matrix A(stateSize, stateSize);
        setModeBlock(A, 0, 0, left, false);
        setModeBlock(A, 0, modeCount, right, true, {-1.0, 0.0});
        X = solveLinear(A, B);
    }
    SMatrix S;
    S.S11 = X.block(0, 0, modeCount, modeCount);
    S.S12 = X.block(0, modeCount, modeCount, modeCount);
    S.S21 = X.block(modeCount, 0, modeCount, modeCount);
    S.S22 = X.block(modeCount, modeCount, modeCount, modeCount);
    return S;
}

SMatrix interfaceSmatrix(const LayerModes& left, const LayerModes& right) {
    return denseInterfaceSmatrix(left, right);
}

struct PartialSMatrix {
    Matrix R;
    Matrix T;
    Matrix S12;
    Matrix S22;
    std::vector<std::size_t> outputRows;
};

[[nodiscard]] bool usesProjectedOutputs(const std::vector<std::size_t>& outputRows) {
    return !outputRows.empty();
}

[[nodiscard]] std::size_t partialOutputRowCount(const PartialSMatrix& partial,
                                                   std::size_t fullRows) {
    return usesProjectedOutputs(partial.outputRows) ? partial.outputRows.size() : fullRows;
}

[[nodiscard]] Matrix selectMatrixRows(const Matrix& source,
                                        const std::vector<std::size_t>& rows) {
    if (!usesProjectedOutputs(rows)) {
        return source;
    }
    Matrix out(rows.size(), source.cols());
    const auto& src = source.data();
    auto& dst = out.data();
    const std::size_t sourceCols = source.cols();
    const std::size_t outCols = out.cols();
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::size_t sourceRow = rows[r];
        if (sourceRow >= source.rows()) {
            throw std::invalid_argument("selected output row is outside source matrix bounds");
        }
        for (std::size_t c = 0; c < sourceCols; ++c) {
            dst[r * outCols + c] = src[sourceRow * sourceCols + c];
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::size_t> makeRowLookup(
    std::size_t fullRows,
    const std::vector<std::size_t>& outputRows) {
    if (!usesProjectedOutputs(outputRows)) {
        return {};
    }
    std::vector<std::size_t> lookup(fullRows, std::numeric_limits<std::size_t>::max());
    for (std::size_t r = 0; r < outputRows.size(); ++r) {
        const std::size_t sourceRow = outputRows[r];
        if (sourceRow >= fullRows) {
            throw std::invalid_argument("selected output row is outside the mode count");
        }
        lookup[sourceRow] = r;
    }
    return lookup;
}

struct LocalScatteringBlocks {
    std::size_t harmonicCount{};
    std::vector<Block2> S11;
    std::vector<Block2> S12;
    std::vector<Block2> S21;
    std::vector<Block2> S22;
};

void requireLocalSegmentBlocks(const LocalScatteringBlocks& segment) {
    const std::size_t N = segment.harmonicCount;
    if (segment.S11.size() != N || segment.S12.size() != N ||
        segment.S21.size() != N || segment.S22.size() != N) {
        throw std::invalid_argument("uniform segment block storage is incomplete");
    }
}

void setLocalBlockDiagonal(Matrix& target,
                              std::size_t colOffset,
                              const std::vector<Block2>& blocks) {
    const std::size_t N = blocks.size();
    const std::size_t n = 2 * N;
    if (target.rows() != n || colOffset > target.cols() || target.cols() - colOffset < n) {
        throw std::invalid_argument("local block diagonal write is outside matrix bounds");
    }
    for (std::size_t h = 0; h < N; ++h) {
        const Block2& b = blocks[h];
        const std::size_t te = h;
        const std::size_t tm = N + h;
        target(te, colOffset + te) = b[0];
        target(te, colOffset + tm) = b[1];
        target(tm, colOffset + te) = b[2];
        target(tm, colOffset + tm) = b[3];
    }
}

[[nodiscard]] Matrix localResponseColumns(const std::vector<Block2>& blocks,
                                            const std::vector<std::size_t>& channels,
                                            const std::vector<std::size_t>& outputRows = {}) {
    const std::size_t N = blocks.size();
    const std::size_t n = 2 * N;
    const std::vector<std::size_t> lookup = makeRowLookup(n, outputRows);
    if (lookup.empty()) {
        Matrix out(n, channels.size());
        for (std::size_t c = 0; c < channels.size(); ++c) {
            const std::size_t source = channels[c];
            if (source >= n) {
                throw std::invalid_argument(
                    "selected incident channel is outside segment size");
            }
            const std::size_t h = source < N ? source : source - N;
            const std::size_t localCol = source < N ? 0 : 1;
            const std::size_t te = h;
            const std::size_t tm = N + h;
            out(te, c) = blocks[h][localCol];
            out(tm, c) = blocks[h][2 + localCol];
        }
        return out;
    }

    Matrix out(outputRows.size(), channels.size());
    for (std::size_t c = 0; c < channels.size(); ++c) {
        const std::size_t source = channels[c];
        if (source >= n) {
            throw std::invalid_argument("selected incident channel is outside segment size");
        }
        const std::size_t h = source < N ? source : source - N;
        const std::size_t localCol = source < N ? 0 : 1;
        const std::size_t te = h;
        const std::size_t tm = N + h;
        const std::size_t teRow = lookup[te];
        if (teRow != std::numeric_limits<std::size_t>::max()) {
            out(teRow, c) = blocks[h][localCol];
        }
        const std::size_t tmRow = lookup[tm];
        if (tmRow != std::numeric_limits<std::size_t>::max()) {
            out(tmRow, c) = blocks[h][2 + localCol];
        }
    }
    return out;
}

[[nodiscard]] Matrix localSquareRows(const std::vector<Block2>& blocks,
                                       const std::vector<std::size_t>& outputRows = {}) {
    const std::size_t n = 2 * blocks.size();
    Matrix full = Matrix::zeros(n, n);
    setLocalBlockDiagonal(full, 0, blocks);
    return selectMatrixRows(full, outputRows);
}

void addLocalBlockDiagonal(Matrix& target, const std::vector<Block2>& blocks) {
    const std::size_t N = blocks.size();
    const std::size_t n = 2 * N;
    if (target.rows() != n || target.cols() != n) {
        throw std::invalid_argument("local block diagonal add expects a matching square matrix");
    }
    for (std::size_t h = 0; h < N; ++h) {
        const Block2& b = blocks[h];
        const std::size_t te = h;
        const std::size_t tm = N + h;
        target(te, te) += b[0];
        target(te, tm) += b[1];
        target(tm, te) += b[2];
        target(tm, tm) += b[3];
    }
}

std::size_t validateUniformSegment(const std::vector<LayerModes>& media,
                                     std::size_t start,
                                     std::size_t end) {
    if (start >= end || end >= media.size()) {
        throw std::invalid_argument("uniform segment requires start < end within media");
    }

    const std::size_t stateSize = layerStateSize(media[start]);
    if (stateSize % 4 != 0) {
        throw std::invalid_argument("uniform segment state matrix must be square 4N x 4N");
    }
    for (std::size_t i = start; i <= end; ++i) {
        if (!isUniformTmm(media[i])) {
            throw std::invalid_argument("uniform segment contains a patterned medium");
        }
        if (layerStateSize(media[i]) != stateSize) {
            throw std::invalid_argument("uniform segment media must have matching dimensions");
        }
    }
    return stateSize / 4;
}

template <typename Fn>
std::size_t forEachUniformSegmentLocal(const std::vector<LayerModes>& media,
                                           std::size_t start,
                                           std::size_t end,
                                           bool includeStartPropagation,
                                           Fn&& fn) {
    const std::size_t N = validateUniformSegment(media, start, end);
    for (std::size_t h = 0; h < N; ++h) {
        LocalSMatrix local =
            includeStartPropagation ? localPropagationSmatrix(media[start], h)
                                      : identityLocalSmatrix();
        for (std::size_t i = start; i < end; ++i) {
            local = localStarProduct(local, localInterfaceSmatrix(media[i], media[i + 1], h));
            if (i + 1 < end) {
                local = localStarProduct(local, localPropagationSmatrix(media[i + 1], h));
            }
        }
        fn(h, N, local);
    }
    return N;
}

LocalScatteringBlocks uniformSegmentBlocks(const std::vector<LayerModes>& media,
                                             std::size_t start,
                                             std::size_t end,
                                             bool includeStartPropagation) {
    LocalScatteringBlocks out;
    const std::size_t N = forEachUniformSegmentLocal(
        media,
        start,
        end,
        includeStartPropagation,
        [&](std::size_t h, std::size_t harmonicCount, const LocalSMatrix& local) {
            if (h == 0) {
                out.S11.resize(harmonicCount);
                out.S12.resize(harmonicCount);
                out.S21.resize(harmonicCount);
                out.S22.resize(harmonicCount);
            }
            out.S11[h] = local.S11;
            out.S12[h] = local.S12;
            out.S21[h] = local.S21;
            out.S22[h] = local.S22;
        });
    out.harmonicCount = N;
    return out;
}

SMatrix uniformSegmentSmatrix(const std::vector<LayerModes>& media,
                                std::size_t start,
                                std::size_t end,
                                bool includeStartPropagation) {
    const LocalScatteringBlocks segment =
        uniformSegmentBlocks(media, start, end, includeStartPropagation);
    requireLocalSegmentBlocks(segment);
    const std::size_t N = segment.harmonicCount;
    const std::size_t block = 2 * N;
    SMatrix out;
    out.S11 = Matrix::zeros(block, block);
    out.S12 = Matrix::zeros(block, block);
    out.S21 = Matrix::zeros(block, block);
    out.S22 = Matrix::zeros(block, block);
    setLocalBlockDiagonal(out.S11, 0, segment.S11);
    setLocalBlockDiagonal(out.S12, 0, segment.S12);
    setLocalBlockDiagonal(out.S21, 0, segment.S21);
    setLocalBlockDiagonal(out.S22, 0, segment.S22);
    return out;
}

Matrix leftMultiplyLocalBlocks(const std::vector<Block2>& blocks, const Matrix& m) {
    const std::size_t N = blocks.size();
    const std::size_t n = 2 * N;
    if (m.rows() != n) {
        throw std::invalid_argument("block-diagonal product row count must match segment size");
    }
    const std::size_t cols = m.cols();
    Matrix out(n, cols);
    const auto& src = m.data();
    auto& dst = out.data();
    for (std::size_t h = 0; h < N; ++h) {
        const auto& b = blocks[h];
        const std::size_t te = h;
        const std::size_t tm = N + h;
        const std::size_t teBase = te * cols;
        const std::size_t tmBase = tm * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            const Complex xTe = src[teBase + c];
            const Complex xTm = src[tmBase + c];
            dst[teBase + c] = b[0] * xTe + b[1] * xTm;
            dst[tmBase + c] = b[2] * xTe + b[3] * xTm;
        }
    }
    return out;
}

void identityMinusLeftMultiplyLocalBlocks(
    Matrix& out,
    const std::vector<Block2>& blocks,
    const Matrix& m) {
    const std::size_t N = blocks.size();
    const std::size_t n = 2 * N;
    if (m.rows() != n || m.cols() != n) {
        throw std::invalid_argument("block-diagonal feedback product must be square");
    }
    out.resizeForOverwrite(n, n);
    const auto& src = m.data();
    auto& dst = out.data();
    for (std::size_t h = 0; h < N; ++h) {
        const auto& b = blocks[h];
        const std::size_t te = h;
        const std::size_t tm = N + h;
        const std::size_t teBase = te * n;
        const std::size_t tmBase = tm * n;
        for (std::size_t c = 0; c < n; ++c) {
            const Complex xTe = src[teBase + c];
            const Complex xTm = src[tmBase + c];
            dst[teBase + c] = -(b[0] * xTe + b[1] * xTm);
            dst[tmBase + c] = -(b[2] * xTe + b[3] * xTm);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        dst[i * n + i] += Complex{1.0, 0.0};
    }
}

struct UniformCascadeWorkspace {
    Matrix feedbackOperator;
    Matrix rhs;
};

thread_local UniformCascadeWorkspace uniformCascadeWorkspace;

PartialSMatrix partialFromTerminalBlocks(
    const LocalScatteringBlocks& segment,
    const std::vector<std::size_t>& channels,
    const std::vector<std::size_t>& bottomChannels,
    const std::vector<std::size_t>& outputRows = {}) {
    requireLocalSegmentBlocks(segment);
    PartialSMatrix out;
    out.R = localResponseColumns(segment.S11, channels, outputRows);
    out.T = localResponseColumns(segment.S21, channels, outputRows);
    out.S12 = localResponseColumns(segment.S12, bottomChannels, outputRows);
    out.outputRows = outputRows;
    return out;
}

PartialSMatrix partialFromUniformSegment(
    const LocalScatteringBlocks& segment,
    const std::vector<std::size_t>& channels,
    const std::vector<std::size_t>& outputRows = {}) {
    const std::size_t n = 2 * segment.harmonicCount;
    PartialSMatrix out;
    out.R = localResponseColumns(segment.S11, channels, outputRows);
    out.T = localResponseColumns(segment.S21, channels);
    out.S12 = localSquareRows(segment.S12, outputRows);
    out.S22 = Matrix::zeros(n, n);
    setLocalBlockDiagonal(out.S22, 0, segment.S22);
    out.outputRows = outputRows;
    return out;
}

void requirePartialForUniformSegment(const PartialSMatrix& partial,
                                         const LocalScatteringBlocks& segment,
                                         const char* op) {
    requireLocalSegmentBlocks(segment);
    const std::size_t n = 2 * segment.harmonicCount;
    const std::size_t outputRows = partialOutputRowCount(partial, n);
    if (partial.R.rows() != outputRows || partial.T.rows() != n ||
        partial.S12.rows() != outputRows || partial.S12.cols() != n ||
        partial.S22.rows() != n || partial.S22.cols() != n ||
        partial.R.cols() != partial.T.cols()) {
        throw std::invalid_argument(op);
    }
}

void cascadeUniformSegment(PartialSMatrix& partial,
                             bool& initialized,
                             const LocalScatteringBlocks& segment,
                             const std::vector<std::size_t>& channels,
                             const std::vector<std::size_t>& outputRows = {}) {
    if (!initialized) {
        partial = partialFromUniformSegment(segment, channels, outputRows);
        initialized = true;
        return;
    }
    requirePartialForUniformSegment(
        partial,
        segment,
        "uniform segment cascade expects matching dimensions");

    const std::size_t n = 2 * segment.harmonicCount;
    const std::size_t incidentCount = partial.T.cols();
    auto& workspace = uniformCascadeWorkspace;
    identityMinusLeftMultiplyLocalBlocks(
        workspace.feedbackOperator, segment.S11, partial.S22);
    workspace.rhs.reset(n, incidentCount + n);
    workspace.rhs.setBlock(
        0, 0, leftMultiplyLocalBlocks(segment.S11, partial.T));
    setLocalBlockDiagonal(workspace.rhs, incidentCount, segment.S12);

    const Matrix D1 = solveLinear(workspace.feedbackOperator, workspace.rhs);
    const Matrix D1_R11_T = D1.block(0, 0, n, incidentCount);
    const Matrix D1_R12 = D1.block(0, incidentCount, n, n);

    const Matrix D2_T = addProduct(partial.T, partial.S22, D1_R11_T);
    const Matrix D2_S22_R12 = partial.S22 * D1_R12;

    PartialSMatrix out;
    out.R = addProduct(partial.R, partial.S12, D1_R11_T);
    out.T = leftMultiplyLocalBlocks(segment.S21, D2_T);
    out.S12 = partial.S12 * D1_R12;
    out.S22 = leftMultiplyLocalBlocks(segment.S21, D2_S22_R12);
    addLocalBlockDiagonal(out.S22, segment.S22);
    out.outputRows = partial.outputRows;
    partial = std::move(out);
}

void cascadeTerminalUniformSegment(PartialSMatrix& partial,
                                      bool& initialized,
                                      const LocalScatteringBlocks& segment,
                                      const std::vector<std::size_t>& channels,
                                      const std::vector<std::size_t>& bottomChannels,
                                      const std::vector<std::size_t>& outputRows = {}) {
    if (!initialized) {
        partial = partialFromTerminalBlocks(
            segment,
            channels,
            bottomChannels,
            outputRows);
        initialized = true;
        return;
    }
    requirePartialForUniformSegment(
        partial,
        segment,
        "terminal uniform segment cascade expects matching dimensions");

    auto& workspace = uniformCascadeWorkspace;
    identityMinusLeftMultiplyLocalBlocks(
        workspace.feedbackOperator, segment.S11, partial.S22);
    workspace.rhs.reset(
        workspace.feedbackOperator.rows(),
        partial.T.cols() + bottomChannels.size());
    workspace.rhs.setBlock(
        0,
        0,
        leftMultiplyLocalBlocks(segment.S11, partial.T));
    if (!bottomChannels.empty()) {
        workspace.rhs.setBlock(
            0,
            partial.T.cols(),
            localResponseColumns(segment.S12, bottomChannels));
    }
    const Matrix solved = solveLinear(
        workspace.feedbackOperator, workspace.rhs);
    const Matrix feedback = solved.block(
        0,
        0,
        workspace.feedbackOperator.rows(),
        partial.T.cols());

    PartialSMatrix out;
    const Matrix reflected = addProduct(partial.R, partial.S12, feedback);
    const Matrix transmittedFeedback =
        addProduct(partial.T, partial.S22, feedback);
    const Matrix fullT = leftMultiplyLocalBlocks(
        segment.S21,
        transmittedFeedback);
    out.R = partial.outputRows.empty() ? selectMatrixRows(reflected, outputRows) : reflected;
    out.T = selectMatrixRows(fullT, outputRows);
    if (!bottomChannels.empty()) {
        out.S12 = partial.S12 * solved.block(
            0,
            partial.T.cols(),
            workspace.feedbackOperator.rows(),
            bottomChannels.size());
    }
    out.outputRows = partial.outputRows.empty() ? outputRows : partial.outputRows;
    partial = std::move(out);
}

std::size_t uniformRunEnd(const std::vector<LayerModes>& media, std::size_t start) {
    std::size_t end = start;
    while (end + 1 < media.size() && isUniformTmm(media[end + 1])) {
        ++end;
    }
    return end;
}

PartialSMatrix partialFromDenseInterface(
    const LayerModes& left,
    const LayerModes& right,
    const std::vector<std::size_t>& channels,
    const std::vector<std::size_t>& outputRows = {},
    bool terminalOutput = false,
    const std::vector<std::size_t>& bottomChannels = {}) {
    const std::size_t stateSize = layerStateSize(left);
    if (layerStateSize(right) != stateSize) {
        throw std::invalid_argument("interface media must have matching modal dimensions");
    }
    const std::size_t modeCount = stateSize / 2;
    if (!terminalOutput && !outputRows.empty() && hasCompactUniformModes(left)) {
        const auto rightDown =
            transformModeBlockByCompactInverse(left, right, true);
        const auto rightUp =
            transformModeBlockByCompactInverse(left, right, false);
        if (rightDown && rightUp) {
            Matrix schur = rightDown->down;
            for (Complex& value : schur.data()) {
                value = -value;
            }

            Matrix reducedRhs(modeCount, channels.size() + modeCount);
            for (std::size_t c = 0; c < channels.size(); ++c) {
                const std::size_t source = channels[c];
                if (source >= modeCount) {
                    throw std::invalid_argument(
                        "selected incident channel is outside interface size");
                }
                reducedRhs(source, c) = Complex{-1.0, 0.0};
            }
            reducedRhs.setBlock(0, channels.size(), rightUp->down);

            if (auto second = trySolveLinearWellConditioned(
                    schur,
                    reducedRhs,
                    schurMinimumReciprocalCondition)) {
                Matrix projectedFirst =
                    selectMatrixRows(rightDown->up, outputRows) * *second;
                const std::size_t sourceCols = rightUp->up.cols();
                auto& firstData = projectedFirst.data();
                const auto& upData = rightUp->up.data();
                const std::size_t firstCols = projectedFirst.cols();
                for (std::size_t r = 0; r < outputRows.size(); ++r) {
                    const std::size_t sourceRow = outputRows[r];
                    if (sourceRow >= modeCount) {
                        throw std::invalid_argument(
                            "selected output row is outside interface size");
                    }
                    for (std::size_t c = 0; c < modeCount; ++c) {
                        firstData[r * firstCols + channels.size() + c] +=
                            upData[sourceRow * sourceCols + c];
                    }
                }

                PartialSMatrix out;
                out.R = projectedFirst.block(
                    0, 0, outputRows.size(), channels.size());
                out.T = second->block(
                    0, 0, modeCount, channels.size());
                out.S12 = projectedFirst.block(
                    0, channels.size(), outputRows.size(), modeCount);
                out.S22 = second->block(
                    0, channels.size(), modeCount, modeCount);
                out.outputRows = outputRows;
                return out;
            }
        }
    }
    const std::size_t rightIncidentCount = terminalOutput
        ? bottomChannels.size()
        : modeCount;
    Matrix rhs(stateSize, channels.size() + rightIncidentCount);
    setSelectedModeColumns(rhs, 0, 0, left, true, channels, {-1.0, 0.0});
    if (terminalOutput) {
        setSelectedModeColumns(
            rhs,
            0,
            channels.size(),
            right,
            false,
            bottomChannels);
    } else {
        setModeBlock(rhs, 0, channels.size(), right, false);
    }

    Matrix X;
    if (hasCompactUniformModes(left) &&
        (!terminalOutput || bottomChannels.empty())) {
        Matrix selected(modeCount, channels.size());
        for (std::size_t c = 0; c < channels.size(); ++c) {
            if (channels[c] >= modeCount) {
                throw std::invalid_argument("selected incident channel is outside interface size");
            }
            selected(channels[c], c) = Complex{1.0, 0.0};
        }
        auto reduced = solveInterfaceWithLeftCompact(
            left,
            Matrix{},
            selected,
            right,
            !terminalOutput);
        if (reduced) {
            X = std::move(*reduced);
        }
    } else if (hasCompactUniformModes(right)) {
        auto reduced = solveInterfaceWithRightCompact(
            modeBlockMatrix(left, false),
            right,
            rhs);
        if (reduced) {
            X = std::move(*reduced);
        }
    }
    if (X.empty()) {
        Matrix A(stateSize, stateSize);
        setModeBlock(A, 0, 0, left, false);
        setModeBlock(A, 0, modeCount, right, true, {-1.0, 0.0});
        X = solveLinear(A, rhs);
    }
    PartialSMatrix out;
    out.R = selectMatrixRows(
        X.block(0, 0, modeCount, channels.size()),
        outputRows);
    const Matrix transmitted = X.block(modeCount, 0, modeCount, channels.size());
    out.T = terminalOutput ? selectMatrixRows(transmitted, outputRows) : transmitted;
    out.S12 = selectMatrixRows(
        X.block(0, channels.size(), modeCount, rightIncidentCount),
        outputRows);
    if (!terminalOutput) {
        out.S22 = X.block(modeCount, channels.size(), modeCount, modeCount);
    }
    out.outputRows = outputRows;
    return out;
}

void cascadePropagation(PartialSMatrix& partial, const LayerModes& layer) {
    const std::size_t stateSize = layerStateSize(layer);
    const std::size_t n = stateSize / 2;
    const std::size_t outputRows = partialOutputRowCount(partial, n);
    if (partial.T.rows() != n ||
        partial.S12.rows() != outputRows || partial.S12.cols() != n ||
        partial.S22.rows() != n || partial.S22.cols() != n) {
        throw std::invalid_argument("partial propagation cascade expects matching dimensions");
    }

    partial.T = leftScaleRowsByPhase(layer, 0, partial.T);
    partial.S12 = rightScaleColumnsByPhase(layer, n, partial.S12);
    partial.S22 = scaleRowsAndColumnsByPhase(layer, 0, n, partial.S22);
}

PartialSMatrix cascadeDenseInterface(const PartialSMatrix& left,
                                       const LayerModes& interfaceLeft,
                                       const LayerModes& interfaceRight) {
    const std::size_t stateSize = layerStateSize(interfaceLeft);
    if (layerStateSize(interfaceRight) != stateSize) {
        throw std::invalid_argument("interface media must have matching modal dimensions");
    }
    const std::size_t modeCount = stateSize / 2;
    const std::size_t outputRows = partialOutputRowCount(left, modeCount);
    if (left.R.rows() != outputRows || left.T.rows() != modeCount ||
        left.S12.rows() != outputRows || left.S12.cols() != modeCount ||
        left.S22.rows() != modeCount || left.S22.cols() != modeCount ||
        left.R.cols() != left.T.cols()) {
        throw std::invalid_argument("partial interface cascade expects matching dimensions");
    }

    Matrix X;
    if (hasCompactUniformModes(interfaceLeft)) {
        auto reduced = solveInterfaceWithLeftCompact(
            interfaceLeft,
            left.S22,
            left.T,
            interfaceRight,
            true);
        if (reduced) {
            X = std::move(*reduced);
        }
    }

    Matrix projected;
    Matrix leftBlock;
    Matrix rhs;
    if (X.empty()) {
        projected = downModesMultiplyColumns(interfaceLeft, left.S22, left.T);
        leftBlock = Matrix(stateSize, modeCount);
        setMatrixColumns(leftBlock, 0, projected, 0, modeCount);
        addModeBlock(leftBlock, 0, 0, interfaceLeft, false);

        rhs = Matrix(stateSize, left.T.cols() + modeCount);
        setMatrixColumns(rhs, 0, projected, modeCount, left.T.cols(), {-1.0, 0.0});
        setModeBlock(rhs, 0, left.T.cols(), interfaceRight, false);
        if (hasCompactUniformModes(interfaceRight)) {
            auto reduced = solveInterfaceWithRightCompact(
                leftBlock,
                interfaceRight,
                rhs);
            if (reduced) {
                X = std::move(*reduced);
            }
        }
    }
    if (X.empty()) {
        Matrix A(stateSize, stateSize);
        A.setBlock(0, 0, leftBlock);
        setModeBlock(A, 0, modeCount, interfaceRight, true, {-1.0, 0.0});
        X = solveLinear(A, rhs);
    }
    const Matrix reflectedFeedback = X.block(0, 0, modeCount, left.T.cols());
    const Matrix rightFeedback = X.block(0, left.T.cols(), modeCount, modeCount);

    PartialSMatrix out;
    out.R = addProduct(left.R, left.S12, reflectedFeedback);
    out.T = X.block(modeCount, 0, modeCount, left.T.cols());
    out.S12 = left.S12 * rightFeedback;
    out.S22 = X.block(modeCount, left.T.cols(), modeCount, modeCount);
    out.outputRows = left.outputRows;
    return out;
}

PartialSMatrix cascadeTerminalDenseInterface(const PartialSMatrix& left,
                                                const LayerModes& interfaceLeft,
                                                const LayerModes& interfaceRight,
                                                const std::vector<std::size_t>& bottomChannels,
                                                const std::vector<std::size_t>& outputRows = {}) {
    const std::size_t stateSize = layerStateSize(interfaceLeft);
    if (layerStateSize(interfaceRight) != stateSize) {
        throw std::invalid_argument("interface media must have matching modal dimensions");
    }
    const std::size_t modeCount = stateSize / 2;
    const std::size_t outputRowCount = partialOutputRowCount(left, modeCount);
    if (left.R.rows() != outputRowCount || left.T.rows() != modeCount ||
        left.S12.rows() != outputRowCount || left.S12.cols() != modeCount ||
        left.S22.rows() != modeCount || left.S22.cols() != modeCount ||
        left.R.cols() != left.T.cols()) {
        throw std::invalid_argument(
            "terminal partial interface cascade expects matching dimensions");
    }

    Matrix X;
    if (hasCompactUniformModes(interfaceLeft) && bottomChannels.empty()) {
        auto reduced = solveInterfaceWithLeftCompact(
            interfaceLeft,
            left.S22,
            left.T,
            interfaceRight,
            false);
        if (reduced) {
            X = std::move(*reduced);
        }
    }

    Matrix projected;
    Matrix leftBlock;
    Matrix rhs;
    if (X.empty()) {
        projected = downModesMultiplyColumns(interfaceLeft, left.S22, left.T);
        leftBlock = Matrix(stateSize, modeCount);
        setMatrixColumns(leftBlock, 0, projected, 0, modeCount);
        addModeBlock(leftBlock, 0, 0, interfaceLeft, false);
        rhs = Matrix(stateSize, left.T.cols() + bottomChannels.size());
        setMatrixColumns(rhs, 0, projected, modeCount, left.T.cols(), {-1.0, 0.0});
        setSelectedModeColumns(
            rhs,
            0,
            left.T.cols(),
            interfaceRight,
            false,
            bottomChannels);
        if (hasCompactUniformModes(interfaceRight)) {
            auto reduced = solveInterfaceWithRightCompact(
                leftBlock,
                interfaceRight,
                rhs);
            if (reduced) {
                X = std::move(*reduced);
            }
        }
    }
    if (X.empty()) {
        Matrix A(stateSize, stateSize);
        A.setBlock(0, 0, leftBlock);
        setModeBlock(A, 0, modeCount, interfaceRight, true, {-1.0, 0.0});
        X = solveLinear(A, rhs);
    }
    const Matrix reflectedFeedback = X.block(0, 0, modeCount, left.T.cols());

    PartialSMatrix out;
    const Matrix reflected = addProduct(left.R, left.S12, reflectedFeedback);
    const Matrix fullT = X.block(modeCount, 0, modeCount, left.T.cols());
    out.R = left.outputRows.empty() ? selectMatrixRows(reflected, outputRows) : reflected;
    out.T = selectMatrixRows(fullT, outputRows);
    if (!bottomChannels.empty()) {
        out.S12 = left.S12 * X.block(
            0,
            left.T.cols(),
            modeCount,
            bottomChannels.size());
    }
    out.outputRows = left.outputRows.empty() ? outputRows : left.outputRows;
    return out;
}

bool cascadeTerminalUniformTailThroughDenseInterface(
    PartialSMatrix& partial,
    const LayerModes& interfaceLeft,
    const LayerModes& interfaceRight,
    const LocalScatteringBlocks& tail,
    const std::vector<std::size_t>& bottomChannels,
    const std::vector<std::size_t>& outputRows = {}) {
    requireLocalSegmentBlocks(tail);
    const std::size_t stateSize = layerStateSize(interfaceLeft);
    if (layerStateSize(interfaceRight) != stateSize || stateSize % 2 != 0) {
        throw std::invalid_argument(
            "loaded terminal interface media must have matching modal dimensions");
    }
    const std::size_t modeCount = stateSize / 2;
    const std::size_t outputRowCount = partialOutputRowCount(partial, modeCount);
    if (tail.harmonicCount * 2 != modeCount ||
        partial.R.rows() != outputRowCount || partial.T.rows() != modeCount ||
        partial.S12.rows() != outputRowCount ||
        partial.S12.cols() != modeCount ||
        partial.S22.rows() != modeCount || partial.S22.cols() != modeCount ||
        partial.R.cols() != partial.T.cols()) {
        throw std::invalid_argument(
            "loaded terminal interface blocks have incompatible dimensions");
    }
    if (!hasCompactUniformModes(interfaceRight)) {
        return false;
    }

    // Let a/b denote down/up amplitudes at the patterned-side interface and
    // c/d the corresponding amplitudes in the first uniform medium.  The
    // uniform tail supplies d = G11*c + G12*b_bottom.  Transforming field
    // continuity by W_uniform^-1 and eliminating c gives one Schur system for
    // b.  This is algebraically the same Redheffer cascade as first forming a
    // dense interface S matrix and then attaching the tail, but it requires
    // only one modeCount factorization instead of two.
    const auto transformedLeftDown =
        transformModeBlockByCompactInverse(interfaceRight, interfaceLeft, true);
    const auto transformedLeftUp =
        transformModeBlockByCompactInverse(interfaceRight, interfaceLeft, false);
    if (!transformedLeftDown || !transformedLeftUp) {
        return false;
    }

    const Matrix& Xdown = transformedLeftDown->down;
    const Matrix& Xup = transformedLeftDown->up;
    const Matrix& Ydown = transformedLeftUp->down;
    const Matrix& Yup = transformedLeftUp->up;

    const Matrix downLoaded = addProduct(Ydown, Xdown, partial.S22);
    const Matrix upLoaded = addProduct(Yup, Xup, partial.S22);
    Matrix schur = upLoaded - leftMultiplyLocalBlocks(tail.S11, downLoaded);

    Matrix topDriving = leftMultiplyLocalBlocks(tail.S11, Xdown) - Xup;
    const std::size_t incidentCount = partial.T.cols();
    Matrix rhs(modeCount, incidentCount + bottomChannels.size());
    rhs.setBlock(0, 0, topDriving * partial.T);
    if (!bottomChannels.empty()) {
        rhs.setBlock(
            0,
            incidentCount,
            localResponseColumns(tail.S12, bottomChannels));
    }

    auto solved = trySolveLinearWellConditioned(
        schur,
        rhs,
        schurMinimumReciprocalCondition);
    if (!solved) {
        return false;
    }
    const Matrix feedback = solved->block(
        0, 0, modeCount, incidentCount);

    Matrix transmittedIntoTail = Xdown * partial.T;
    multiplyAccumulate(transmittedIntoTail, downLoaded, feedback);

    PartialSMatrix out;
    const Matrix reflected = addProduct(
        partial.R,
        partial.S12,
        feedback);
    const Matrix transmitted = leftMultiplyLocalBlocks(
        tail.S21,
        transmittedIntoTail);
    out.R = partial.outputRows.empty()
        ? selectMatrixRows(reflected, outputRows)
        : reflected;
    out.T = selectMatrixRows(transmitted, outputRows);
    if (!bottomChannels.empty()) {
        out.S12 = partial.S12 * solved->block(
            0,
            incidentCount,
            modeCount,
            bottomChannels.size());
    }
    out.outputRows = partial.outputRows.empty()
        ? outputRows
        : partial.outputRows;
    partial = std::move(out);
    return true;
}

PartialSMatrix stackToPartialSmatrix(const std::vector<LayerModes>& media,
                                        const std::vector<std::size_t>& channels,
                                        const std::vector<std::size_t>& outputRows = {},
                                        const std::vector<std::size_t>& bottomChannels = {}) {
    if (media.size() < 2) {
        throw std::invalid_argument("partial stack solve expects at least two media");
    }
    const std::size_t last = media.size() - 1;
    PartialSMatrix S;
    bool initialized = false;
    std::size_t pos = 0;
    while (pos < last) {
        if (isUniformTmm(media[pos])) {
            const std::size_t runEnd = uniformRunEnd(media, pos);
            if (runEnd > pos) {
                const bool includeStartPropagation = pos != 0;
                const LocalScatteringBlocks segment =
                    uniformSegmentBlocks(media, pos, runEnd, includeStartPropagation);
                if (runEnd == last) {
                    cascadeTerminalUniformSegment(
                        S,
                        initialized,
                        segment,
                        channels,
                        bottomChannels,
                        outputRows);
                } else {
                    cascadeUniformSegment(S, initialized, segment, channels, outputRows);
                }
                pos = runEnd;
                continue;
            }
        }

        if (pos != 0) {
            if (!initialized) {
                throw std::logic_error("stack propagation encountered before first interface");
            }
            cascadePropagation(S, media[pos]);
        }
        const bool terminal = pos + 1 == last;
        if (initialized && !terminal && isUniformTmm(media[pos + 1])) {
            const std::size_t tailEnd = uniformRunEnd(media, pos + 1);
            if (tailEnd == last) {
                const LocalScatteringBlocks tail = uniformSegmentBlocks(
                    media,
                    pos + 1,
                    tailEnd,
                    true);
                if (cascadeTerminalUniformTailThroughDenseInterface(
                        S,
                        media[pos],
                        media[pos + 1],
                        tail,
                        bottomChannels,
                        outputRows)) {
                    pos = last;
                    continue;
                }
            }
        }
        if (initialized) {
            S = terminal
                ? cascadeTerminalDenseInterface(
                    S,
                    media[pos],
                    media[pos + 1],
                    bottomChannels,
                    outputRows)
                : cascadeDenseInterface(S, media[pos], media[pos + 1]);
        } else {
            S = partialFromDenseInterface(
                media[pos],
                media[pos + 1],
                channels,
                outputRows,
                terminal,
                bottomChannels);
            initialized = true;
        }
        ++pos;
    }
    if (!initialized) {
        throw std::logic_error("stack solve did not produce a scattering matrix");
    }
    return S;
}

SMatrix stackToFullSmatrix(const std::vector<LayerModes>& media) {
    if (media.size() < 2) {
        throw std::invalid_argument("full stack S-parameter solve expects at least two media");
    }

    SMatrix S;
    bool initialized = false;
    const auto cascade = [&](const SMatrix& next) {
        if (initialized) {
            S = fullStarProduct(S, next);
        } else {
            S = next;
            initialized = true;
        }
    };

    const std::size_t last = media.size() - 1;
    std::size_t pos = 0;
    while (pos < last) {
        if (isUniformTmm(media[pos])) {
            const std::size_t runEnd = uniformRunEnd(media, pos);
            if (runEnd > pos) {
                cascade(uniformSegmentSmatrix(media, pos, runEnd, pos != 0));
                pos = runEnd;
                continue;
            }
        }
        if (pos != 0) {
            cascade(propagationSmatrix(media[pos]));
        }
        cascade(interfaceSmatrix(media[pos], media[pos + 1]));
        ++pos;
    }
    if (!initialized) {
        throw std::logic_error("full stack solve did not produce a scattering matrix");
    }
    return S;
}

struct DirectionalRT {
    Matrix R;
    Matrix T;
};

Matrix swapModeHalves(const Matrix& amplitudes) {
    if (amplitudes.rows() % 2 != 0) {
        throw std::invalid_argument("modal amplitudes must contain two equal direction blocks");
    }
    const std::size_t block = amplitudes.rows() / 2;
    Matrix out(amplitudes.rows(), amplitudes.cols());
    out.setBlock(0, 0, amplitudes.block(block, 0, block, amplitudes.cols()));
    out.setBlock(block, 0, amplitudes.block(0, 0, block, amplitudes.cols()));
    return out;
}

Matrix solveDirectionalModeBasis(const LayerModes& modes,
                                 const Matrix& rhs,
                                 bool reverse) {
    const std::size_t stateSize = layerStateSize(modes);
    if (stateSize == 0 || stateSize % 2 != 0 || rhs.rows() != stateSize) {
        throw std::invalid_argument(
            "enhanced T-matrix basis solve received incompatible modal dimensions");
    }

    Matrix amplitudes;
    if (auto compact = applyCompactUniformInverse(modes, rhs)) {
        amplitudes = std::move(*compact);
    } else {
        amplitudes = solveLinear(materializeModeMatrix(modes), rhs);
    }
    return reverse ? swapModeHalves(amplitudes) : amplitudes;
}

Matrix directionalModeBlock(const LayerModes& modes, bool forward, bool reverse) {
    return modeBlockMatrix(modes, reverse ? !forward : forward);
}

Matrix directionalPropagationBlock(const LayerModes& modes,
                                   bool forward,
                                   bool reverse) {
    const std::size_t modeCount = layerStateSize(modes) / 2;
    const bool originalDown = reverse ? !forward : forward;
    return propagationMatrixBlock(
        modes,
        originalDown ? std::size_t{0} : modeCount,
        modeCount);
}

DirectionalRT enhancedTransmittanceDirectional(
    const std::vector<LayerModes>& media,
    bool reverse) {
    if (media.size() < 2) {
        throw std::invalid_argument(
            "enhanced T-matrix stack solve expects at least two exterior media");
    }
    const std::size_t stateSize = layerStateSize(media.front());
    if (stateSize == 0 || stateSize % 2 != 0) {
        throw std::invalid_argument(
            "enhanced T-matrix stack solve requires a nonempty even modal state");
    }
    for (const LayerModes& modes : media) {
        if (layerStateSize(modes) != stateSize) {
            throw std::invalid_argument(
                "enhanced T-matrix stack media must have matching modal dimensions");
        }
    }

    const std::size_t modeCount = stateSize / 2;
    const std::size_t last = media.size() - 1;
    const std::size_t incidentIndex = reverse ? last : 0;
    const std::size_t exitIndex = reverse ? 0 : last;
    Matrix effectiveField = directionalModeBlock(media[exitIndex], true, reverse);
    Matrix transmissionMap = Matrix::identity(modeCount);

    // Moharam et al., JOSA A 12, 1077-1086 (1995), Eqs. (28)-(32),
    // doi:10.1364/JOSAA.12.001077.
    // The factors X_l remain on solve right-hand sides, so no exponentially
    // large X_l^-1 is ever formed. Separate forward/backward phase blocks make
    // the recurrence valid for this backend's general 4N vector modal basis.
    const auto loadLayer = [&](const LayerModes& layer) {
        const Matrix matched = solveDirectionalModeBasis(layer, effectiveField, reverse);
        const Matrix a = matched.block(0, 0, modeCount, modeCount);
        const Matrix b = matched.block(modeCount, 0, modeCount, modeCount);
        const Matrix forwardPhase =
            directionalPropagationBlock(layer, true, reverse);
        const Matrix backwardPhase =
            directionalPropagationBlock(layer, false, reverse);
        const Matrix normalizedTransmission = solveLinear(a, forwardPhase);
        const Matrix loadedBackward =
            backwardPhase * (b * normalizedTransmission);
        effectiveField = addProduct(
            directionalModeBlock(layer, true, reverse),
            directionalModeBlock(layer, false, reverse),
            loadedBackward);
        transmissionMap = transmissionMap * normalizedTransmission;
    };

    if (reverse) {
        for (std::size_t index = 1; index < last; ++index) {
            loadLayer(media[index]);
        }
    } else {
        for (std::size_t index = last; index-- > 1;) {
            loadLayer(media[index]);
        }
    }

    const Matrix exteriorMatched =
        solveDirectionalModeBasis(media[incidentIndex], effectiveField, reverse);
    const Matrix a = exteriorMatched.block(0, 0, modeCount, modeCount);
    const Matrix b = exteriorMatched.block(modeCount, 0, modeCount, modeCount);
    const Matrix normalizedTransmission = solveLinear(a, Matrix::identity(modeCount));
    return {
        b * normalizedTransmission,
        transmissionMap * normalizedTransmission,
    };
}

SMatrix stackToEnhancedTransmittanceSmatrix(const std::vector<LayerModes>& media) {
    DirectionalRT top = enhancedTransmittanceDirectional(media, false);
    DirectionalRT bottom = enhancedTransmittanceDirectional(media, true);
    return {
        std::move(top.R),
        std::move(bottom.T),
        std::move(top.T),
        std::move(bottom.R),
    };
}

ZeroOrderAmplitudes stackToZeroOrderAmplitudes(const std::vector<LayerModes>& media,
                                                   int harmonicCount,
                                                   int centerIdx) {
    if (media.size() < 2) {
        throw std::invalid_argument("zero-order amplitude solve expects at least two media");
    }
    if (harmonicCount <= 0 || centerIdx < 0 || centerIdx >= harmonicCount) {
        throw std::invalid_argument("zero-order amplitude solve received invalid harmonic indices");
    }

    const std::size_t N = static_cast<std::size_t>(harmonicCount);
    const std::size_t te0 = static_cast<std::size_t>(centerIdx);
    const std::size_t tm0 = N + static_cast<std::size_t>(centerIdx);
    const PartialSMatrix S = stackToPartialSmatrix(media, {te0, tm0});

    ZeroOrderAmplitudes out;
    out.R = Matrix(2, 2);
    out.T = Matrix(2, 2);
    out.R(0, 0) = S.R(te0, 0);
    out.R(0, 1) = S.R(te0, 1);
    out.R(1, 0) = S.R(tm0, 0);
    out.R(1, 1) = S.R(tm0, 1);
    out.T(0, 0) = S.T(te0, 0);
    out.T(0, 1) = S.T(te0, 1);
    out.T(1, 0) = S.T(tm0, 0);
    out.T(1, 1) = S.T(tm0, 1);
    return out;
}

template <bool IncludeOrders>
using DiffractionOutput = std::conditional_t<IncludeOrders, DiffractionResult, DiffractionTotals>;

struct IncidentChannelPlan {
    std::array<std::size_t, 2> channels{};
    std::array<std::size_t, 2> columns{};
    std::array<Complex, 2> amplitudes{};
    std::size_t count{};
};

struct SpectrumSolvePlan {
    std::vector<std::size_t> channels;
    std::vector<IncidentChannelPlan> incidents;
};

SpectrumSolvePlan makeSpectrumSolvePlan(const std::vector<Polarization>& polarizations,
                                           std::size_t te0,
                                           std::size_t tm0) {
    SpectrumSolvePlan plan;
    plan.channels.reserve(2);
    bool needTe = false;
    bool needTm = false;
    for (Polarization pol : polarizations) {
        const PolarizationAmplitudes amplitudes = polarizationAmplitudes(pol);
        needTe = needTe || std::abs(amplitudes.te) > Real{0};
        needTm = needTm || std::abs(amplitudes.tm) > Real{0};
    }
    std::size_t teColumn = 0;
    std::size_t tmColumn = 0;
    if (needTe) {
        teColumn = plan.channels.size();
        plan.channels.push_back(te0);
    }
    if (needTm) {
        tmColumn = plan.channels.size();
        plan.channels.push_back(tm0);
    }
    if (plan.channels.empty()) {
        return plan;
    }

    plan.incidents.reserve(polarizations.size());
    for (Polarization pol : polarizations) {
        const PolarizationAmplitudes amplitudes = polarizationAmplitudes(pol);
        IncidentChannelPlan incident;
        if (std::abs(amplitudes.te) > Real{0} &&
            std::abs(amplitudes.tm) > Real{0}) {
            incident.channels = {te0, tm0};
            incident.columns = {teColumn, tmColumn};
            incident.amplitudes = {amplitudes.te, amplitudes.tm};
            incident.count = 2;
        } else if (std::abs(amplitudes.te) > Real{0}) {
            incident.channels = {te0, 0};
            incident.columns = {teColumn, 0};
            incident.amplitudes = {amplitudes.te, {}};
            incident.count = 1;
        } else {
            incident.channels = {tm0, 0};
            incident.columns = {tmColumn, 0};
            incident.amplitudes = {amplitudes.tm, {}};
            incident.count = 1;
        }
        plan.incidents.push_back(incident);
    }
    return plan;
}

struct FluxWeights {
    std::vector<Real> incident;
    std::vector<Real> reflected;
    std::vector<Real> transmitted;
};

FluxWeights makeFluxWeights(const std::vector<LayerModes>& media, std::size_t block) {
    FluxWeights weights;
    weights.incident.resize(block);
    weights.reflected.resize(block);
    weights.transmitted.resize(block);
    for (std::size_t i = 0; i < block; ++i) {
        weights.incident[i] = propagatingFluxWeight(media.front(), i);
        weights.reflected[i] = propagatingFluxWeight(media.front(), block + i);
        weights.transmitted[i] = propagatingFluxWeight(media.back(), i);
    }
    return weights;
}

Real radiativePortWeight(const LayerModes& modes, std::size_t channel) {
    if (channel >= modes.gamma.size()) {
        throw std::invalid_argument(
            "directional thermal channel is outside exterior gamma storage");
    }
    const Complex gamma = modes.gamma[channel];
    const Real scale = std::max(Real{1.0}, std::abs(gamma));
    if (std::abs(std::imag(gamma)) > Real{1e-10} * scale) {
        return Real{0.0};
    }
    return propagatingFluxWeight(modes, channel);
}

std::vector<std::size_t> makeObservedOutputRows(const FluxWeights& weights,
                                                   std::size_t block) {
    std::vector<std::size_t> rows;
    rows.reserve(block);
    for (std::size_t row = 0; row < block; ++row) {
        if (weights.reflected[row] > Real{0.0} ||
            weights.transmitted[row] > Real{0.0}) {
            rows.push_back(row);
        }
    }
    return rows.size() == block ? std::vector<std::size_t>{} : rows;
}

std::size_t checkedHarmonicCount(int harmonicCount, int centerIdx, const char* op) {
    if (harmonicCount <= 0 || centerIdx < 0 || centerIdx >= harmonicCount) {
        throw std::invalid_argument(op);
    }
    return static_cast<std::size_t>(harmonicCount);
}

Real incidentFluxFor(const IncidentChannelPlan& incident, const FluxWeights& weights) {
    Real flux = 0.0;
    for (std::size_t i = 0; i < incident.count; ++i) {
        flux += std::norm(incident.amplitudes[i]) * weights.incident[incident.channels[i]];
    }
    return flux < 1e-14 ? Real{1.0} : flux;
}

struct DiffractionOrderPower {
    Real R{};
    Real T{};
};

struct ScatteringMatrixView {
    const std::vector<Complex>& R;
    const std::vector<Complex>& T;
    const std::vector<std::size_t>& rowLookup;
    std::size_t cols{};

    [[nodiscard]] Complex valueAt(const std::vector<Complex>& values,
                                   std::size_t row,
                                   std::size_t col) const {
        if (rowLookup.empty()) {
            return values[row * cols + col];
        }
        const std::size_t projected = rowLookup[row];
        return projected == std::numeric_limits<std::size_t>::max()
            ? Complex{}
            : values[projected * cols + col];
    }

    DiffractionOrderPower rowPower(const IncidentChannelPlan& incident,
                                   const FluxWeights& weights,
                                   std::size_t row,
                                   Real incidentFlux) const {
        Complex reflected{};
        Complex transmitted{};
        for (std::size_t i = 0; i < incident.count; ++i) {
            const std::size_t col = incident.columns[i];
            const Complex amplitude = incident.amplitudes[i];
            reflected += valueAt(R, row, col) * amplitude;
            transmitted += valueAt(T, row, col) * amplitude;
        }
        return {
            std::norm(reflected) * weights.reflected[row] / incidentFlux,
            std::norm(transmitted) * weights.transmitted[row] / incidentFlux,
        };
    }

    DiffractionOrderPower orderPower(const IncidentChannelPlan& incident,
                                      const FluxWeights& weights,
                                      std::size_t te,
                                      std::size_t tm,
                                      Real incidentFlux) const {
        Complex reflTe{};
        Complex reflTm{};
        Complex tranTe{};
        Complex tranTm{};
        if (incident.count == 1) {
            const std::size_t col = incident.columns[0];
            const Complex amplitude = incident.amplitudes[0];
            reflTe = valueAt(R, te, col) * amplitude;
            reflTm = valueAt(R, tm, col) * amplitude;
            tranTe = valueAt(T, te, col) * amplitude;
            tranTm = valueAt(T, tm, col) * amplitude;
        } else {
            const std::size_t col0 = incident.columns[0];
            const std::size_t col1 = incident.columns[1];
            const Complex amplitude0 = incident.amplitudes[0];
            const Complex amplitude1 = incident.amplitudes[1];
            reflTe = valueAt(R, te, col0) * amplitude0 +
                valueAt(R, te, col1) * amplitude1;
            reflTm = valueAt(R, tm, col0) * amplitude0 +
                valueAt(R, tm, col1) * amplitude1;
            tranTe = valueAt(T, te, col0) * amplitude0 +
                valueAt(T, te, col1) * amplitude1;
            tranTm = valueAt(T, tm, col0) * amplitude0 +
                valueAt(T, tm, col1) * amplitude1;
        }
        return {
            (std::norm(reflTe) * weights.reflected[te] +
             std::norm(reflTm) * weights.reflected[tm]) /
                incidentFlux,
            (std::norm(tranTe) * weights.transmitted[te] +
             std::norm(tranTm) * weights.transmitted[tm]) /
                incidentFlux,
        };
    }
};

template <bool IncludeOrders>
std::vector<DiffractionOutput<IncludeOrders>> efficienciesFromScattering(
    std::size_t N,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    SpectrumSolvePlan plan,
    const ScatteringMatrixView& scattering,
    const FluxWeights& weights,
    const std::vector<std::size_t>& projectedOutputRows) {
    const std::size_t block = 2 * N;
    const std::size_t center = static_cast<std::size_t>(centerIdx);
    const std::size_t te0 = center;
    const std::size_t tm0 = N + center;
    std::vector<DiffractionOutput<IncludeOrders>> out;
    out.reserve(polarizations.size());
    for (const IncidentChannelPlan& incident : plan.incidents) {
        const Real incidentFlux = incidentFluxFor(incident, weights);

        DiffractionOutput<IncludeOrders> d;
        if constexpr (IncludeOrders) {
            d.R.resize(N);
            d.T.resize(N);
        }
        Real Rtot = 0.0;
        Real Ttot = 0.0;
        Real R0 = 0.0;
        Real T0 = 0.0;
        if constexpr (IncludeOrders) {
            for (std::size_t i = 0; i < N; ++i) {
                const DiffractionOrderPower power = scattering.orderPower(
                    incident,
                    weights,
                    i,
                    N + i,
                    incidentFlux);
                d.R[i] = power.R;
                d.T[i] = power.T;
                if (i == center) {
                    R0 = power.R;
                    T0 = power.T;
                }
                Rtot += power.R;
                Ttot += power.T;
            }
        } else if (!projectedOutputRows.empty()) {
            for (const std::size_t row : projectedOutputRows) {
                const DiffractionOrderPower power =
                    scattering.rowPower(incident, weights, row, incidentFlux);
                if (row == te0 || row == tm0) {
                    R0 += power.R;
                    T0 += power.T;
                }
                Rtot += power.R;
                Ttot += power.T;
            }
        } else {
            for (std::size_t i = 0; i < N; ++i) {
                const DiffractionOrderPower power = scattering.orderPower(
                    incident,
                    weights,
                    i,
                    N + i,
                    incidentFlux);
                if (i == center) {
                    R0 = power.R;
                    T0 = power.T;
                }
                Rtot += power.R;
                Ttot += power.T;
            }
        }

        d.R0 = R0;
        d.T0 = T0;
        d.rTotal = Rtot;
        d.tTotal = Ttot;
        d.conservation = Rtot + Ttot;
        out.push_back(std::move(d));
    }
    return out;
}

template <bool IncludeOrders>
std::vector<DiffractionOutput<IncludeOrders>> computeStackEfficienciesImpl(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm) {
    if (media.size() < 2) {
        throw std::invalid_argument("efficiency stack solve expects at least two media");
    }
    const std::size_t N = checkedHarmonicCount(
        harmonicCount,
        centerIdx,
        "efficiency stack solve received invalid harmonic indices");
    const std::size_t block = 2 * N;
    const std::size_t te0 = static_cast<std::size_t>(centerIdx);
    const std::size_t tm0 = N + static_cast<std::size_t>(centerIdx);

    SpectrumSolvePlan plan = makeSpectrumSolvePlan(polarizations, te0, tm0);
    if (plan.channels.empty()) {
        return {};
    }

    const FluxWeights weights = makeFluxWeights(media, block);
    if (algorithm == StackingAlgorithm::EnhancedTransmittanceMatrix) {
        DirectionalRT directional = enhancedTransmittanceDirectional(media, false);
        for (IncidentChannelPlan& incident : plan.incidents) {
            for (std::size_t i = 0; i < incident.count; ++i) {
                incident.columns[i] = incident.channels[i];
            }
        }
        const std::vector<std::size_t> noProjection;
        const ScatteringMatrixView scattering{
            directional.R.data(),
            directional.T.data(),
            noProjection,
            block};
        return efficienciesFromScattering<IncludeOrders>(
            N,
            centerIdx,
            polarizations,
            std::move(plan),
            scattering,
            weights,
            noProjection);
    }
    if (algorithm != StackingAlgorithm::ScatteringMatrix) {
        throw std::invalid_argument("unsupported RCWA stacking algorithm");
    }
    const std::vector<std::size_t> outputRows = makeObservedOutputRows(weights, block);
    const PartialSMatrix S = stackToPartialSmatrix(media, plan.channels, outputRows);
    const std::vector<std::size_t> rowLookup = makeRowLookup(block, S.outputRows);
    const ScatteringMatrixView scattering{S.R.data(), S.T.data(), rowLookup, S.R.cols()};
    return efficienciesFromScattering<IncludeOrders>(
        N,
        centerIdx,
        polarizations,
        plan,
        scattering,
        weights,
        S.outputRows);
}

} // namespace

std::vector<DiffractionResult> computeStackEfficienciesForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm) {
    return computeStackEfficienciesImpl<true>(
        media,
        harmonicCount,
        centerIdx,
        polarizations,
        algorithm);
}

std::vector<DiffractionTotals> computeStackTotalsForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm) {
    return computeStackEfficienciesImpl<false>(
        media,
        harmonicCount,
        centerIdx,
        polarizations,
        algorithm);
}

std::vector<DiffractionTotals> computeStackTotalsForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    const SMatrix& smatrix) {
    if (media.size() < 2) {
        throw std::invalid_argument("efficiency stack solve expects at least two media");
    }
    const std::size_t N = checkedHarmonicCount(
        harmonicCount,
        centerIdx,
        "efficiency stack solve received invalid harmonic indices");
    const std::size_t block = 2 * N;
    if (smatrix.S11.rows() != block || smatrix.S11.cols() != block ||
        smatrix.S21.rows() != block || smatrix.S21.cols() != block) {
        throw std::invalid_argument(
            "precomputed scattering matrix does not match the harmonic basis");
    }

    SpectrumSolvePlan plan = makeSpectrumSolvePlan(
        polarizations,
        static_cast<std::size_t>(centerIdx),
        N + static_cast<std::size_t>(centerIdx));
    for (IncidentChannelPlan& incident : plan.incidents) {
        for (std::size_t i = 0; i < incident.count; ++i) {
            incident.columns[i] = incident.channels[i];
        }
    }
    const std::vector<std::size_t> noProjection;
    const FluxWeights weights = makeFluxWeights(media, block);
    const ScatteringMatrixView scattering{
        smatrix.S11.data(),
        smatrix.S21.data(),
        noProjection,
        block};
    return efficienciesFromScattering<false>(
        N,
        centerIdx,
        polarizations,
        std::move(plan),
        scattering,
        weights,
        noProjection);
}

SpectrumThermalChannels computeStackTotalsAndTopDirectionalThermalChannels(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm) {
    if (media.size() < 2) {
        throw std::invalid_argument(
            "combined spectrum/thermal solve expects at least two media");
    }
    const std::size_t N = checkedHarmonicCount(
        harmonicCount,
        centerIdx,
        "combined spectrum/thermal solve received invalid harmonic indices");
    const std::size_t block = 2 * N;
    const std::size_t center = static_cast<std::size_t>(centerIdx);
    const std::size_t te0 = center;
    const std::size_t tm0 = N + center;

    if (algorithm == StackingAlgorithm::EnhancedTransmittanceMatrix) {
        const SMatrix smatrix = stackToEnhancedTransmittanceSmatrix(media);
        return {
            computeStackTotalsForPolarizations(
                media,
                harmonicCount,
                centerIdx,
                polarizations,
                smatrix),
            computeTopDirectionalThermalChannels(media, smatrix),
        };
    }
    if (algorithm != StackingAlgorithm::ScatteringMatrix) {
        throw std::invalid_argument("unsupported RCWA stacking algorithm");
    }

    std::vector<std::size_t> topChannels;
    std::vector<std::size_t> bottomChannels;
    std::vector<Real> topIncidentWeights(block);
    std::vector<Real> topOutgoingWeights(block);
    std::vector<Real> bottomOutgoingWeights(block);
    std::vector<Real> bottomIncidentWeights(block);
    std::vector<std::size_t> topColumnByChannel(
        block,
        std::numeric_limits<std::size_t>::max());
    topChannels.reserve(block);
    bottomChannels.reserve(block);
    for (std::size_t channel = 0; channel < block; ++channel) {
        topIncidentWeights[channel] =
            radiativePortWeight(media.front(), channel);
        topOutgoingWeights[channel] =
            radiativePortWeight(media.front(), block + channel);
        bottomOutgoingWeights[channel] =
            radiativePortWeight(media.back(), channel);
        bottomIncidentWeights[channel] =
            radiativePortWeight(media.back(), block + channel);
        if (topIncidentWeights[channel] > Real{0.0}) {
            topColumnByChannel[channel] = topChannels.size();
            topChannels.push_back(channel);
        }
        if (bottomIncidentWeights[channel] > Real{0.0}) {
            bottomChannels.push_back(channel);
        }
    }

    const FluxWeights spectrumWeights = makeFluxWeights(media, block);
    std::vector<std::size_t> outputRows =
        makeObservedOutputRows(spectrumWeights, block);
    if (!outputRows.empty()) {
        for (std::size_t row = 0; row < block; ++row) {
            if (topOutgoingWeights[row] > Real{0.0} ||
                bottomOutgoingWeights[row] > Real{0.0}) {
                if (std::find(outputRows.begin(), outputRows.end(), row) ==
                    outputRows.end()) {
                    outputRows.push_back(row);
                }
            }
        }
        std::sort(outputRows.begin(), outputRows.end());
    }

    const PartialSMatrix S = stackToPartialSmatrix(
        media,
        topChannels,
        outputRows,
        bottomChannels);
    const std::vector<std::size_t> rowLookup = makeRowLookup(block, S.outputRows);
    const ScatteringMatrixView scattering{
        S.R.data(),
        S.T.data(),
        rowLookup,
        S.R.cols()};

    SpectrumSolvePlan spectrumPlan = makeSpectrumSolvePlan(
        polarizations,
        te0,
        tm0);
    for (IncidentChannelPlan& incident : spectrumPlan.incidents) {
        for (std::size_t i = 0; i < incident.count; ++i) {
            const std::size_t column =
                topColumnByChannel[incident.channels[i]];
            if (column == std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error(
                    "requested incident polarization is not a radiative channel");
            }
            incident.columns[i] = column;
        }
    }

    SpectrumThermalChannels out;
    out.spectra = efficienciesFromScattering<false>(
        N,
        centerIdx,
        polarizations,
        std::move(spectrumPlan),
        scattering,
        spectrumWeights,
        S.outputRows);

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Complex nanComplex{nan, nan};
    out.thermal.absorptivity.assign(block, nan);
    out.thermal.emissivity.assign(block, nan);
    out.thermal.incidentScattering.assign(block, nan);
    out.thermal.outgoingScattering.assign(block, nan);
    out.thermal.teTmAbsorptivityCoherence.assign(N, nanComplex);
    out.thermal.teTmEmissivityCoherence.assign(N, nanComplex);

    const auto projectedRow = [&](std::size_t row) {
        if (S.outputRows.empty()) {
            return row;
        }
        const std::size_t projected = rowLookup[row];
        if (projected == std::numeric_limits<std::size_t>::max()) {
            throw std::logic_error(
                "required thermal output row was not retained");
        }
        return projected;
    };
    const auto topColumn = [&](std::size_t channel) {
        const std::size_t column = topColumnByChannel[channel];
        if (column == std::numeric_limits<std::size_t>::max()) {
            throw std::logic_error(
                "required thermal incident channel was not retained");
        }
        return column;
    };

    const Real teIncidentWeight = topIncidentWeights[te0];
    const Real tmIncidentWeight = topIncidentWeights[tm0];
    const std::size_t teColumn = topColumn(te0);
    const std::size_t tmColumn = topColumn(tm0);
    Real teIncidentScattering = 0.0;
    Real tmIncidentScattering = 0.0;
    Complex absorptionCross{};
    for (std::size_t row = 0; row < block; ++row) {
        const Real topWeight = topOutgoingWeights[row];
        const Real bottomWeight = bottomOutgoingWeights[row];
        if (topWeight == Real{0.0} && bottomWeight == Real{0.0}) {
            continue;
        }
        const std::size_t projected = projectedRow(row);
        if (topWeight > Real{0.0}) {
            const Complex te = S.R(projected, teColumn);
            const Complex tm = S.R(projected, tmColumn);
            teIncidentScattering +=
                std::norm(te) * topWeight / teIncidentWeight;
            tmIncidentScattering +=
                std::norm(tm) * topWeight / tmIncidentWeight;
            absorptionCross +=
                std::conj(te) * tm * topWeight;
        }
        if (bottomWeight > Real{0.0}) {
            const Complex te = S.T(projected, teColumn);
            const Complex tm = S.T(projected, tmColumn);
            teIncidentScattering +=
                std::norm(te) * bottomWeight / teIncidentWeight;
            tmIncidentScattering +=
                std::norm(tm) * bottomWeight / tmIncidentWeight;
            absorptionCross +=
                std::conj(te) * tm * bottomWeight;
        }
    }
    out.thermal.incidentScattering[te0] = teIncidentScattering;
    out.thermal.absorptivity[te0] = Real{1.0} - teIncidentScattering;
    out.thermal.incidentScattering[tm0] = tmIncidentScattering;
    out.thermal.absorptivity[tm0] = Real{1.0} - tmIncidentScattering;
    out.thermal.teTmAbsorptivityCoherence[center] =
        -absorptionCross / std::sqrt(teIncidentWeight * tmIncidentWeight);

    const Real teOutgoingWeight = topOutgoingWeights[te0];
    const Real tmOutgoingWeight = topOutgoingWeights[tm0];
    const Real outgoingNormalization =
        std::sqrt(teOutgoingWeight * tmOutgoingWeight);
    const std::size_t teRow = projectedRow(te0);
    const std::size_t tmRow = projectedRow(tm0);
    Real teOutgoingScattering = 0.0;
    Real tmOutgoingScattering = 0.0;
    Complex emissionCross{};
    for (std::size_t col = 0; col < topChannels.size(); ++col) {
        const Real weight = topIncidentWeights[topChannels[col]];
        const Complex te = S.R(teRow, col);
        const Complex tm = S.R(tmRow, col);
        teOutgoingScattering += std::norm(te) * teOutgoingWeight / weight;
        tmOutgoingScattering += std::norm(tm) * tmOutgoingWeight / weight;
        emissionCross += te * std::conj(tm) * outgoingNormalization / weight;
    }
    for (std::size_t col = 0; col < bottomChannels.size(); ++col) {
        const Real weight = bottomIncidentWeights[bottomChannels[col]];
        const Complex te = S.S12(teRow, col);
        const Complex tm = S.S12(tmRow, col);
        teOutgoingScattering += std::norm(te) * teOutgoingWeight / weight;
        tmOutgoingScattering += std::norm(tm) * tmOutgoingWeight / weight;
        emissionCross += te * std::conj(tm) * outgoingNormalization / weight;
    }
    out.thermal.outgoingScattering[te0] = teOutgoingScattering;
    out.thermal.emissivity[te0] = Real{1.0} - teOutgoingScattering;
    out.thermal.outgoingScattering[tm0] = tmOutgoingScattering;
    out.thermal.emissivity[tm0] = Real{1.0} - tmOutgoingScattering;
    out.thermal.teTmEmissivityCoherence[center] = -emissionCross;
    return out;
}

SMatrix computeStackSmatrix(const std::vector<LayerModes>& media,
                            StackingAlgorithm algorithm) {
    switch (algorithm) {
    case StackingAlgorithm::ScatteringMatrix:
        return stackToFullSmatrix(media);
    case StackingAlgorithm::EnhancedTransmittanceMatrix:
        return stackToEnhancedTransmittanceSmatrix(media);
    }
    throw std::invalid_argument("unsupported RCWA stacking algorithm");
}

ZeroOrderAmplitudes computeZeroOrderAmplitudes(const std::vector<LayerModes>& media,
                                                   int harmonicCount,
                                                   int centerIdx,
                                                   StackingAlgorithm algorithm) {
    if (algorithm == StackingAlgorithm::ScatteringMatrix) {
        return stackToZeroOrderAmplitudes(media, harmonicCount, centerIdx);
    }
    if (algorithm != StackingAlgorithm::EnhancedTransmittanceMatrix) {
        throw std::invalid_argument("unsupported RCWA stacking algorithm");
    }
    if (media.size() < 2 || harmonicCount <= 0 || centerIdx < 0 ||
        centerIdx >= harmonicCount) {
        throw std::invalid_argument(
            "zero-order enhanced T-matrix solve received invalid stack dimensions");
    }
    const std::size_t N = static_cast<std::size_t>(harmonicCount);
    const std::size_t te0 = static_cast<std::size_t>(centerIdx);
    const std::size_t tm0 = N + te0;
    const DirectionalRT directional = enhancedTransmittanceDirectional(media, false);
    ZeroOrderAmplitudes out;
    out.R = Matrix(2, 2);
    out.T = Matrix(2, 2);
    for (std::size_t row = 0; row < 2; ++row) {
        const std::size_t sourceRow = row == 0 ? te0 : tm0;
        for (std::size_t col = 0; col < 2; ++col) {
            const std::size_t sourceCol = col == 0 ? te0 : tm0;
            out.R(row, col) = directional.R(sourceRow, sourceCol);
            out.T(row, col) = directional.T(sourceRow, sourceCol);
        }
    }
    return out;
}

TopDirectionalThermalChannels computeTopDirectionalThermalChannels(
    const std::vector<LayerModes>& media,
    const SMatrix& S) {
    if (media.size() < 2) {
        throw std::invalid_argument(
            "directional thermal-channel solve expects at least two exterior media");
    }
    const std::size_t stateSize = layerStateSize(media.front());
    if (stateSize == 0 || stateSize % 2 != 0 ||
        layerStateSize(media.back()) != stateSize) {
        throw std::invalid_argument(
            "directional thermal-channel solve expects matching exterior mode dimensions");
    }
    const std::size_t block = stateSize / 2;
    if (S.S11.rows() != block || S.S11.cols() != block ||
        S.S12.rows() != block || S.S12.cols() != block ||
        S.S21.rows() != block || S.S21.cols() != block ||
        S.S22.rows() != block || S.S22.cols() != block) {
        throw std::logic_error(
            "directional thermal-channel scattering blocks have inconsistent dimensions");
    }

    TopDirectionalThermalChannels out;
    out.absorptivity.resize(block);
    out.emissivity.resize(block);
    out.incidentScattering.resize(block);
    out.outgoingScattering.resize(block);
    for (std::size_t channel = 0; channel < block; ++channel) {
        const Real incidentWeight = radiativePortWeight(media.front(), channel);
        if (incidentWeight == Real{0.0}) {
            out.absorptivity[channel] = std::numeric_limits<Real>::quiet_NaN();
            out.incidentScattering[channel] = std::numeric_limits<Real>::quiet_NaN();
        } else {
            Real scattering = 0.0;
            for (std::size_t row = 0; row < block; ++row) {
                const Real topOutgoingWeight =
                    radiativePortWeight(media.front(), block + row);
                if (topOutgoingWeight > Real{0.0}) {
                    scattering += std::norm(S.S11(row, channel)) *
                        topOutgoingWeight / incidentWeight;
                }
                const Real bottomOutgoingWeight =
                    radiativePortWeight(media.back(), row);
                if (bottomOutgoingWeight > Real{0.0}) {
                    scattering += std::norm(S.S21(row, channel)) *
                        bottomOutgoingWeight / incidentWeight;
                }
            }
            out.incidentScattering[channel] = scattering;
            Real absorptivity = Real{1.0} - scattering;
            if (absorptivity < Real{0.0} && absorptivity > Real{-1e-10}) {
                absorptivity = Real{0.0};
            } else if (absorptivity > Real{1.0} &&
                       absorptivity < Real{1.0 + 1e-10}) {
                absorptivity = Real{1.0};
            }
            out.absorptivity[channel] = absorptivity;
        }

        const std::size_t row = channel;
        const Real outgoingWeight = radiativePortWeight(media.front(), block + row);
        if (outgoingWeight == Real{0.0}) {
            out.emissivity[row] = std::numeric_limits<Real>::quiet_NaN();
            out.outgoingScattering[row] = std::numeric_limits<Real>::quiet_NaN();
            continue;
        }

        Real scattering = 0.0;
        for (std::size_t col = 0; col < block; ++col) {
            const Real topIncidentWeight = radiativePortWeight(media.front(), col);
            if (topIncidentWeight > Real{0.0}) {
                scattering += std::norm(S.S11(row, col)) *
                    outgoingWeight / topIncidentWeight;
            }
            const Real bottomIncidentWeight =
                radiativePortWeight(media.back(), block + col);
            if (bottomIncidentWeight > Real{0.0}) {
                scattering += std::norm(S.S12(row, col)) *
                    outgoingWeight / bottomIncidentWeight;
            }
        }
        out.outgoingScattering[row] = scattering;
        Real emissivity = Real{1.0} - scattering;
        if (emissivity < Real{0.0} && emissivity > Real{-1e-10}) {
            emissivity = Real{0.0};
        } else if (emissivity > Real{1.0} && emissivity < Real{1.0 + 1e-10}) {
            emissivity = Real{1.0};
        }
        out.emissivity[row] = emissivity;
    }

    if (block % 2 != 0) {
        throw std::logic_error(
            "directional thermal channels require paired TE/TM exterior channels");
    }
    const std::size_t harmonicCount = block / 2;
    out.teTmAbsorptivityCoherence.resize(harmonicCount);
    out.teTmEmissivityCoherence.resize(harmonicCount);
    const Complex nanComplex{
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN()};
    for (std::size_t harmonic = 0; harmonic < harmonicCount; ++harmonic) {
        const std::size_t te = harmonic;
        const std::size_t tm = harmonicCount + harmonic;
        const Real teIncidentWeight = radiativePortWeight(media.front(), te);
        const Real tmIncidentWeight = radiativePortWeight(media.front(), tm);
        if (teIncidentWeight == Real{0.0} || tmIncidentWeight == Real{0.0}) {
            out.teTmAbsorptivityCoherence[harmonic] = nanComplex;
        } else {
            Complex scattering{};
            const Real incidentNormalization =
                std::sqrt(teIncidentWeight * tmIncidentWeight);
            for (std::size_t row = 0; row < block; ++row) {
                const Real topOutgoingWeight =
                    radiativePortWeight(media.front(), block + row);
                if (topOutgoingWeight > Real{0.0}) {
                    scattering += std::conj(S.S11(row, te)) * S.S11(row, tm) *
                        topOutgoingWeight / incidentNormalization;
                }
                const Real bottomOutgoingWeight =
                    radiativePortWeight(media.back(), row);
                if (bottomOutgoingWeight > Real{0.0}) {
                    scattering += std::conj(S.S21(row, te)) * S.S21(row, tm) *
                        bottomOutgoingWeight / incidentNormalization;
                }
            }
            out.teTmAbsorptivityCoherence[harmonic] = -scattering;
        }

        const Real teOutgoingWeight =
            radiativePortWeight(media.front(), block + te);
        const Real tmOutgoingWeight =
            radiativePortWeight(media.front(), block + tm);
        if (teOutgoingWeight == Real{0.0} || tmOutgoingWeight == Real{0.0}) {
            out.teTmEmissivityCoherence[harmonic] = nanComplex;
        } else {
            Complex scattering{};
            const Real outgoingNormalization =
                std::sqrt(teOutgoingWeight * tmOutgoingWeight);
            for (std::size_t col = 0; col < block; ++col) {
                const Real topIncidentWeight =
                    radiativePortWeight(media.front(), col);
                if (topIncidentWeight > Real{0.0}) {
                    scattering += S.S11(te, col) * std::conj(S.S11(tm, col)) *
                        outgoingNormalization / topIncidentWeight;
                }
                const Real bottomIncidentWeight =
                    radiativePortWeight(media.back(), block + col);
                if (bottomIncidentWeight > Real{0.0}) {
                    scattering += S.S12(te, col) * std::conj(S.S12(tm, col)) *
                        outgoingNormalization / bottomIncidentWeight;
                }
            }
            out.teTmEmissivityCoherence[harmonic] = -scattering;
        }
    }
    return out;
}

} // namespace rcwa
