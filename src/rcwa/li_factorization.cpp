#include "rcwa/li_factorization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace rcwa::detail {

namespace {

constexpr Complex one{1.0, 0.0};

std::size_t checkedGridSize(int countX, int countY, std::string_view context) {
    if (countX <= 0 || countY <= 0) {
        throw std::invalid_argument(std::string(context) + " counts must be positive");
    }
    const auto x = static_cast<std::size_t>(countX);
    const auto y = static_cast<std::size_t>(countY);
    if (x > std::numeric_limits<std::size_t>::max() / y) {
        throw std::invalid_argument(std::string(context) + " size exceeds addressable range");
    }
    return x * y;
}

int oddCountForOrder(int order, std::string_view context) {
    if (order < 0 || order > (std::numeric_limits<int>::max() - 1) / 2) {
        throw std::invalid_argument(std::string(context) + " order is outside the supported range");
    }
    return 2 * order + 1;
}

int convolutionCount(int retainedCount, std::string_view context) {
    if (retainedCount <= 0 || retainedCount % 2 == 0 ||
        retainedCount > (std::numeric_limits<int>::max() + std::int64_t{1}) / 2) {
        throw std::invalid_argument(
            std::string(context) + " retained count must be a supported positive odd integer");
    }
    return 2 * retainedCount - 1;
}

void requireSquare(const Matrix& matrix, std::string_view context) {
    if (matrix.empty() || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument(std::string(context) + " must be a non-empty square matrix");
    }
}

Matrix inverseBlock(const Matrix& matrix,
                    std::string_view tensorName,
                    std::string_view blockName) {
    requireSquare(matrix, std::string(tensorName) + " " + std::string(blockName));
    try {
        return inverse(matrix);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(tensorName) + " " + std::string(blockName) +
            " is singular during Li factorization: " + error.what());
    }
}

Complex pointwisePivotInverse(const Tensor3& tensor,
                              std::size_t pivot,
                              std::string_view tensorName) {
    const Complex pivotValue = tensor(pivot, pivot);
    if (!isFinite(pivotValue) || std::abs(pivotValue) == Real{0}) {
        throw std::runtime_error(
            std::string(tensorName) + " has a singular pointwise Li pivot");
    }
    return one / pivotValue;
}

Complex pointwiseLiMinusComponent(const Tensor3& tensor,
                                  std::size_t pivot,
                                  std::size_t row,
                                  std::size_t column,
                                  Complex pivotInverse) {
    if (row == pivot && column == pivot) {
        return pivotInverse;
    }
    if (row == pivot) {
        return pivotInverse * tensor(row, column);
    }
    if (column == pivot) {
        return tensor(row, column) * pivotInverse;
    }
    return tensor(row, column) -
        tensor(row, pivot) * pivotInverse * tensor(pivot, column);
}

Tensor3 pointwiseLiMinus(const Tensor3& tensor,
                         std::size_t pivot,
                         std::string_view tensorName) {
    const Complex pivotInverse = pointwisePivotInverse(
        tensor, pivot, tensorName);
    Tensor3 transformed;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            transformed(row, column) = pointwiseLiMinusComponent(
                tensor, pivot, row, column, pivotInverse);
        }
    }
    return transformed;
}

LiTensorBlocks liBlockTransform(const LiTensorBlocks& input,
                                std::size_t pivot,
                                bool plus,
                                std::string_view tensorName) {
    LiTensorBlocks output;
    output[pivot][pivot] = inverseBlock(
        input[pivot][pivot], tensorName, "pivot block");

    for (std::size_t column = 0; column < 3; ++column) {
        if (column != pivot) {
            output[pivot][column] =
                output[pivot][pivot] * input[pivot][column];
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        if (row != pivot) {
            output[row][pivot] =
                input[row][pivot] * output[pivot][pivot];
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        if (row == pivot) {
            continue;
        }
        for (std::size_t column = 0; column < 3; ++column) {
            if (column == pivot) {
                continue;
            }
            const Matrix correction =
                input[row][pivot] * output[pivot][column];
            output[row][column] = plus
                ? input[row][column] + correction
                : input[row][column] - correction;
        }
    }
    return output;
}

std::vector<std::vector<Complex>> centeredFourierWeights(int sampleCount,
                                                          int coefficientCount) {
    if (sampleCount <= 0 || coefficientCount <= 0 || coefficientCount % 2 == 0) {
        throw std::invalid_argument(
            "partial Fourier transform requires positive samples and an odd coefficient count");
    }
    const int half = coefficientCount / 2;
    std::vector<std::vector<Complex>> weights(
        static_cast<std::size_t>(coefficientCount),
        std::vector<Complex>(static_cast<std::size_t>(sampleCount)));
    const Real normalization = Real{1} / static_cast<Real>(sampleCount);
    for (int order = -half; order <= half; ++order) {
        auto& row = weights[static_cast<std::size_t>(order + half)];
        for (int sample = 0; sample < sampleCount; ++sample) {
            const Real coordinate =
                (static_cast<Real>(sample) + Real{0.5}) /
                    static_cast<Real>(sampleCount) -
                Real{0.5};
            row[static_cast<std::size_t>(sample)] =
                normalization *
                std::exp(Complex{0.0, -twoPi * order * coordinate});
        }
    }
    return weights;
}

std::vector<std::vector<Complex>> piecewiseFourierWeights(
    const std::vector<Real>& boundaries,
    int coefficientCount) {
    if (boundaries.size() < 2 || coefficientCount <= 0 || coefficientCount % 2 == 0) {
        throw std::invalid_argument(
            "piecewise Fourier transform requires boundaries and an odd coefficient count");
    }
    const Real tolerance = Real{256} * std::numeric_limits<Real>::epsilon();
    if (!std::isfinite(boundaries.front()) || !std::isfinite(boundaries.back()) ||
        std::abs(boundaries.front() + Real{0.5}) > tolerance ||
        std::abs(boundaries.back() - Real{0.5}) > tolerance) {
        throw std::invalid_argument(
            "piecewise Fourier boundaries must span normalized coordinates [-0.5, 0.5]");
    }
    for (std::size_t i = 1; i < boundaries.size(); ++i) {
        if (!std::isfinite(boundaries[i]) || !(boundaries[i] > boundaries[i - 1])) {
            throw std::invalid_argument(
                "piecewise Fourier boundaries must be finite and strictly increasing");
        }
    }

    const int half = coefficientCount / 2;
    const std::size_t cellCount = boundaries.size() - 1;
    std::vector<std::vector<Complex>> weights(
        static_cast<std::size_t>(coefficientCount),
        std::vector<Complex>(cellCount));
    for (int order = -half; order <= half; ++order) {
        auto& row = weights[static_cast<std::size_t>(order + half)];
        for (std::size_t cell = 0; cell < cellCount; ++cell) {
            const Real start = boundaries[cell];
            const Real end = boundaries[cell + 1];
            const Real width = end - start;
            const Real center = Real{0.5} * (start + end);
            const Real integral = order == 0
                ? width
                : std::sin(pi * static_cast<Real>(order) * width) /
                    (pi * static_cast<Real>(order));
            row[cell] = integral * std::exp(
                Complex{0.0, -twoPi * static_cast<Real>(order) * center});
        }
    }
    return weights;
}

std::vector<std::size_t> rectangularProjectionIndices(
    int retainedCountX,
    int retainedCountY,
    const HarmonicBasis& basis) {
    const int orderX = retainedCountX / 2;
    const int orderY = retainedCountY / 2;
    std::vector<std::size_t> indices;
    indices.reserve(basis.size());
    for (const HarmonicIndex harmonic : basis.orders) {
        if (std::abs(harmonic.m) > orderX || std::abs(harmonic.n) > orderY) {
            throw std::invalid_argument(
                "requested harmonic lies outside the ordered Li envelope");
        }
        indices.push_back(
            static_cast<std::size_t>(harmonic.n + orderY) *
                static_cast<std::size_t>(retainedCountX) +
            static_cast<std::size_t>(harmonic.m + orderX));
    }
    return indices;
}

Matrix projectRectangularMatrix(const Matrix& rectangular,
                                const std::vector<std::size_t>& indices) {
    Matrix projected(indices.size(), indices.size());
    for (std::size_t row = 0; row < indices.size(); ++row) {
        for (std::size_t column = 0; column < indices.size(); ++column) {
            projected(row, column) = rectangular(indices[row], indices[column]);
        }
    }
    return projected;
}

using FourierBlockCoefficients =
    std::array<std::array<std::vector<Matrix>, 3>, 3>;

std::vector<Matrix> allocateMatrixCoefficients(int coefficientCount,
                                               int blockSize) {
    std::vector<Matrix> coefficients;
    coefficients.reserve(static_cast<std::size_t>(coefficientCount));
    for (int order = 0; order < coefficientCount; ++order) {
        coefficients.emplace_back(
            static_cast<std::size_t>(blockSize),
            static_cast<std::size_t>(blockSize));
    }
    return coefficients;
}

void accumulateMatrixTransform(
    std::vector<Matrix>& coefficients,
    const Matrix& value,
    const std::vector<std::vector<Complex>>& weights,
    int cellIndex) {
    if (coefficients.size() != weights.size()) {
        throw std::invalid_argument(
            "RETICOLO matrix transform coefficient envelope is inconsistent");
    }
    for (std::size_t order = 0; order < coefficients.size(); ++order) {
        const Complex weight = weights[order][static_cast<std::size_t>(cellIndex)];
        auto& coefficientData = coefficients[order].data();
        const auto& valueData = value.data();
        for (std::size_t index = 0; index < coefficientData.size(); ++index) {
            coefficientData[index] += weight * valueData[index];
        }
    }
}

Matrix makeXOuterMatrix(std::vector<Matrix> coefficients,
                        int retainedCountX,
                        int retainedCountY) {
    const std::size_t rectangularSize = checkedGridSize(
        retainedCountX, retainedCountY, "RETICOLO x-outer harmonic");
    const int coefficientHalfX = static_cast<int>(coefficients.size() / 2);
    Matrix transformed(rectangularSize, rectangularSize);
    for (int rowY = 0; rowY < retainedCountY; ++rowY) {
        for (int rowX = 0; rowX < retainedCountX; ++rowX) {
            const std::size_t row = static_cast<std::size_t>(rowY) *
                static_cast<std::size_t>(retainedCountX) +
                static_cast<std::size_t>(rowX);
            for (int columnY = 0; columnY < retainedCountY; ++columnY) {
                for (int columnX = 0; columnX < retainedCountX; ++columnX) {
                    const std::size_t column =
                        static_cast<std::size_t>(columnY) *
                            static_cast<std::size_t>(retainedCountX) +
                        static_cast<std::size_t>(columnX);
                    transformed(row, column) = coefficients[
                        static_cast<std::size_t>(
                            rowX - columnX + coefficientHalfX)](
                                static_cast<std::size_t>(rowY),
                                static_cast<std::size_t>(columnY));
                }
            }
        }
    }
    return transformed;
}

bool scalarCellValues(const std::vector<Tensor3>& cells,
                      std::vector<Complex>* values) {
    values->clear();
    values->reserve(cells.size());
    for (const Tensor3& tensor : cells) {
        Complex value{};
        if (!tensorIsScalar(tensor, &value)) {
            values->clear();
            return false;
        }
        if (!isFinite(value) || std::abs(value) == Real{0}) {
            throw std::runtime_error(
                "RETICOLO scalar Li factorization has a singular cell value");
        }
        values->push_back(value);
    }
    return true;
}

LiFactorization liFactorizeScalar2dReticolo(
    const std::vector<Complex>& cells,
    int cellCountX,
    int cellCountY,
    int retainedCountX,
    int retainedCountY,
    const HarmonicBasis& basis,
    std::string_view tensorName,
    const std::vector<std::vector<Complex>>& xWeights,
    const std::vector<std::vector<Complex>>& yWeights) {
    const int coefficientCountX = convolutionCount(
        retainedCountX, "RETICOLO scalar x");
    const int coefficientCountY = convolutionCount(
        retainedCountY, "RETICOLO scalar y");
    const int coefficientHalfY = coefficientCountY / 2;

    // retcouche.m calls ret2li(..., li=1) for both epx and epy.  For epy,
    // invert the local y Toeplitz matrix of 1/u, then assemble along x.  For
    // epx, invert the local y Toeplitz matrix of u, assemble along x, and
    // invert the complete matrix once more.
    std::vector<Matrix> qxxInnerInverseCoefficients = allocateMatrixCoefficients(
        coefficientCountX, retainedCountY);
    std::vector<Matrix> qyyCoefficients = allocateMatrixCoefficients(
        coefficientCountX, retainedCountY);
    std::vector<Matrix> directCoefficients = allocateMatrixCoefficients(
        coefficientCountX, retainedCountY);
    for (int cellX = 0; cellX < cellCountX; ++cellX) {
        std::vector<Complex> reciprocalY(
            static_cast<std::size_t>(coefficientCountY));
        std::vector<Complex> directY(
            static_cast<std::size_t>(coefficientCountY));
        for (int orderY = -coefficientHalfY;
             orderY <= coefficientHalfY;
             ++orderY) {
            Complex reciprocalCoefficient{};
            Complex directCoefficient{};
            const auto& weights = yWeights[
                static_cast<std::size_t>(orderY + coefficientHalfY)];
            for (int cellY = 0; cellY < cellCountY; ++cellY) {
                const Complex value = cells[static_cast<std::size_t>(
                    cellY * cellCountX + cellX)];
                reciprocalCoefficient +=
                    weights[static_cast<std::size_t>(cellY)] / value;
                directCoefficient +=
                    weights[static_cast<std::size_t>(cellY)] * value;
            }
            const std::size_t coefficientIndex = static_cast<std::size_t>(
                orderY + coefficientHalfY);
            reciprocalY[coefficientIndex] = reciprocalCoefficient;
            directY[coefficientIndex] = directCoefficient;
        }
        const Matrix localYReciprocalInverse = inverseBlock(
            toeplitzConvMatrix(reciprocalY, retainedCountY),
            tensorName,
            "local y reciprocal block");
        const Matrix localYDirect = toeplitzConvMatrix(
            directY, retainedCountY);
        const Matrix localYDirectInverse = inverseBlock(
            localYDirect,
            tensorName,
            "local y direct block");
        accumulateMatrixTransform(
            qxxInnerInverseCoefficients,
            localYDirectInverse,
            xWeights,
            cellX);
        accumulateMatrixTransform(
            qyyCoefficients,
            localYReciprocalInverse,
            xWeights,
            cellX);
        accumulateMatrixTransform(
            directCoefficients,
            localYDirect,
            xWeights,
            cellX);
    }

    const Matrix rectangularQxx = inverseBlock(
        makeXOuterMatrix(
            std::move(qxxInnerInverseCoefficients),
            retainedCountX,
            retainedCountY),
        tensorName,
        "assembled x-normal block");
    const Matrix rectangularQyy = makeXOuterMatrix(
        std::move(qyyCoefficients), retainedCountX, retainedCountY);
    const Matrix rectangularDirect = makeXOuterMatrix(
        std::move(directCoefficients), retainedCountX, retainedCountY);
    const std::vector<std::size_t> projection = rectangularProjectionIndices(
        retainedCountX, retainedCountY, basis);

    const Matrix qxx = projectRectangularMatrix(rectangularQxx, projection);
    const Matrix qyy = projectRectangularMatrix(rectangularQyy, projection);
    const Matrix direct = projectRectangularMatrix(rectangularDirect, projection);
    const Matrix zero(basis.size(), basis.size());
    LiFactorization result;
    for (auto& row : result.q) {
        for (Matrix& block : row) {
            block = zero;
        }
    }
    result.q[0][0] = qxx;
    result.q[1][1] = qyy;
    result.q[2][2] = direct;
    result.zzInverse = inverseBlock(
        result.q[2][2], tensorName, "zz block");
    return result;
}

FourierBlockCoefficients allocateBlockCoefficients(
    int coefficientCount,
    int blockSize) {
    FourierBlockCoefficients coefficients;
    for (auto& tensorRow : coefficients) {
        for (auto& blocks : tensorRow) {
            blocks.reserve(static_cast<std::size_t>(coefficientCount));
            for (int order = 0; order < coefficientCount; ++order) {
                blocks.emplace_back(
                    static_cast<std::size_t>(blockSize),
                    static_cast<std::size_t>(blockSize));
            }
        }
    }
    return coefficients;
}

void accumulatePartialTransform(FourierBlockCoefficients& coefficients,
                                const LiTensorBlocks& sampled,
                                const std::vector<std::vector<Complex>>& weights,
                                int sampleIndex) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const auto& sampleData = sampled[row][column].data();
            for (std::size_t order = 0; order < weights.size(); ++order) {
                auto& coefficientData = coefficients[row][column][order].data();
                const Complex weight = weights[order][static_cast<std::size_t>(sampleIndex)];
                for (std::size_t index = 0; index < coefficientData.size(); ++index) {
                    coefficientData[index] += weight * sampleData[index];
                }
            }
        }
    }
}

LiTensorBlocks makeXOuterBlockToeplitz(
    FourierBlockCoefficients coefficients,
    int retainedCountX,
    int retainedCountY) {
    const auto rectangularSize = checkedGridSize(
        retainedCountX, retainedCountY, "ordered Li rectangular harmonic");
    const int coefficientHalfX = static_cast<int>(
        coefficients[0][0].size() / 2);
    LiTensorBlocks transformed;
    for (std::size_t tensorRow = 0; tensorRow < 3; ++tensorRow) {
        for (std::size_t tensorColumn = 0; tensorColumn < 3; ++tensorColumn) {
            Matrix convolution(rectangularSize, rectangularSize);
            for (int rowY = 0; rowY < retainedCountY; ++rowY) {
                for (int rowX = 0; rowX < retainedCountX; ++rowX) {
                    const std::size_t row = static_cast<std::size_t>(rowY) *
                        static_cast<std::size_t>(retainedCountX) +
                        static_cast<std::size_t>(rowX);
                    for (int columnY = 0; columnY < retainedCountY; ++columnY) {
                        for (int columnX = 0; columnX < retainedCountX; ++columnX) {
                            const std::size_t column =
                                static_cast<std::size_t>(columnY) *
                                    static_cast<std::size_t>(retainedCountX) +
                                static_cast<std::size_t>(columnX);
                            const int difference = rowX - columnX;
                            convolution(row, column) =
                                coefficients[tensorRow][tensorColumn][
                                    static_cast<std::size_t>(
                                        difference + coefficientHalfX)](
                                            static_cast<std::size_t>(rowY),
                                            static_cast<std::size_t>(columnY));
                        }
                    }
                }
            }
            transformed[tensorRow][tensorColumn] = std::move(convolution);
        }
    }
    return transformed;
}

LiFactorization liFactorizeTensor2dReticoloImpl(
    const std::vector<Tensor3>& cells,
    int cellCountX,
    int cellCountY,
    const HarmonicBasis& basis,
    std::string_view tensorName,
    const std::vector<std::vector<Complex>>& xWeights,
    const std::vector<std::vector<Complex>>& yWeights) {
    const auto expected = checkedGridSize(
        cellCountX, cellCountY, "RETICOLO Li cell");
    if (cells.size() != expected) {
        throw std::invalid_argument(
            "RETICOLO Li cell array size must equal cell_count_x * cell_count_y");
    }
    if (basis.orders.empty()) {
        throw std::invalid_argument("RETICOLO Li factorization requires a non-empty harmonic basis");
    }

    const int retainedCountX = oddCountForOrder(basis.orderX, "RETICOLO Li x");
    const int retainedCountY = oddCountForOrder(basis.orderY, "RETICOLO Li y");
    const int coefficientCountX = convolutionCount(retainedCountX, "RETICOLO Li x");
    const int coefficientCountY = convolutionCount(retainedCountY, "RETICOLO Li y");
    if (xWeights.size() != static_cast<std::size_t>(coefficientCountX) ||
        yWeights.size() != static_cast<std::size_t>(coefficientCountY)) {
        throw std::invalid_argument("RETICOLO Li Fourier weight envelope is inconsistent");
    }

    std::vector<Complex> scalarCells;
    if (scalarCellValues(cells, &scalarCells)) {
        return liFactorizeScalar2dReticolo(
            scalarCells,
            cellCountX,
            cellCountY,
            retainedCountX,
            retainedCountY,
            basis,
            tensorName,
            xWeights,
            yWeights);
    }

    // Ordered full-tensor extension: form/invert the local y Toeplitz
    // operators first, Fourier transform their matrix entries along x, then
    // apply the outer x block transform.
    const int coefficientHalfY = coefficientCountY / 2;
    FourierBlockCoefficients xCoefficients = allocateBlockCoefficients(
        coefficientCountX, retainedCountY);
    std::vector<Tensor3> yTransformed(static_cast<std::size_t>(cellCountY));

    for (int cellX = 0; cellX < cellCountX; ++cellX) {
        for (int cellY = 0; cellY < cellCountY; ++cellY) {
            const std::size_t cellIndex =
                static_cast<std::size_t>(cellY) *
                    static_cast<std::size_t>(cellCountX) +
                static_cast<std::size_t>(cellX);
            yTransformed[static_cast<std::size_t>(cellY)] =
                pointwiseLiMinus(cells[cellIndex], 1, tensorName);
        }

        LiTensorBlocks afterYFourier;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                std::vector<Complex> coefficients(
                    static_cast<std::size_t>(coefficientCountY));
                for (int order = -coefficientHalfY;
                     order <= coefficientHalfY;
                     ++order) {
                    Complex coefficient{};
                    const auto& weights = yWeights[
                        static_cast<std::size_t>(order + coefficientHalfY)];
                    if (weights.size() != static_cast<std::size_t>(cellCountY)) {
                        throw std::invalid_argument(
                            "RETICOLO Li y weights do not match the cell partition");
                    }
                    for (int cellY = 0; cellY < cellCountY; ++cellY) {
                        coefficient +=
                            yTransformed[static_cast<std::size_t>(cellY)](row, column) *
                            weights[static_cast<std::size_t>(cellY)];
                    }
                    coefficients[static_cast<std::size_t>(order + coefficientHalfY)] =
                        coefficient;
                }
                afterYFourier[row][column] =
                    toeplitzConvMatrix(coefficients, retainedCountY);
            }
        }

        const LiTensorBlocks afterY = liBlockTransform(
            afterYFourier, 1, true, tensorName);
        const LiTensorBlocks beforeX = liBlockTransform(
            afterY, 0, false, tensorName);
        if (xWeights.front().size() != static_cast<std::size_t>(cellCountX)) {
            throw std::invalid_argument(
                "RETICOLO Li x weights do not match the cell partition");
        }
        accumulatePartialTransform(
            xCoefficients, beforeX, xWeights, cellX);
    }

    LiTensorBlocks afterXFourier = makeXOuterBlockToeplitz(
        std::move(xCoefficients), retainedCountX, retainedCountY);
    LiTensorBlocks rectangular = liBlockTransform(
        afterXFourier, 0, true, tensorName);
    const auto projection = rectangularProjectionIndices(
        retainedCountX, retainedCountY, basis);

    LiFactorization result;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            result.q[row][column] = projectRectangularMatrix(
                rectangular[row][column], projection);
        }
    }
    result.zzInverse = inverseBlock(
        result.q[2][2], tensorName, "zz block");
    return result;
}

} // namespace

LiFactorization liFactorizeUniform(const Tensor3& tensor,
                                   std::size_t harmonicCount,
                                   std::string_view tensorName) {
    if (harmonicCount == 0) {
        throw std::invalid_argument("uniform Li factorization requires at least one harmonic");
    }
    const Complex zz = tensor(2, 2);
    if (!isFinite(zz) || std::abs(zz) == Real{0}) {
        throw std::runtime_error(
            std::string(tensorName) + " zz block is singular during Li factorization");
    }
    LiFactorization result;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            result.q[row][column] =
                Matrix::scaledIdentity(harmonicCount, tensor(row, column));
        }
    }
    result.zzInverse = Matrix::scaledIdentity(harmonicCount, one / zz);
    return result;
}

LiFactorization liFactorizeDirect(const LiTensorBlocks& direct,
                                  std::string_view tensorName) {
    LiFactorization result;
    result.q = direct;
    result.zzInverse = inverseBlock(
        result.q[2][2], tensorName, "zz block");
    return result;
}

LiFactorization liFactorizeScalarLamellar(const Matrix& direct,
                                          const Matrix& reciprocal,
                                          LiInterfaceAxis normalAxis,
                                          std::string_view tensorName) {
    requireSquare(direct, std::string(tensorName) + " direct convolution");
    requireSquare(reciprocal, std::string(tensorName) + " reciprocal convolution");
    if (direct.rows() != reciprocal.rows()) {
        throw std::invalid_argument(
            std::string(tensorName) + " direct and reciprocal convolutions must have equal size");
    }

    const std::size_t size = direct.rows();
    const Matrix zero(size, size);
    LiFactorization result;
    for (auto& row : result.q) {
        for (Matrix& block : row) {
            block = zero;
        }
    }
    result.q[0][0] = direct;
    result.q[1][1] = direct;
    result.q[2][2] = direct;
    const std::size_t normal = static_cast<std::size_t>(normalAxis);
    result.q[normal][normal] = inverseBlock(
        reciprocal, tensorName, "reciprocal normal block");
    result.zzInverse = inverseBlock(
        result.q[2][2], tensorName, "zz block");
    return result;
}

LiFactorization liFactorizeTensorLamellar(
    LiInterfaceAxis normalAxis,
    const LiConvolutionBuilder& convolution,
    std::string_view tensorName) {
    if (!convolution) {
        throw std::invalid_argument("tensor Li factorization requires a convolution builder");
    }
    const std::size_t pivot = static_cast<std::size_t>(normalAxis);
    LiTensorBlocks afterFourier;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            afterFourier[row][column] = convolution(
                [=](const Tensor3& tensor) {
                    return pointwiseLiMinusComponent(
                        tensor,
                        pivot,
                        row,
                        column,
                        pointwisePivotInverse(tensor, pivot, tensorName));
                });
        }
    }

    LiFactorization result;
    result.q = liBlockTransform(afterFourier, pivot, true, tensorName);
    result.zzInverse = inverseBlock(
        result.q[2][2], tensorName, "zz block");
    return result;
}

LiFactorization liFactorizeTensor2dReticolo(
    const std::vector<Tensor3>& samples,
    int sampleCountX,
    int sampleCountY,
    const HarmonicBasis& basis,
    std::string_view tensorName) {
    const auto expected = checkedGridSize(
        sampleCountX, sampleCountY, "RETICOLO Li sample");
    if (samples.size() != expected) {
        throw std::invalid_argument(
            "RETICOLO Li sample array size must equal sample_count_x * sample_count_y");
    }
    const int retainedCountX = oddCountForOrder(basis.orderX, "RETICOLO Li x");
    const int retainedCountY = oddCountForOrder(basis.orderY, "RETICOLO Li y");
    const int coefficientCountX = convolutionCount(retainedCountX, "RETICOLO Li x");
    const int coefficientCountY = convolutionCount(retainedCountY, "RETICOLO Li y");
    const auto xWeights = centeredFourierWeights(sampleCountX, coefficientCountX);
    const auto yWeights = centeredFourierWeights(sampleCountY, coefficientCountY);
    return liFactorizeTensor2dReticoloImpl(
        samples,
        sampleCountX,
        sampleCountY,
        basis,
        tensorName,
        xWeights,
        yWeights);
}

LiFactorization liFactorizeTensor2dReticoloCells(
    const std::vector<Tensor3>& cells,
    const std::vector<Real>& xBoundaries,
    const std::vector<Real>& yBoundaries,
    const HarmonicBasis& basis,
    std::string_view tensorName) {
    if (xBoundaries.size() < 2 || yBoundaries.size() < 2) {
        throw std::invalid_argument(
            "RETICOLO Li cell factorization requires x and y boundaries");
    }
    const int cellCountX = static_cast<int>(xBoundaries.size() - 1);
    const int cellCountY = static_cast<int>(yBoundaries.size() - 1);
    const int retainedCountX = oddCountForOrder(basis.orderX, "RETICOLO Li x");
    const int retainedCountY = oddCountForOrder(basis.orderY, "RETICOLO Li y");
    const auto xWeights = piecewiseFourierWeights(
        xBoundaries, convolutionCount(retainedCountX, "RETICOLO Li x"));
    const auto yWeights = piecewiseFourierWeights(
        yBoundaries, convolutionCount(retainedCountY, "RETICOLO Li y"));
    return liFactorizeTensor2dReticoloImpl(
        cells,
        cellCountX,
        cellCountY,
        basis,
        tensorName,
        xWeights,
        yWeights);
}

} // namespace rcwa::detail
