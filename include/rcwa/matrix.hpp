#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <optional>
#include <vector>

#include "rcwa/types.hpp"

namespace rcwa {

[[nodiscard]] const char* linearAlgebraBackendDetail();
[[nodiscard]] const char* matrixMultiplyBackendDetail();

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, Complex fill);
    Matrix(std::size_t rows, std::size_t cols, std::vector<Complex> data);

    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix identity(std::size_t n);
    static Matrix scaledIdentity(std::size_t n, Complex value);
    static Matrix diagonal(const std::vector<Complex>& d);

    [[nodiscard]] std::size_t rows() const noexcept { return mRows; }
    [[nodiscard]] std::size_t cols() const noexcept { return mCols; }
    [[nodiscard]] bool empty() const noexcept { return mRows == 0 || mCols == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return mData.size(); }

    Complex& operator()(std::size_t r, std::size_t c);
    const Complex& operator()(std::size_t r, std::size_t c) const;

    [[nodiscard]] std::vector<Complex>& data() noexcept { return mData; }
    [[nodiscard]] const std::vector<Complex>& data() const noexcept { return mData; }

    [[nodiscard]] Matrix block(std::size_t r,
                               std::size_t c,
                               std::size_t nr,
                               std::size_t nc) const;
    void setBlock(std::size_t r, std::size_t c, const Matrix& src);

    // Internal workspace helpers. resizeForOverwrite retains allocation and
    // deliberately preserves existing entries when the element count is
    // unchanged; callers must overwrite the complete logical matrix.
    void reset(std::size_t rows, std::size_t cols, Complex fill = {});
    void resizeForOverwrite(std::size_t rows, std::size_t cols);

private:
    std::size_t mRows{};
    std::size_t mCols{};
    std::vector<Complex> mData{};
};

namespace detail {

class Fingerprint64 {
public:
    explicit Fingerprint64(std::uint64_t seed) noexcept : mState(seed) {}

    void append(std::uint64_t value) noexcept {
        mState ^= value + UINT64_C(0x9e3779b97f4a7c15) +
            (mState << 6) + (mState >> 2);
    }

    void append(Real value) noexcept {
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(value));
        append(bits);
    }

    void append(Complex value) noexcept {
        append(std::real(value));
        append(std::imag(value));
    }

    void append(const Matrix& matrix) noexcept {
        append(static_cast<std::uint64_t>(matrix.rows()));
        append(static_cast<std::uint64_t>(matrix.cols()));
        for (const Complex value : matrix.data()) {
            append(value);
        }
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return mState; }
    [[nodiscard]] std::uint64_t nonzeroValue() const noexcept {
        return mState == 0 ? UINT64_C(1) : mState;
    }

private:
    std::uint64_t mState{};
};

} // namespace detail

// Exact O(N) storage for diagonal Fourier-space operators such as Kx, Ky,
// and modal propagation factors.  RCWA convolution, modal, and scattering
// matrices remain dense; materializing this operator is therefore explicit.
class DiagonalOperator {
public:
    DiagonalOperator() = default;
    explicit DiagonalOperator(std::vector<Complex> values);

    [[nodiscard]] std::size_t size() const noexcept { return mValues.size(); }
    [[nodiscard]] std::size_t rows() const noexcept { return size(); }
    [[nodiscard]] std::size_t cols() const noexcept { return size(); }
    [[nodiscard]] bool empty() const noexcept { return mValues.empty(); }
    [[nodiscard]] const std::vector<Complex>& values() const noexcept {
        return mValues;
    }
    [[nodiscard]] const Complex& operator[](std::size_t index) const {
        return mValues[index];
    }
    [[nodiscard]] Complex operator()(std::size_t row, std::size_t column) const {
        return row == column ? mValues[row] : Complex{};
    }
    [[nodiscard]] Matrix toDense() const;

private:
    std::vector<Complex> mValues{};
};

[[nodiscard]] Matrix add(const Matrix& a, const Matrix& b);
[[nodiscard]] Matrix subtract(const Matrix& a, const Matrix& b);
[[nodiscard]] Matrix negate(const Matrix& a);
[[nodiscard]] Matrix scale(Complex s, const Matrix& a);
[[nodiscard]] Matrix multiply(const Matrix& a, const Matrix& b);
[[nodiscard]] Matrix multiply(const DiagonalOperator& a, const Matrix& b);
[[nodiscard]] Matrix multiply(const Matrix& a, const DiagonalOperator& b);
[[nodiscard]] Matrix leftScaleRows(
    const std::vector<Complex>& diagonal,
    const Matrix& matrix);
[[nodiscard]] Matrix rightScaleColumns(
    const Matrix& matrix,
    const std::vector<Complex>& diagonal);
void multiplyAccumulate(Matrix& target,
                        const Matrix& a,
                        const Matrix& b,
                        Complex alpha = {1.0, 0.0});
[[nodiscard]] Matrix multiplyColumnBlock(const Matrix& a,
                                           std::size_t columnOffset,
                                           std::size_t columnCount,
                                           const Matrix& b);

[[nodiscard]] Matrix solveLinear(const Matrix& A, const Matrix& B);
[[nodiscard]] std::optional<Matrix> trySolveLinearWellConditioned(
    const Matrix& A,
    const Matrix& B,
    Real minimumReciprocalCondition);
[[nodiscard]] Matrix inverse(const Matrix& A);

struct Eigensystem {
    std::vector<Complex> values;
    Matrix vectors;
};

[[nodiscard]] Eigensystem eig(const Matrix& A);

[[nodiscard]] Real maxAbs(const Matrix& a);
[[nodiscard]] Real maxAbs(const DiagonalOperator& a);

[[nodiscard]] inline Matrix operator+(const Matrix& a, const Matrix& b) {
    return add(a, b);
}

[[nodiscard]] inline Matrix operator-(const Matrix& a, const Matrix& b) {
    return subtract(a, b);
}

[[nodiscard]] inline Matrix operator-(const Matrix& a) {
    return negate(a);
}

[[nodiscard]] inline Matrix operator*(const Matrix& a, const Matrix& b) {
    return multiply(a, b);
}

[[nodiscard]] inline Matrix operator*(Complex s, const Matrix& a) {
    return scale(s, a);
}

[[nodiscard]] inline Matrix operator*(const Matrix& a, Complex s) {
    return scale(s, a);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace rcwa
