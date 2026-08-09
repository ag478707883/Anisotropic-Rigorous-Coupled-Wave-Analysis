#include "rcwa/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef RCWA_HAS_LAPACKE
#ifdef RCWA_HAS_MKL
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#endif

#ifdef RCWA_HAS_CBLAS
#ifdef RCWA_HAS_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef RCWA_HAS_LAPACKE
#error "RCWA requires LAPACKE for solve_linear() and eig()."
#endif

#ifndef RCWA_HAS_CBLAS
#error "RCWA requires CBLAS for dense matrix multiplication."
#endif

namespace rcwa {

namespace {

void requireSameShape(const Matrix& a, const Matrix& b, const char* op) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        std::ostringstream message;
        message << op << " expects matching matrix shapes, got "
                << a.rows() << "x" << a.cols() << " and "
                << b.rows() << "x" << b.cols();
        throw std::invalid_argument(message.str());
    }
}

void requireBlockInBounds(const Matrix& m,
                             std::size_t r,
                             std::size_t c,
                             std::size_t nr,
                             std::size_t nc,
                             const char* op) {
    if (r > m.rows() || c > m.cols() || nr > m.rows() - r || nc > m.cols() - c) {
        std::ostringstream message;
        message << op << " range is outside matrix bounds: matrix="
                << m.rows() << "x" << m.cols()
                << ", row=" << r << ", col=" << c
                << ", rows=" << nr << ", cols=" << nc;
        throw std::out_of_range(message.str());
    }
}

} // namespace

const char* linearAlgebraBackendDetail() {
#ifdef RCWA_HAS_MKL
    return "Intel oneAPI MKL LAPACKE";
#else
    return "LAPACKE";
#endif
}

const char* matrixMultiplyBackendDetail() {
#ifdef RCWA_HAS_MKL
    return "Intel oneAPI MKL CBLAS";
#else
    return "CBLAS";
#endif
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : mRows(rows), mCols(cols), mData(checkedProduct(rows, cols, "matrix"), Complex{}) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, Complex fill)
    : mRows(rows), mCols(cols), mData(checkedProduct(rows, cols, "matrix"), fill) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<Complex> data)
    : mRows(rows), mCols(cols), mData(std::move(data)) {
    const std::size_t expected = checkedProduct(rows, cols, "matrix");
    if (mData.size() != expected) {
        std::ostringstream message;
        message << "matrix data size must equal rows * cols, got "
                << mData.size() << " values for "
                << rows << "x" << cols;
        throw std::invalid_argument(message.str());
    }
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return {rows, cols};
}

Matrix Matrix::identity(std::size_t n) {
    return scaledIdentity(n, {1.0, 0.0});
}

Matrix Matrix::scaledIdentity(std::size_t n, Complex value) {
    Matrix out(n, n);
    auto& data = out.mData;
    for (std::size_t i = 0; i < n; ++i) {
        data[i * n + i] = value;
    }
    return out;
}

Matrix Matrix::diagonal(const std::vector<Complex>& d) {
    Matrix out(d.size(), d.size());
    auto& data = out.mData;
    const std::size_t n = d.size();
    for (std::size_t i = 0; i < d.size(); ++i) {
        data[i * n + i] = d[i];
    }
    return out;
}

Complex& Matrix::operator()(std::size_t r, std::size_t c) {
    return mData[r * mCols + c];
}

const Complex& Matrix::operator()(std::size_t r, std::size_t c) const {
    return mData[r * mCols + c];
}

Matrix Matrix::block(std::size_t r, std::size_t c, std::size_t nr, std::size_t nc) const {
    requireBlockInBounds(*this, r, c, nr, nc, "matrix block");
    Matrix out(nr, nc);
    for (std::size_t i = 0; i < nr; ++i) {
        const auto* src = mData.data() + (r + i) * mCols + c;
        auto* dst = out.mData.data() + i * nc;
        std::copy_n(src, nc, dst);
    }
    return out;
}

void Matrix::setBlock(std::size_t r, std::size_t c, const Matrix& src) {
    requireBlockInBounds(*this, r, c, src.rows(), src.cols(), "matrix set_block");
    for (std::size_t i = 0; i < src.rows(); ++i) {
        const auto* srcRow = src.mData.data() + i * src.mCols;
        auto* dstRow = mData.data() + (r + i) * mCols + c;
        std::copy_n(srcRow, src.mCols, dstRow);
    }
}

void Matrix::reset(std::size_t rows, std::size_t cols, Complex fill) {
    const std::size_t count = checkedProduct(rows, cols, "matrix workspace");
    mRows = rows;
    mCols = cols;
    mData.resize(count);
    std::fill(mData.begin(), mData.end(), fill);
}

void Matrix::resizeForOverwrite(std::size_t rows, std::size_t cols) {
    const std::size_t count = checkedProduct(rows, cols, "matrix workspace");
    mRows = rows;
    mCols = cols;
    mData.resize(count);
}

DiagonalOperator::DiagonalOperator(std::vector<Complex> values)
    : mValues(std::move(values)) {}

Matrix DiagonalOperator::toDense() const {
    return Matrix::diagonal(mValues);
}

Matrix add(const Matrix& a, const Matrix& b) {
    requireSameShape(a, b, "matrix addition");
    Matrix out(a.rows(), a.cols());
    const auto& aData = a.data();
    const auto& bData = b.data();
    auto& outData = out.data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        outData[i] = aData[i] + bData[i];
    }
    return out;
}

Matrix subtract(const Matrix& a, const Matrix& b) {
    requireSameShape(a, b, "matrix subtraction");
    Matrix out(a.rows(), a.cols());
    const auto& aData = a.data();
    const auto& bData = b.data();
    auto& outData = out.data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        outData[i] = aData[i] - bData[i];
    }
    return out;
}

Matrix negate(const Matrix& a) {
    Matrix out(a.rows(), a.cols());
    const auto& aData = a.data();
    auto& outData = out.data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        outData[i] = -aData[i];
    }
    return out;
}

Matrix scale(Complex s, const Matrix& a) {
    Matrix out(a.rows(), a.cols());
    const auto& aData = a.data();
    auto& outData = out.data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        outData[i] = s * aData[i];
    }
    return out;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    const std::size_t aRows = a.rows();
    const std::size_t aCols = a.cols();
    const std::size_t bCols = b.cols();
    if (aCols != b.rows()) {
        std::ostringstream message;
        message << "matrix multiplication expects lhs columns to match rhs rows, got "
                << aRows << "x" << aCols << " and "
                << b.rows() << "x" << b.cols();
        throw std::invalid_argument(message.str());
    }
    Matrix out(aRows, bCols);
    if (aRows == 0 || aCols == 0 || bCols == 0) {
        return out;
    }
    if (aRows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        aCols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        bCols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed CBLAS integer range");
    }
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};
    const Complex* aPtr = a.data().data();
    const Complex* bPtr = b.data().data();
    Complex* outPtr = out.data().data();
    if (bCols == 1) {
#ifdef RCWA_SINGLE_PRECISION
        cblas_cgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(aCols),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            outPtr,
            1);
#else
        cblas_zgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(aCols),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            outPtr,
            1);
#endif
        return out;
    }
#ifdef RCWA_SINGLE_PRECISION
    cblas_cgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(aCols),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        outPtr,
        static_cast<int>(bCols));
#else
    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(aCols),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        outPtr,
        static_cast<int>(bCols));
#endif
    return out;
}

Matrix multiply(const DiagonalOperator& a, const Matrix& b) {
    return leftScaleRows(a.values(), b);
}

Matrix leftScaleRows(
    const std::vector<Complex>& diagonal,
    const Matrix& matrix) {
    if (diagonal.size() != matrix.rows()) {
        std::ostringstream message;
        message << "left diagonal multiplication expects operator size to match "
                << "matrix rows, got " << diagonal.size() << " and "
                << matrix.rows() << "x" << matrix.cols();
        throw std::invalid_argument(message.str());
    }
    Matrix out(matrix.rows(), matrix.cols());
    const auto& source = matrix.data();
    auto& target = out.data();
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const Complex factor = diagonal[row];
        const std::size_t offset = row * matrix.cols();
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            target[offset + column] = factor * source[offset + column];
        }
    }
    return out;
}

Matrix multiply(const Matrix& a, const DiagonalOperator& b) {
    return rightScaleColumns(a, b.values());
}

Matrix rightScaleColumns(
    const Matrix& matrix,
    const std::vector<Complex>& diagonal) {
    if (matrix.cols() != diagonal.size()) {
        std::ostringstream message;
        message << "right diagonal multiplication expects operator size to match "
                << "matrix columns, got " << matrix.rows() << "x" << matrix.cols()
                << " and " << diagonal.size();
        throw std::invalid_argument(message.str());
    }
    Matrix out(matrix.rows(), matrix.cols());
    const auto& source = matrix.data();
    auto& target = out.data();
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const std::size_t offset = row * matrix.cols();
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            target[offset + column] = source[offset + column] * diagonal[column];
        }
    }
    return out;
}

void multiplyAccumulate(Matrix& target,
                        const Matrix& a,
                        const Matrix& b,
                        Complex alpha) {
    const std::size_t aRows = a.rows();
    const std::size_t aCols = a.cols();
    const std::size_t bCols = b.cols();
    if (aCols != b.rows() || target.rows() != aRows || target.cols() != bCols) {
        std::ostringstream message;
        message << "matrix multiply-accumulate expects target="
                << aRows << "x" << bCols << ", lhs="
                << aRows << "x" << aCols << ", rhs="
                << b.rows() << "x" << bCols << ", got target="
                << target.rows() << "x" << target.cols();
        throw std::invalid_argument(message.str());
    }
    if (aRows == 0 || aCols == 0 || bCols == 0 || alpha == Complex{}) {
        return;
    }
    if (aRows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        aCols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        bCols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed CBLAS integer range");
    }
    const Complex beta{1.0, 0.0};
    const Complex* aPtr = a.data().data();
    const Complex* bPtr = b.data().data();
    Complex* targetPtr = target.data().data();
    if (bCols == 1) {
#ifdef RCWA_SINGLE_PRECISION
        cblas_cgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(aCols),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            targetPtr,
            1);
#else
        cblas_zgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(aCols),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            targetPtr,
            1);
#endif
        return;
    }
#ifdef RCWA_SINGLE_PRECISION
    cblas_cgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(aCols),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        targetPtr,
        static_cast<int>(bCols));
#else
    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(aCols),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        targetPtr,
        static_cast<int>(bCols));
#endif
}

Matrix multiplyColumnBlock(const Matrix& a,
                             std::size_t columnOffset,
                             std::size_t columnCount,
                             const Matrix& b) {
    const std::size_t aRows = a.rows();
    const std::size_t aCols = a.cols();
    const std::size_t bCols = b.cols();
    if (columnOffset > aCols || columnCount > aCols - columnOffset) {
        throw std::invalid_argument("matrix column block lies outside lhs bounds");
    }
    if (columnCount != b.rows()) {
        throw std::invalid_argument(
            "matrix column block multiplication expects rhs rows to match block width");
    }
    Matrix out(aRows, bCols);
    if (aRows == 0 || columnCount == 0 || bCols == 0) {
        return out;
    }
    if (aRows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        aCols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        columnCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        bCols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed CBLAS integer range");
    }
    const Complex alpha{1.0, 0.0};
    const Complex beta{0.0, 0.0};
    const Complex* aPtr = a.data().data() + columnOffset;
    const Complex* bPtr = b.data().data();
    Complex* outPtr = out.data().data();
    if (bCols == 1) {
#ifdef RCWA_SINGLE_PRECISION
        cblas_cgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(columnCount),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            outPtr,
            1);
#else
        cblas_zgemv(
            CblasRowMajor,
            CblasNoTrans,
            static_cast<int>(aRows),
            static_cast<int>(columnCount),
            &alpha,
            aPtr,
            static_cast<int>(aCols),
            bPtr,
            1,
            &beta,
            outPtr,
            1);
#endif
        return out;
    }
#ifdef RCWA_SINGLE_PRECISION
    cblas_cgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(columnCount),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        outPtr,
        static_cast<int>(bCols));
#else
    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(aRows),
        static_cast<int>(bCols),
        static_cast<int>(columnCount),
        &alpha,
        aPtr,
        static_cast<int>(aCols),
        bPtr,
        static_cast<int>(bCols),
        &beta,
        outPtr,
        static_cast<int>(bCols));
#endif
    return out;
}

Real maxAbs(const Matrix& a) {
    Real m = 0.0;
    for (const auto& v : a.data()) {
        m = std::max(m, std::abs(v));
    }
    return m;
}

Real maxAbs(const DiagonalOperator& a) {
    Real maximum = 0.0;
    for (const Complex value : a.values()) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

namespace {

typedef
#ifdef RCWA_SINGLE_PRECISION
    lapack_complex_float
#else
    lapack_complex_double
#endif
        LapackComplex;

Real lapackReal(LapackComplex z) {
#ifdef RCWA_HAS_MKL
    return static_cast<Real>(z.real());
#else
#ifdef RCWA_SINGLE_PRECISION
    return lapack_complex_float_real(z);
#else
    return lapack_complex_double_real(z);
#endif
#endif
}

Real lapackImag(LapackComplex z) {
#ifdef RCWA_HAS_MKL
    return static_cast<Real>(z.imag());
#else
#ifdef RCWA_SINGLE_PRECISION
    return lapack_complex_float_imag(z);
#else
    return lapack_complex_double_imag(z);
#endif
#endif
}

lapack_int lapackGetrf(int matrixLayout,
                        lapack_int m,
                        lapack_int n,
                        LapackComplex* a,
                        lapack_int lda,
                        lapack_int* ipiv) {
#ifdef RCWA_SINGLE_PRECISION
    return LAPACKE_cgetrf(matrixLayout, m, n, a, lda, ipiv);
#else
    return LAPACKE_zgetrf(matrixLayout, m, n, a, lda, ipiv);
#endif
}

lapack_int lapackGetrs(int matrixLayout,
                        lapack_int n,
                        lapack_int nrhs,
                        const LapackComplex* a,
                        lapack_int lda,
                        const lapack_int* ipiv,
                        LapackComplex* b,
                        lapack_int ldb) {
#ifdef RCWA_SINGLE_PRECISION
    return LAPACKE_cgetrs(
        matrixLayout, 'N', n, nrhs, a, lda, ipiv, b, ldb);
#else
    return LAPACKE_zgetrs(
        matrixLayout, 'N', n, nrhs, a, lda, ipiv, b, ldb);
#endif
}

lapack_int lapackGecon(int matrixLayout,
                        lapack_int n,
                        const LapackComplex* a,
                        lapack_int lda,
                        Real anorm,
                        Real* reciprocalCondition) {
#ifdef RCWA_SINGLE_PRECISION
    return LAPACKE_cgecon(
        matrixLayout, '1', n, a, lda, anorm, reciprocalCondition);
#else
    return LAPACKE_zgecon(
        matrixLayout, '1', n, a, lda, anorm, reciprocalCondition);
#endif
}

lapack_int lapackGeevWork(lapack_int n,
                           LapackComplex* a,
                           lapack_int lda,
                           LapackComplex* w,
                           LapackComplex* vr,
                           lapack_int ldvr,
                           LapackComplex* work,
                           lapack_int lwork,
                           Real* rwork) {
#ifdef RCWA_SINGLE_PRECISION
    return LAPACKE_cgeev_work(
        LAPACK_COL_MAJOR,
        'N',
        'V',
        n,
        a,
        lda,
        w,
        nullptr,
        n,
        vr,
        ldvr,
        work,
        lwork,
        rwork);
#else
    return LAPACKE_zgeev_work(
        LAPACK_COL_MAJOR,
        'N',
        'V',
        n,
        a,
        lda,
        w,
        nullptr,
        n,
        vr,
        ldvr,
        work,
        lwork,
        rwork);
#endif
}

lapack_int lapackRealGeevWork(lapack_int n,
                              Real* a,
                              Real* realValues,
                              Real* imagValues,
                              Real* vectors,
                              Real* work,
                              lapack_int lwork) {
#ifdef RCWA_SINGLE_PRECISION
    return LAPACKE_sgeev_work(
        LAPACK_COL_MAJOR,
        'N',
        'V',
        n,
        a,
        n,
        realValues,
        imagValues,
        nullptr,
        n,
        vectors,
        n,
        work,
        lwork);
#else
    return LAPACKE_dgeev_work(
        LAPACK_COL_MAJOR,
        'N',
        'V',
        n,
        a,
        n,
        realValues,
        imagValues,
        nullptr,
        n,
        vectors,
        n,
        work,
        lwork);
#endif
}

void copyToColMajorLapack(const Matrix& source,
                           std::vector<LapackComplex>& destination) {
    const std::size_t rows = source.rows();
    const std::size_t cols = source.cols();
    destination.resize(rows * cols);
    const auto& src = source.data();
    // LAPACK is called in column-major mode to avoid the allocation and
    // transpose performed by LAPACKE's row-major wrappers.  Tile our explicit
    // transpose so both the row-major source and column-major destination stay
    // resident in cache for the whole block.
    constexpr std::size_t tileSize = 32;
    for (std::size_t r0 = 0; r0 < rows; r0 += tileSize) {
        const std::size_t rEnd = std::min(rows, r0 + tileSize);
        for (std::size_t c0 = 0; c0 < cols; c0 += tileSize) {
            const std::size_t cEnd = std::min(cols, c0 + tileSize);
            for (std::size_t r = r0; r < rEnd; ++r) {
                const std::size_t srcBase = r * cols;
                for (std::size_t c = c0; c < cEnd; ++c) {
                    const Complex z = src[srcBase + c];
#ifdef RCWA_HAS_MKL
                    destination[c * rows + r] = z;
#else
#ifdef RCWA_SINGLE_PRECISION
                    destination[c * rows + r] =
                        lapack_make_complex_float(std::real(z), std::imag(z));
#else
                    destination[c * rows + r] =
                        lapack_make_complex_double(std::real(z), std::imag(z));
#endif
#endif
                }
            }
        }
    }
}

Matrix fromColMajorLapack(const std::vector<LapackComplex>& data,
                           std::size_t rows,
                           std::size_t cols) {
    Matrix out(rows, cols);
    auto& dst = out.data();
    constexpr std::size_t tileSize = 32;
    for (std::size_t r0 = 0; r0 < rows; r0 += tileSize) {
        const std::size_t rEnd = std::min(rows, r0 + tileSize);
        for (std::size_t c0 = 0; c0 < cols; c0 += tileSize) {
            const std::size_t cEnd = std::min(cols, c0 + tileSize);
            for (std::size_t r = r0; r < rEnd; ++r) {
                const std::size_t dstBase = r * cols;
                for (std::size_t c = c0; c < cEnd; ++c) {
                    const auto z = data[c * rows + r];
#ifdef RCWA_HAS_MKL
                    dst[dstBase + c] = z;
#else
                    dst[dstBase + c] =
                        Complex{lapackReal(z), lapackImag(z)};
#endif
                }
            }
        }
    }
    return out;
}

bool allFinite(const Matrix& m) {
    for (const auto& z : m.data()) {
        if (!std::isfinite(std::real(z)) || !std::isfinite(std::imag(z))) {
            return false;
        }
    }
    return true;
}

Real lapackComplexOneAbs(Complex value) noexcept {
    return std::abs(std::real(value)) + std::abs(std::imag(value));
}

Real matrixOneNorm(const Matrix& matrix) {
    Real norm = 0.0;
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        Real sum = 0.0;
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            sum += lapackComplexOneAbs(matrix(row, col));
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

struct LinearSolveWorkspace {
    std::vector<LapackComplex> a;
    std::vector<LapackComplex> b;
    std::vector<lapack_int> pivots;
};

struct EigenWorkspace {
    std::vector<LapackComplex> a;
    std::vector<LapackComplex> values;
    std::vector<LapackComplex> vectors;
    std::vector<LapackComplex> work;
    std::vector<Real> realWork;
    std::vector<std::pair<lapack_int, lapack_int>> optimalWorkSizes;
};

struct RealEigenWorkspace {
    std::vector<Real> a;
    std::vector<Real> realValues;
    std::vector<Real> imagValues;
    std::vector<Real> vectors;
    std::vector<Real> work;
    std::vector<std::pair<lapack_int, lapack_int>> optimalWorkSizes;
};

thread_local LinearSolveWorkspace linearSolveWorkspace;
thread_local EigenWorkspace eigenWorkspace;
thread_local RealEigenWorkspace realEigenWorkspace;

lapack_int optimalGeevWorkSize(EigenWorkspace& workspace,
                                lapack_int n,
                                LapackComplex* a,
                                LapackComplex* values,
                                LapackComplex* vectors) {
    for (const auto& [cachedN, cachedSize] : workspace.optimalWorkSizes) {
        if (cachedN == n) {
            return cachedSize;
        }
    }

    LapackComplex query{};
    workspace.realWork.resize(std::max<std::size_t>(1, 2 * static_cast<std::size_t>(n)));
    const lapack_int info = lapackGeevWork(
        n,
        a,
        n,
        values,
        vectors,
        n,
        &query,
        -1,
        workspace.realWork.data());
    if (info != 0) {
        throw std::runtime_error("LAPACKE geev workspace query failed");
    }
    const auto requested = static_cast<lapack_int>(std::max<Real>(
        Real{2} * static_cast<Real>(n),
        lapackReal(query)));
    workspace.optimalWorkSizes.emplace_back(n, requested);
    return requested;
}

lapack_int optimalRealGeevWorkSize(RealEigenWorkspace& workspace,
                                   lapack_int n,
                                   Real* a,
                                   Real* realValues,
                                   Real* imagValues,
                                   Real* vectors) {
    for (const auto& [cachedN, cachedSize] : workspace.optimalWorkSizes) {
        if (cachedN == n) {
            return cachedSize;
        }
    }

    Real query{};
    const lapack_int info = lapackRealGeevWork(
        n,
        a,
        realValues,
        imagValues,
        vectors,
        &query,
        -1);
    if (info != 0) {
        throw std::runtime_error("LAPACKE real geev workspace query failed");
    }
    const auto requested = static_cast<lapack_int>(std::max<Real>(
        Real{4} * static_cast<Real>(n),
        query));
    workspace.optimalWorkSizes.emplace_back(n, requested);
    return requested;
}

bool matrixIsExactlyReal(const Matrix& matrix) {
    return std::all_of(
        matrix.data().begin(),
        matrix.data().end(),
        [](Complex value) { return std::imag(value) == Real{0}; });
}

Eigensystem realEig(const Matrix& A) {
    const std::size_t size = A.rows();
    const auto n = static_cast<lapack_int>(size);
    auto& workspace = realEigenWorkspace;
    workspace.a.resize(size * size);
    const auto& source = A.data();
    constexpr std::size_t tileSize = 32;
    for (std::size_t r0 = 0; r0 < size; r0 += tileSize) {
        const std::size_t rEnd = std::min(size, r0 + tileSize);
        for (std::size_t c0 = 0; c0 < size; c0 += tileSize) {
            const std::size_t cEnd = std::min(size, c0 + tileSize);
            for (std::size_t r = r0; r < rEnd; ++r) {
                const std::size_t sourceBase = r * size;
                for (std::size_t c = c0; c < cEnd; ++c) {
                    workspace.a[c * size + r] = std::real(source[sourceBase + c]);
                }
            }
        }
    }
    workspace.realValues.resize(size);
    workspace.imagValues.resize(size);
    workspace.vectors.resize(size * size);
    const lapack_int lwork = optimalRealGeevWorkSize(
        workspace,
        n,
        workspace.a.data(),
        workspace.realValues.data(),
        workspace.imagValues.data(),
        workspace.vectors.data());
    workspace.work.resize(static_cast<std::size_t>(lwork));
    const lapack_int info = lapackRealGeevWork(
        n,
        workspace.a.data(),
        workspace.realValues.data(),
        workspace.imagValues.data(),
        workspace.vectors.data(),
        workspace.work.data(),
        lwork);
    if (info < 0) {
        throw std::runtime_error("LAPACKE real geev received an invalid argument");
    }
    if (info > 0) {
        throw std::runtime_error("real eigenvalue decomposition failed to converge");
    }

    Eigensystem out;
    out.values.resize(size);
    out.vectors = Matrix(size, size);
    auto& vectors = out.vectors.data();
    std::size_t column = 0;
    while (column < size) {
        const Real imag = workspace.imagValues[column];
        out.values[column] = {workspace.realValues[column], imag};
        if (imag == Real{0}) {
            for (std::size_t row = 0; row < size; ++row) {
                vectors[row * size + column] = {
                    workspace.vectors[column * size + row],
                    Real{0}};
            }
            ++column;
            continue;
        }
        if (!(imag > Real{0}) || column + 1 >= size ||
            workspace.imagValues[column + 1] != -imag) {
            throw std::runtime_error(
                "real eigenvalue decomposition returned a malformed conjugate pair");
        }
        out.values[column + 1] = {
            workspace.realValues[column + 1],
            workspace.imagValues[column + 1]};
        for (std::size_t row = 0; row < size; ++row) {
            const Real realPart = workspace.vectors[column * size + row];
            const Real imagPart = workspace.vectors[(column + 1) * size + row];
            vectors[row * size + column] = {realPart, imagPart};
            vectors[row * size + column + 1] = {realPart, -imagPart};
        }
        column += 2;
    }
    if (!allFinite(out.vectors)) {
        throw std::runtime_error(
            "real eigenvalue decomposition produced non-finite vectors");
    }
    return out;
}

std::optional<Matrix> solveLinearImpl(const Matrix& A,
                                       const Matrix& B,
                                       std::optional<Real> minimumReciprocalCondition) {
    const auto n = static_cast<lapack_int>(A.rows());
    const auto nrhs = static_cast<lapack_int>(B.cols());
    auto& workspace = linearSolveWorkspace;
    copyToColMajorLapack(A, workspace.a);
    copyToColMajorLapack(B, workspace.b);
    workspace.pivots.resize(static_cast<std::size_t>(n));

    const lapack_int factorInfo = lapackGetrf(
        LAPACK_COL_MAJOR,
        n,
        n,
        workspace.a.data(),
        n,
        workspace.pivots.data());
    if (factorInfo < 0) {
        throw std::runtime_error("LAPACKE getrf received an invalid argument");
    }
    if (factorInfo > 0) {
        if (minimumReciprocalCondition) {
            return std::nullopt;
        }
        throw std::runtime_error("linear solve failed: matrix is singular");
    }

    if (minimumReciprocalCondition) {
        Real reciprocalCondition{};
        const lapack_int conditionInfo = lapackGecon(
            LAPACK_COL_MAJOR,
            n,
            workspace.a.data(),
            n,
            matrixOneNorm(A),
            &reciprocalCondition);
        if (conditionInfo != 0 || !std::isfinite(reciprocalCondition) ||
            reciprocalCondition < *minimumReciprocalCondition) {
            return std::nullopt;
        }
    }

    const lapack_int solveInfo = lapackGetrs(
        LAPACK_COL_MAJOR,
        n,
        nrhs,
        workspace.a.data(),
        n,
        workspace.pivots.data(),
        workspace.b.data(),
        n);
    if (solveInfo != 0) {
        throw std::runtime_error("LAPACKE getrs failed to solve the factored system");
    }
    Matrix solution = fromColMajorLapack(workspace.b, A.rows(), B.cols());
    if (!allFinite(solution)) {
        throw std::runtime_error("linear solve failed: non-finite result");
    }
    return solution;
}

} // namespace

Matrix solveLinear(const Matrix& A, const Matrix& B) {
    if (A.rows() != A.cols() || A.rows() != B.rows()) {
        throw std::invalid_argument("solve_linear expects A square and compatible with B");
    }
    if (A.rows() > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max()) ||
        B.cols() > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed LAPACK integer range");
    }
    if (B.cols() == 0) {
        return Matrix(A.rows(), 0);
    }
    if (!allFinite(A) || !allFinite(B)) {
        throw std::runtime_error("linear solve failed: input matrix contains non-finite values");
    }

    return std::move(*solveLinearImpl(A, B, std::nullopt));
}

std::optional<Matrix> trySolveLinearWellConditioned(
    const Matrix& A,
    const Matrix& B,
    Real minimumReciprocalCondition) {
    if (A.rows() != A.cols() || A.rows() != B.rows()) {
        throw std::invalid_argument(
            "conditioned linear solve expects A square and compatible with B");
    }
    if (!(minimumReciprocalCondition >= Real{0}) ||
        !std::isfinite(minimumReciprocalCondition)) {
        throw std::invalid_argument(
            "minimum reciprocal condition must be finite and non-negative");
    }
    if (A.rows() > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max()) ||
        B.cols() > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed LAPACK integer range");
    }
    if (B.cols() == 0) {
        return Matrix(A.rows(), 0);
    }
    if (!allFinite(A) || !allFinite(B)) {
        return std::nullopt;
    }
    return solveLinearImpl(A, B, minimumReciprocalCondition);
}

Eigensystem eig(const Matrix& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("eig expects a square matrix");
    }
    if (A.rows() == 0) {
        return {};
    }
    if (A.rows() > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw std::invalid_argument("matrix dimensions exceed LAPACK integer range");
    }
    if (matrixIsExactlyReal(A)) {
        return realEig(A);
    }

    const auto n = static_cast<lapack_int>(A.rows());
    auto& workspace = eigenWorkspace;
    copyToColMajorLapack(A, workspace.a);
    workspace.values.resize(static_cast<std::size_t>(n));
    workspace.vectors.resize(
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    const lapack_int lwork = optimalGeevWorkSize(
        workspace,
        n,
        workspace.a.data(),
        workspace.values.data(),
        workspace.vectors.data());
    workspace.work.resize(static_cast<std::size_t>(lwork));
    workspace.realWork.resize(std::max<std::size_t>(
        1,
        2 * static_cast<std::size_t>(n)));
    const lapack_int info = lapackGeevWork(
        n,
        workspace.a.data(),
        n,
        workspace.values.data(),
        workspace.vectors.data(),
        n,
        workspace.work.data(),
        lwork,
        workspace.realWork.data());
    if (info < 0) {
        throw std::runtime_error("LAPACKE geev received an invalid argument");
    }
    if (info > 0) {
        throw std::runtime_error("complex eigenvalue decomposition failed to converge");
    }

    Eigensystem out;
    out.values.resize(static_cast<std::size_t>(n));
    for (lapack_int i = 0; i < n; ++i) {
        const auto z = workspace.values[static_cast<std::size_t>(i)];
        out.values[static_cast<std::size_t>(i)] =
            Complex{lapackReal(z), lapackImag(z)};
    }
    out.vectors = fromColMajorLapack(workspace.vectors, A.rows(), A.rows());
    if (!allFinite(out.vectors)) {
        throw std::runtime_error("complex eigenvalue decomposition produced non-finite vectors");
    }
    return out;
}

Matrix inverse(const Matrix& A) {
    return solveLinear(A, Matrix::identity(A.rows()));
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        os << "[ ";
        for (std::size_t j = 0; j < m.cols(); ++j) {
            const auto& v = m(i, j);
            os << "(" << std::real(v);
            if (std::imag(v) >= 0.0) {
                os << "+";
            }
            os << std::imag(v) << "j)";
            if (j + 1 < m.cols()) {
                os << "  ";
            }
        }
        os << " ]\n";
    }
    return os;
}

} // namespace rcwa
