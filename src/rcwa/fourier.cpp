#include "rcwa/fourier.hpp"

#include "rcwa/li_factorization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#ifdef RCWA_HAS_FFTW
#include <fftw3.h>
#endif

#ifdef RCWA_HAS_FFTW
#ifdef RCWA_SINGLE_PRECISION
#define RCWA_FFTW_COMPLEX fftwf_complex
#define RCWA_FFTW_MALLOC fftwf_malloc
#define RCWA_FFTW_FREE fftwf_free
#define RCWA_FFTW_PLAN fftwf_plan
#define RCWA_FFTW_PLAN_DFT_1D fftwf_plan_dft_1d
#define RCWA_FFTW_PLAN_DFT_2D fftwf_plan_dft_2d
#define RCWA_FFTW_EXECUTE fftwf_execute
#define RCWA_FFTW_EXECUTE_DFT fftwf_execute_dft
#define RCWA_FFTW_DESTROY_PLAN fftwf_destroy_plan
#else
#define RCWA_FFTW_COMPLEX fftw_complex
#define RCWA_FFTW_MALLOC fftw_malloc
#define RCWA_FFTW_FREE fftw_free
#define RCWA_FFTW_PLAN fftw_plan
#define RCWA_FFTW_PLAN_DFT_1D fftw_plan_dft_1d
#define RCWA_FFTW_PLAN_DFT_2D fftw_plan_dft_2d
#define RCWA_FFTW_EXECUTE fftw_execute
#define RCWA_FFTW_EXECUTE_DFT fftw_execute_dft
#define RCWA_FFTW_DESTROY_PLAN fftw_destroy_plan
#endif
#endif

namespace rcwa {

namespace {

void requireFiniteSamples(const std::vector<Complex>& samples, const char* context) {
    for (const Complex value : samples) {
        if (!isFinite(value)) {
            throw std::invalid_argument(
                std::string(context) + " must contain only finite values");
        }
    }
}

int halfOrder(int N) {
    if (N <= 0 || N % 2 == 0) {
        throw std::invalid_argument("RCWA harmonic count N must be a positive odd integer");
    }
    return N / 2;
}

std::size_t coeffIndex2d(int p, int q, int Nx, int Ny) {
    const int Mx = halfOrder(Nx);
    const int My = halfOrder(Ny);
    return static_cast<std::size_t>(q + My) * static_cast<std::size_t>(Nx) +
        static_cast<std::size_t>(p + Mx);
}

std::size_t harmonicIndex2d(int ix, int iy, int Nx) {
    return static_cast<std::size_t>(iy) * static_cast<std::size_t>(Nx) +
        static_cast<std::size_t>(ix);
}

void storeLiFactorization(detail::LiTensorBlocks& blocks,
                          Matrix& zzInverse,
                          detail::LiFactorization factorization) {
    blocks = std::move(factorization.q);
    zzInverse = std::move(factorization.zzInverse);
}

void buildUniformMaterial(detail::LiTensorBlocks& direct,
                          detail::LiTensorBlocks& factorized,
                          Matrix& zzInverse,
                          const Tensor3& tensor,
                          std::size_t harmonicCount,
                          std::string_view tensorName) {
    auto li = detail::liFactorizeUniform(
        tensor, harmonicCount, tensorName);
    direct = li.q;
    storeLiFactorization(factorized, zzInverse, std::move(li));
}

void buildLamellarMaterial(detail::LiTensorBlocks& direct,
                           detail::LiTensorBlocks& factorized,
                           Matrix& zzInverse,
                           detail::LiInterfaceAxis normalAxis,
                           const detail::LiConvolutionBuilder& convolution,
                           std::string_view tensorName) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            direct[row][column] = convolution(
                [=](const Tensor3& tensor) {
                    return tensor(row, column);
                });
        }
    }
    storeLiFactorization(
        factorized,
        zzInverse,
        detail::liFactorizeTensorLamellar(
            normalAxis, convolution, tensorName));
}

std::vector<Complex> coeffsForBinary(Real fillFactor,
                                       Complex ridge,
                                       Complex groove,
                                       int N,
                                       Real centerOffsetFraction) {
    if (!std::isfinite(fillFactor) || fillFactor < 0.0 || fillFactor > 1.0) {
        throw std::invalid_argument("fill_factor must be finite and in [0, 1]");
    }
    requireFinite(ridge, "binary grating ridge value");
    requireFinite(groove, "binary grating groove value");
    if (!std::isfinite(centerOffsetFraction)) {
        throw std::invalid_argument("binary grating center offset must be finite");
    }
    const int M = halfOrder(N);
    std::vector<Complex> coeffs(static_cast<std::size_t>(N));
    for (int p = -M; p <= M; ++p) {
        Complex value{};
        if (p == 0) {
            value = groove + (ridge - groove) * fillFactor;
        } else {
            const Real x = pi * static_cast<Real>(p) * fillFactor;
            const Real sinc = std::sin(x) / (pi * static_cast<Real>(p));
            value = (ridge - groove) * sinc;
            const Real phase = -twoPi * static_cast<Real>(p) * centerOffsetFraction;
            value *= std::exp(Complex{0.0, phase});
        }
        coeffs[static_cast<std::size_t>(p + M)] = value;
    }
    return coeffs;
}

int convolutionCoeffCount(int retainedCount) {
    halfOrder(retainedCount);
    if (retainedCount > (std::numeric_limits<int>::max() + std::int64_t{1}) / 2) {
        throw std::invalid_argument("convolution coefficient count exceeds integer range");
    }
    return 2 * retainedCount - 1;
}

int coeffCountForSpan(int span) {
    if (span < 0) {
        throw std::invalid_argument("harmonic basis span must be non-negative");
    }
    if (span > (std::numeric_limits<int>::max() - 1) / 2) {
        throw std::invalid_argument("harmonic basis span exceeds integer range");
    }
    return 2 * span + 1;
}

void updateBasisOrders(HarmonicBasis& basis) {
    basis.orderX = 0;
    basis.orderY = 0;
    for (const auto& order : basis.orders) {
        basis.orderX = std::max(basis.orderX, std::abs(order.m));
        basis.orderY = std::max(basis.orderY, std::abs(order.n));
    }
}

struct ReciprocalMetric2d {
    long double xWeight{};
    long double yWeight{};
};

ReciprocalMetric2d reciprocalMetric2d(Real periodXUm, Real periodYUm) {
    if (!(periodXUm > Real{0}) || !(periodYUm > Real{0}) ||
        !std::isfinite(periodXUm) || !std::isfinite(periodYUm)) {
        throw std::invalid_argument("harmonic-basis lattice periods must be positive and finite");
    }

    // The common reciprocal-length scale is irrelevant for ordering.  Scale
    // by the shorter real-space period so that both weights are in (0, 1]
    // and the comparison remains invariant if both periods change units.
    const long double px = static_cast<long double>(periodXUm);
    const long double py = static_cast<long double>(periodYUm);
    const long double minimumPeriod = std::min(px, py);
    const long double x = minimumPeriod / px;
    const long double y = minimumPeriod / py;
    return {x * x, y * y};
}

long double reciprocalRadiusSquared(const HarmonicIndex& index,
                                    const ReciprocalMetric2d& metric) {
    const long double m = static_cast<long double>(index.m);
    const long double n = static_cast<long double>(index.n);
    return m * m * metric.xWeight + n * n * metric.yWeight;
}

bool reciprocalRadiiEqual(long double a, long double b) {
    const long double scale = std::max({std::abs(a), std::abs(b), 1.0L});
    return std::abs(a - b) <=
        64.0L * std::numeric_limits<long double>::epsilon() * scale;
}

int estimatedCircularExtent(int harmonicCount,
                            long double periodRatio,
                            int maximumExtent) {
    const long double estimate =
        std::sqrt(static_cast<long double>(harmonicCount) * periodRatio /
                  static_cast<long double>(pi)) +
        2.0L;
    if (!std::isfinite(estimate) || estimate >= maximumExtent) {
        return maximumExtent;
    }
    return std::max(1, static_cast<int>(std::ceil(estimate)));
}

struct MetricHarmonicIndex {
    HarmonicIndex index;
    long double radiusSquared{};
};

std::vector<MetricHarmonicIndex> circularMetricCandidates(
    int harmonicCount,
    Real periodXUm,
    Real periodYUm) {
    const ReciprocalMetric2d metric = reciprocalMetric2d(periodXUm, periodYUm);
    const int maximumExtent = harmonicCount / 2;
    const long double px = static_cast<long double>(periodXUm);
    const long double py = static_cast<long double>(periodYUm);
    int extentX = estimatedCircularExtent(
        harmonicCount, px / py, maximumExtent);
    int extentY = estimatedCircularExtent(
        harmonicCount, py / px, maximumExtent);

    for (;;) {
        const std::size_t widthX =
            std::size_t{2} * static_cast<std::size_t>(extentX) + 1;
        const std::size_t widthY =
            std::size_t{2} * static_cast<std::size_t>(extentY) + 1;
        const std::size_t candidateCount = checkedProduct(
            widthX, widthY, "circular harmonic basis candidate");
        if (candidateCount >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "circular harmonic basis candidate grid exceeds the supported index range");
        }

        std::vector<MetricHarmonicIndex> candidates;
        candidates.reserve(candidateCount);
        for (int n = -extentY; n <= extentY; ++n) {
            for (int m = -extentX; m <= extentX; ++m) {
                const HarmonicIndex index{m, n};
                candidates.push_back({
                    index,
                    reciprocalRadiusSquared(index, metric),
                });
            }
        }
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](const MetricHarmonicIndex& a, const MetricHarmonicIndex& b) {
                if (a.radiusSquared != b.radiusSquared) {
                    return a.radiusSquared < b.radiusSquared;
                }
                if (a.index.n != b.index.n) {
                    return a.index.n < b.index.n;
                }
                return a.index.m < b.index.m;
            });

        if (candidates.size() > static_cast<std::size_t>(harmonicCount)) {
            const long double excludedRadius =
                candidates[static_cast<std::size_t>(harmonicCount)].radiusSquared;
            const HarmonicIndex firstOutsideX{extentX + 1, 0};
            const HarmonicIndex firstOutsideY{0, extentY + 1};
            const long double outsideX =
                reciprocalRadiusSquared(firstOutsideX, metric);
            const long double outsideY =
                reciprocalRadiusSquared(firstOutsideY, metric);
            if (outsideX > excludedRadius && outsideY > excludedRadius &&
                !reciprocalRadiiEqual(outsideX, excludedRadius) &&
                !reciprocalRadiiEqual(outsideY, excludedRadius)) {
                return candidates;
            }

            bool grew = false;
            if ((outsideX < excludedRadius ||
                 reciprocalRadiiEqual(outsideX, excludedRadius)) &&
                extentX < maximumExtent) {
                extentX = std::min(maximumExtent, std::max(extentX + 1, 2 * extentX));
                grew = true;
            }
            if ((outsideY < excludedRadius ||
                 reciprocalRadiiEqual(outsideY, excludedRadius)) &&
                extentY < maximumExtent) {
                extentY = std::min(maximumExtent, std::max(extentY + 1, 2 * extentY));
                grew = true;
            }
            if (grew) {
                continue;
            }
        }

        if (extentX == maximumExtent && extentY == maximumExtent) {
            return candidates;
        }
        extentX = std::min(maximumExtent, std::max(extentX + 1, 2 * extentX));
        extentY = std::min(maximumExtent, std::max(extentY + 1, 2 * extentY));
    }
}

std::vector<Complex> binaryComposite(Real fillFactor,
                                       const Tensor3& a,
                                       const Tensor3& b,
                                       const detail::LiTensorFunction& fn,
                                      int N,
                                      Real centerOffsetFraction) {
    return fourierCoeffsBinaryGrating(
        fillFactor, fn(a), fn(b), N, centerOffsetFraction);
}

std::vector<Complex> lift1dCoeffsTo2d(const std::vector<Complex>& coeffs1d,
                                          int coeffCountX,
                                          int coeffCountY) {
    const int Mx = coeffCountX / 2;
    const int My = coeffCountY / 2;
    if (coeffs1d.size() != static_cast<std::size_t>(coeffCountX)) {
        throw std::invalid_argument("1D coefficient count must match x coefficient grid");
    }
    std::vector<Complex> out(checkedGridSize(
        coeffCountX,
        coeffCountY,
        "2D Fourier coefficient"));
    for (int p = -Mx; p <= Mx; ++p) {
        out[static_cast<std::size_t>(My * coeffCountX + (p + Mx))] =
            coeffs1d[static_cast<std::size_t>(p + Mx)];
    }
    return out;
}

std::vector<Complex> lift1dYCoeffsTo2d(
    const std::vector<Complex>& coeffs1d,
    int coeffCountX,
    int coeffCountY) {
    const int Mx = coeffCountX / 2;
    const int My = coeffCountY / 2;
    if (coeffs1d.size() != static_cast<std::size_t>(coeffCountY)) {
        throw std::invalid_argument(
            "1D coefficient count must match y coefficient grid");
    }
    std::vector<Complex> out(checkedGridSize(
        coeffCountX,
        coeffCountY,
        "2D Fourier coefficient"));
    for (int q = -My; q <= My; ++q) {
        out[static_cast<std::size_t>((q + My) * coeffCountX + Mx)] =
            coeffs1d[static_cast<std::size_t>(q + My)];
    }
    return out;
}

Matrix liftedBinaryConvMatrix(Real fillFactor,
                                  const Tensor3& ridge,
                                  const Tensor3& groove,
                                  const detail::LiTensorFunction& component,
                                 int coeffCountX,
                                 int coeffCountY,
                                 int retainedCountX,
                                 int retainedCountY,
                                 Real centerOffsetFraction) {
    const auto coeffs = lift1dCoeffsTo2d(
        fourierCoeffsBinaryGrating(
            fillFactor,
            component(ridge),
            component(groove),
            coeffCountX,
            centerOffsetFraction),
        coeffCountX,
        coeffCountY);
    return convMatrix2d(
        coeffs, coeffCountX, coeffCountY, retainedCountX, retainedCountY);
}

std::vector<Complex> tensorComponentSamples(const std::vector<Tensor3>& samples,
                                              std::size_t r,
                                              std::size_t c) {
    std::vector<Complex> out(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) out[i] = samples[i](r, c);
    return out;
}

std::vector<Complex> tensorFunctionSamples(const std::vector<Tensor3>& samples,
                                             const detail::LiTensorFunction& fn) {
    std::vector<Complex> out(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        out[i] = fn(samples[i]);
    }
    return out;
}

bool allScalarTensors(const std::vector<Tensor3>& samples,
                        std::vector<Complex>* values = nullptr) {
    if (values != nullptr) {
        values->clear();
        values->reserve(samples.size());
    }
    for (const auto& t : samples) {
        Complex value{};
        if (!tensorIsScalar(t, &value)) {
            if (values != nullptr) {
                values->clear();
            }
            return false;
        }
        if (values != nullptr) {
            values->push_back(value);
        }
    }
    return true;
}

void annotateAnalyticLamellarApplicability(
    TensorFourierMatrices& tensors,
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove) {
    const bool noMaterialBoundary = fillFactor <= Real{0} || fillFactor >= Real{1} ||
        (epsRidge.v == epsGroove.v && muRidge.v == muGroove.v);
    const bool tensorMaterial = !tensorIsScalar(epsRidge) ||
        !tensorIsScalar(epsGroove) || !tensorIsScalar(muRidge) ||
        !tensorIsScalar(muGroove);
    tensors.metadata.tensorFactorizationApplicability =
        !noMaterialBoundary && tensorMaterial
        ? TensorFactorizationApplicability::CoordinateAligned
        : TensorFactorizationApplicability::NotApplicable;
    tensors.metadata.factorizationUsesSampledGeometry = false;
    tensors.metadata.requiresJointConvergence = false;
}

bool tensorSamplesAreConstant(const std::vector<Tensor3>& samples, Tensor3* value) {
    if (samples.empty()) {
        return false;
    }
    const Tensor3 first = samples.front();
    for (const auto& sample : samples) {
        if (sample.v != first.v) {
            return false;
        }
    }
    if (value != nullptr) {
        *value = first;
    }
    return true;
}

bool complexDependencyClose(Complex a, Complex b) {
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= Real{1e-12} * scale;
}

enum class SampleDependenceAxis { X, Y };

template <typename Sample, typename Equal>
bool samplesDependOnlyAlong(const std::vector<Sample>& samples,
                            int sampleCountX,
                            int sampleCountY,
                            SampleDependenceAxis axis,
                            Equal&& equal) {
    for (int y = 0; y < sampleCountY; ++y) {
        for (int x = 0; x < sampleCountX; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) *
                static_cast<std::size_t>(sampleCountX) +
                static_cast<std::size_t>(x);
            const std::size_t reference = axis == SampleDependenceAxis::X
                ? static_cast<std::size_t>(x)
                : static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(sampleCountX);
            if (!equal(samples[index], samples[reference])) {
                return false;
            }
        }
    }
    return true;
}

std::vector<Complex> reciprocalSamples(const std::vector<Complex>& samples,
                                        const char* context) {
    std::vector<Complex> out;
    out.reserve(samples.size());
    for (const Complex value : samples) {
        if (std::abs(value) == Real{0.0}) {
            throw std::runtime_error(std::string(context) + " contains a singular scalar value");
        }
        out.push_back(Complex{1.0, 0.0} / value);
    }
    return out;
}

template <typename ConvFromSamples>
void fillSampledMaterialFactorization(
    std::array<std::array<Matrix, 3>, 3>& direct,
    std::array<std::array<Matrix, 3>, 3>& q,
    Matrix& zzInverse,
    const std::vector<Tensor3>& samples,
    int sampleCountX,
    int sampleCountY,
    std::size_t retainedSize,
    const HarmonicBasis& basis,
    bool factorizeX,
    bool factorizeY,
    const char* tensorName,
    ConvFromSamples& convFromSamples,
    bool* sequentialLiApplied) {
    if (sequentialLiApplied != nullptr) {
        *sequentialLiApplied = false;
    }
    Tensor3 uniform;
    if (tensorSamplesAreConstant(samples, &uniform)) {
        buildUniformMaterial(
            direct, q, zzInverse, uniform, retainedSize, tensorName);
        return;
    }

    std::vector<Complex> scalarSamples;
    if (allScalarTensors(samples, &scalarSamples)) {
        const Matrix materialConv = convFromSamples(scalarSamples);
        detail::assignIsotropicTensorBlocks(direct, materialConv);
        const auto scalarEqual = [](Complex left, Complex right) {
            return complexDependencyClose(left, right);
        };
        const bool dependsOnlyOnX = samplesDependOnlyAlong(
            scalarSamples,
            sampleCountX,
            sampleCountY,
            SampleDependenceAxis::X,
            scalarEqual);
        const bool dependsOnlyOnY = samplesDependOnlyAlong(
            scalarSamples,
            sampleCountX,
            sampleCountY,
            SampleDependenceAxis::Y,
            scalarEqual);
        const bool xOnly = factorizeX && dependsOnlyOnX && !dependsOnlyOnY;
        const bool yOnly = factorizeY && dependsOnlyOnY && !dependsOnlyOnX;
        if (!xOnly && !yOnly && factorizeX && factorizeY) {
            storeLiFactorization(
                q,
                zzInverse,
                detail::liFactorizeTensor2dReticolo(
                    samples,
                    sampleCountX,
                    sampleCountY,
                    basis,
                    tensorName));
            if (sequentialLiApplied != nullptr) {
                *sequentialLiApplied = true;
            }
        } else if (xOnly || yOnly) {
            const Matrix reciprocalConv = convFromSamples(
                reciprocalSamples(scalarSamples, tensorName));
            storeLiFactorization(
                q,
                zzInverse,
                detail::liFactorizeScalarLamellar(
                    materialConv,
                    reciprocalConv,
                    xOnly
                        ? detail::LiInterfaceAxis::X
                        : detail::LiInterfaceAxis::Y,
                    tensorName));
        } else {
            storeLiFactorization(
                q,
                zzInverse,
                detail::liFactorizeDirect(direct, tensorName));
        }
        return;
    }

    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            direct[row][column] =
                convFromSamples(tensorComponentSamples(samples, row, column));
        }
    }
    detail::LiConvolutionBuilder tensorConv = [&](const detail::LiTensorFunction& fn) {
        return convFromSamples(tensorFunctionSamples(samples, fn));
    };
    const auto tensorEqual = [](const Tensor3& left, const Tensor3& right) {
        return left.v == right.v;
    };
    const bool dependsOnlyOnX = samplesDependOnlyAlong(
        samples,
        sampleCountX,
        sampleCountY,
        SampleDependenceAxis::X,
        tensorEqual);
    const bool dependsOnlyOnY = samplesDependOnlyAlong(
        samples,
        sampleCountX,
        sampleCountY,
        SampleDependenceAxis::Y,
        tensorEqual);
    const bool xOnly = factorizeX && dependsOnlyOnX && !dependsOnlyOnY;
    const bool yOnly = factorizeY && dependsOnlyOnY && !dependsOnlyOnX;
    if (xOnly || yOnly) {
        storeLiFactorization(
            q,
            zzInverse,
            detail::liFactorizeTensorLamellar(
                xOnly
                    ? detail::LiInterfaceAxis::X
                    : detail::LiInterfaceAxis::Y,
                tensorConv,
                tensorName));
    } else if (factorizeX && factorizeY) {
        storeLiFactorization(
            q,
            zzInverse,
            detail::liFactorizeTensor2dReticolo(
                samples,
                sampleCountX,
                sampleCountY,
                basis,
                tensorName));
        if (sequentialLiApplied != nullptr) {
            *sequentialLiApplied = true;
        }
    } else {
        storeLiFactorization(
            q,
            zzInverse,
            detail::liFactorizeDirect(direct, tensorName));
    }
}

template <typename ConvFromSamples>
TensorFourierMatrices tensorFourierMatricesFromSamples2dImpl(
    const std::vector<Tensor3>& epsSamples,
    const std::vector<Tensor3>& muSamples,
    int sampleCountX,
    int sampleCountY,
    std::size_t retainedSize,
    const HarmonicBasis& basis,
    bool factorizeX,
    bool factorizeY,
    ConvFromSamples&& convFromSamples) {
    if (epsSamples.empty() || muSamples.empty()) {
        throw std::invalid_argument("2D tensor samples must not be empty");
    }
    const auto expected = checkedGridSize(sampleCountX, sampleCountY, "2D tensor");
    if (epsSamples.size() != expected || muSamples.size() != expected) {
        throw std::invalid_argument(
            "2D tensor sample sizes must equal sample_count_x * sample_count_y");
    }

    TensorFourierMatrices out;
    bool sequentialEpsilon = false;
    bool sequentialMu = false;
    fillSampledMaterialFactorization(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        epsSamples,
        sampleCountX,
        sampleCountY,
        retainedSize,
        basis,
        factorizeX,
        factorizeY,
        "epsilon",
        convFromSamples,
        &sequentialEpsilon);
    fillSampledMaterialFactorization(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        muSamples,
        sampleCountX,
        sampleCountY,
        retainedSize,
        basis,
        factorizeX,
        factorizeY,
        "mu",
        convFromSamples,
        &sequentialMu);
    out.metadata.sequentialLiFactorization2d = sequentialEpsilon || sequentialMu;
    Tensor3 uniformEpsilon;
    Tensor3 uniformMu;
    const bool uniformMaterial =
        tensorSamplesAreConstant(epsSamples, &uniformEpsilon) &&
        tensorSamplesAreConstant(muSamples, &uniformMu);
    const bool tensorMaterial = !allScalarTensors(epsSamples) ||
        !allScalarTensors(muSamples);
    out.metadata.tensorFactorizationApplicability =
        tensorMaterial && !uniformMaterial
        ? TensorFactorizationApplicability::Unknown
        : TensorFactorizationApplicability::NotApplicable;
    out.metadata.factorizationUsesSampledGeometry = !uniformMaterial;
    out.metadata.requiresJointConvergence = !uniformMaterial;
    return out;
}
std::size_t periodicIndex(int k, int size) {
    const int wrapped = ((k % size) + size) % size;
    return static_cast<std::size_t>(wrapped);
}

#ifdef RCWA_HAS_FFTW

struct FftwComplexDeleter {
    void operator()(RCWA_FFTW_COMPLEX* ptr) const {
        RCWA_FFTW_FREE(ptr);
    }
};

using FftwBuffer = std::unique_ptr<RCWA_FFTW_COMPLEX, FftwComplexDeleter>;

FftwBuffer makeFftwBuffer(std::size_t size) {
    const std::size_t bytes = checkedProduct(
        size,
        sizeof(RCWA_FFTW_COMPLEX),
        "FFTW buffer byte");
    FftwBuffer buffer(reinterpret_cast<RCWA_FFTW_COMPLEX*>(
        RCWA_FFTW_MALLOC(bytes)));
    if (!buffer) {
        throw std::runtime_error("FFTW allocation failed");
    }
    return buffer;
}

struct FftwPlanKey {
    bool twoDimensional{};
    int x{};
    int y{};

    bool operator<(const FftwPlanKey& other) const {
        return std::tie(twoDimensional, x, y) <
            std::tie(other.twoDimensional, other.x, other.y);
    }
};

class CachedFftwPlan {
public:
    CachedFftwPlan(bool twoDimensional, int x, int y)
        : mScratchIn(makeFftwBuffer(checkedProduct(
              static_cast<std::size_t>(x),
              static_cast<std::size_t>(y),
              "FFTW plan scratch"))),
          mScratchOut(makeFftwBuffer(checkedProduct(
              static_cast<std::size_t>(x),
              static_cast<std::size_t>(y),
              "FFTW plan scratch"))) {
        if (twoDimensional) {
            mPlan = RCWA_FFTW_PLAN_DFT_2D(
                y, x, mScratchIn.get(), mScratchOut.get(), FFTW_FORWARD, FFTW_ESTIMATE);
        } else {
            mPlan = RCWA_FFTW_PLAN_DFT_1D(
                x, mScratchIn.get(), mScratchOut.get(), FFTW_FORWARD, FFTW_ESTIMATE);
        }
        if (mPlan == nullptr) {
            throw std::runtime_error(twoDimensional
                ? "FFTW 2D plan creation failed"
                : "FFTW 1D plan creation failed");
        }
    }

    ~CachedFftwPlan() {
        if (mPlan != nullptr) {
            RCWA_FFTW_DESTROY_PLAN(mPlan);
        }
    }

    CachedFftwPlan(const CachedFftwPlan&) = delete;
    CachedFftwPlan& operator=(const CachedFftwPlan&) = delete;

    void execute(RCWA_FFTW_COMPLEX* in, RCWA_FFTW_COMPLEX* out) const {
        RCWA_FFTW_EXECUTE_DFT(mPlan, in, out);
    }

private:
    FftwBuffer mScratchIn;
    FftwBuffer mScratchOut;
    RCWA_FFTW_PLAN mPlan{};
};

struct FftwCoeffMap1D {
    std::vector<std::size_t> fftIndices;
};

struct FftwCoeffMap2D {
    std::vector<std::size_t> fftIndices;
    std::vector<Complex> shifts;
};

struct FftwCoeffMap1DKey {
    int sampleCount{};
    int harmonicCount{};

    bool operator<(const FftwCoeffMap1DKey& other) const {
        return std::tie(sampleCount, harmonicCount) <
            std::tie(other.sampleCount, other.harmonicCount);
    }
};

struct FftwCoeffMap2DKey {
    int sampleCountX{};
    int sampleCountY{};
    int harmonicCountX{};
    int harmonicCountY{};

    bool operator<(const FftwCoeffMap2DKey& other) const {
        return std::tie(sampleCountX, sampleCountY, harmonicCountX, harmonicCountY) <
            std::tie(
                other.sampleCountX,
                other.sampleCountY,
                other.harmonicCountX,
                other.harmonicCountY);
    }
};

std::mutex& fftwPlanCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::mutex& fftwCoeffMapCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<FftwPlanKey, std::unique_ptr<CachedFftwPlan>>& fftwPlanCache() {
    // Keep FFTW plans alive until process teardown.  When rcwa is loaded as a
    // Python extension on Windows, DLL unload order can otherwise destroy FFTW
    // before this static cache runs its destructors.
    static auto* cache = new std::map<FftwPlanKey, std::unique_ptr<CachedFftwPlan>>();
    return *cache;
}

std::map<FftwCoeffMap1DKey, std::shared_ptr<const FftwCoeffMap1D>>&
fftwCoeffMap1dCache() {
    static auto* cache =
        new std::map<FftwCoeffMap1DKey, std::shared_ptr<const FftwCoeffMap1D>>();
    return *cache;
}

std::map<FftwCoeffMap2DKey, std::shared_ptr<const FftwCoeffMap2D>>&
fftwCoeffMap2dCache() {
    static auto* cache =
        new std::map<FftwCoeffMap2DKey, std::shared_ptr<const FftwCoeffMap2D>>();
    return *cache;
}

const CachedFftwPlan& cachedFftwPlan(bool twoDimensional, int x, int y = 1) {
    const FftwPlanKey key{twoDimensional, x, y};
    std::lock_guard<std::mutex> lock(fftwPlanCacheMutex());
    auto& cache = fftwPlanCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return *found->second;
    }
    auto inserted = cache.emplace(key, std::make_unique<CachedFftwPlan>(twoDimensional, x, y));
    return *inserted.first->second;
}

std::shared_ptr<const FftwCoeffMap1D> makeFftwCoeffMap1d(int sampleCount,
                                                             int harmonicCount) {
    const int M = halfOrder(harmonicCount);
    auto map = std::make_shared<FftwCoeffMap1D>();
    map->fftIndices.reserve(static_cast<std::size_t>(harmonicCount));
    for (int p = -M; p <= M; ++p) {
        map->fftIndices.push_back(periodicIndex(p, sampleCount));
    }
    return map;
}

std::shared_ptr<const FftwCoeffMap2D> makeFftwCoeffMap2d(int sampleCountX,
                                                             int sampleCountY,
                                                             int harmonicCountX,
                                                             int harmonicCountY) {
    const int Mx = halfOrder(harmonicCountX);
    const int My = halfOrder(harmonicCountY);
    const auto total = checkedGridSize(harmonicCountX, harmonicCountY, "2D harmonic");
    auto map = std::make_shared<FftwCoeffMap2D>();
    map->fftIndices.resize(total);
    map->shifts.resize(total);
    for (int q = -My; q <= My; ++q) {
        const auto ky = periodicIndex(q, sampleCountY);
        for (int p = -Mx; p <= Mx; ++p) {
            const auto kx = periodicIndex(p, sampleCountX);
            const auto fftIndex = ky * static_cast<std::size_t>(sampleCountX) + kx;
            const Real shiftPhase = twoPi * (
                static_cast<Real>(p) *
                    (Real{0.5} - Real{0.5} / static_cast<Real>(sampleCountX)) +
                static_cast<Real>(q) *
                    (Real{0.5} - Real{0.5} / static_cast<Real>(sampleCountY)));
            const std::size_t coeffIndex =
                coeffIndex2d(p, q, harmonicCountX, harmonicCountY);
            map->fftIndices[coeffIndex] = fftIndex;
            map->shifts[coeffIndex] = Complex{std::cos(shiftPhase), std::sin(shiftPhase)};
        }
    }
    return map;
}

std::shared_ptr<const FftwCoeffMap1D> cachedFftwCoeffMap1d(int sampleCount,
                                                               int harmonicCount) {
    const FftwCoeffMap1DKey key{sampleCount, harmonicCount};
    {
        std::lock_guard<std::mutex> lock(fftwCoeffMapCacheMutex());
        auto& cache = fftwCoeffMap1dCache();
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
    }

    auto map = makeFftwCoeffMap1d(sampleCount, harmonicCount);
    std::lock_guard<std::mutex> lock(fftwCoeffMapCacheMutex());
    auto& cache = fftwCoeffMap1dCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    cache.emplace(key, map);
    return map;
}

std::shared_ptr<const FftwCoeffMap2D> cachedFftwCoeffMap2d(int sampleCountX,
                                                               int sampleCountY,
                                                               int harmonicCountX,
                                                               int harmonicCountY) {
    const FftwCoeffMap2DKey key{
        sampleCountX,
        sampleCountY,
        harmonicCountX,
        harmonicCountY};
    {
        std::lock_guard<std::mutex> lock(fftwCoeffMapCacheMutex());
        auto& cache = fftwCoeffMap2dCache();
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
    }

    auto map = makeFftwCoeffMap2d(
        sampleCountX,
        sampleCountY,
        harmonicCountX,
        harmonicCountY);
    std::lock_guard<std::mutex> lock(fftwCoeffMapCacheMutex());
    auto& cache = fftwCoeffMap2dCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    cache.emplace(key, map);
    return map;
}

std::vector<Complex> fftwCoeffsFromSamples(
    const std::vector<Complex>& samples,
    int sampleCount,
    int harmonicCount) {
    halfOrder(harmonicCount);
    const std::size_t S = static_cast<std::size_t>(sampleCount);
    std::vector<Complex> coeffs(static_cast<std::size_t>(harmonicCount));
    FftwBuffer in = makeFftwBuffer(S);
    FftwBuffer out = makeFftwBuffer(S);
    auto* inData = in.get();
    auto* outData = out.get();
    for (std::size_t s = 0; s < S; ++s) {
        inData[s][0] = std::real(samples[s]);
        inData[s][1] = std::imag(samples[s]);
    }
    cachedFftwPlan(false, sampleCount).execute(inData, outData);
    const Complex norm{static_cast<Real>(sampleCount), 0.0};
    const auto coeffMap = cachedFftwCoeffMap1d(sampleCount, harmonicCount);
    for (std::size_t i = 0; i < coeffMap->fftIndices.size(); ++i) {
        const auto k = coeffMap->fftIndices[i];
        coeffs[i] = Complex{outData[k][0], outData[k][1]} / norm;
    }
    return coeffs;
}

std::vector<Complex> fftwCoeffsFromSamples2d(
    const std::vector<Complex>& samples,
    int sampleCountX,
    int sampleCountY,
    int Nx,
    int Ny) {
    halfOrder(Nx);
    halfOrder(Ny);
    const auto S = checkedGridSize(sampleCountX, sampleCountY, "2D FFTW");
    std::vector<Complex> coeffs(checkedGridSize(Nx, Ny, "2D harmonic"));
    FftwBuffer in = makeFftwBuffer(S);
    FftwBuffer out = makeFftwBuffer(S);
    auto* inData = in.get();
    auto* outData = out.get();
    for (std::size_t s = 0; s < S; ++s) {
        inData[s][0] = std::real(samples[s]);
        inData[s][1] = std::imag(samples[s]);
    }
    cachedFftwPlan(true, sampleCountX, sampleCountY).execute(inData, outData);
    const Complex norm{static_cast<Real>(S), 0.0};
    const auto coeffMap = cachedFftwCoeffMap2d(sampleCountX, sampleCountY, Nx, Ny);
    for (std::size_t i = 0; i < coeffMap->fftIndices.size(); ++i) {
        const auto sampleIdx = coeffMap->fftIndices[i];
        coeffs[i] =
            coeffMap->shifts[i] * Complex{outData[sampleIdx][0], outData[sampleIdx][1]} /
            norm;
    }
    return coeffs;
}

#endif

constexpr std::size_t missingConvIndex = std::numeric_limits<std::size_t>::max();

struct RectangularConvIndexKey {
    int coeffCountX{};
    int coeffCountY{};
    int retainedCountX{};
    int retainedCountY{};

    bool operator<(const RectangularConvIndexKey& other) const {
        return std::tie(coeffCountX, coeffCountY, retainedCountX, retainedCountY) <
            std::tie(
                other.coeffCountX,
                other.coeffCountY,
                other.retainedCountX,
                other.retainedCountY);
    }
};

struct BasisConvIndexKey {
    int coeffCountX{};
    int coeffCountY{};
    std::vector<int> orderPairs;

    bool operator<(const BasisConvIndexKey& other) const {
        return std::tie(coeffCountX, coeffCountY, orderPairs) <
            std::tie(other.coeffCountX, other.coeffCountY, other.orderPairs);
    }
};

std::mutex& convIndexCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<RectangularConvIndexKey, std::shared_ptr<const std::vector<std::size_t>>>&
rectangularConvIndexCache() {
    static std::map<RectangularConvIndexKey, std::shared_ptr<const std::vector<std::size_t>>> cache;
    return cache;
}

std::map<BasisConvIndexKey, std::shared_ptr<const std::vector<std::size_t>>>&
basisConvIndexCache() {
    static std::map<BasisConvIndexKey, std::shared_ptr<const std::vector<std::size_t>>> cache;
    return cache;
}

std::shared_ptr<const std::vector<std::size_t>> cachedRectangularConvIndices(
    int coeffCountX,
    int coeffCountY,
    int retainedCountX,
    int retainedCountY) {
    const RectangularConvIndexKey key{
        coeffCountX,
        coeffCountY,
        retainedCountX,
        retainedCountY};
    {
        std::lock_guard<std::mutex> lock(convIndexCacheMutex());
        auto& cache = rectangularConvIndexCache();
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
    }

    const int coeffHalfX = halfOrder(coeffCountX);
    const int coeffHalfY = halfOrder(coeffCountY);
    const int retainedHalfX = halfOrder(retainedCountX);
    const int retainedHalfY = halfOrder(retainedCountY);
    const auto total = checkedGridSize(retainedCountX, retainedCountY, "retained harmonic");
    auto map = std::make_shared<std::vector<std::size_t>>(
        checkedProduct(total, total, "convolution index"),
        missingConvIndex);
    for (int ry = 0; ry < retainedCountY; ++ry) {
        const int rn = ry - retainedHalfY;
        for (int rx = 0; rx < retainedCountX; ++rx) {
            const int rm = rx - retainedHalfX;
            const auto r = harmonicIndex2d(rx, ry, retainedCountX);
            const auto rowBase = r * total;
            for (int cy = 0; cy < retainedCountY; ++cy) {
                const int cn = cy - retainedHalfY;
                for (int cx = 0; cx < retainedCountX; ++cx) {
                    const int cm = cx - retainedHalfX;
                    const int dp = rm - cm;
                    const int dq = rn - cn;
                    if (std::abs(dp) <= coeffHalfX && std::abs(dq) <= coeffHalfY) {
                        const auto c = harmonicIndex2d(cx, cy, retainedCountX);
                        (*map)[rowBase + c] = static_cast<std::size_t>(
                            (dq + coeffHalfY) * coeffCountX + (dp + coeffHalfX));
                    }
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(convIndexCacheMutex());
    auto& cache = rectangularConvIndexCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    cache.emplace(key, map);
    return map;
}

BasisConvIndexKey basisConvIndexKey(int coeffCountX,
                                       int coeffCountY,
                                       const HarmonicBasis& basis) {
    BasisConvIndexKey key;
    key.coeffCountX = coeffCountX;
    key.coeffCountY = coeffCountY;
    key.orderPairs.reserve(2 * basis.orders.size());
    for (const auto& order : basis.orders) {
        key.orderPairs.push_back(order.m);
        key.orderPairs.push_back(order.n);
    }
    return key;
}

std::shared_ptr<const std::vector<std::size_t>> cachedBasisConvIndices(
    int coeffCountX,
    int coeffCountY,
    const HarmonicBasis& basis) {
    const BasisConvIndexKey key = basisConvIndexKey(coeffCountX, coeffCountY, basis);
    {
        std::lock_guard<std::mutex> lock(convIndexCacheMutex());
        auto& cache = basisConvIndexCache();
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
    }

    const int coeffHalfX = halfOrder(coeffCountX);
    const int coeffHalfY = halfOrder(coeffCountY);
    const auto total = basis.orders.size();
    auto map = std::make_shared<std::vector<std::size_t>>(
        checkedProduct(total, total, "convolution index"),
        missingConvIndex);
    for (std::size_t r = 0; r < total; ++r) {
        const auto rowBase = r * total;
        for (std::size_t c = 0; c < total; ++c) {
            const int dp = basis.orders[r].m - basis.orders[c].m;
            const int dq = basis.orders[r].n - basis.orders[c].n;
            if (std::abs(dp) <= coeffHalfX && std::abs(dq) <= coeffHalfY) {
                (*map)[rowBase + c] = static_cast<std::size_t>(
                    (dq + coeffHalfY) * coeffCountX + (dp + coeffHalfX));
            }
        }
    }

    std::lock_guard<std::mutex> lock(convIndexCacheMutex());
    auto& cache = basisConvIndexCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    cache.emplace(key, map);
    return map;
}

} // namespace

FourierFormulation effectiveFourierFormulation(
    const FourierConvergenceOptions& options) noexcept {
    if (options.subpixelSmoothing) {
        return FourierFormulation::Kottke;
    }
    if (!options.polarizationDecomposition) {
        return FourierFormulation::Default;
    }
    switch (options.polarizationBasis) {
    case PolarizationBasis::VectorField:
        return FourierFormulation::PolBasisVL;
    case PolarizationBasis::Normal:
        return FourierFormulation::PolBasisNV;
    case PolarizationBasis::Jones:
        return FourierFormulation::PolBasisJones;
    }
    return FourierFormulation::Default;
}

namespace {

struct LanczosScale {
    Real bx{};
    Real by{};
    Real denominator{};
};

LanczosScale makeLanczosScale(const HarmonicBasis& basis,
                              Real periodXUm,
                              Real periodYUm,
                              Real lanczosWidth) {
    if (!(periodXUm > Real{0.0}) || !(periodYUm > Real{0.0}) ||
        !std::isfinite(periodXUm) || !std::isfinite(periodYUm)) {
        throw std::invalid_argument("Lanczos smoothing periods must be positive and finite");
    }
    const Real bx = Real{1.0} / periodXUm;
    const Real by = Real{1.0} / periodYUm;
    Real maximumRadius{};
    for (const HarmonicIndex order : basis.orders) {
        maximumRadius = std::max(
            maximumRadius,
            std::hypot(static_cast<Real>(order.m) * bx,
                       static_cast<Real>(order.n) * by));
    }
    const Real primitive = std::min(bx, by);
    const Real smoothingOrder =
        (maximumRadius + primitive) * lanczosWidth;
    return {bx, by, Real{2.0} * smoothingOrder + primitive};
}

Real lanczosFactorFromRadius(Real radius,
                             Real denominator,
                             int lanczosPower) {
    const Real argument = twoPi * radius / denominator;
    Real jinc = Real{1.0};
    if (std::abs(argument) >= Real{1e-6}) {
        jinc = Real{2.0} * std::cyl_bessel_j(Real{1.0}, argument) / argument;
    } else {
        const Real argument2 = argument * argument;
        jinc = Real{1.0} - argument2 / Real{8.0} +
            argument2 * argument2 / Real{192.0};
    }
    return std::pow(jinc, lanczosPower);
}

} // namespace

void validateFourierConvergenceOptions(const FourierConvergenceOptions& options) {
    if (options.resolution < 2 || options.resolution > 4096) {
        throw std::invalid_argument("Fourier formulation resolution must be in [2, 4096]");
    }
    if (options.lanczosPower < 1 || options.lanczosPower > 64) {
        throw std::invalid_argument("Lanczos smoothing power must be in [1, 64]");
    }
    if (!(options.lanczosWidth > Real{0.0}) ||
        !std::isfinite(options.lanczosWidth)) {
        throw std::invalid_argument("Lanczos smoothing width must be positive and finite");
    }
}

Real lanczosSmoothingFactor(const HarmonicBasis& basis,
                            int deltaOrderX,
                            int deltaOrderY,
                            Real periodXUm,
                            Real periodYUm,
                            const FourierConvergenceOptions& options) {
    validateFourierConvergenceOptions(options);
    if (!options.lanczosSmoothing ||
        (deltaOrderX == 0 && deltaOrderY == 0)) {
        return Real{1.0};
    }
    const LanczosScale scale = makeLanczosScale(
        basis,
        periodXUm,
        periodYUm,
        options.lanczosWidth);
    const Real radius = std::hypot(
        static_cast<Real>(deltaOrderX) * scale.bx,
        static_cast<Real>(deltaOrderY) * scale.by);
    return lanczosFactorFromRadius(
        radius,
        scale.denominator,
        options.lanczosPower);
}

void applyLanczosSmoothing(Matrix& convolution,
                           const HarmonicBasis& basis,
                           Real periodXUm,
                           Real periodYUm,
                           const FourierConvergenceOptions& options) {
    if (!options.lanczosSmoothing) {
        return;
    }
    if (convolution.rows() != basis.size() ||
        convolution.cols() != basis.size()) {
        throw std::invalid_argument(
            "Lanczos smoothing convolution size must match the harmonic basis");
    }
    validateFourierConvergenceOptions(options);
    const LanczosScale scale = makeLanczosScale(
        basis,
        periodXUm,
        periodYUm,
        options.lanczosWidth);

    const int maximumDeltaX = 2 * basis.orderX;
    const int maximumDeltaY = 2 * basis.orderY;
    const std::size_t deltaCountX = static_cast<std::size_t>(
        2 * maximumDeltaX + 1);
    const std::size_t deltaCountY = static_cast<std::size_t>(
        2 * maximumDeltaY + 1);
    std::vector<Real> factors(checkedProduct(
        deltaCountX,
        deltaCountY,
        "Lanczos smoothing kernel"));
    for (int dn = -maximumDeltaY; dn <= maximumDeltaY; ++dn) {
        for (int dm = -maximumDeltaX; dm <= maximumDeltaX; ++dm) {
            const Real radius = std::hypot(
                static_cast<Real>(dm) * scale.bx,
                static_cast<Real>(dn) * scale.by);
            const std::size_t factorIndex =
                static_cast<std::size_t>(dn + maximumDeltaY) * deltaCountX +
                static_cast<std::size_t>(dm + maximumDeltaX);
            factors[factorIndex] = lanczosFactorFromRadius(
                radius,
                scale.denominator,
                options.lanczosPower);
        }
    }
    for (std::size_t row = 0; row < basis.size(); ++row) {
        for (std::size_t col = 0; col < basis.size(); ++col) {
            const int dm = basis.orders[row].m - basis.orders[col].m;
            const int dn = basis.orders[row].n - basis.orders[col].n;
            const std::size_t factorIndex =
                static_cast<std::size_t>(dn + maximumDeltaY) * deltaCountX +
                static_cast<std::size_t>(dm + maximumDeltaX);
            convolution(row, col) *= factors[factorIndex];
        }
    }
}

bool fourierUsesFftw() {
#ifdef RCWA_HAS_FFTW
    return true;
#else
    return false;
#endif
}

std::vector<Complex> fourierCoeffsUniform(Complex eps, int numHarmonics) {
    requireFinite(eps, "uniform Fourier value");
    const int M = halfOrder(numHarmonics);
    std::vector<Complex> coeffs(static_cast<std::size_t>(numHarmonics), Complex{});
    coeffs[static_cast<std::size_t>(M)] = eps;
    return coeffs;
}

std::vector<Complex> fourierCoeffsBinaryGrating(
    Real fillFactor,
    Complex epsRidge,
    Complex epsGroove,
    int numHarmonics,
    Real centerOffsetFraction) {
    return coeffsForBinary(
        fillFactor,
        epsRidge,
        epsGroove,
        numHarmonics,
        centerOffsetFraction);
}

std::vector<Complex> fourierCoeffsFromSamples(
    const std::vector<Complex>& epsSamples,
    int numHarmonics) {
    if (epsSamples.empty()) {
        throw std::invalid_argument("eps_samples must not be empty");
    }
    if (epsSamples.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("eps_samples size exceeds FFTW integer range");
    }
    requireFiniteSamples(epsSamples, "eps_samples");
#ifdef RCWA_HAS_FFTW
    return fftwCoeffsFromSamples(
        epsSamples, static_cast<int>(epsSamples.size()), numHarmonics);
#else
    const int M = halfOrder(numHarmonics);
    const std::size_t S = epsSamples.size();
    std::vector<Complex> coeffs(static_cast<std::size_t>(numHarmonics));
    for (int p = -M; p <= M; ++p) {
        Complex sum{};
        for (std::size_t s = 0; s < S; ++s) {
            const Real x = static_cast<Real>(s) / static_cast<Real>(S);
            sum += epsSamples[s] * std::exp(Complex{0.0, -twoPi * p * x});
        }
        coeffs[static_cast<std::size_t>(p + M)] = sum / Complex{static_cast<Real>(S), 0.0};
    }
    return coeffs;
#endif
}

std::vector<HarmonicIndex> harmonicIndices2d(int Nx, int Ny) {
    const int Mx = halfOrder(Nx);
    const int My = halfOrder(Ny);
    std::vector<HarmonicIndex> out;
    out.reserve(checkedGridSize(Nx, Ny, "2D harmonic"));
    for (int ny = -My; ny <= My; ++ny) {
        for (int mx = -Mx; mx <= Mx; ++mx) {
            out.push_back({mx, ny});
        }
    }
    return out;
}

bool HarmonicBasis::is1dX() const noexcept {
    if (orders.empty()) {
        return false;
    }
    for (const auto& order : orders) {
        if (order.n != 0) {
            return false;
        }
    }
    return true;
}

int HarmonicBasis::centerIndex() const {
    for (std::size_t i = 0; i < orders.size(); ++i) {
        if (orders[i].m == 0 && orders[i].n == 0) {
            return static_cast<int>(i);
        }
    }
    throw std::invalid_argument("harmonic basis must include the zero order");
}

HarmonicBasis makeHarmonicBasisOrders(
    int orderX,
    int orderY,
    LatticeTruncation truncation,
    Real periodXUm,
    Real periodYUm) {
    if (orderX < 0 || orderY < 0) {
        throw std::invalid_argument("harmonic orders must be non-negative");
    }
    if (orderX == 0 && orderY == 0) {
        HarmonicBasis basis;
        basis.orders.push_back({0, 0});
        basis.truncation = truncation;
        return basis;
    }

    const std::size_t widthX =
        std::size_t{2} * static_cast<std::size_t>(orderX) + 1;
    const std::size_t widthY =
        std::size_t{2} * static_cast<std::size_t>(orderY) + 1;
    const std::size_t candidateCount = checkedProduct(
        widthX,
        widthY,
        "harmonic basis candidate");
    if (candidateCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "harmonic basis exceeds the supported integer index range");
    }

    HarmonicBasis basis;
    basis.truncation = truncation;
    basis.orders.reserve(candidateCount);

    const ReciprocalMetric2d metric = reciprocalMetric2d(periodXUm, periodYUm);
    long double circularCutoffSquared = 0.0L;
    if (truncation == LatticeTruncation::Circular) {
        if (orderX == 0) {
            circularCutoffSquared =
                static_cast<long double>(orderY) * orderY * metric.yWeight;
        } else if (orderY == 0) {
            circularCutoffSquared =
                static_cast<long double>(orderX) * orderX * metric.xWeight;
        } else {
            circularCutoffSquared = std::min(
                static_cast<long double>(orderX) * orderX * metric.xWeight,
                static_cast<long double>(orderY) * orderY * metric.yWeight);
        }
    }

    for (int n = -orderY; n <= orderY; ++n) {
        for (int m = -orderX; m <= orderX; ++m) {
            bool keep = true;
            if (truncation == LatticeTruncation::Circular) {
                const long double radiusSquared =
                    reciprocalRadiusSquared({m, n}, metric);
                keep = radiusSquared < circularCutoffSquared ||
                    reciprocalRadiiEqual(radiusSquared, circularCutoffSquared);
            }
            if (keep) {
                basis.orders.push_back({m, n});
            }
        }
    }

    updateBasisOrders(basis);
    (void)basis.centerIndex();
    return basis;
}

HarmonicBasis makeHarmonicBasisCount(
    int harmonicCount,
    LatticeTruncation truncation,
    Real periodXUm,
    Real periodYUm) {
    if (harmonicCount <= 0) {
        throw std::invalid_argument("harmonic count must be positive");
    }
    if (harmonicCount == 1) {
        return makeHarmonicBasisOrders(
            0, 0, truncation, periodXUm, periodYUm);
    }

    if (truncation == LatticeTruncation::Parallelogramic) {
        int retainedWidth = static_cast<int>(
            std::floor(std::sqrt(static_cast<Real>(harmonicCount))));
        if (retainedWidth % 2 == 0) {
            --retainedWidth;
        }
        retainedWidth = std::max(retainedWidth, 1);
        const int order = retainedWidth / 2;
        return makeHarmonicBasisOrders(
            order,
            order,
            LatticeTruncation::Parallelogramic,
            periodXUm,
            periodYUm);
    }

    std::vector<MetricHarmonicIndex> metricCandidates =
        circularMetricCandidates(harmonicCount, periodXUm, periodYUm);
    std::size_t retained = static_cast<std::size_t>(harmonicCount);
    if (retained < metricCandidates.size()) {
        const long double excludedRadius = metricCandidates[retained].radiusSquared;
        while (retained > 1 &&
               reciprocalRadiiEqual(
                   metricCandidates[retained - 1].radiusSquared,
                   excludedRadius)) {
            --retained;
        }
    }
    std::vector<HarmonicIndex> candidates;
    candidates.reserve(retained);
    for (std::size_t i = 0; i < retained; ++i) {
        candidates.push_back(metricCandidates[i].index);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const HarmonicIndex& a,
                                                              const HarmonicIndex& b) {
        if (a.n != b.n) {
            return a.n < b.n;
        }
        return a.m < b.m;
    });

    HarmonicBasis basis;
    basis.orders = std::move(candidates);
    basis.truncation = truncation;
    updateBasisOrders(basis);
    (void)basis.centerIndex();
    return basis;
}

std::vector<Complex> fourierCoeffsFromSamples2d(
    const std::vector<Complex>& samples,
    int sampleCountX,
    int sampleCountY,
    int Nx,
    int Ny) {
    if (sampleCountX <= 0 || sampleCountY <= 0) {
        throw std::invalid_argument("2D sample counts must be positive");
    }
    const auto expected = checkedGridSize(sampleCountX, sampleCountY, "2D");
    if (samples.size() != expected) {
        throw std::invalid_argument("2D sample array size must equal sample_count_x * sample_count_y");
    }
    requireFiniteSamples(samples, "2D sample array");
#ifdef RCWA_HAS_FFTW
    return fftwCoeffsFromSamples2d(samples, sampleCountX, sampleCountY, Nx, Ny);
#else
    const int Mx = halfOrder(Nx);
    const int My = halfOrder(Ny);
    std::vector<Complex> coeffs(checkedGridSize(Nx, Ny, "2D harmonic"));
    for (int q = -My; q <= My; ++q) {
        for (int p = -Mx; p <= Mx; ++p) {
            Complex sum{};
            for (int sy = 0; sy < sampleCountY; ++sy) {
                const Real y =
                    (static_cast<Real>(sy) + Real{0.5}) /
                        static_cast<Real>(sampleCountY) -
                    Real{0.5};
                for (int sx = 0; sx < sampleCountX; ++sx) {
                    const Real x =
                        (static_cast<Real>(sx) + Real{0.5}) /
                            static_cast<Real>(sampleCountX) -
                        Real{0.5};
                    const auto sampleIdx = static_cast<std::size_t>(sy) *
                        static_cast<std::size_t>(sampleCountX) +
                        static_cast<std::size_t>(sx);
                    const Real phase = -twoPi * (static_cast<Real>(p) * x + static_cast<Real>(q) * y);
                    sum += samples[sampleIdx] * std::exp(Complex{0.0, phase});
                }
            }
            coeffs[coeffIndex2d(p, q, Nx, Ny)] =
                sum / Complex{static_cast<Real>(expected), 0.0};
        }
    }
    return coeffs;
#endif
}

Matrix toeplitzConvMatrix(const std::vector<Complex>& coeffs) {
    return toeplitzConvMatrix(coeffs, static_cast<int>(coeffs.size()));
}

Matrix toeplitzConvMatrix(const std::vector<Complex>& coeffs, int retainedHarmonicCount) {
    const int Nc = static_cast<int>(coeffs.size());
    const int Mc = halfOrder(Nc);
    const int N = retainedHarmonicCount;
    halfOrder(N);
    Matrix out(static_cast<std::size_t>(N), static_cast<std::size_t>(N));
    auto& dst = out.data();
    for (int r = 0; r < N; ++r) {
        const auto rowBase = static_cast<std::size_t>(r) * static_cast<std::size_t>(N);
        for (int c = 0; c < N; ++c) {
            const int idx = r - c + Mc;
            if (idx >= 0 && idx < Nc) {
                dst[rowBase + static_cast<std::size_t>(c)] =
                    coeffs[static_cast<std::size_t>(idx)];
            }
        }
    }
    return out;
}

Matrix convMatrix2d(const std::vector<Complex>& coeffs, int Nx, int Ny) {
    return convMatrix2d(coeffs, Nx, Ny, Nx, Ny);
}

Matrix convMatrix2d(
    const std::vector<Complex>& coeffs,
    int coeffCountX,
    int coeffCountY,
    int retainedCountX,
    int retainedCountY) {
    halfOrder(coeffCountX);
    halfOrder(coeffCountY);
    halfOrder(retainedCountX);
    halfOrder(retainedCountY);
    const auto expected = checkedGridSize(
        coeffCountX,
        coeffCountY,
        "2D Fourier coefficient");
    if (coeffs.size() != expected) {
        throw std::invalid_argument("2D Fourier coefficient count must match coefficient grid");
    }
    const auto total = checkedGridSize(
        retainedCountX,
        retainedCountY,
        "retained harmonic");
    const auto indexMap = cachedRectangularConvIndices(
        coeffCountX,
        coeffCountY,
        retainedCountX,
        retainedCountY);
    Matrix out(total, total);
    const auto& indices = *indexMap;
    auto& dst = out.data();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const std::size_t coeffIndex = indices[i];
        if (coeffIndex != missingConvIndex) {
            dst[i] = coeffs[coeffIndex];
        }
    }
    return out;
}

Matrix convMatrix2d(
    const std::vector<Complex>& coeffs,
    int coeffCountX,
    int coeffCountY,
    const HarmonicBasis& basis) {
    halfOrder(coeffCountX);
    halfOrder(coeffCountY);
    const auto expected = checkedGridSize(
        coeffCountX,
        coeffCountY,
        "2D Fourier coefficient");
    if (coeffs.size() != expected) {
        throw std::invalid_argument("2D Fourier coefficient count must match coefficient grid");
    }
    const auto total = basis.orders.size();
    const auto indexMap = cachedBasisConvIndices(coeffCountX, coeffCountY, basis);
    Matrix out(total, total);
    const auto& indices = *indexMap;
    auto& dst = out.data();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const std::size_t coeffIndex = indices[i];
        if (coeffIndex != missingConvIndex) {
            dst[i] = coeffs[coeffIndex];
        }
    }
    return out;
}

TensorFourierMatrices tensorFourierMatricesUniform(
    const Tensor3& eps,
    const Tensor3& mu,
    int N) {
    halfOrder(N);
    TensorFourierMatrices out;
    const auto size = static_cast<std::size_t>(N);
    buildUniformMaterial(
        out.eps, out.liEpsQ, out.liEpsQZzInverse, eps, size, "epsilon");
    buildUniformMaterial(
        out.mu, out.liMuQ, out.liMuQZzInverse, mu, size, "mu");
    out.metadata.tensorFactorizationApplicability =
        TensorFactorizationApplicability::NotApplicable;
    return out;
}

TensorFourierMatrices tensorFourierMatricesUniform(
    const Tensor3& eps,
    const Tensor3& mu,
    const HarmonicBasis& basis) {
    TensorFourierMatrices out;
    buildUniformMaterial(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        eps,
        basis.size(),
        "epsilon");
    buildUniformMaterial(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        mu,
        basis.size(),
        "mu");
    out.metadata.tensorFactorizationApplicability =
        TensorFactorizationApplicability::NotApplicable;
    return out;
}

TensorFourierMatrices tensorFourierMatricesBinaryGrating(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    int N,
    Real centerOffsetFraction) {
    TensorFourierMatrices out;
    const int Nc = convolutionCoeffCount(N);
    detail::LiConvolutionBuilder epsConvolution =
        [&](const detail::LiTensorFunction& function) {
            return toeplitzConvMatrix(
                binaryComposite(
                    fillFactor,
                    epsRidge,
                    epsGroove,
                    function,
                    Nc,
                    centerOffsetFraction),
                N);
        };
    detail::LiConvolutionBuilder muConvolution =
        [&](const detail::LiTensorFunction& function) {
            return toeplitzConvMatrix(
                binaryComposite(
                    fillFactor,
                    muRidge,
                    muGroove,
                    function,
                    Nc,
                    centerOffsetFraction),
                N);
        };
    buildLamellarMaterial(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        detail::LiInterfaceAxis::X,
        epsConvolution,
        "epsilon");
    buildLamellarMaterial(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        detail::LiInterfaceAxis::X,
        muConvolution,
        "mu");
    annotateAnalyticLamellarApplicability(
        out, fillFactor, epsRidge, epsGroove, muRidge, muGroove);
    return out;
}

TensorFourierMatrices tensorFourierMatricesBinaryGrating2d(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    int retainedCountX,
    int retainedCountY,
    Real centerOffsetFraction) {
    const int coeffCountX = convolutionCoeffCount(retainedCountX);
    const int coeffCountY = convolutionCoeffCount(retainedCountY);

    TensorFourierMatrices out;
    detail::LiConvolutionBuilder epsConvolution =
        [&](const detail::LiTensorFunction& function) {
            return liftedBinaryConvMatrix(
                fillFactor,
                epsRidge,
                epsGroove,
                function,
                coeffCountX,
                coeffCountY,
                retainedCountX,
                retainedCountY,
                centerOffsetFraction);
        };
    detail::LiConvolutionBuilder muConvolution =
        [&](const detail::LiTensorFunction& function) {
            return liftedBinaryConvMatrix(
                fillFactor,
                muRidge,
                muGroove,
                function,
                coeffCountX,
                coeffCountY,
                retainedCountX,
                retainedCountY,
                centerOffsetFraction);
        };
    buildLamellarMaterial(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        detail::LiInterfaceAxis::X,
        epsConvolution,
        "epsilon");
    buildLamellarMaterial(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        detail::LiInterfaceAxis::X,
        muConvolution,
        "mu");
    annotateAnalyticLamellarApplicability(
        out, fillFactor, epsRidge, epsGroove, muRidge, muGroove);
    return out;
}

TensorFourierMatrices tensorFourierMatricesBinaryGrating(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    const HarmonicBasis& basis,
    Real centerOffsetFraction) {
    const int coeffCountX = coeffCountForSpan(2 * basis.orderX);
    const int coeffCountY = coeffCountForSpan(2 * basis.orderY);

    TensorFourierMatrices out;
    const auto convolution = [&](const Tensor3& ridge,
                                 const Tensor3& groove,
                                 const detail::LiTensorFunction& function) {
        const auto coefficients = lift1dCoeffsTo2d(
            fourierCoeffsBinaryGrating(
                fillFactor,
                function(ridge),
                function(groove),
                coeffCountX,
                centerOffsetFraction),
            coeffCountX,
            coeffCountY);
        return convMatrix2d(
            coefficients, coeffCountX, coeffCountY, basis);
    };
    detail::LiConvolutionBuilder epsConvolution =
        [&](const detail::LiTensorFunction& function) {
            return convolution(epsRidge, epsGroove, function);
        };
    detail::LiConvolutionBuilder muConvolution =
        [&](const detail::LiTensorFunction& function) {
            return convolution(muRidge, muGroove, function);
        };
    buildLamellarMaterial(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        detail::LiInterfaceAxis::X,
        epsConvolution,
        "epsilon");
    buildLamellarMaterial(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        detail::LiInterfaceAxis::X,
        muConvolution,
        "mu");
    annotateAnalyticLamellarApplicability(
        out, fillFactor, epsRidge, epsGroove, muRidge, muGroove);
    return out;
}

TensorFourierMatrices tensorFourierMatricesBinaryGratingY(
    Real fillFactor,
    const Tensor3& epsRidge,
    const Tensor3& epsGroove,
    const Tensor3& muRidge,
    const Tensor3& muGroove,
    const HarmonicBasis& basis,
    Real centerOffsetFraction) {
    const int coeffCountX = coeffCountForSpan(2 * basis.orderX);
    const int coeffCountY = coeffCountForSpan(2 * basis.orderY);
    const auto convolution = [&](const Tensor3& ridge,
                                 const Tensor3& groove,
                                 const detail::LiTensorFunction& component) {
        const auto coefficients = lift1dYCoeffsTo2d(
            fourierCoeffsBinaryGrating(
                fillFactor,
                component(ridge),
                component(groove),
                coeffCountY,
                centerOffsetFraction),
            coeffCountX,
            coeffCountY);
        return convMatrix2d(coefficients, coeffCountX, coeffCountY, basis);
    };

    TensorFourierMatrices out;
    detail::LiConvolutionBuilder epsConvolution =
        [&](const detail::LiTensorFunction& function) {
            return convolution(epsRidge, epsGroove, function);
        };
    detail::LiConvolutionBuilder muConvolution =
        [&](const detail::LiTensorFunction& function) {
            return convolution(muRidge, muGroove, function);
        };
    buildLamellarMaterial(
        out.eps,
        out.liEpsQ,
        out.liEpsQZzInverse,
        detail::LiInterfaceAxis::Y,
        epsConvolution,
        "epsilon");
    buildLamellarMaterial(
        out.mu,
        out.liMuQ,
        out.liMuQZzInverse,
        detail::LiInterfaceAxis::Y,
        muConvolution,
        "mu");
    annotateAnalyticLamellarApplicability(
        out, fillFactor, epsRidge, epsGroove, muRidge, muGroove);
    return out;
}

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
    const FourierConvergenceOptions& options) {
    validateFourierConvergenceOptions(options);
    TensorFourierMatrices out = tensorFourierMatricesBinaryGrating(
        fillFactor,
        epsRidge,
        epsGroove,
        muRidge,
        muGroove,
        basis,
        centerOffsetFraction);

    if (options.lanczosSmoothing) {
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                applyLanczosSmoothing(
                    out.eps[row][col], basis, periodXUm, periodYUm, options);
                applyLanczosSmoothing(
                    out.mu[row][col], basis, periodXUm, periodYUm, options);
            }
        }

        const int coeffCountX = coeffCountForSpan(2 * basis.orderX);
        const int coeffCountY = coeffCountForSpan(2 * basis.orderY);
        const auto smoothedConv = [&](const Tensor3& ridge,
                                      const Tensor3& groove,
                                      const detail::LiTensorFunction& component) {
            const auto coeffs = lift1dCoeffsTo2d(
                fourierCoeffsBinaryGrating(
                    fillFactor,
                    component(ridge),
                    component(groove),
                    coeffCountX,
                    centerOffsetFraction),
                coeffCountX,
                coeffCountY);
            Matrix convolution =
                convMatrix2d(coeffs, coeffCountX, coeffCountY, basis);
            applyLanczosSmoothing(
                convolution, basis, periodXUm, periodYUm, options);
            return convolution;
        };
        detail::LiConvolutionBuilder epsConv =
            [&](const detail::LiTensorFunction& component) {
                return smoothedConv(epsRidge, epsGroove, component);
            };
        detail::LiConvolutionBuilder muConv =
            [&](const detail::LiTensorFunction& component) {
                return smoothedConv(muRidge, muGroove, component);
            };
        storeLiFactorization(
            out.liEpsQ,
            out.liEpsQZzInverse,
            detail::liFactorizeTensorLamellar(
                detail::LiInterfaceAxis::X, epsConv, "epsilon"));
        storeLiFactorization(
            out.liMuQ,
            out.liMuQZzInverse,
            detail::liFactorizeTensorLamellar(
                detail::LiInterfaceAxis::X, muConv, "mu"));
    }

    out.metadata.formulation = effectiveFourierFormulation(options);
    out.metadata.lanczosSmoothingApplied = options.lanczosSmoothing;
    out.metadata.formulationResolution = options.resolution;
    return out;
}

TensorFourierMatrices tensorFourierMatricesFromSamples2d(
    const std::vector<Tensor3>& epsSamples,
    const std::vector<Tensor3>& muSamples,
    int sampleCountX,
    int sampleCountY,
    int Nx,
    int Ny) {
    const int coeffCountX = convolutionCoeffCount(Nx);
    const int coeffCountY = convolutionCoeffCount(Ny);
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        halfOrder(Nx),
        halfOrder(Ny),
        LatticeTruncation::Parallelogramic);
    return tensorFourierMatricesFromSamples2dImpl(
        epsSamples,
        muSamples,
        sampleCountX,
        sampleCountY,
        checkedGridSize(Nx, Ny, "2D harmonic"),
        basis,
        Nx > 1,
        Ny > 1,
        [&](const std::vector<Complex>& samples) {
            return convMatrix2d(
                fourierCoeffsFromSamples2d(
                    samples,
                    sampleCountX,
                    sampleCountY,
                    coeffCountX,
                    coeffCountY),
                coeffCountX,
                coeffCountY,
                Nx,
                Ny);
        });
}

TensorFourierMatrices tensorFourierMatricesFromSamples2d(
    const std::vector<Tensor3>& epsSamples,
    const std::vector<Tensor3>& muSamples,
    int sampleCountX,
    int sampleCountY,
    const HarmonicBasis& basis) {
    const int coeffCountX = coeffCountForSpan(2 * basis.orderX);
    const int coeffCountY = coeffCountForSpan(2 * basis.orderY);
    return tensorFourierMatricesFromSamples2dImpl(
        epsSamples,
        muSamples,
        sampleCountX,
        sampleCountY,
        basis.size(),
        basis,
        basis.orderX > 0,
        basis.orderY > 0,
        [&](const std::vector<Complex>& samples) {
            return convMatrix2d(
                fourierCoeffsFromSamples2d(
                    samples,
                    sampleCountX,
                    sampleCountY,
                    coeffCountX,
                    coeffCountY),
                coeffCountX,
                coeffCountY,
                basis);
        });
}

std::pair<DiagonalOperator, DiagonalOperator>
makeTransverseWavevectorOperators(
    Real kx0Normalized,
    Real ky0Normalized,
    Real gxNormalized,
    Real gyNormalized,
    const HarmonicBasis& basis) {
    std::vector<Complex> kx(basis.orders.size());
    std::vector<Complex> ky(basis.orders.size());
    for (std::size_t i = 0; i < basis.orders.size(); ++i) {
        kx[i] = Complex{
            kx0Normalized + basis.orders[i].m * gxNormalized,
            0.0};
        ky[i] = Complex{
            ky0Normalized + basis.orders[i].n * gyNormalized,
            0.0};
    }
    return {
        DiagonalOperator(std::move(kx)),
        DiagonalOperator(std::move(ky)),
    };
}

} // namespace rcwa
