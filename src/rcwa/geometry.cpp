#include "rcwa/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "rcwa/fourier.hpp"
#include "rcwa/layer.hpp"
#include "rcwa/physics.hpp"
#include "rcwa/solver.hpp"

namespace rcwa {

// Geometry owns the allocation path from user layer specs to modal stack data.
namespace {
void validateAdaptiveGratingControls(Real fillStepTolerance,
                                        int minSlices,
                                        int maxSlices,
                                        const char* context) {
    if (!(fillStepTolerance > Real{0.0}) || !std::isfinite(fillStepTolerance)) {
        throw std::invalid_argument(std::string(context) +
            " fill_step_tolerance must be positive and finite");
    }
    if (minSlices < 1) {
        throw std::invalid_argument(std::string(context) +
            " min_slices must be at least 1");
    }
    if (maxSlices < minSlices) {
        throw std::invalid_argument(std::string(context) +
            " max_slices must be >= min_slices");
    }
    if (maxSlices > 4096) {
        throw std::invalid_argument(std::string(context) +
            " max_slices must not exceed 4096");
    }
}

void validateFillProfile(const std::vector<FillFactorProfilePoint>& fillProfile) {
    if (fillProfile.size() < 2) {
        throw std::invalid_argument(
            "adaptive profiled grating fill profile must contain at least two points");
    }
    for (const auto& point : fillProfile) {
        if (!std::isfinite(point.zFraction) || point.zFraction < Real{0.0} ||
            point.zFraction > Real{1.0}) {
            throw std::invalid_argument(
                "adaptive profiled grating z fractions must be finite and in [0, 1]");
        }
        if (!std::isfinite(point.fillFactor) || point.fillFactor < Real{0.0} ||
            point.fillFactor > Real{1.0}) {
            throw std::invalid_argument(
                "adaptive profiled grating fill factors must be finite and in [0, 1]");
        }
    }
    if (std::abs(fillProfile.front().zFraction) > Real{1e-12} ||
        std::abs(fillProfile.back().zFraction - Real{1.0}) > Real{1e-12}) {
        throw std::invalid_argument(
            "adaptive profiled grating fill profile must start at z_fraction 0 and end at 1");
    }
    for (std::size_t i = 1; i < fillProfile.size(); ++i) {
        if (!(fillProfile[i].zFraction > fillProfile[i - 1].zFraction)) {
            throw std::invalid_argument(
                "adaptive profiled grating z fractions must be strictly increasing");
        }
    }
}

Real fillAtValidatedProfile(const std::vector<FillFactorProfilePoint>& fillProfile,
                               Real zFraction) {
    if (zFraction <= fillProfile.front().zFraction) {
        return fillProfile.front().fillFactor;
    }
    if (zFraction >= fillProfile.back().zFraction) {
        return fillProfile.back().fillFactor;
    }
    for (std::size_t i = 1; i < fillProfile.size(); ++i) {
        const auto& upper = fillProfile[i];
        if (zFraction <= upper.zFraction) {
            const auto& lower = fillProfile[i - 1];
            const Real t = (zFraction - lower.zFraction) /
                (upper.zFraction - lower.zFraction);
            return lower.fillFactor + (upper.fillFactor - lower.fillFactor) * t;
        }
    }
    return fillProfile.back().fillFactor;
}

std::vector<int> profileSegmentSliceCounts(
    const std::vector<FillFactorProfilePoint>& fillProfile,
    Real fillStepTolerance,
    int minSlices,
    int maxSlices) {
    validateFillProfile(fillProfile);
    validateAdaptiveGratingControls(
        fillStepTolerance,
        minSlices,
        maxSlices,
        "adaptive profiled grating");

    const std::size_t segmentCount = fillProfile.size() - 1;
    if (segmentCount > static_cast<std::size_t>(maxSlices)) {
        throw std::invalid_argument(
            "adaptive profiled grating max_slices must cover all profile segments");
    }

    std::vector<int> desired(segmentCount, 1);
    int desiredTotal = 0;
    for (std::size_t i = 0; i < segmentCount; ++i) {
        const Real fillDelta =
            std::abs(fillProfile[i + 1].fillFactor - fillProfile[i].fillFactor);
        const Real rawSlices = std::ceil(fillDelta / fillStepTolerance);
        desired[i] = std::max(
            1,
            rawSlices > static_cast<Real>(maxSlices)
                ? maxSlices
                : static_cast<int>(rawSlices));
        desiredTotal += desired[i];
    }

    const int mandatory = static_cast<int>(segmentCount);
    const int targetTotal = std::max(
        mandatory,
        std::min(std::max(desiredTotal, minSlices), maxSlices));
    if (targetTotal == desiredTotal) {
        return desired;
    }

    if (targetTotal > desiredTotal) {
        std::vector<int> counts = desired;
        for (int extra = 0; extra < targetTotal - desiredTotal; ++extra) {
            counts[static_cast<std::size_t>(extra) % segmentCount] += 1;
        }
        return counts;
    }

    std::vector<int> counts(segmentCount, 1);
    const int extraBudget = targetTotal - mandatory;
    int weightSum = 0;
    for (int count : desired) {
        weightSum += count - 1;
    }
    if (extraBudget == 0 || weightSum == 0) {
        return counts;
    }

    std::vector<Real> remainders(segmentCount, Real{0.0});
    int assigned = 0;
    for (std::size_t i = 0; i < segmentCount; ++i) {
        const Real exactExtra =
            static_cast<Real>(extraBudget) * static_cast<Real>(desired[i] - 1) /
            static_cast<Real>(weightSum);
        const int wholeExtra = static_cast<int>(std::floor(exactExtra));
        counts[i] += wholeExtra;
        assigned += wholeExtra;
        remainders[i] = exactExtra - static_cast<Real>(wholeExtra);
    }
    for (int left = extraBudget - assigned; left > 0; --left) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < segmentCount; ++i) {
            if (remainders[i] > remainders[best]) {
                best = i;
            }
        }
        counts[best] += 1;
        remainders[best] = Real{-1.0};
    }
    return counts;
}

} // namespace

int adaptiveTaperedGratingSliceCount(Real topFillFactor,
                                         Real bottomFillFactor,
                                         Real fillStepTolerance,
                                         int minSlices,
                                         int maxSlices) {
    if (topFillFactor < Real{0.0} || topFillFactor > Real{1.0} ||
        bottomFillFactor < Real{0.0} || bottomFillFactor > Real{1.0} ||
        !std::isfinite(topFillFactor) || !std::isfinite(bottomFillFactor)) {
        throw std::invalid_argument("adaptive tapered grating fill factors must be finite and in [0, 1]");
    }
    validateAdaptiveGratingControls(
        fillStepTolerance,
        minSlices,
        maxSlices,
        "adaptive tapered grating");

    const Real fillSpan = std::abs(bottomFillFactor - topFillFactor);
    const Real rawSlices = std::ceil(fillSpan / fillStepTolerance);
    const int adaptiveSlices =
        rawSlices > static_cast<Real>(maxSlices)
            ? maxSlices
            : static_cast<int>(rawSlices);
    const int requested = std::max(
        minSlices,
        adaptiveSlices);
    return std::min(requested, maxSlices);
}

Real adaptiveTaperedGratingMidpointFill(Real topFillFactor,
                                            Real bottomFillFactor,
                                            int slice,
                                            int slices) {
    if (slices < 1 || slice < 0 || slice >= slices) {
        throw std::invalid_argument("adaptive tapered grating slice index is outside the slice count");
    }
    const Real zFraction =
        (static_cast<Real>(slice) + Real{0.5}) / static_cast<Real>(slices);
    return topFillFactor + (bottomFillFactor - topFillFactor) * zFraction;
}

std::vector<GratingLayer> expandAdaptiveTaperedGratingLayer(
    const AdaptiveTaperedGratingLayer& layer) {
    if (!(layer.thicknessUm >= Real{0.0}) || !std::isfinite(layer.thicknessUm)) {
        throw std::invalid_argument("adaptive tapered grating thickness must be finite and non-negative");
    }
    if (!(layer.periodUm > Real{0.0}) || !(layer.periodYUm > Real{0.0}) ||
        !std::isfinite(layer.periodUm) || !std::isfinite(layer.periodYUm)) {
        throw std::invalid_argument("adaptive tapered grating periods must be positive and finite");
    }
    if (!std::isfinite(layer.ridgeOffsetUm)) {
        throw std::invalid_argument(
            "adaptive tapered grating ridge_offset_um must be finite");
    }
    const int slices = adaptiveTaperedGratingSliceCount(
        layer.topFillFactor,
        layer.bottomFillFactor,
        layer.fillStepTolerance,
        layer.minSlices,
        layer.maxSlices);
    const Real sliceThickness = layer.thicknessUm / static_cast<Real>(slices);

    std::vector<GratingLayer> out;
    out.reserve(static_cast<std::size_t>(slices));
    for (int i = 0; i < slices; ++i) {
        GratingLayer slice;
        slice.materialRidge = layer.materialRidge;
        slice.materialGroove = layer.materialGroove;
        slice.fillFactor = adaptiveTaperedGratingMidpointFill(
            layer.topFillFactor,
            layer.bottomFillFactor,
            i,
            slices);
        slice.periodUm = layer.periodUm;
        slice.periodYUm = layer.periodYUm;
        slice.ridgeOffsetUm = layer.ridgeOffsetUm;
        slice.thicknessUm = sliceThickness;
        out.push_back(std::move(slice));
    }
    return out;
}

int adaptiveProfiledGratingSliceCount(
    const std::vector<FillFactorProfilePoint>& fillProfile,
    Real fillStepTolerance,
    int minSlices,
    int maxSlices) {
    int total = 0;
    for (int count : profileSegmentSliceCounts(
             fillProfile,
             fillStepTolerance,
             minSlices,
             maxSlices)) {
        total += count;
    }
    return total;
}

Real adaptiveProfiledGratingFillAt(
    const std::vector<FillFactorProfilePoint>& fillProfile,
    Real zFraction) {
    validateFillProfile(fillProfile);
    if (!std::isfinite(zFraction) || zFraction < Real{0.0} || zFraction > Real{1.0}) {
        throw std::invalid_argument(
            "adaptive profiled grating query z_fraction must be finite and in [0, 1]");
    }
    return fillAtValidatedProfile(fillProfile, zFraction);
}

std::vector<GratingLayer> expandAdaptiveProfiledGratingLayer(
    const AdaptiveProfiledGratingLayer& layer) {
    if (!(layer.thicknessUm >= Real{0.0}) || !std::isfinite(layer.thicknessUm)) {
        throw std::invalid_argument(
            "adaptive profiled grating thickness must be finite and non-negative");
    }
    if (!(layer.periodUm > Real{0.0}) || !(layer.periodYUm > Real{0.0}) ||
        !std::isfinite(layer.periodUm) || !std::isfinite(layer.periodYUm)) {
        throw std::invalid_argument(
            "adaptive profiled grating periods must be positive and finite");
    }
    if (!std::isfinite(layer.ridgeOffsetUm)) {
        throw std::invalid_argument(
            "adaptive profiled grating ridge_offset_um must be finite");
    }
    const auto segmentCounts = profileSegmentSliceCounts(
        layer.fillProfile,
        layer.fillStepTolerance,
        layer.minSlices,
        layer.maxSlices);

    int totalSlices = 0;
    for (int count : segmentCounts) {
        totalSlices += count;
    }

    std::vector<GratingLayer> out;
    out.reserve(static_cast<std::size_t>(totalSlices));
    for (std::size_t segment = 0; segment < segmentCounts.size(); ++segment) {
        const auto& lower = layer.fillProfile[segment];
        const auto& upper = layer.fillProfile[segment + 1];
        const Real segmentZSpan = upper.zFraction - lower.zFraction;
        const int count = segmentCounts[segment];
        const Real sliceThickness =
            layer.thicknessUm * segmentZSpan / static_cast<Real>(count);
        for (int i = 0; i < count; ++i) {
            const Real zFraction = lower.zFraction +
                segmentZSpan * (static_cast<Real>(i) + Real{0.5}) /
                    static_cast<Real>(count);
            GratingLayer slice;
            slice.materialRidge = layer.materialRidge;
            slice.materialGroove = layer.materialGroove;
            slice.fillFactor = fillAtValidatedProfile(layer.fillProfile, zFraction);
            slice.periodUm = layer.periodUm;
            slice.periodYUm = layer.periodYUm;
            slice.ridgeOffsetUm = layer.ridgeOffsetUm;
            slice.thicknessUm = sliceThickness;
            out.push_back(std::move(slice));
        }
    }
    return out;
}

void requireNearlyEqual(Real a, Real b, const char* context) {
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    if (std::abs(a - b) > 1e-12 * scale) {
        throw std::invalid_argument(std::string(context) +
            " periodic layers must share one x/y lattice for a single RCWA solve");
    }
}

template <typename... Fn>
struct Overloaded : Fn... {
    using Fn::operator()...;
};

template <typename... Fn>
Overloaded(Fn...) -> Overloaded<Fn...>;

struct LayerLatticeView {
    Real periodXUm{};
    Real periodYUm{};
    const char* layerLabel{};
};

LayerLatticeView latticeView(const GratingLayer& layer) {
    requirePositive(layer.periodUm, "grating period_x");
    requirePositive(layer.periodYUm, "grating period_y");
    return {layer.periodUm, layer.periodYUm, "grating"};
}

LayerLatticeView latticeView(const PeriodicLayer2D& layer) {
    requirePositive(layer.periodXUm, "2D layer period_x");
    requirePositive(layer.periodYUm, "2D layer period_y");
    return {layer.periodXUm, layer.periodYUm, "2D layer"};
}

LayerLatticeView latticeView(const PatternedLayer2D& layer) {
    requirePositive(layer.periodXUm, "patterned layer period_x");
    requirePositive(layer.periodYUm, "patterned layer period_y");
    return {layer.periodXUm, layer.periodYUm, "patterned layer"};
}

LayerLatticeView latticeView(const PrecomputedPeriodicLayer2DPtr& layer) {
    if (!layer) {
        throw std::invalid_argument("precomputed 2D periodic layer pointer must not be null");
    }
    requirePositive(layer->periodXUm, "precomputed 2D layer period_x");
    requirePositive(layer->periodYUm, "precomputed 2D layer period_y");
    return {layer->periodXUm, layer->periodYUm, "precomputed 2D layer"};
}

std::optional<LayerLatticeView> latticeView(const LayerSpec& layer) {
    return std::visit(Overloaded{
        [](const UniformLayer&) -> std::optional<LayerLatticeView> {
            return std::nullopt;
        },
        [](const auto& periodicLayer) -> std::optional<LayerLatticeView> {
            return latticeView(periodicLayer);
        },
    }, layer);
}

class StackAmplitudeSolver {
public:
    explicit StackAmplitudeSolver(const PreparedStack& stack);

    [[nodiscard]] StackAmplitudes solve(const Matrix& incidentAmplitudes) const;
    [[nodiscard]] StackAmplitudes solve(Polarization pol) const;

private:
    [[nodiscard]] Matrix modeBlockProduct(const LayerModes& modes,
                                            bool down,
                                            const Matrix& rhs,
                                            Complex scale = {1.0, 0.0}) const;
    [[nodiscard]] std::size_t layerDownOffset(std::size_t layerIndex) const;
    [[nodiscard]] std::size_t layerUpOffset(std::size_t layerIndex) const;
    void setModeBlock(Matrix& target,
                        std::size_t row,
                        std::size_t col,
                        const LayerModes& modes,
                        bool down,
                        Complex scale = {1.0, 0.0}) const;
    void setPropagatedModeBlock(Matrix& target,
                                   std::size_t row,
                                   std::size_t col,
                                   const LayerModes& modes,
                                   bool down,
                                   Complex scale = {1.0, 0.0}) const;

    const PreparedStack& mStack;
    std::size_t mLayerCount{};
    std::size_t mStateSize{};
    std::size_t mModeCount{};
};

void mergeLattice(StackLattice& lattice, const LayerLatticeView& view) {
    if (!lattice.hasPeriodicLayer) {
        lattice.periodXUm = view.periodXUm;
        lattice.periodYUm = view.periodYUm;
        lattice.hasPeriodicLayer = true;
        return;
    }
    requireNearlyEqual(lattice.periodXUm, view.periodXUm, view.layerLabel);
    requireNearlyEqual(lattice.periodYUm, view.periodYUm, view.layerLabel);
}

StackLattice inferStackLattice(const std::vector<LayerSpec>& layers) {
    StackLattice lattice;
    for (const auto& layer : layers) {
        if (const auto view = latticeView(layer)) {
            mergeLattice(lattice, *view);
        }
    }
    return lattice;
}

void validateThickness(Real thicknessUm, const char* context) {
    if (thicknessUm < 0.0 || !std::isfinite(thicknessUm)) {
        throw std::invalid_argument(std::string(context) +
            " thickness must be a finite non-negative value");
    }
}

void validateGratingLayer(const GratingLayer& g) {
    validateThickness(g.thicknessUm, "grating layer");
    requirePositive(g.periodUm, "grating period_x");
    requirePositive(g.periodYUm, "grating period_y");
    requireFinite(g.ridgeOffsetUm, "grating ridge_offset_um");
    if (g.fillFactor < 0.0 || g.fillFactor > 1.0 || !std::isfinite(g.fillFactor)) {
        throw std::invalid_argument("grating fill_factor must be in [0, 1]");
    }
}

void validatePeriodicLayer2d(const PeriodicLayer2D& layer) {
    validateThickness(layer.thicknessUm, "2D periodic layer");
    const std::size_t expected = checkedGridSize(
        layer.sampleCountX, layer.sampleCountY, "2D periodic layer sample grid");
    if (layer.cells.size() != expected) {
        throw std::invalid_argument(
            "2D periodic layer cells size must equal sample_count_x * sample_count_y");
    }
    requirePositive(layer.periodXUm, "2D layer period_x");
    requirePositive(layer.periodYUm, "2D layer period_y");
}

void validatePeriodicSampling(const PeriodicLayer2D& layer, const HarmonicBasis& basis) {
    const int requiredX = 4 * basis.orderX + 1;
    const int requiredY = 4 * basis.orderY + 1;
    if (layer.sampleCountX < requiredX || layer.sampleCountY < requiredY) {
        throw std::invalid_argument(
            "2D periodic layer sampling must cover the retained harmonic differences "
            "to build unaliased RCWA convolution matrices");
    }
}

TensorFourierMatrices tensorPeriodicMatrices2d(const PeriodicLayer2D& layer,
                                                  const HarmonicBasis& basis,
                                                  Real wavelengthUm) {
    const std::size_t expected = checkedGridSize(
        layer.sampleCountX, layer.sampleCountY, "2D periodic layer sample grid");
    if (layer.cells.size() != expected) {
        throw std::invalid_argument(
            "2D periodic layer cells size must equal sample_count_x * sample_count_y");
    }
    if (layer.fourierOptions.polarizationDecomposition ||
        layer.fourierOptions.subpixelSmoothing ||
        layer.fourierOptions.lanczosSmoothing) {
        throw std::invalid_argument(
            "advanced Fourier convergence options require PatternedLayer2D boundary "
            "geometry (or an analytic GratingLayer); raw PeriodicLayer2D cells do not "
            "contain the subpixel interface geometry needed by these formulations");
    }

    std::vector<Tensor3> epsSamples(layer.cells.size());
    std::vector<Tensor3> muSamples(layer.cells.size());
    for (std::size_t i = 0; i < layer.cells.size(); ++i) {
        const auto [eps, mu] = layer.cells[i].tensors(wavelengthUm);
        epsSamples[i] = eps;
        muSamples[i] = mu;
    }
    return tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        layer.sampleCountX,
        layer.sampleCountY,
        basis);
}

bool tensorExactlyEqual(const Tensor3& a, const Tensor3& b) {
    return a.v == b.v;
}

using UniformTensorPair = std::pair<Tensor3, Tensor3>;

std::optional<UniformTensorPair> exactUniformTensors(
    const UniformLayer& layer,
    Real wavelengthUm) {
    validateThickness(layer.thicknessUm, "uniform layer");
    return layer.material.tensors(wavelengthUm);
}

std::optional<UniformTensorPair> exactUniformTensors(
    const GratingLayer& layer,
    Real wavelengthUm) {
    validateGratingLayer(layer);
    const auto [ridgeEps, ridgeMu] = layer.materialRidge.tensors(wavelengthUm);
    const auto [grooveEps, grooveMu] = layer.materialGroove.tensors(wavelengthUm);
    if (layer.fillFactor == Real{0.0} ||
        (tensorExactlyEqual(ridgeEps, grooveEps) &&
         tensorExactlyEqual(ridgeMu, grooveMu))) {
        return UniformTensorPair{grooveEps, grooveMu};
    }
    if (layer.fillFactor == Real{1.0}) {
        return UniformTensorPair{ridgeEps, ridgeMu};
    }
    return std::nullopt;
}

std::optional<UniformTensorPair> exactUniformTensors(
    const PeriodicLayer2D& layer,
    Real wavelengthUm) {
    validatePeriodicLayer2d(layer);
    if (layer.cells.empty()) {
        return std::nullopt;
    }
    const auto first = layer.cells.front().tensors(wavelengthUm);
    for (std::size_t i = 1; i < layer.cells.size(); ++i) {
        const auto current = layer.cells[i].tensors(wavelengthUm);
        if (!tensorExactlyEqual(first.first, current.first) ||
            !tensorExactlyEqual(first.second, current.second)) {
            return std::nullopt;
        }
    }
    return UniformTensorPair{first.first, first.second};
}

std::optional<UniformTensorPair> exactUniformTensors(
    const PatternedLayer2D& layer,
    Real wavelengthUm) {
    validateThickness(layer.thicknessUm, "patterned layer");
    requirePositive(layer.periodXUm, "patterned layer period_x");
    requirePositive(layer.periodYUm, "patterned layer period_y");
    const auto background = layer.background.tensors(wavelengthUm);
    for (const auto& region : layer.regions) {
        const auto regionTensors = region.material.tensors(wavelengthUm);
        if (!tensorExactlyEqual(background.first, regionTensors.first) ||
            !tensorExactlyEqual(background.second, regionTensors.second)) {
            return std::nullopt;
        }
    }
    return UniformTensorPair{background.first, background.second};
}

LayerModes modesForMaterial(const Material& mat,
                              const DiagonalOperator& Kx,
                              const DiagonalOperator& Ky,
                              int totalHarmonics,
                              Real thicknessUm,
                              Real wavelengthUm) {
    if (Kx.rows() == static_cast<std::size_t>(totalHarmonics)) {
        const auto [eps, mu] = mat.tensors(wavelengthUm);
        return computeUniformModes(
            eps,
            mu,
            Kx,
            Ky,
            thicknessUm,
            wavelengthUm);
    }

    throw std::invalid_argument("invalid harmonic matrix size");
}

void validatePrecomputedPeriodicLayer2d(const PrecomputedPeriodicLayer2D& layer) {
    validateThickness(layer.thicknessUm, "precomputed 2D periodic layer");
    requirePositive(layer.periodXUm, "precomputed 2D layer period_x");
    requirePositive(layer.periodYUm, "precomputed 2D layer period_y");
}

bool exactScalarIdentityValue(const Matrix& matrix, Complex& value) {
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
        return false;
    }
    value = matrix(0, 0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            const Complex expected = row == column ? value : Complex{};
            if (matrix(row, column) != expected) {
                return false;
            }
        }
    }
    return true;
}

std::optional<UniformTensorPair> exactUniformTensors(
    const PrecomputedPeriodicLayer2D& layer) {
    validatePrecomputedPeriodicLayer2d(layer);
    Tensor3 eps;
    Tensor3 mu;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            Complex epsValue{};
            Complex muValue{};
            Complex factorizedEpsValue{};
            Complex factorizedMuValue{};
            if (!exactScalarIdentityValue(
                    layer.tensors.eps[row][column], epsValue) ||
                !exactScalarIdentityValue(
                    layer.tensors.mu[row][column], muValue) ||
                !exactScalarIdentityValue(
                    layer.tensors.liEpsQ[row][column], factorizedEpsValue) ||
                !exactScalarIdentityValue(
                    layer.tensors.liMuQ[row][column], factorizedMuValue) ||
                factorizedEpsValue != epsValue ||
                factorizedMuValue != muValue) {
                return std::nullopt;
            }
            eps(row, column) = epsValue;
            mu(row, column) = muValue;
        }
    }
    if (eps(2, 2) == Complex{} || mu(2, 2) == Complex{}) {
        return std::nullopt;
    }
    Complex epsZzInverse{};
    Complex muZzInverse{};
    if (!exactScalarIdentityValue(
            layer.tensors.liEpsQZzInverse, epsZzInverse) ||
        !exactScalarIdentityValue(
            layer.tensors.liMuQZzInverse, muZzInverse) ||
        epsZzInverse != Complex{1.0, 0.0} / eps(2, 2) ||
        muZzInverse != Complex{1.0, 0.0} / mu(2, 2)) {
        return std::nullopt;
    }
    return UniformTensorPair{eps, mu};
}

std::shared_ptr<const TensorFourierMatrices> makeCachedTensorMatrices(
    TensorFourierMatrices tensors,
    const HarmonicBasis& basis) {
    annotateTensorFourierMetadata(tensors, basis.size());
    return std::make_shared<const TensorFourierMatrices>(std::move(tensors));
}

const PrecomputedPeriodicLayer2D& requirePrecomputedLayer(
    const PrecomputedPeriodicLayer2DPtr& layer) {
    if (!layer) {
        throw std::invalid_argument("precomputed 2D periodic layer pointer must not be null");
    }
    return *layer;
}

std::optional<UniformTensorPair> exactUniformTensors(
    const PrecomputedPeriodicLayer2DPtr& layer,
    Real) {
    return exactUniformTensors(requirePrecomputedLayer(layer));
}

std::optional<UniformTensorPair> exactUniformTensors(
    const LayerSpec& spec,
    Real wavelengthUm) {
    return std::visit(
        [&](const auto& layer) {
            return exactUniformTensors(layer, wavelengthUm);
        },
        spec);
}

std::optional<std::uint64_t> layerModalInput(
    const LayerSpec& spec,
    Real wavelengthUm,
    const TensorFourierMatrices* cachedTensors) {
    if (cachedTensors != nullptr) {
        if (cachedTensors->metadata.fingerprint == 0) {
            return std::nullopt;
        }
        return cachedTensors->metadata.fingerprint;
    }
    if (auto uniform = exactUniformTensors(spec, wavelengthUm)) {
        return modalFingerprintForUniformTensors(uniform->first, uniform->second);
    }
    return std::nullopt;
}

template <typename Layer>
TensorFourierMatrices makeLayerTensorMatrices(
    const Layer& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm,
    bool reduceUniform) {
    using LayerType = std::decay_t<Layer>;
    if constexpr (std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>) {
        return requirePrecomputedLayer(layer).tensors;
    } else {
        if (reduceUniform) {
            if (const auto uniform = exactUniformTensors(layer, wavelengthUm)) {
                return tensorFourierMatricesUniform(
                    uniform->first,
                    uniform->second,
                    basis);
            }
        }
        if constexpr (std::is_same_v<LayerType, UniformLayer>) {
            const auto [eps, mu] = layer.material.tensors(wavelengthUm);
            return tensorFourierMatricesUniform(eps, mu, basis);
        } else if constexpr (std::is_same_v<LayerType, GratingLayer>) {
            const auto [epsRidge, muRidge] =
                layer.materialRidge.tensors(wavelengthUm);
            const auto [epsGroove, muGroove] =
                layer.materialGroove.tensors(wavelengthUm);
            return tensorFourierMatricesBinaryGrating(
                layer.fillFactor,
                epsRidge,
                epsGroove,
                muRidge,
                muGroove,
                basis,
                layer.ridgeOffsetUm / layer.periodUm,
                layer.periodUm,
                layer.periodYUm,
                layer.fourierOptions);
        } else if constexpr (std::is_same_v<LayerType, PeriodicLayer2D>) {
            validatePeriodicSampling(layer, basis);
            return tensorPeriodicMatrices2d(layer, basis, wavelengthUm);
        } else {
            static_assert(std::is_same_v<LayerType, PatternedLayer2D>);
            return tensorFourierMatricesPatternedLayer2d(
                layer,
                basis,
                wavelengthUm);
        }
    }
}

TensorFourierMatrices tensorMatricesForLayer(const LayerSpec& spec,
                                                const HarmonicBasis& basis,
                                                Real wavelengthUm) {
    return std::visit(
        [&](const auto& layer) {
            return makeLayerTensorMatrices(layer, basis, wavelengthUm, true);
        },
        spec);
}

Real layerThicknessUm(const LayerSpec& spec) {
    return std::visit(
        [](const auto& layer) {
            using LayerType = std::decay_t<decltype(layer)>;
            if constexpr (std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>) {
                return requirePrecomputedLayer(layer).thicknessUm;
            } else {
                return layer.thicknessUm;
            }
        },
        spec);
}

std::shared_ptr<const TensorFourierMatrices> periodicTensorCacheForLayer(
    const LayerSpec& spec,
    const HarmonicBasis& basis,
    Real wavelengthUm) {
    return std::visit(
        [&](const auto& layer) -> std::shared_ptr<const TensorFourierMatrices> {
            using LayerType = std::decay_t<decltype(layer)>;
            if constexpr (std::is_same_v<LayerType, UniformLayer>) {
                return {};
            } else if constexpr (
                std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>) {
                const auto& precomputed = requirePrecomputedLayer(layer);
                validatePrecomputedPeriodicLayer2d(precomputed);
                if (exactUniformTensors(precomputed)) {
                    return {};
                }
                if (!precomputed.tensors.metadata.valid ||
                    precomputed.tensors.metadata.harmonicCount != basis.size() ||
                    precomputed.tensors.metadata.fingerprint == 0) {
                    return makeCachedTensorMatrices(precomputed.tensors, basis);
                }
                return std::shared_ptr<const TensorFourierMatrices>(
                    layer,
                    &precomputed.tensors);
            } else {
                if (exactUniformTensors(layer, wavelengthUm)) {
                    return {};
                }
                return makeCachedTensorMatrices(
                    makeLayerTensorMatrices(layer, basis, wavelengthUm, false),
                    basis);
            }
        },
        spec);
}

LayerModes computeLayerModes(const LayerSpec& spec,
                               const DiagonalOperator& Kx,
                               const DiagonalOperator& Ky,
                               const HarmonicBasis& basis,
                               Real wavelengthUm,
                               const TensorFourierMatrices* cachedTensors) {
    const Real thicknessUm = layerThicknessUm(spec);
    return std::visit(
        [&](const auto& layer) -> LayerModes {
            if (const auto uniform = exactUniformTensors(layer, wavelengthUm)) {
                return computeUniformModes(
                    uniform->first,
                    uniform->second,
                    Kx,
                    Ky,
                    thicknessUm,
                    wavelengthUm);
            }
            if (cachedTensors != nullptr) {
                return computePeriodicModes(
                    *cachedTensors,
                    Kx,
                    Ky,
                    thicknessUm,
                    wavelengthUm);
            }
            TensorFourierMatrices tensors = makeLayerTensorMatrices(
                layer,
                basis,
                wavelengthUm,
                false);
            annotateTensorFourierMetadata(tensors, basis.size());
            return computePeriodicModes(
                tensors,
                Kx,
                Ky,
                thicknessUm,
                wavelengthUm);
        },
        spec);
}

std::vector<Real> finiteLayerThicknesses(const std::vector<LayerSpec>& layers) {
    std::vector<Real> thicknesses;
    thicknesses.reserve(layers.size());
    for (const auto& layer : layers) {
        thicknesses.push_back(layerThicknessUm(layer));
    }
    return thicknesses;
}

LocatedDepth locateDepthInFiniteStack(const std::vector<Real>& thicknesses, Real zUm) {
    requireFinite(zUm, "field z coordinate");
    Real top = 0.0;
    for (std::size_t layer = 0; layer < thicknesses.size(); ++layer) {
        const Real bottom = top + thicknesses[layer];
        const bool lastLayer = layer + 1 == thicknesses.size();
        if ((zUm >= top - 1e-12 && zUm <= bottom + 1e-12) ||
            (lastLayer && std::abs(zUm - bottom) <= 1e-12)) {
            return {
                static_cast<int>(layer),
                std::min(std::max(zUm - top, Real{0.0}), thicknesses[layer]),
            };
        }
        top = bottom;
    }
    throw std::out_of_range("field z coordinate is outside the finite layer stack");
}

namespace {

template <typename Materials>
bool materialsAreStatic(const Materials& materials) {
    for (const auto& material : materials) {
        if (material.isDispersive()) {
            return false;
        }
    }
    return true;
}

bool layerTensorsStatic(const LayerSpec& layer) {
    return std::visit(
        [](const auto& concrete) {
            using LayerType = std::decay_t<decltype(concrete)>;
            if constexpr (std::is_same_v<LayerType, UniformLayer>) {
                return false;
            } else if constexpr (std::is_same_v<LayerType, GratingLayer>) {
                return !concrete.materialRidge.isDispersive() &&
                       !concrete.materialGroove.isDispersive();
            } else if constexpr (std::is_same_v<LayerType, PeriodicLayer2D>) {
                return materialsAreStatic(concrete.cells);
            } else if constexpr (std::is_same_v<LayerType, PatternedLayer2D>) {
                if (concrete.background.isDispersive()) {
                    return false;
                }
                return std::all_of(
                    concrete.regions.begin(),
                    concrete.regions.end(),
                    [](const PatternRegion& region) {
                        return !region.material.isDispersive();
                    });
            } else {
                static_assert(
                    std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>);
                return true;
            }
        },
        layer);
}

bool layerHasDispersiveMaterial(const LayerSpec& layer) {
    return std::visit(
        [](const auto& concrete) {
            using LayerType = std::decay_t<decltype(concrete)>;
            if constexpr (std::is_same_v<LayerType, UniformLayer>) {
                return concrete.material.isDispersive();
            } else if constexpr (std::is_same_v<LayerType, GratingLayer>) {
                return concrete.materialRidge.isDispersive() ||
                       concrete.materialGroove.isDispersive();
            } else if constexpr (std::is_same_v<LayerType, PeriodicLayer2D>) {
                return !materialsAreStatic(concrete.cells);
            } else if constexpr (std::is_same_v<LayerType, PatternedLayer2D>) {
                return concrete.background.isDispersive() ||
                       std::any_of(
                           concrete.regions.begin(),
                           concrete.regions.end(),
                           [](const PatternRegion& region) {
                               return region.material.isDispersive();
                           });
            } else {
                static_assert(
                    std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>);
                return false;
            }
        },
        layer);
}

bool tensorsEqual(const std::pair<Tensor3, Tensor3>& left,
                  const std::pair<Tensor3, Tensor3>& right) {
    return left.first.v == right.first.v && left.second.v == right.second.v;
}

void validatePassiveMaterial(
    const Material& material,
    Real wavelengthUm,
    const char* location,
    std::vector<std::pair<Tensor3, Tensor3>>& checkedTensors) {
    const auto tensors = material.tensors(wavelengthUm);
    if (std::any_of(
            checkedTensors.begin(),
            checkedTensors.end(),
            [&](const auto& checked) { return tensorsEqual(checked, tensors); })) {
        return;
    }
    checkedTensors.push_back(tensors);
    const MaterialPassivityResult result = materialPassivity(
        tensors.first,
        tensors.second);
    if (result.passive) {
        return;
    }

    std::ostringstream message;
    message.precision(8);
    message << "directional thermal channels require a passive local stack with "
               "positive-semidefinite loss operators under the exp(-i*omega*t) "
               "convention; material '"
            << material.name() << "' in " << location
            << " has minimum epsilon/mu loss-operator eigenvalues "
            << result.epsilonMinimumLossEigenvalue << " and "
            << result.muMinimumLossEigenvalue << " (tolerance "
            << result.tolerance << "). Use the ordinary spectrum/S-parameter APIs "
               "for active media; their scattering deficits are not thermodynamic "
               "emissivities.";
    throw std::invalid_argument(message.str());
}

void validatePassiveLayer(
    const LayerSpec& layer,
    Real wavelengthUm,
    std::vector<std::pair<Tensor3, Tensor3>>& checkedTensors) {
    const auto validateMaterial = [&](const Material& material, const char* location) {
        validatePassiveMaterial(
            material,
            wavelengthUm,
            location,
            checkedTensors);
    };
    std::visit(
        [&](const auto& concrete) {
            using LayerType = std::decay_t<decltype(concrete)>;
            if constexpr (std::is_same_v<LayerType, UniformLayer>) {
                validateMaterial(concrete.material, "uniform layer");
            } else if constexpr (std::is_same_v<LayerType, GratingLayer>) {
                validateMaterial(concrete.materialRidge, "grating ridge");
                validateMaterial(concrete.materialGroove, "grating groove");
            } else if constexpr (std::is_same_v<LayerType, PeriodicLayer2D>) {
                for (const Material& material : concrete.cells) {
                    validateMaterial(material, "sampled 2D layer");
                }
            } else if constexpr (std::is_same_v<LayerType, PatternedLayer2D>) {
                validateMaterial(
                    concrete.background,
                    "patterned-layer background");
                for (const PatternRegion& region : concrete.regions) {
                    validateMaterial(region.material, "patterned-layer region");
                }
            } else {
                static_assert(
                    std::is_same_v<LayerType, PrecomputedPeriodicLayer2DPtr>);
                const auto& precomputed = requirePrecomputedLayer(concrete);
                if (precomputed.sourceMaterials.empty()) {
                    throw std::invalid_argument(
                        "directional thermal channels cannot establish passivity for an "
                        "opaque PrecomputedPeriodicLayer2D without sourceMaterials; use "
                        "precomputePatternedLayer2d or a material-based layer");
                }
                for (const Material& material : precomputed.sourceMaterials) {
                    validateMaterial(material, "precomputed 2D layer");
                }
            }
        },
        layer);
}

StackGeometry makeStackGeometry(int harmonicCount,
                                  int orderX,
                                  int orderY,
                                  bool harmonicCountMode,
                                  LatticeTruncation truncation,
                                  const std::vector<LayerSpec>& layers) {
    StackGeometry geometry;
    geometry.lattice = inferStackLattice(layers);
    geometry.basis = harmonicCountMode
        ? makeHarmonicBasisCount(
            harmonicCount,
            truncation,
            geometry.lattice.periodXUm,
            geometry.lattice.periodYUm)
        : makeHarmonicBasisOrders(
            orderX,
            orderY,
            truncation,
            geometry.lattice.periodXUm,
            geometry.lattice.periodYUm);
    geometry.totalHarmonics = static_cast<int>(geometry.basis.size());
    geometry.centerIdx = geometry.basis.centerIndex();
    return geometry;
}

} // namespace

StackBuilder::StackBuilder(int harmonicCount,
                           int orderX,
                           int orderY,
                           bool harmonicCountMode,
                           LatticeTruncation truncation,
                           const std::vector<LayerSpec>& layers)
    : mLayers(layers),
      mGeometry(makeStackGeometry(
          harmonicCount,
          orderX,
          orderY,
          harmonicCountMode,
          truncation,
          layers)) {
    mPeriodicTensorPolicies.reserve(mLayers.size());
    for (const LayerSpec& layer : mLayers) {
        if (!latticeView(layer).has_value()) {
            mPeriodicTensorPolicies.push_back(PeriodicTensorPolicy::None);
        } else if (layerTensorsStatic(layer)) {
            mPeriodicTensorPolicies.push_back(PeriodicTensorPolicy::Static);
            mHasPeriodicLayers = true;
        } else {
            mPeriodicTensorPolicies.push_back(PeriodicTensorPolicy::Dynamic);
            mHasPeriodicLayers = true;
            mHasDynamicPeriodicLayers = true;
        }
        mHasDispersiveMaterials =
            mHasDispersiveMaterials || layerHasDispersiveMaterial(layer);
    }
}

bool StackBuilder::hasPeriodicLayers() const noexcept {
    return mHasPeriodicLayers;
}

bool StackBuilder::hasDynamicPeriodicLayers() const noexcept {
    return mHasDynamicPeriodicLayers;
}

bool StackBuilder::hasDispersiveMaterials() const noexcept {
    return mHasDispersiveMaterials;
}

void StackBuilder::validateThermalPassivity(
    Real wavelengthUm,
    const Material& superstrate,
    const Material& substrate) const {
    requirePositive(wavelengthUm, "thermal-channel wavelength");
    std::vector<std::pair<Tensor3, Tensor3>> checkedTensors;
    checkedTensors.reserve(mLayers.size() + 2);
    validatePassiveMaterial(
        superstrate, wavelengthUm, "superstrate", checkedTensors);
    validatePassiveMaterial(
        substrate, wavelengthUm, "substrate", checkedTensors);
    for (const LayerSpec& layer : mLayers) {
        validatePassiveLayer(layer, wavelengthUm, checkedTensors);
    }
}

PeriodicTensorCache StackBuilder::makePeriodicTensorCache(Real wavelengthUm) const {
    return makePeriodicTensorCache(wavelengthUm, nullptr);
}

PeriodicTensorCache StackBuilder::makePeriodicTensorCache(
    Real wavelengthUm,
    const PeriodicTensorCache* staticCache) const {
    if (staticCache != nullptr && !staticCache->empty() &&
        staticCache->size() != mLayers.size()) {
        throw std::invalid_argument("reusable periodic tensor cache must match layer count");
    }
    PeriodicTensorCache cache;
    cache.reserve(mLayers.size());
    for (std::size_t layerIndex = 0; layerIndex < mLayers.size(); ++layerIndex) {
        if (staticCache != nullptr && !staticCache->empty() &&
            staticCache->at(layerIndex)) {
            cache.push_back(staticCache->at(layerIndex));
            continue;
        }
        if (mPeriodicTensorPolicies[layerIndex] == PeriodicTensorPolicy::None) {
            cache.emplace_back();
            continue;
        }
        cache.push_back(
            periodicTensorCacheForLayer(mLayers[layerIndex], mGeometry.basis, wavelengthUm));
    }
    return cache;
}

PeriodicTensorCache StackBuilder::makeStaticPeriodicCache(
    Real referenceWavelengthUm) const {
    PeriodicTensorCache cache;
    cache.reserve(mLayers.size());
    for (std::size_t layerIndex = 0; layerIndex < mLayers.size(); ++layerIndex) {
        cache.push_back(
            mPeriodicTensorPolicies[layerIndex] == PeriodicTensorPolicy::Static
                ? periodicTensorCacheForLayer(
                    mLayers[layerIndex],
                    mGeometry.basis,
                    referenceWavelengthUm)
                : std::shared_ptr<const TensorFourierMatrices>{});
    }
    return std::any_of(
               cache.begin(),
               cache.end(),
               [](const auto& entry) { return static_cast<bool>(entry); })
        ? cache
        : PeriodicTensorCache{};
}

PreparedStack StackBuilder::prepare(Real wavelengthUm,
                                    Real thetaDeg,
                                    Real phiDeg,
                                    const Material& superstrate,
                                    const Material& substrate,
                                    const PeriodicTensorCache* periodicCache,
                                    bool analyticContinuation,
                                    const PreparedStack* xMirrorSource,
                                    Real blochKxReduced,
                                    Real blochKyReduced) const {
    requirePositive(wavelengthUm, "wavelength");
    requireFinite(thetaDeg, "theta_deg");
    requireFinite(phiDeg, "phi_deg");
    if (std::abs(thetaDeg) >= Real{90}) {
        throw std::invalid_argument(
            "theta_deg must be strictly between -90 and 90 degrees");
    }
    if (periodicCache != nullptr && periodicCache->size() != mLayers.size()) {
        throw std::invalid_argument("periodic tensor cache must match layer count");
    }

    PeriodicTensorCache ownedPeriodicCache;
    const PeriodicTensorCache* effectivePeriodicCache = periodicCache;
    if (effectivePeriodicCache == nullptr && mHasPeriodicLayers) {
        ownedPeriodicCache.resize(mLayers.size());
        effectivePeriodicCache = &ownedPeriodicCache;
    }

    Real kx0{};
    Real ky0{};
    if (analyticContinuation) {
        requireFinite(blochKxReduced, "bloch_kx_reduced");
        requireFinite(blochKyReduced, "bloch_ky_reduced");
        const auto requireScalarExterior = [&](const Material& material,
                                               const char* location) {
            const auto [eps, mu] = material.tensors(wavelengthUm);
            Complex epsScalar{};
            Complex muScalar{};
            if (!tensorIsScalar(eps, &epsScalar) ||
                !tensorIsScalar(mu, &muScalar) ||
                !(std::abs(epsScalar) > Real{0}) ||
                !(std::abs(muScalar) > Real{0})) {
                throw std::invalid_argument(
                    std::string("analytic-continuation ") + location +
                    " must have finite nonzero isotropic scalar epsilon and mu");
            }
        };
        requireScalarExterior(superstrate, "superstrate");
        requireScalarExterior(substrate, "substrate");
        // blochK*Reduced is expressed in reciprocal-lattice coordinates:
        // kx = blochKxReduced * 2*pi/periodX and likewise for y.  Kx/Ky in
        // the layer eigensystems are normalized by k0 = 2*pi/wavelength.
        kx0 = blochKxReduced * wavelengthUm / mGeometry.lattice.periodXUm;
        ky0 = blochKyReduced * wavelengthUm / mGeometry.lattice.periodYUm;
    } else {
        const Real theta = degToRad(thetaDeg);
        const Real phi = degToRad(std::remainder(phiDeg, Real{360}));
        const Complex nInc = validatedIncidentIndex(superstrate, wavelengthUm);
        kx0 = std::real(nInc) * std::sin(theta) * std::cos(phi);
        ky0 = std::real(nInc) * std::sin(theta) * std::sin(phi);
    }

    PreparedStack out;
    out.basis = mGeometry.basis;
    out.totalHarmonics = mGeometry.totalHarmonics;
    out.centerIdx = mGeometry.centerIdx;
    out.lattice = mGeometry.lattice;
    auto wavevectors = makeTransverseWavevectorOperators(
        kx0,
        ky0,
        wavelengthUm / mGeometry.lattice.periodXUm,
        wavelengthUm / mGeometry.lattice.periodYUm,
        mGeometry.basis);
    out.Kx = std::move(wavevectors.first);
    out.Ky = std::move(wavevectors.second);
    out.media.reserve(mLayers.size() + 2);
    out.media.push_back(modesForMaterial(
        superstrate, out.Kx, out.Ky, mGeometry.totalHarmonics, 0.0, wavelengthUm));
    std::optional<std::vector<std::size_t>> xMirrorMap;
    const bool canReuseMirroredModes =
        xMirrorSource != nullptr &&
        xMirrorSource->media.size() == mLayers.size() + 2;
    if (canReuseMirroredModes) {
        xMirrorMap = transverseSymmetryHarmonicMap(
            xMirrorSource->Kx,
            xMirrorSource->Ky,
            out.Kx,
            out.Ky,
            Real{-1},
            Real{1});
    }
    std::optional<std::vector<std::size_t>> inPlaneInversionMap;
    bool inPlaneInversionMapAttempted = false;
    constexpr TransverseSymmetryTransform xMirrorTransform{
        Real{-1},
        Real{1},
        std::array<Real, 4>{Real{-1}, Real{1}, Real{1}, Real{-1}},
        std::array<Real, 3>{Real{-1}, Real{1}, Real{1}},
    };
    constexpr TransverseSymmetryTransform inPlaneInversionTransform{
        Real{-1},
        Real{-1},
        std::array<Real, 4>{Real{-1}, Real{-1}, Real{-1}, Real{-1}},
        std::array<Real, 3>{Real{-1}, Real{-1}, Real{1}},
    };
    std::unordered_map<std::uint64_t, std::size_t> modalBasisCache;
    modalBasisCache.reserve(mLayers.size());
    for (std::size_t layerIndex = 0; layerIndex < mLayers.size(); ++layerIndex) {
        const auto& spec = mLayers[layerIndex];
        if (periodicCache == nullptr && mHasPeriodicLayers) {
            ownedPeriodicCache[layerIndex] = periodicTensorCacheForLayer(
                spec,
                mGeometry.basis,
                wavelengthUm);
        }
        const TensorFourierMatrices* cachedTensors = effectivePeriodicCache == nullptr
            ? nullptr
            : effectivePeriodicCache->at(layerIndex).get();
        const auto modalInput = layerModalInput(spec, wavelengthUm, cachedTensors);
        const auto cachedBasis = modalInput
            ? modalBasisCache.find(*modalInput)
            : modalBasisCache.end();
        if (cachedBasis != modalBasisCache.end()) {
            out.media.push_back(layerModesWithThickness(
                out.media[cachedBasis->second],
                layerThicknessUm(spec),
                wavelengthUm));
            ++out.finiteLayerModalBasisReuses;
            continue;
        }

        std::optional<LayerModes> mirrored;
        if (canReuseMirroredModes) {
            std::optional<UniformTensorPair> uniformMirrorInput;
            bool uniformMirrorInputChecked = false;
            const auto tryMirrorWithMap =
                [&](const TransverseSymmetryTransform& transform,
                    const std::vector<std::size_t>& harmonicMap)
                    -> std::optional<LayerModes> {
                    if (cachedTensors != nullptr) {
                        return tryMakeTransverseSymmetryPeriodicModes(
                            xMirrorSource->media[layerIndex + 1],
                            *cachedTensors,
                            xMirrorSource->Kx,
                            xMirrorSource->Ky,
                            out.Kx,
                            out.Ky,
                            transform,
                            &harmonicMap);
                    }
                    if (!uniformMirrorInputChecked) {
                        uniformMirrorInputChecked = true;
                        uniformMirrorInput = exactUniformTensors(
                            spec,
                            wavelengthUm);
                    }
                    if (!uniformMirrorInput) {
                        return std::nullopt;
                    }
                    return tryMakeTransverseSymmetryUniformModes(
                        xMirrorSource->media[layerIndex + 1],
                        uniformMirrorInput->first,
                        uniformMirrorInput->second,
                        xMirrorSource->Kx,
                        xMirrorSource->Ky,
                        out.Kx,
                        out.Ky,
                        transform,
                        &harmonicMap);
                };
            if (xMirrorMap) {
                mirrored = tryMirrorWithMap(xMirrorTransform, *xMirrorMap);
            }
            if (!mirrored) {
                if (!inPlaneInversionMapAttempted) {
                    inPlaneInversionMapAttempted = true;
                    inPlaneInversionMap = transverseSymmetryHarmonicMap(
                        xMirrorSource->Kx,
                        xMirrorSource->Ky,
                        out.Kx,
                        out.Ky,
                        Real{-1},
                        Real{-1});
                }
            }
            if (!mirrored && inPlaneInversionMap) {
                mirrored = tryMirrorWithMap(
                    inPlaneInversionTransform,
                    *inPlaneInversionMap);
            }
        }
        LayerModes modes = mirrored
            ? std::move(*mirrored)
            : computeLayerModes(
                spec,
                out.Kx,
                out.Ky,
                mGeometry.basis,
                wavelengthUm,
                cachedTensors);
        out.media.push_back(std::move(modes));
        ++out.finiteLayerModalBasisBuilds;
        if (modalInput) {
            modalBasisCache.emplace(
                *modalInput,
                out.media.size() - 1);
        }
    }
    out.media.push_back(modesForMaterial(
        substrate, out.Kx, out.Ky, mGeometry.totalHarmonics, 0.0, wavelengthUm));
    return out;
}

PreparedStack prepareStack(Real wavelengthUm,
                            Real thetaDeg,
                            Real phiDeg,
                            int harmonicCount,
                            int orderX,
                            int orderY,
                            bool harmonicCountMode,
                            LatticeTruncation truncation,
                            const Material& superstrate,
                            const Material& substrate,
                            const std::vector<LayerSpec>& layers) {
    return StackBuilder{
        harmonicCount,
        orderX,
        orderY,
        harmonicCountMode,
        truncation,
        layers}
        .prepare(
            wavelengthUm,
            thetaDeg,
            phiDeg,
            superstrate,
            substrate);
}

StackBuilder makeStackBuilder(const SolverState& s) {
    return StackBuilder(
        s.harmonicCount,
        s.orderX,
        s.orderY,
        s.harmonicCountMode,
        s.truncation,
        s.layers);
}

PreparedStack prepareStackFromState(const SolverState& s,
                                       const StackBuilder& builder,
                                       Real wavelengthUm,
                                       Real thetaDeg,
                                       Real phiDeg,
                                       const PeriodicTensorCache* periodicCache,
                                       bool analyticContinuation,
                                       Real blochKxReduced,
                                       Real blochKyReduced) {
    return builder.prepare(
        wavelengthUm,
        thetaDeg,
        phiDeg,
        s.superstrate,
        s.substrate,
        periodicCache,
        analyticContinuation,
        nullptr,
        blochKxReduced,
        blochKyReduced);
}

PreparedStack prepareStackFromState(const SolverState& s) {
    const StackBuilder builder = makeStackBuilder(s);
    return prepareStackFromState(
        s,
        builder,
        s.wavelengthUm,
        s.thetaDeg,
        s.phiDeg);
}

StackAmplitudeSolver::StackAmplitudeSolver(const PreparedStack& stack)
    : mStack(stack),
      mLayerCount(stack.media.size() - 2),
      mStateSize(layerStateSize(stack.media.front())),
      mModeCount(mStateSize / 2) {}

Matrix makeIncidentAmplitudes(std::size_t totalHarmonics,
                                int centerIdx,
                                Polarization pol) {
    Matrix inc(2 * totalHarmonics, 1);
    const std::size_t te0 = static_cast<std::size_t>(centerIdx);
    const std::size_t tm0 = totalHarmonics + static_cast<std::size_t>(centerIdx);
    const PolarizationAmplitudes amplitudes = polarizationAmplitudes(pol);
    inc(te0, 0) = amplitudes.te;
    inc(tm0, 0) = amplitudes.tm;
    return inc;
}

Matrix StackAmplitudeSolver::modeBlockProduct(const LayerModes& modes,
                                                bool down,
                                                const Matrix& rhs,
                                                Complex scale) const {
    if (rhs.rows() != mModeCount) {
        throw std::invalid_argument("mode block product rhs row count must match mode count");
    }
    const std::size_t sourceOffset = down ? 0 : mModeCount;
    Matrix out(mStateSize, rhs.cols());
    const auto& rhsData = rhs.data();
    auto& outData = out.data();
    const std::size_t rhsCols = rhs.cols();
    for (std::size_t r = 0; r < mStateSize; ++r) {
        const std::size_t outBase = r * rhsCols;
        for (std::size_t c = 0; c < rhsCols; ++c) {
            Complex sum{};
            for (std::size_t k = 0; k < mModeCount; ++k) {
                sum += modeStateValue(modes, r, sourceOffset + k) *
                    rhsData[k * rhsCols + c];
            }
            outData[outBase + c] = scale * sum;
        }
    }
    return out;
}

std::size_t StackAmplitudeSolver::layerDownOffset(std::size_t layerIndex) const {
    return mModeCount + 2 * layerIndex * mModeCount;
}

std::size_t StackAmplitudeSolver::layerUpOffset(std::size_t layerIndex) const {
    return mModeCount + (2 * layerIndex + 1) * mModeCount;
}

void StackAmplitudeSolver::setModeBlock(Matrix& target,
                                          std::size_t row,
                                          std::size_t col,
                                          const LayerModes& modes,
                                          bool down,
                                          Complex scale) const {
    const std::size_t sourceOffset = down ? 0 : mModeCount;
    const std::size_t targetCols = target.cols();
    auto& dst = target.data();
    for (std::size_t r = 0; r < mStateSize; ++r) {
        const std::size_t targetBase = (row + r) * targetCols + col;
        for (std::size_t c = 0; c < mModeCount; ++c) {
            dst[targetBase + c] =
                scale * modeStateValue(modes, r, sourceOffset + c);
        }
    }
}

void StackAmplitudeSolver::setPropagatedModeBlock(Matrix& target,
                                                     std::size_t row,
                                                     std::size_t col,
                                                     const LayerModes& modes,
                                                     bool down,
                                                     Complex scale) const {
    const auto& phase = propagationPhase(modes);
    const std::size_t sourceOffset = down ? 0 : mModeCount;
    if (sourceOffset + mModeCount > phase.size()) {
        throw std::invalid_argument("propagation phase block is outside layer mode storage");
    }
    const std::size_t targetCols = target.cols();
    auto& dst = target.data();
    for (std::size_t r = 0; r < mStateSize; ++r) {
        const std::size_t targetBase = (row + r) * targetCols + col;
        for (std::size_t c = 0; c < mModeCount; ++c) {
            dst[targetBase + c] =
                scale * modeStateValue(modes, r, sourceOffset + c) *
                phase[sourceOffset + c];
        }
    }
}

StackAmplitudes StackAmplitudeSolver::solve(const Matrix& inc) const {
    if (inc.rows() != mModeCount || inc.cols() != 1) {
        throw std::invalid_argument(
            "incident amplitude vector size must match modal channel count");
    }
    const std::size_t unknownCount = (2 * mLayerCount + 2) * mModeCount;
    const std::size_t equationCount = (mLayerCount + 1) * mStateSize;

    Matrix A(equationCount, unknownCount);
    Matrix rhs(equationCount, 1);

    setModeBlock(A, 0, 0, mStack.media.front(), false);

    if (mLayerCount == 0) {
        setModeBlock(A, 0, mModeCount, mStack.media[1], true, {-1.0, 0.0});
    } else {
        setModeBlock(A, 0, layerDownOffset(0), mStack.media[1], true, {-1.0, 0.0});
        setPropagatedModeBlock(
            A, 0, layerUpOffset(0), mStack.media[1], false, {-1.0, 0.0});
    }
    rhs.setBlock(0, 0, modeBlockProduct(mStack.media.front(), true, inc, {-1.0, 0.0}));

    for (std::size_t layer = 0; layer + 1 < mLayerCount; ++layer) {
        const std::size_t row = (layer + 1) * mStateSize;
        const LayerModes& left = mStack.media[layer + 1];
        const LayerModes& right = mStack.media[layer + 2];
        setPropagatedModeBlock(A, row, layerDownOffset(layer), left, true);
        setModeBlock(A, row, layerUpOffset(layer), left, false);
        setModeBlock(A, row, layerDownOffset(layer + 1), right, true, {-1.0, 0.0});
        setPropagatedModeBlock(
            A, row, layerUpOffset(layer + 1), right, false, {-1.0, 0.0});
    }

    if (mLayerCount > 0) {
        const std::size_t row = mLayerCount * mStateSize;
        const LayerModes& lastLayer = mStack.media[mLayerCount];
        const LayerModes& substrate = mStack.media.back();
        const std::size_t tOffset = mModeCount + 2 * mLayerCount * mModeCount;
        setPropagatedModeBlock(A, row, layerDownOffset(mLayerCount - 1), lastLayer, true);
        setModeBlock(A, row, layerUpOffset(mLayerCount - 1), lastLayer, false);
        setModeBlock(A, row, tOffset, substrate, true, {-1.0, 0.0});
    }

    const Matrix x = solveLinear(A, rhs);
    StackAmplitudes out;
    out.reflected = x.block(0, 0, mModeCount, 1);
    const std::size_t tOffset = mModeCount + 2 * mLayerCount * mModeCount;
    out.transmitted = x.block(tOffset, 0, mModeCount, 1);
    out.layerDownTop.reserve(mLayerCount);
    out.layerUpBottom.reserve(mLayerCount);
    for (std::size_t layer = 0; layer < mLayerCount; ++layer) {
        out.layerDownTop.push_back(
            x.block(layerDownOffset(layer), 0, mModeCount, 1));
        out.layerUpBottom.push_back(
            x.block(layerUpOffset(layer), 0, mModeCount, 1));
    }
    return out;
}

StackAmplitudes StackAmplitudeSolver::solve(Polarization pol) const {
    return solve(makeIncidentAmplitudes(
        static_cast<std::size_t>(mStack.totalHarmonics),
        mStack.centerIdx,
        pol));
}

StackAmplitudes solveStackAmplitudes(const PreparedStack& stack,
                                       const Matrix& inc) {
    return StackAmplitudeSolver(stack).solve(inc);
}

StackAmplitudes solveStackAmplitudes(const PreparedStack& stack, Polarization pol) {
    return StackAmplitudeSolver(stack).solve(pol);
}

} // namespace rcwa
