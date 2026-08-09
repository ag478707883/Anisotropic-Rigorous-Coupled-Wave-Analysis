#include "rcwa/field.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rcwa/fourier.hpp"
#include "rcwa/solver.hpp"

namespace rcwa {

namespace {

struct FieldLayerOperators {
    Matrix E31;
    Matrix E32;
    Matrix invE33;
    Matrix M31;
    Matrix M32;
    Matrix invM33;
};

struct FieldHarmonicState {
    Matrix Ex;
    Matrix Ey;
    Matrix Ez;
    Matrix Hx;
    Matrix Hy;
    Matrix Hz;
};

struct FieldComponentMask {
    bool ex{};
    bool ey{};
    bool ez{};
    bool hx{};
    bool hy{};
    bool hz{};
};

enum class LocatedRegion {
    Superstrate,
    Layer,
    Substrate,
};

struct LocatedGlobalDepth {
    LocatedRegion region{LocatedRegion::Layer};
    int layerIndex{};
    Real zLocalUm{};
};

struct LocalFieldPoint {
    int layerIndex{};
    Real zUm{};
};

struct ComputedDepthState {
    LocatedGlobalDepth depth;
    FieldHarmonicState fields;
};

struct SpatialPhaseCache {
    std::size_t harmonicCount{};
    std::vector<Complex> uPhase;
    std::vector<Complex> vPhase;
    bool vVaries{};

    [[nodiscard]] Complex phase(std::size_t iu, std::size_t iv, std::size_t h) const {
        const std::size_t vIndex = vVaries ? iv : std::size_t{0};
        return uPhase[iu * harmonicCount + h] *
               vPhase[vIndex * harmonicCount + h];
    }
};

bool inverseMatchesBlock(const Matrix& inverseBlock, const Matrix& block) {
    return !inverseBlock.empty() &&
           inverseBlock.rows() == block.rows() &&
           inverseBlock.cols() == block.cols();
}

FieldLayerOperators fieldLayerOperatorsFromTensors(const TensorFourierMatrices& tensors) {
    const Matrix& E31 = tensors.liEpsQ[2][0].empty()
        ? tensors.eps[2][0]
        : tensors.liEpsQ[2][0];
    const Matrix& E32 = tensors.liEpsQ[2][1].empty()
        ? tensors.eps[2][1]
        : tensors.liEpsQ[2][1];
    const Matrix& E33 = tensors.liEpsQ[2][2].empty()
        ? tensors.eps[2][2]
        : tensors.liEpsQ[2][2];
    const Matrix& M31 = tensors.liMuQ[2][0].empty()
        ? tensors.mu[2][0]
        : tensors.liMuQ[2][0];
    const Matrix& M32 = tensors.liMuQ[2][1].empty()
        ? tensors.mu[2][1]
        : tensors.liMuQ[2][1];
    const Matrix& M33 = tensors.liMuQ[2][2].empty()
        ? tensors.mu[2][2]
        : tensors.liMuQ[2][2];
    Matrix invE33 = inverseMatchesBlock(tensors.liEpsQZzInverse, E33)
        ? tensors.liEpsQZzInverse
        : inverse(E33);
    Matrix invM33 = inverseMatchesBlock(tensors.liMuQZzInverse, M33)
        ? tensors.liMuQZzInverse
        : inverse(M33);
    return {
        E31,
        E32,
        std::move(invE33),
        M31,
        M32,
        std::move(invM33),
    };
}

FieldLayerOperators buildFieldLayerOperators(const LayerSpec& spec,
                                                const HarmonicBasis& basis,
                                                Real wavelengthUm,
                                                const TensorFourierMatrices* cachedTensors) {
    if (cachedTensors != nullptr) {
        return fieldLayerOperatorsFromTensors(*cachedTensors);
    }
    TensorFourierMatrices tensors = tensorMatricesForLayer(spec, basis, wavelengthUm);
    return fieldLayerOperatorsFromTensors(tensors);
}

FieldComponentMask requestedFieldComponents(
    const std::vector<FieldComponent>& components) {
    FieldComponentMask mask;
    for (const FieldComponent component : components) {
        mask.ex = mask.ex || component == FieldComponent::Ex || component == FieldComponent::E;
        mask.ey = mask.ey || component == FieldComponent::Ey || component == FieldComponent::E;
        mask.ez = mask.ez || component == FieldComponent::Ez || component == FieldComponent::E;
        mask.hx = mask.hx || component == FieldComponent::Hx || component == FieldComponent::H;
        mask.hy = mask.hy || component == FieldComponent::Hy || component == FieldComponent::H;
        mask.hz = mask.hz || component == FieldComponent::Hz || component == FieldComponent::H;
    }
    return mask;
}

FieldHarmonicState harmonicFieldComponents(const Matrix& state,
                                             const FieldLayerOperators& ops,
                                             const std::vector<Complex>& kx,
                                             const std::vector<Complex>& ky,
                                             std::size_t harmonicCount,
                                             const FieldComponentMask& mask) {
    FieldHarmonicState out{
        state.block(0, 0, harmonicCount, 1),
        state.block(harmonicCount, 0, harmonicCount, 1),
        Matrix(harmonicCount, 1),
        state.block(2 * harmonicCount, 0, harmonicCount, 1),
        state.block(3 * harmonicCount, 0, harmonicCount, 1),
        Matrix(harmonicCount, 1),
    };

    if (mask.ez) {
        Matrix rhs(harmonicCount, 1);
        for (std::size_t h = 0; h < harmonicCount; ++h) {
            rhs(h, 0) = ky[h] * out.Hx(h, 0) - kx[h] * out.Hy(h, 0);
        }
        rhs = rhs - ops.E31 * out.Ex - ops.E32 * out.Ey;
        out.Ez = ops.invE33 * rhs;
    }
    if (mask.hz) {
        Matrix rhs(harmonicCount, 1);
        for (std::size_t h = 0; h < harmonicCount; ++h) {
            rhs(h, 0) = kx[h] * out.Ey(h, 0) - ky[h] * out.Ex(h, 0);
        }
        rhs = rhs - ops.M31 * out.Hx - ops.M32 * out.Hy;
        out.Hz = ops.invM33 * rhs;
    }
    return out;
}

Complex isotropicScalar(const Tensor3& tensor, const char* name, Real tol = 1e-12) {
    const Complex value = tensor(0, 0);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const Complex expected = (r == c) ? value : Complex{};
            if (std::abs(tensor(r, c) - expected) > tol) {
                throw std::runtime_error(
                    std::string("physical incident field normalization requires isotropic ") +
                    name);
            }
        }
    }
    return value;
}

Complex forwardSqrt(Complex value) {
    Complex root = std::sqrt(value);
    if (std::imag(root) < -1e-14 ||
        (std::abs(std::imag(root)) <= 1e-14 && std::real(root) < 0.0)) {
        root = -root;
    }
    return root;
}

std::array<Complex, 4> physicalPlaneWaveTangentialState(Complex kx,
                                                            Complex ky,
                                                            Complex kz,
                                                            Complex eps,
                                                            Complex mu,
                                                            bool te) {
    const Complex kp = std::sqrt(kx * kx + ky * ky);
    Complex sx{};
    Complex sy{1.0, 0.0};
    if (std::abs(kp) >= 1e-14) {
        sx = -ky / kp;
        sy = kx / kp;
    }

    if (te) {
        return {
            sx,
            sy,
            -kz * sy / mu,
            kz * sx / mu,
        };
    }

    const Complex n = forwardSqrt(eps * mu);
    const Complex px = sy * kz / n;
    const Complex py = -sx * kz / n;
    const Complex pz = (sx * ky - sy * kx) / n;
    return {
        px,
        py,
        (ky * pz - kz * py) / mu,
        (kz * px - kx * pz) / mu,
    };
}

Complex physicalChannelScale(const LayerModes& superstrateModes,
                               std::size_t harmonicCount,
                               std::size_t harmonic,
                               std::size_t column,
                               const std::array<Complex, 4>& target) {
    std::size_t best = 0;
    Real bestAbs = 0.0;
    for (std::size_t component = 0; component < target.size(); ++component) {
        const Real magnitude = std::abs(target[component]);
        if (magnitude > bestAbs) {
            bestAbs = magnitude;
            best = component;
        }
    }
    if (bestAbs < 1e-300) {
        return {1.0, 0.0};
    }
    const std::size_t row = best * harmonicCount + harmonic;
    const Complex normalized = modeStateValue(superstrateModes, row, column);
    if (std::abs(normalized) < 1e-300) {
        throw std::runtime_error("could not normalize incident field amplitude from modal basis");
    }
    return target[best] / normalized;
}

Matrix physicalIncidentAmplitudes(const PreparedStack& stack,
                                    const Material& superstrate,
                                    Real wavelengthUm,
                                    Polarization pol) {
    Matrix inc = makeIncidentAmplitudes(
        static_cast<std::size_t>(stack.totalHarmonics), stack.centerIdx, pol);
    const Tensor3 epsTensor = superstrate.epsilonTensor(wavelengthUm);
    const Tensor3 muTensor = superstrate.muTensor(wavelengthUm);
    const Complex eps = isotropicScalar(epsTensor, "superstrate epsilon");
    const Complex mu = isotropicScalar(muTensor, "superstrate permeability");
    const auto& kx = stack.Kx.values();
    const auto& ky = stack.Ky.values();
    const std::size_t center = static_cast<std::size_t>(stack.centerIdx);
    const std::size_t harmonicCount = static_cast<std::size_t>(stack.totalHarmonics);
    const Complex kz = forwardSqrt(eps * mu - kx[center] * kx[center] -
                                    ky[center] * ky[center]);

    const Complex teScale = physicalChannelScale(
        stack.media.front(),
        harmonicCount,
        center,
        center,
        physicalPlaneWaveTangentialState(kx[center], ky[center], kz, eps, mu, true));
    const Complex tmScale = physicalChannelScale(
        stack.media.front(),
        harmonicCount,
        center,
        harmonicCount + center,
        physicalPlaneWaveTangentialState(kx[center], ky[center], kz, eps, mu, false));

    inc(center, 0) *= teScale;
    inc(harmonicCount + center, 0) *= tmScale;
    return inc;
}

Real interpolateAxis(Real minValue, Real maxValue, int count, int index) {
    if (count <= 1) {
        return minValue;
    }
    const Real t = static_cast<Real>(index) / static_cast<Real>(count - 1);
    return minValue + (maxValue - minValue) * t;
}

Real totalStackThickness(const std::vector<Real>& thicknesses) {
    Real total = 0.0;
    for (Real thickness : thicknesses) {
        total += thickness;
    }
    return total;
}

LocatedGlobalDepth locateGlobalDepth(const std::vector<Real>& thicknesses, Real zUm) {
    requireFinite(zUm, "field z coordinate");
    if (zUm < 0.0) {
        return {LocatedRegion::Superstrate, -1, zUm};
    }
    const Real total = totalStackThickness(thicknesses);
    if (zUm > total) {
        return {LocatedRegion::Substrate, static_cast<int>(thicknesses.size()), zUm - total};
    }
    const LocatedDepth depth = locateDepthInFiniteStack(thicknesses, zUm);
    return {LocatedRegion::Layer, depth.layerIndex, depth.zLocalUm};
}

Matrix modalStateCoefficients(const LayerModes& modes,
                              const Matrix& modalAmplitudes) {
    const std::size_t harmonicCount = layerHarmonicCount(modes);
    const std::size_t stateSize = 4 * harmonicCount;
    if (modalAmplitudes.rows() != stateSize || modalAmplitudes.cols() != 1) {
        throw std::invalid_argument(
            "modal field amplitudes must contain one value per mode");
    }
    if (!hasCompactUniformModes(modes)) {
        return modes.W * modalAmplitudes;
    }

    Matrix state(stateSize, 1);
    for (std::size_t harmonic = 0; harmonic < harmonicCount; ++harmonic) {
        const auto& block = modes.uniformBlocks[harmonic].values;
        for (std::size_t component = 0; component < 4; ++component) {
            Complex value{};
            for (std::size_t localColumn = 0; localColumn < 4; ++localColumn) {
                value += block[component * 4 + localColumn] *
                         modalAmplitudes(localColumn * harmonicCount + harmonic, 0);
            }
            state(component * harmonicCount + harmonic, 0) = value;
        }
    }
    return state;
}

Matrix fieldStateCoefficients(const PreparedStack& stack,
                                const StackAmplitudes& amplitudes,
                                const std::vector<Real>& thicknesses,
                                const LocalFieldPoint& point,
                                Real wavelengthUm) {
    if (point.layerIndex < 0 ||
        point.layerIndex >= static_cast<int>(stack.media.size()) - 2) {
        throw std::out_of_range("field sample layer_index is outside the finite layer stack");
    }
    const auto layerIndex = static_cast<std::size_t>(point.layerIndex);
    const Real thickness = thicknesses[layerIndex];
    if (point.zUm < -1e-12 || point.zUm > thickness + 1e-12) {
        throw std::out_of_range("field sample z_um is outside the requested finite layer");
    }

    const LayerModes& modes = stack.media[layerIndex + 1];
    const std::size_t harmonicCount = static_cast<std::size_t>(stack.totalHarmonics);
    const std::size_t modeCount = 2 * harmonicCount;
    Matrix modalAmplitudes(4 * harmonicCount, 1);
    const Real k0z = twoPi * point.zUm / wavelengthUm;
    const Real k0Up = twoPi * (thickness - point.zUm) / wavelengthUm;
    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        const Complex downPhase = boundedExp(iu * modes.gamma[mode] * k0z);
        const Complex downAmp = amplitudes.layerDownTop[layerIndex](mode, 0) * downPhase;
        const std::size_t upCol = modeCount + mode;
        // Up-mode gamma is stored with the backward sign, so convert it to the
        // bottom-to-point propagation factor.
        const Complex upPhase = boundedExp(iu * (-modes.gamma[upCol]) * k0Up);
        const Complex upAmp = amplitudes.layerUpBottom[layerIndex](mode, 0) * upPhase;
        modalAmplitudes(mode, 0) = downAmp;
        modalAmplitudes(upCol, 0) = upAmp;
    }
    return modalStateCoefficients(modes, modalAmplitudes);
}

Matrix halfspaceStateCoefficients(const LayerModes& modes,
                                    const Matrix& downAmplitudes,
                                    const Matrix& upAmplitudes,
                                    Real zUm,
                                    Real wavelengthUm) {
    const std::size_t harmonicCount = layerHarmonicCount(modes);
    const std::size_t modeCount = 2 * harmonicCount;
    Matrix modalAmplitudes(4 * harmonicCount, 1);
    const Real k0z = twoPi * zUm / wavelengthUm;

    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        const Complex downPhase = boundedExp(iu * modes.gamma[mode] * k0z);
        const Complex downAmp = downAmplitudes.empty()
            ? Complex{}
            : downAmplitudes(mode, 0) * downPhase;
        const std::size_t upCol = modeCount + mode;
        const Complex upPhase = boundedExp(iu * modes.gamma[upCol] * k0z);
        const Complex upAmp = upAmplitudes.empty()
            ? Complex{}
            : upAmplitudes(mode, 0) * upPhase;
        modalAmplitudes(mode, 0) = downAmp;
        modalAmplitudes(upCol, 0) = upAmp;
    }
    return modalStateCoefficients(modes, modalAmplitudes);
}

Matrix globalFieldStateCoefficients(const PreparedStack& stack,
                                       const StackAmplitudes& amplitudes,
                                       const std::vector<Real>& thicknesses,
                                       const Matrix& incident,
                                       const LocatedGlobalDepth& depth,
                                       Real wavelengthUm) {
    if (depth.region == LocatedRegion::Superstrate) {
        return halfspaceStateCoefficients(
            stack.media.front(), incident, amplitudes.reflected, depth.zLocalUm, wavelengthUm);
    }
    if (depth.region == LocatedRegion::Substrate) {
        return halfspaceStateCoefficients(
            stack.media.back(),
            amplitudes.transmitted,
            Matrix{},
            depth.zLocalUm,
            wavelengthUm);
    }
    const LocalFieldPoint point{depth.layerIndex, depth.zLocalUm};
    return fieldStateCoefficients(stack, amplitudes, thicknesses, point, wavelengthUm);
}

void fillSpatialPhaseRow(std::vector<Complex>& phases,
                            std::size_t row,
                            const std::vector<Complex>& wavevectors,
                            Real coordinateUm,
                            Real phaseScale) {
    Complex* dst = phases.data() + row * wavevectors.size();
    for (std::size_t h = 0; h < wavevectors.size(); ++h) {
        dst[h] = std::exp(iu * wavevectors[h] * coordinateUm * phaseScale);
    }
}

SpatialPhaseCache buildSpatialPhaseCache(const FieldPlaneRequest& request,
                                            const std::vector<Complex>& kx,
                                            const std::vector<Complex>& ky,
                                            Real wavelengthUm) {
    SpatialPhaseCache cache;
    cache.harmonicCount = kx.size();
    const Real phaseScale = twoPi / wavelengthUm;
    const auto uPoints = static_cast<std::size_t>(request.uPoints);
    const auto vPoints = static_cast<std::size_t>(request.vPoints);
    cache.uPhase.resize(uPoints * cache.harmonicCount);

    switch (request.plane) {
    case FieldPlane::XY:
        cache.vVaries = true;
        cache.vPhase.resize(vPoints * cache.harmonicCount);
        for (std::size_t iu = 0; iu < uPoints; ++iu) {
            const Real x = interpolateAxis(
                request.uMinUm, request.uMaxUm, request.uPoints, static_cast<int>(iu));
            fillSpatialPhaseRow(cache.uPhase, iu, kx, x, phaseScale);
        }
        for (std::size_t iv = 0; iv < vPoints; ++iv) {
            const Real y = interpolateAxis(
                request.vMinUm, request.vMaxUm, request.vPoints, static_cast<int>(iv));
            fillSpatialPhaseRow(cache.vPhase, iv, ky, y, phaseScale);
        }
        break;
    case FieldPlane::XZ:
        cache.vVaries = false;
        cache.vPhase.resize(cache.harmonicCount);
        for (std::size_t iu = 0; iu < uPoints; ++iu) {
            const Real x = interpolateAxis(
                request.uMinUm, request.uMaxUm, request.uPoints, static_cast<int>(iu));
            fillSpatialPhaseRow(cache.uPhase, iu, kx, x, phaseScale);
        }
        fillSpatialPhaseRow(cache.vPhase, 0, ky, request.fixedUm, phaseScale);
        break;
    case FieldPlane::YZ:
        cache.vVaries = false;
        cache.vPhase.resize(cache.harmonicCount);
        for (std::size_t iu = 0; iu < uPoints; ++iu) {
            const Real y = interpolateAxis(
                request.uMinUm, request.uMaxUm, request.uPoints, static_cast<int>(iu));
            fillSpatialPhaseRow(cache.uPhase, iu, ky, y, phaseScale);
        }
        fillSpatialPhaseRow(cache.vPhase, 0, kx, request.fixedUm, phaseScale);
        break;
    }
    return cache;
}

FieldValue accumulateFieldValueAt(const FieldHarmonicState& state,
                                     const SpatialPhaseCache& phases,
                                     std::size_t iu,
                                     std::size_t iv,
                                     const FieldComponentMask& mask) {
    FieldValue out;
    for (std::size_t h = 0; h < phases.harmonicCount; ++h) {
        const Complex phase = phases.phase(iu, iv, h);
        if (mask.ex) out.Ex += state.Ex(h, 0) * phase;
        if (mask.ey) out.Ey += state.Ey(h, 0) * phase;
        if (mask.ez) out.Ez += state.Ez(h, 0) * phase;
        if (mask.hx) out.Hx += state.Hx(h, 0) * phase;
        if (mask.hy) out.Hy += state.Hy(h, 0) * phase;
        if (mask.hz) out.Hz += state.Hz(h, 0) * phase;
    }
    return out;
}

void validateFieldPlaneRequest(const FieldPlaneRequest& request) {
    if (request.uPoints <= 0 || request.vPoints <= 0) {
        throw std::invalid_argument("field plane point counts must be positive");
    }
    requireFinite(request.uMinUm, "field plane u_min_um");
    requireFinite(request.uMaxUm, "field plane u_max_um");
    requireFinite(request.vMinUm, "field plane v_min_um");
    requireFinite(request.vMaxUm, "field plane v_max_um");
    requireFinite(request.fixedUm, "field plane fixed_um");
    if (request.components.empty()) {
        throw std::invalid_argument("field plane component list must not be empty");
    }
    if (request.workers < 0) {
        throw std::invalid_argument("field plane workers must be non-negative");
    }
}

std::vector<FieldLayerOperators> buildFieldLayerOperators(
    const std::vector<LayerSpec>& layers,
    const HarmonicBasis& basis,
    Real wavelengthUm,
    const PeriodicTensorCache* periodicCache) {
    std::vector<FieldLayerOperators> fieldLayers;
    fieldLayers.reserve(layers.size());
    if (periodicCache != nullptr && periodicCache->size() != layers.size()) {
        throw std::invalid_argument("field periodic tensor cache must match layer count");
    }
    for (std::size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        const TensorFourierMatrices* cachedTensors = periodicCache == nullptr
            ? nullptr
            : periodicCache->at(layerIndex).get();
        fieldLayers.push_back(buildFieldLayerOperators(
            layers[layerIndex],
            basis,
            wavelengthUm,
            cachedTensors));
    }
    return fieldLayers;
}

const FieldLayerOperators& operatorsForDepth(
    const std::vector<FieldLayerOperators>& fieldLayers,
    const LocatedGlobalDepth& depth,
    const FieldLayerOperators& superstrateOps,
    const FieldLayerOperators& substrateOps) {
    if (depth.region == LocatedRegion::Superstrate) {
        return superstrateOps;
    }
    if (depth.region == LocatedRegion::Substrate) {
        return substrateOps;
    }
    return fieldLayers[static_cast<std::size_t>(depth.layerIndex)];
}

Vec3 fieldPlanePoint(const FieldPlaneRequest& request, int iu, int iv) {
    const Real u = interpolateAxis(request.uMinUm, request.uMaxUm, request.uPoints, iu);
    const Real v = interpolateAxis(request.vMinUm, request.vMaxUm, request.vPoints, iv);
    switch (request.plane) {
    case FieldPlane::XY:
        return {u, v, request.fixedUm};
    case FieldPlane::XZ:
        return {u, request.fixedUm, v};
    case FieldPlane::YZ:
        return {request.fixedUm, u, v};
    }
    throw std::invalid_argument("unsupported field plane");
}

void rejectExactRayleighCutoff(const PreparedStack& stack) {
    const auto hasCutoff = [](const LayerModes& modes) {
        return std::any_of(
            modes.gamma.begin(),
            modes.gamma.end(),
            [](Complex gamma) { return std::abs(gamma) <= Real{1e-12}; });
    };
    if (hasCutoff(stack.media.front()) || hasCutoff(stack.media.back())) {
        throw std::invalid_argument(
            "field reconstruction is undefined at an exact exterior Rayleigh cutoff; "
            "shift the wavelength or incidence angle slightly");
    }
}

} // namespace

const char* fieldComponentName(FieldComponent component) {
    switch (component) {
    case FieldComponent::Ex:
        return "Ex";
    case FieldComponent::Ey:
        return "Ey";
    case FieldComponent::Ez:
        return "Ez";
    case FieldComponent::E:
        return "E";
    case FieldComponent::Hx:
        return "Hx";
    case FieldComponent::Hy:
        return "Hy";
    case FieldComponent::Hz:
        return "Hz";
    case FieldComponent::H:
        return "H";
    }
    return "Unknown";
}

const char* fieldPlaneName(FieldPlane plane) {
    switch (plane) {
    case FieldPlane::XY:
        return "xy";
    case FieldPlane::XZ:
        return "xz";
    case FieldPlane::YZ:
        return "yz";
    }
    return "unknown";
}

Complex fieldComponentValue(const FieldValue& value, FieldComponent component) {
    switch (component) {
    case FieldComponent::Ex:
        return value.Ex;
    case FieldComponent::Ey:
        return value.Ey;
    case FieldComponent::Ez:
        return value.Ez;
    case FieldComponent::E:
        return {std::sqrt(std::norm(value.Ex) + std::norm(value.Ey) + std::norm(value.Ez)), 0.0};
    case FieldComponent::Hx:
        return value.Hx;
    case FieldComponent::Hy:
        return value.Hy;
    case FieldComponent::Hz:
        return value.Hz;
    case FieldComponent::H:
        return {std::sqrt(std::norm(value.Hx) + std::norm(value.Hy) + std::norm(value.Hz)), 0.0};
    }
    throw std::invalid_argument("unsupported field component");
}

FieldGridResult RcwaSolver::solveFieldGrid(const FieldPlaneRequest& request) const {
    validateFieldPlaneRequest(request);
    const SolverState& s = mState;

    const StackBuilder builder(
        s.harmonicCount,
        s.orderX,
        s.orderY,
        s.harmonicCountMode,
        s.truncation,
        s.layers);
    PeriodicTensorCache periodicCache = builder.hasPeriodicLayers()
        ? builder.makePeriodicTensorCache(s.wavelengthUm)
        : PeriodicTensorCache{};
    const PeriodicTensorCache* periodicCachePtr = periodicCache.empty()
        ? nullptr
        : &periodicCache;

    PreparedStack stack = builder.prepare(
        s.wavelengthUm,
        s.thetaDeg,
        s.phiDeg,
        s.superstrate,
        s.substrate,
        periodicCachePtr);
    rejectExactRayleighCutoff(stack);
    const Matrix incident = physicalIncidentAmplitudes(
        stack, s.superstrate, s.wavelengthUm, s.pol);
    StackAmplitudes amplitudes = solveStackAmplitudes(stack, incident);
    const std::vector<Real> thicknesses = finiteLayerThicknesses(s.layers);

    std::vector<FieldLayerOperators> fieldLayers =
        buildFieldLayerOperators(s.layers, stack.basis, s.wavelengthUm, periodicCachePtr);
    const FieldLayerOperators superstrateOps = buildFieldLayerOperators(
        UniformLayer{s.superstrate, 0.0}, stack.basis, s.wavelengthUm, nullptr);
    const FieldLayerOperators substrateOps = buildFieldLayerOperators(
        UniformLayer{s.substrate, 0.0}, stack.basis, s.wavelengthUm, nullptr);

    const auto& kx = stack.Kx.values();
    const auto& ky = stack.Ky.values();
    const auto harmonicCount = static_cast<std::size_t>(stack.totalHarmonics);
    const SpatialPhaseCache phaseCache =
        buildSpatialPhaseCache(request, kx, ky, s.wavelengthUm);
    const FieldComponentMask componentMask =
        requestedFieldComponents(request.components);

    FieldGridResult out;
    out.plane = request.plane;
    out.fixedUm = request.fixedUm;
    out.wavelengthUm = s.wavelengthUm;
    out.thetaDeg = s.thetaDeg;
    out.phiDeg = s.phiDeg;
    out.N = stack.totalHarmonics;
    out.uPoints = request.uPoints;
    out.vPoints = request.vPoints;
    out.components = request.components;
    out.uUm.resize(static_cast<std::size_t>(request.uPoints));
    out.vUm.resize(static_cast<std::size_t>(request.vPoints));
    for (int uIndex = 0; uIndex < request.uPoints; ++uIndex) {
        out.uUm[static_cast<std::size_t>(uIndex)] = interpolateAxis(
            request.uMinUm,
            request.uMaxUm,
            request.uPoints,
            uIndex);
    }
    for (int vIndex = 0; vIndex < request.vPoints; ++vIndex) {
        out.vUm[static_cast<std::size_t>(vIndex)] = interpolateAxis(
            request.vMinUm,
            request.vMaxUm,
            request.vPoints,
            vIndex);
    }

    const auto pointCount = static_cast<std::size_t>(request.uPoints) *
        static_cast<std::size_t>(request.vPoints);
    const auto componentCount = request.components.size();
    out.values.resize(checkedProduct(pointCount, componentCount, "field grid output value"));
    const auto makeDepthState = [&](int iv) {
        const Vec3 point = fieldPlanePoint(request, 0, iv);
        const LocatedGlobalDepth depth = locateGlobalDepth(thicknesses, point.z);
        const Matrix state =
            globalFieldStateCoefficients(
                stack, amplitudes, thicknesses, incident, depth, s.wavelengthUm);
        const FieldLayerOperators& ops = operatorsForDepth(fieldLayers, depth, superstrateOps, substrateOps);
        ComputedDepthState rowState;
        rowState.depth = depth;
        rowState.fields = harmonicFieldComponents(
            state,
            ops,
            kx,
            ky,
            harmonicCount,
            componentMask);
        return rowState;
    };

    const auto writePoint = [&](const ComputedDepthState& rowState,
                                int uIndex,
                                int vIndex) {
        const std::size_t linear = static_cast<std::size_t>(vIndex) *
            static_cast<std::size_t>(request.uPoints) +
            static_cast<std::size_t>(uIndex);
        const FieldValue value = accumulateFieldValueAt(
            rowState.fields,
            phaseCache,
            static_cast<std::size_t>(uIndex),
            static_cast<std::size_t>(vIndex),
            componentMask);
        for (std::size_t componentIndex = 0; componentIndex < componentCount;
             ++componentIndex) {
            const FieldComponent component = request.components[componentIndex];
            const std::size_t valueIndex = componentIndex * pointCount + linear;
            out.values[valueIndex] = fieldComponentValue(value, component);
        }
    };

    const std::size_t depthCount = request.plane == FieldPlane::XY
        ? std::size_t{1}
        : static_cast<std::size_t>(request.vPoints);
    std::vector<ComputedDepthState> depthStates(depthCount);
    runIndexedJobs(depthCount, request.workers, [&](std::size_t depthIndex) {
        depthStates[depthIndex] = makeDepthState(static_cast<int>(depthIndex));
    });
    out.layerIndexByV.resize(static_cast<std::size_t>(request.vPoints));
    out.zLocalUmByV.resize(static_cast<std::size_t>(request.vPoints));
    for (int vIndex = 0; vIndex < request.vPoints; ++vIndex) {
        const std::size_t depthIndex = request.plane == FieldPlane::XY
            ? std::size_t{0}
            : static_cast<std::size_t>(vIndex);
        out.layerIndexByV[static_cast<std::size_t>(vIndex)] =
            depthStates[depthIndex].depth.layerIndex;
        out.zLocalUmByV[static_cast<std::size_t>(vIndex)] =
            depthStates[depthIndex].depth.zLocalUm;
    }
    runIndexedJobs(pointCount, request.workers, [&](std::size_t linear) {
        const int uIndex =
            static_cast<int>(linear % static_cast<std::size_t>(request.uPoints));
        const int vIndex =
            static_cast<int>(linear / static_cast<std::size_t>(request.uPoints));
        writePoint(
            depthStates[request.plane == FieldPlane::XY
                ? std::size_t{0}
                : static_cast<std::size_t>(vIndex)],
            uIndex,
            vIndex);
    });
    return out;
}

FieldPlaneResult RcwaSolver::solveFieldPlane(const FieldPlaneRequest& request) const {
    const FieldGridResult grid = solveFieldGrid(request);
    FieldPlaneResult out;
    out.plane = grid.plane;
    out.wavelengthUm = grid.wavelengthUm;
    out.thetaDeg = grid.thetaDeg;
    out.phiDeg = grid.phiDeg;
    out.N = grid.N;
    out.uPoints = grid.uPoints;
    out.vPoints = grid.vPoints;

    const std::size_t pointCount = checkedProduct(
        static_cast<std::size_t>(grid.uPoints),
        static_cast<std::size_t>(grid.vPoints),
        "field plane output point");
    const std::size_t componentCount = grid.components.size();
    out.samples.resize(checkedProduct(
        pointCount,
        componentCount,
        "field plane output sample"));
    runIndexedJobs(pointCount, request.workers, [&](std::size_t linear) {
        const int uIndex = static_cast<int>(
            linear % static_cast<std::size_t>(grid.uPoints));
        const int vIndex = static_cast<int>(
            linear / static_cast<std::size_t>(grid.uPoints));
        const Vec3 point = fieldPlanePoint(request, uIndex, vIndex);
        const std::size_t sampleBase = linear * componentCount;
        for (std::size_t componentIndex = 0; componentIndex < componentCount;
             ++componentIndex) {
            const Complex value = grid.values[componentIndex * pointCount + linear];
            out.samples[sampleBase + componentIndex] = FieldPlaneSample{
                uIndex,
                vIndex,
                grid.layerIndexByV[static_cast<std::size_t>(vIndex)],
                point.x,
                point.y,
                point.z,
                grid.zLocalUmByV[static_cast<std::size_t>(vIndex)],
                grid.components[componentIndex],
                value,
                std::abs(value),
            };
        }
    });
    return out;
}

} // namespace rcwa
