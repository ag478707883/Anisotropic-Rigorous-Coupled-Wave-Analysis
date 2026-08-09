#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "rcwa/fourier.hpp"
#include "rcwa/layer.hpp"
#include "rcwa/material.hpp"
#include "rcwa/matrix.hpp"
#include "rcwa/pattern.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

struct SolverState;

// A homogeneous finite layer in the z stack.
struct UniformLayer {
    Material material;
    Real thicknessUm{};
};

// A binary lamellar grating with x-periodic ridges and a shared y lattice.
struct GratingLayer {
    Material materialRidge;
    Material materialGroove;
    Real fillFactor{0.5};
    Real periodUm{1.0};
    Real periodYUm{1.0};
    Real ridgeOffsetUm{0.0};
    Real thicknessUm{0.3};
    FourierConvergenceOptions fourierOptions{};
};

// A tapered grating facade that expands into explicit lamellar slices.
struct AdaptiveTaperedGratingLayer {
    Material materialRidge;
    Material materialGroove;
    Real topFillFactor{0.5};
    Real bottomFillFactor{0.5};
    Real periodUm{1.0};
    Real periodYUm{1.0};
    Real ridgeOffsetUm{0.0};
    Real thicknessUm{0.3};
    Real fillStepTolerance{0.02};
    int minSlices{1};
    int maxSlices{256};
};

struct FillFactorProfilePoint {
    Real zFraction{};
    Real fillFactor{};
};

// A profile-driven grating facade that expands into explicit lamellar slices.
struct AdaptiveProfiledGratingLayer {
    Material materialRidge;
    Material materialGroove;
    std::vector<FillFactorProfilePoint> fillProfile;
    Real periodUm{1.0};
    Real periodYUm{1.0};
    Real ridgeOffsetUm{0.0};
    Real thicknessUm{0.3};
    Real fillStepTolerance{0.02};
    int minSlices{1};
    int maxSlices{256};
};

[[nodiscard]] int adaptiveTaperedGratingSliceCount(Real topFillFactor,
                                                       Real bottomFillFactor,
                                                       Real fillStepTolerance,
                                                       int minSlices,
                                                       int maxSlices);
[[nodiscard]] Real adaptiveTaperedGratingMidpointFill(Real topFillFactor,
                                                         Real bottomFillFactor,
                                                         int slice,
                                                         int slices);
[[nodiscard]] std::vector<GratingLayer> expandAdaptiveTaperedGratingLayer(
    const AdaptiveTaperedGratingLayer& layer);
[[nodiscard]] int adaptiveProfiledGratingSliceCount(
    const std::vector<FillFactorProfilePoint>& fillProfile,
    Real fillStepTolerance,
    int minSlices,
    int maxSlices);
[[nodiscard]] Real adaptiveProfiledGratingFillAt(
    const std::vector<FillFactorProfilePoint>& fillProfile,
    Real zFraction);
[[nodiscard]] std::vector<GratingLayer> expandAdaptiveProfiledGratingLayer(
    const AdaptiveProfiledGratingLayer& layer);

// Every finite z-stack entry is normalized into one of these explicit layer forms.
using LayerSpec = std::variant<
    UniformLayer,
    GratingLayer,
    PeriodicLayer2D,
    PatternedLayer2D,
    PrecomputedPeriodicLayer2DPtr>;

// Shared x/y lattice inferred from every periodic layer in one finite stack.
struct StackLattice {
    Real periodXUm{1.0};
    Real periodYUm{1.0};
    bool hasPeriodicLayer{false};
};

// Geometry and modal state prepared for one wavelength/incidence solve.
struct PreparedStack {
    DiagonalOperator Kx;
    DiagonalOperator Ky;
    std::vector<LayerModes> media;
    HarmonicBasis basis;
    int totalHarmonics{};
    int centerIdx{};
    StackLattice lattice;
    std::size_t finiteLayerModalBasisBuilds{};
    std::size_t finiteLayerModalBasisReuses{};
};

// Solved modal amplitudes for reflected/transmitted waves and finite layers.
struct StackAmplitudes {
    Matrix reflected;
    Matrix transmitted;
    std::vector<Matrix> layerDownTop;
    std::vector<Matrix> layerUpBottom;
};

struct LocatedDepth {
    int layerIndex{};
    Real zLocalUm{};
};

using PeriodicTensorCache = std::vector<std::shared_ptr<const TensorFourierMatrices>>;

// The immutable harmonic/lattice geometry shared by all wavelengths in a stack builder.
struct StackGeometry {
    HarmonicBasis basis;
    StackLattice lattice;
    int totalHarmonics{};
    int centerIdx{};
};

// Builds wavelength-specific PreparedStack objects from reusable stack geometry.
class StackBuilder {
public:
    StackBuilder(int harmonicCount,
                 int orderX,
                 int orderY,
                 bool harmonicCountMode,
                 LatticeTruncation truncation,
                 const std::vector<LayerSpec>& layers);

    [[nodiscard]] const StackGeometry& geometry() const noexcept { return mGeometry; }
    [[nodiscard]] bool hasPeriodicLayers() const noexcept;
    [[nodiscard]] bool hasDynamicPeriodicLayers() const noexcept;
    [[nodiscard]] bool hasDispersiveMaterials() const noexcept;
    void validateThermalPassivity(Real wavelengthUm,
                                   const Material& superstrate,
                                   const Material& substrate) const;
    [[nodiscard]] PeriodicTensorCache makePeriodicTensorCache(Real wavelengthUm) const;
    [[nodiscard]] PeriodicTensorCache makePeriodicTensorCache(
        Real wavelengthUm,
        const PeriodicTensorCache* staticCache) const;
    [[nodiscard]] PeriodicTensorCache makeStaticPeriodicCache(
        Real referenceWavelengthUm) const;
    [[nodiscard]] PreparedStack prepare(Real wavelengthUm,
                                        Real thetaDeg,
                                        Real phiDeg,
                                        const Material& superstrate,
                                        const Material& substrate,
                                        const PeriodicTensorCache* periodicCache = nullptr,
                                        bool analyticContinuation = false,
                                        const PreparedStack* xMirrorSource = nullptr,
                                        Real blochKxReduced = 0.0,
                                        Real blochKyReduced = 0.0) const;

private:
    enum class PeriodicTensorPolicy : unsigned char {
        None,
        Static,
        Dynamic,
    };

    const std::vector<LayerSpec>& mLayers;
    StackGeometry mGeometry;
    std::vector<PeriodicTensorPolicy> mPeriodicTensorPolicies;
    bool mHasPeriodicLayers{};
    bool mHasDynamicPeriodicLayers{};
    bool mHasDispersiveMaterials{};
};

[[nodiscard]] PreparedStack prepareStack(Real wavelengthUm,
                                          Real thetaDeg,
                                          Real phiDeg,
                                          int harmonicCount,
                                          int orderX,
                                          int orderY,
                                          bool harmonicCountMode,
                                          LatticeTruncation truncation,
                                          const Material& superstrate,
                                          const Material& substrate,
                                          const std::vector<LayerSpec>& layers);
[[nodiscard]] StackBuilder makeStackBuilder(const SolverState& s);
[[nodiscard]] PreparedStack prepareStackFromState(
    const SolverState& s,
    const StackBuilder& builder,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const PeriodicTensorCache* periodicCache = nullptr,
    bool analyticContinuation = false,
    Real blochKxReduced = 0.0,
    Real blochKyReduced = 0.0);
[[nodiscard]] PreparedStack prepareStackFromState(const SolverState& s);

[[nodiscard]] TensorFourierMatrices tensorMatricesForLayer(const LayerSpec& spec,
                                                              const HarmonicBasis& basis,
                                                              Real wavelengthUm);
[[nodiscard]] LayerModes computeLayerModes(
    const LayerSpec& spec,
    const DiagonalOperator& Kx,
    const DiagonalOperator& Ky,
    const HarmonicBasis& basis,
    Real wavelengthUm,
    const TensorFourierMatrices* cachedTensors = nullptr);
[[nodiscard]] std::vector<Real> finiteLayerThicknesses(const std::vector<LayerSpec>& layers);
[[nodiscard]] LocatedDepth locateDepthInFiniteStack(const std::vector<Real>& thicknesses,
                                                        Real zUm);
[[nodiscard]] StackAmplitudes solveStackAmplitudes(const PreparedStack& stack,
                                                     Polarization pol);
[[nodiscard]] StackAmplitudes solveStackAmplitudes(const PreparedStack& stack,
                                                     const Matrix& incidentAmplitudes);
[[nodiscard]] Matrix makeIncidentAmplitudes(std::size_t totalHarmonics,
                                              int centerIdx,
                                              Polarization pol);

} // namespace rcwa
