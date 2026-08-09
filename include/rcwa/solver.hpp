#pragma once

#include <vector>

#include "rcwa/field.hpp"
#include "rcwa/geometry.hpp"
#include "rcwa/matrix.hpp"
#include "rcwa/smatrix.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

// Mutable configuration behind RcwaSolver. It is declared here because solver,
// spectrum, and field implementations are intentionally split across files.
struct SolverState {
    Real wavelengthUm{0.633};
    Real thetaDeg{0.0};
    Real phiDeg{0.0};
    int harmonicCount{11};
    int orderX{0};
    int orderY{0};
    bool harmonicCountMode{true};
    LatticeTruncation truncation{LatticeTruncation::Circular};
    Polarization pol{Polarization::TE};
    StackingAlgorithm stackingAlgorithm{StackingAlgorithm::ScatteringMatrix};
    FourierConvergenceOptions fourierOptions{};
    Material superstrate{Material::air()};
    Material substrate{Material::air()};
    std::vector<LayerSpec> layers;
};

// Per-order and total diffraction efficiencies for one wavelength/incidence solve.
struct SpectrumResult {
    std::vector<Real> rOrders;
    std::vector<Real> tOrders;
    int centerIdx{};

    Real R0{};
    Real T0{};
    Real rTotal{};
    Real tTotal{};
    Real conservation{};

    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
};

// Lightweight spectrum result for scans that only need zero-order and totals.
struct SpectrumTotalsResult {
    int centerIdx{};

    Real R0{};
    Real T0{};
    Real rTotal{};
    Real tTotal{};
    Real conservation{};

    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
};

// Batch request item for angle scans where each angle can use a different polarization.
struct SpectrumAnglePolarization {
    Real thetaDeg{};
    Polarization polarization{Polarization::TE};
};

// Dense scattering matrix for the retained TE/TM diffraction channels.
struct SParameterResult {
    Matrix S11;
    Matrix S12;
    Matrix S21;
    Matrix S22;
    std::vector<HarmonicIndex> orders;
    int centerIdx{};

    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
};

// Coherent zero-order reflection/transmission amplitude matrices.
struct ZeroOrderAmplitudeResult {
    Matrix R;
    Matrix T;
    int centerIdx{};

    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
};

// Directional absorptivity/emissivity for the zero-order TE/TM and coherent
// LCP/RCP channels above the stack. Emissivity is a row deficiency, while
// absorptivity is a column deficiency, of the power-normalized complete
// scattering matrix.
struct DirectionalThermalChannelResult {
    Real teAbsorptivity{};
    Real tmAbsorptivity{};
    Real teEmissivity{};
    Real tmEmissivity{};
    Real teIncidentScattering{};
    Real tmIncidentScattering{};
    Real teOutgoingScattering{};
    Real tmOutgoingScattering{};
    Real lcpAbsorptivity{};
    Real rcpAbsorptivity{};
    Real lcpEmissivity{};
    Real rcpEmissivity{};
    Real lcpIncidentScattering{};
    Real rcpIncidentScattering{};
    Real lcpOutgoingScattering{};
    Real rcpOutgoingScattering{};
    int centerIdx{};

    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
};

struct SpectrumThermalBatchResult {
    std::vector<SpectrumTotalsResult> spectra;
    std::vector<DirectionalThermalChannelResult> thermalChannels;
};

// Result shapers and spectrum runners shared by split implementation files.
[[nodiscard]] SParameterResult makeSParameterResult(
    const SolverState& state,
    PreparedStack stack,
    Real thetaDeg,
    Real phiDeg);
[[nodiscard]] SpectrumResult makeSpectrumResult(const PreparedStack& stack,
                                                  DiffractionResult d,
                                                  Real wavelengthUm,
                                                  Real thetaDeg,
                                                  Real phiDeg);
[[nodiscard]] SpectrumResult makeSpectrumResult(const StackGeometry& geometry,
                                                  DiffractionResult d,
                                                  Real wavelengthUm,
                                                  Real thetaDeg,
                                                  Real phiDeg);
[[nodiscard]] SpectrumTotalsResult makeSpectrumTotalsResult(const PreparedStack& stack,
                                                               DiffractionTotals d,
                                                               Real wavelengthUm,
                                                               Real thetaDeg,
                                                               Real phiDeg);
[[nodiscard]] SpectrumTotalsResult makeSpectrumTotalsResult(
    const StackGeometry& geometry,
    DiffractionTotals d,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg);
[[nodiscard]] ZeroOrderAmplitudeResult makeZeroOrderAmplitudeResult(
    const PreparedStack& stack,
    ZeroOrderAmplitudes amplitudes,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg);
[[nodiscard]] DirectionalThermalChannelResult makeDirectionalThermalChannelResult(
    const TopDirectionalThermalChannels& channel,
    const PreparedStack& stack,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg);

// Public facade for configuring an RCWA stack and evaluating spectra, amplitudes,
// S-parameters, and field planes. Geometry allocation and modal stack preparation
// live behind the facade in geometry.cpp so new layer types can extend one route.
class RcwaSolver {
public:
    void setWavelength(Real lambdaUm);
    void setIncidence(Real thetaDeg, Real phiDeg = 0.0);
    void setHarmonicCount(
        int harmonicCount,
        LatticeTruncation truncation = LatticeTruncation::Circular);
    void setHarmonicOrders(
        int orderX,
        int orderY,
        LatticeTruncation truncation = LatticeTruncation::Circular);
    void setPolarization(Polarization pol);
    void setStackingAlgorithm(StackingAlgorithm algorithm);
    [[nodiscard]] StackingAlgorithm stackingAlgorithm() const noexcept;
    void setFourierConvergenceOptions(FourierConvergenceOptions options);
    [[nodiscard]] const FourierConvergenceOptions& fourierConvergenceOptions() const;

    void setSuperstrate(Material m);
    void setSubstrate(Material m);

    void addLayer(LayerSpec layer);
    void addUniformLayer(Material mat, Real thicknessUm);
    void addGratingLayer(GratingLayer g);
    void addAdaptiveTaperedGratingLayer(AdaptiveTaperedGratingLayer layer);
    void addAdaptiveProfiledGratingLayer(AdaptiveProfiledGratingLayer layer);
    void addPeriodicLayer2d(PeriodicLayer2D layer);
    void addPatternedLayer2d(PatternedLayer2D layer);
    void addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2D layer);
    void addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2DPtr layer);
    void clearLayers();

    [[nodiscard]] SpectrumResult solve() const;
    [[nodiscard]] SpectrumResult solveSpectrumOnly() const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumOnlyForPolarizations(
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchOnly(
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchOnly(
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations,
        int workers) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchForAngles(
        const std::vector<Real>& thetaDegs,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchForAngles(
        const std::vector<Real>& thetaDegs,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations,
        int workers) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchForAnglePolarizations(
        const std::vector<SpectrumAnglePolarization>& requests,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm) const;
    [[nodiscard]] std::vector<SpectrumResult> solveSpectrumBatchForAnglePolarizations(
        const std::vector<SpectrumAnglePolarization>& requests,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        int workers) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult> solveSpectrumBatchTotalsOnly(
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult> solveSpectrumBatchTotalsOnly(
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations,
        int workers) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult> solveSpectrumBatchTotalsForAngles(
        const std::vector<Real>& thetaDegs,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult> solveSpectrumBatchTotalsForAngles(
        const std::vector<Real>& thetaDegs,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations,
        int workers) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult>
    solveSpectrumBatchTotalsForAnglePolarizations(
        const std::vector<SpectrumAnglePolarization>& requests,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm) const;
    [[nodiscard]] std::vector<SpectrumTotalsResult>
    solveSpectrumBatchTotalsForAnglePolarizations(
        const std::vector<SpectrumAnglePolarization>& requests,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        int workers) const;
    [[nodiscard]] SParameterResult solveSParameters() const;
    [[nodiscard]] SParameterResult solveSParametersAnalyticContinuation(
        Real blochKxReduced,
        Real blochKyReduced) const;
    [[nodiscard]] SParameterResult solveSParametersAnalyticContinuationAtGamma() const;
    [[nodiscard]] SpectrumResult findAbsorptivityPeak(
        Real wavelengthMinUm,
        Real wavelengthMaxUm,
        Polarization polarization,
        int gridPoints = 17,
        int refinements = 3,
        int workers = 1) const;
    [[nodiscard]] ZeroOrderAmplitudeResult solveZeroOrderAmplitudes() const;
    [[nodiscard]] std::vector<ZeroOrderAmplitudeResult> solveZeroOrderAmplitudesBatchOnly(
        const std::vector<Real>& wavelengthsUm) const;
    [[nodiscard]] std::vector<ZeroOrderAmplitudeResult> solveZeroOrderAmplitudesBatchOnly(
        const std::vector<Real>& wavelengthsUm,
        int workers) const;
    [[nodiscard]] DirectionalThermalChannelResult solveDirectionalThermalChannels() const;
    [[nodiscard]] std::vector<DirectionalThermalChannelResult>
    solveDirectionalThermalChannelsBatchOnly(const std::vector<Real>& wavelengthsUm) const;
    [[nodiscard]] std::vector<DirectionalThermalChannelResult>
    solveDirectionalThermalChannelsBatchOnly(const std::vector<Real>& wavelengthsUm,
                                               int workers) const;
    [[nodiscard]] SpectrumThermalBatchResult solveSpectrumThermalBatchForAngles(
        const std::vector<Real>& thetaDegs,
        Real phiDeg,
        const std::vector<Real>& wavelengthsUm,
        const std::vector<Polarization>& polarizations,
        int workers) const;
    [[nodiscard]] FieldPlaneResult solveFieldPlane(const FieldPlaneRequest& request) const;
    [[nodiscard]] FieldGridResult solveFieldGrid(const FieldPlaneRequest& request) const;

private:
    SolverState mState;
};

} // namespace rcwa
