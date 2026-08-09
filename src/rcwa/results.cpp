#include "rcwa/solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rcwa {

namespace {

Real clampThermalDeficit(Real value) {
    if (value < Real{0.0} && value > Real{-1e-10}) {
        return Real{0.0};
    }
    if (value > Real{1.0} && value < Real{1.0 + 1e-10}) {
        return Real{1.0};
    }
    return value;
}

} // namespace

SParameterResult makeSParameterResult(const SolverState& state,
                                      PreparedStack stack,
                                      Real thetaDeg,
                                      Real phiDeg) {
    SMatrix smatrix = computeStackSmatrix(stack.media, state.stackingAlgorithm);
    SParameterResult result;
    result.S11 = std::move(smatrix.S11);
    result.S12 = std::move(smatrix.S12);
    result.S21 = std::move(smatrix.S21);
    result.S22 = std::move(smatrix.S22);
    result.orders = stack.basis.orders;
    result.centerIdx = stack.centerIdx;
    result.wavelengthUm = state.wavelengthUm;
    result.thetaDeg = thetaDeg;
    result.phiDeg = phiDeg;
    result.N = stack.totalHarmonics;
    return result;
}

namespace {

SpectrumResult makeSpectrumResultImpl(int centerIdx,
                                      int totalHarmonics,
                                      DiffractionResult d,
                                      Real wavelengthUm,
                                      Real thetaDeg,
                                      Real phiDeg) {
    SpectrumResult result;
    result.rOrders = std::move(d.R);
    result.tOrders = std::move(d.T);
    result.centerIdx = centerIdx;
    result.R0 = d.R0;
    result.T0 = d.T0;
    result.rTotal = d.rTotal;
    result.tTotal = d.tTotal;
    result.conservation = d.conservation;
    result.wavelengthUm = wavelengthUm;
    result.thetaDeg = thetaDeg;
    result.phiDeg = phiDeg;
    result.N = totalHarmonics;
    return result;
}

} // namespace

SpectrumResult makeSpectrumResult(const PreparedStack& stack,
                                  DiffractionResult d,
                                  Real wavelengthUm,
                                  Real thetaDeg,
                                  Real phiDeg) {
    return makeSpectrumResultImpl(
        stack.centerIdx,
        stack.totalHarmonics,
        std::move(d),
        wavelengthUm,
        thetaDeg,
        phiDeg);
}

SpectrumResult makeSpectrumResult(const StackGeometry& geometry,
                                  DiffractionResult d,
                                  Real wavelengthUm,
                                  Real thetaDeg,
                                  Real phiDeg) {
    return makeSpectrumResultImpl(
        geometry.centerIdx,
        geometry.totalHarmonics,
        std::move(d),
        wavelengthUm,
        thetaDeg,
        phiDeg);
}

namespace {

SpectrumTotalsResult makeSpectrumTotalsResultImpl(int centerIdx,
                                                  int totalHarmonics,
                                                  DiffractionTotals d,
                                                  Real wavelengthUm,
                                                  Real thetaDeg,
                                                  Real phiDeg) {
    SpectrumTotalsResult result;
    result.centerIdx = centerIdx;
    result.R0 = d.R0;
    result.T0 = d.T0;
    result.rTotal = d.rTotal;
    result.tTotal = d.tTotal;
    result.conservation = d.conservation;
    result.wavelengthUm = wavelengthUm;
    result.thetaDeg = thetaDeg;
    result.phiDeg = phiDeg;
    result.N = totalHarmonics;
    return result;
}

} // namespace

SpectrumTotalsResult makeSpectrumTotalsResult(const PreparedStack& stack,
                                              DiffractionTotals d,
                                              Real wavelengthUm,
                                              Real thetaDeg,
                                              Real phiDeg) {
    return makeSpectrumTotalsResultImpl(
        stack.centerIdx,
        stack.totalHarmonics,
        std::move(d),
        wavelengthUm,
        thetaDeg,
        phiDeg);
}

SpectrumTotalsResult makeSpectrumTotalsResult(
    const StackGeometry& geometry,
    DiffractionTotals d,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg) {
    return makeSpectrumTotalsResultImpl(
        geometry.centerIdx,
        geometry.totalHarmonics,
        std::move(d),
        wavelengthUm,
        thetaDeg,
        phiDeg);
}

ZeroOrderAmplitudeResult makeZeroOrderAmplitudeResult(
    const PreparedStack& stack,
    ZeroOrderAmplitudes amplitudes,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg) {
    ZeroOrderAmplitudeResult result;
    result.R = std::move(amplitudes.R);
    result.T = std::move(amplitudes.T);
    result.centerIdx = stack.centerIdx;
    result.wavelengthUm = wavelengthUm;
    result.thetaDeg = thetaDeg;
    result.phiDeg = phiDeg;
    result.N = stack.totalHarmonics;
    return result;
}

DirectionalThermalChannelResult makeDirectionalThermalChannelResult(
    const TopDirectionalThermalChannels& channel,
    const PreparedStack& stack,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg) {
    const std::size_t N = static_cast<std::size_t>(stack.totalHarmonics);
    const std::size_t te0 = static_cast<std::size_t>(stack.centerIdx);
    const std::size_t tm0 = N + te0;
    if (tm0 >= channel.absorptivity.size() ||
        channel.absorptivity.size() != channel.emissivity.size() ||
        channel.emissivity.size() != channel.incidentScattering.size() ||
        channel.emissivity.size() != channel.outgoingScattering.size() ||
        channel.teTmAbsorptivityCoherence.size() != N ||
        channel.teTmEmissivityCoherence.size() != N) {
        throw std::logic_error(
            "directional thermal-channel result does not match the retained harmonic basis");
    }

    // For a Jones vector (E_TE + s*i E_TM)/sqrt(2), the absorption power is
    // 1/2*(A_TE+A_TM) + s*Im(C_TE,TM).  LCP uses s=-1 for the exp(-iwt)
    // convention; RCP uses s=+1.  The same quadratic form applies to the
    // emissivity row deficiency I-S*S^H.
    const auto circularProjection = [](Real te,
                                       Real tm,
                                       Complex teTm,
                                       bool lcp) {
        const Real helicityTerm = std::imag(teTm);
        return clampThermalDeficit(
            Real{0.5} * (te + tm) + (lcp ? -helicityTerm : helicityTerm));
    };
    const Real lcpAbsorptivity = circularProjection(
        channel.absorptivity[te0],
        channel.absorptivity[tm0],
        channel.teTmAbsorptivityCoherence[te0],
        true);
    const Real rcpAbsorptivity = circularProjection(
        channel.absorptivity[te0],
        channel.absorptivity[tm0],
        channel.teTmAbsorptivityCoherence[te0],
        false);
    const Real lcpEmissivity = circularProjection(
        channel.emissivity[te0],
        channel.emissivity[tm0],
        channel.teTmEmissivityCoherence[te0],
        true);
    const Real rcpEmissivity = circularProjection(
        channel.emissivity[te0],
        channel.emissivity[tm0],
        channel.teTmEmissivityCoherence[te0],
        false);

    return {
        channel.absorptivity[te0],
        channel.absorptivity[tm0],
        channel.emissivity[te0],
        channel.emissivity[tm0],
        channel.incidentScattering[te0],
        channel.incidentScattering[tm0],
        channel.outgoingScattering[te0],
        channel.outgoingScattering[tm0],
        lcpAbsorptivity,
        rcpAbsorptivity,
        lcpEmissivity,
        rcpEmissivity,
        Real{1.0} - lcpAbsorptivity,
        Real{1.0} - rcpAbsorptivity,
        Real{1.0} - lcpEmissivity,
        Real{1.0} - rcpEmissivity,
        stack.centerIdx,
        wavelengthUm,
        thetaDeg,
        phiDeg,
        stack.totalHarmonics,
    };
}

} // namespace rcwa
