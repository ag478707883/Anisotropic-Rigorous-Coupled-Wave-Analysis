#include "rcwa/engine.hpp"

#include "rcwa/solver.hpp"

namespace rcwa {

StackEngine::StackEngine(const SolverState& state)
    : mState(state), mBuilder(makeStackBuilder(state)) {}

PreparedStack StackEngine::prepare() const {
    return prepareStackFromState(
        mState,
        mBuilder,
        mState.wavelengthUm,
        mState.thetaDeg,
        mState.phiDeg);
}

PreparedStack StackEngine::prepare(Real wavelengthUm,
                                   Real thetaDeg,
                                   Real phiDeg,
                                   const PeriodicTensorCache* cache,
                                   bool analyticContinuation,
                                   Real blochKxReduced,
                                   Real blochKyReduced) const {
    return prepareStackFromState(
        mState,
        mBuilder,
        wavelengthUm,
        thetaDeg,
        phiDeg,
        cache,
        analyticContinuation,
        blochKxReduced,
        blochKyReduced);
}

std::vector<DiffractionResult> StackEngine::efficiencies(
    const PreparedStack& stack,
    const std::vector<Polarization>& polarizations) const {
    return computeStackEfficienciesForPolarizations(
        stack.media,
        stack.totalHarmonics,
        stack.centerIdx,
        polarizations,
        mState.stackingAlgorithm);
}

std::optional<std::vector<DiffractionResult>> StackEngine::uniformScalarSpectra(
    const std::vector<Polarization>& polarizations) const {
    return solveUniformScalarSpectra(
        mState,
        mBuilder.geometry(),
        mState.wavelengthUm,
        mState.thetaDeg,
        mState.phiDeg,
        polarizations);
}

SMatrix StackEngine::scattering(const PreparedStack& stack) const {
    return computeStackSmatrix(stack.media, mState.stackingAlgorithm);
}

} // namespace rcwa
