#include "rcwa/solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rcwa/fourier.hpp"
#include "rcwa/engine.hpp"
#include "rcwa/layer.hpp"

namespace rcwa {

namespace {

void assignFourierOptions(LayerSpec& layer,
                           const FourierConvergenceOptions& options) {
    std::visit(
        [&](auto& concrete) {
            if constexpr (requires { concrete.fourierOptions; }) {
                concrete.fourierOptions = options;
            }
        },
        layer);
}

template <typename Result, typename SolveFn>
std::vector<Result> runPreparedWavelengthBatch(
    const SolverState& state,
    const std::vector<Real>& wavelengthsUm,
    int workers,
    SolveFn&& solveFn) {
    if (wavelengthsUm.empty()) {
        return {};
    }

    const StackBuilder builder = makeStackBuilder(state);
    const PeriodicTensorCache staticCache = builder.hasPeriodicLayers()
        ? builder.makeStaticPeriodicCache(wavelengthsUm.front())
        : PeriodicTensorCache{};
    const PeriodicTensorCache* staticCachePtr = staticCache.empty()
        ? nullptr
        : &staticCache;
    const bool hasDynamicPeriodicLayers = builder.hasDynamicPeriodicLayers();

    std::vector<Result> out(wavelengthsUm.size());
    runIndexedJobs(wavelengthsUm.size(), workers, [&](std::size_t index) {
        const Real wavelengthUm = wavelengthsUm[index];
        PeriodicTensorCache dynamicCache = hasDynamicPeriodicLayers
            ? builder.makePeriodicTensorCache(wavelengthUm, staticCachePtr)
            : PeriodicTensorCache{};
        const PeriodicTensorCache* cache = hasDynamicPeriodicLayers
            ? (dynamicCache.empty() ? nullptr : &dynamicCache)
            : staticCachePtr;
        PreparedStack stack = prepareStackFromState(
            state,
            builder,
            wavelengthUm,
            state.thetaDeg,
            state.phiDeg,
            cache);
        out[index] = solveFn(stack, wavelengthUm);
    });
    return out;
}

Real absorptivity(const SpectrumResult& result) {
    return Real{1.0} - result.rTotal - result.tTotal;
}

std::vector<SpectrumResult> solveCurrentSpectra(
    const SolverState& state,
    const std::vector<Polarization>& polarizations) {
    if (polarizations.empty()) {
        return {};
    }

    const StackEngine engine(state);
    auto diffraction = engine.uniformScalarSpectra(polarizations);
    std::optional<PreparedStack> stack;
    if (!diffraction) {
        stack = engine.prepare();
        diffraction = engine.efficiencies(*stack, polarizations);
    }

    std::vector<SpectrumResult> results;
    results.reserve(diffraction->size());
    for (auto& item : *diffraction) {
        results.push_back(stack
            ? makeSpectrumResult(
                *stack,
                std::move(item),
                state.wavelengthUm,
                state.thetaDeg,
                state.phiDeg)
            : makeSpectrumResult(
                engine.builder().geometry(),
                std::move(item),
                state.wavelengthUm,
                state.thetaDeg,
                state.phiDeg));
    }
    return results;
}

} // namespace

void RcwaSolver::setWavelength(Real lambdaUm) {
    mState.wavelengthUm = lambdaUm;
}

void RcwaSolver::setIncidence(Real thetaDeg, Real phiDeg) {
    SolverState& s = mState;
    s.thetaDeg = thetaDeg;
    s.phiDeg = phiDeg;
}

void RcwaSolver::setHarmonicCount(int harmonicCount, LatticeTruncation truncation) {
    SolverState& s = mState;
    s.harmonicCount = harmonicCount;
    s.orderX = 0;
    s.orderY = 0;
    s.harmonicCountMode = true;
    s.truncation = truncation;
}

void RcwaSolver::setHarmonicOrders(int orderX,
                                     int orderY,
                                     LatticeTruncation truncation) {
    SolverState& s = mState;
    s.harmonicCount = 0;
    s.orderX = orderX;
    s.orderY = orderY;
    s.harmonicCountMode = false;
    s.truncation = truncation;
}

void RcwaSolver::setPolarization(Polarization pol) {
    mState.pol = pol;
}

void RcwaSolver::setStackingAlgorithm(StackingAlgorithm algorithm) {
    switch (algorithm) {
    case StackingAlgorithm::ScatteringMatrix:
    case StackingAlgorithm::EnhancedTransmittanceMatrix:
        mState.stackingAlgorithm = algorithm;
        return;
    }
    throw std::invalid_argument("unsupported RCWA stacking algorithm");
}

StackingAlgorithm RcwaSolver::stackingAlgorithm() const noexcept {
    return mState.stackingAlgorithm;
}

void RcwaSolver::setFourierConvergenceOptions(FourierConvergenceOptions options) {
    validateFourierConvergenceOptions(options);
    SolverState& s = mState;
    s.fourierOptions = options;
    for (LayerSpec& layer : s.layers) {
        assignFourierOptions(layer, options);
    }
}

const FourierConvergenceOptions& RcwaSolver::fourierConvergenceOptions() const {
    return mState.fourierOptions;
}

void RcwaSolver::setSuperstrate(Material m) {
    mState.superstrate = std::move(m);
}

void RcwaSolver::setSubstrate(Material m) {
    mState.substrate = std::move(m);
}

void RcwaSolver::addLayer(LayerSpec layer) {
    assignFourierOptions(layer, mState.fourierOptions);
    mState.layers.push_back(std::move(layer));
}

void RcwaSolver::addUniformLayer(Material mat, Real thicknessUm) {
    mState.layers.push_back(UniformLayer{std::move(mat), thicknessUm});
}

void RcwaSolver::addGratingLayer(GratingLayer g) {
    g.fourierOptions = mState.fourierOptions;
    mState.layers.push_back(std::move(g));
}

void RcwaSolver::addAdaptiveTaperedGratingLayer(AdaptiveTaperedGratingLayer layer) {
    for (auto& slice : expandAdaptiveTaperedGratingLayer(layer)) {
        slice.fourierOptions = mState.fourierOptions;
        mState.layers.push_back(std::move(slice));
    }
}

void RcwaSolver::addAdaptiveProfiledGratingLayer(AdaptiveProfiledGratingLayer layer) {
    for (auto& slice : expandAdaptiveProfiledGratingLayer(layer)) {
        slice.fourierOptions = mState.fourierOptions;
        mState.layers.push_back(std::move(slice));
    }
}

void RcwaSolver::addPeriodicLayer2d(PeriodicLayer2D layer) {
    layer.fourierOptions = mState.fourierOptions;
    mState.layers.push_back(std::move(layer));
}

void RcwaSolver::addPatternedLayer2d(PatternedLayer2D layer) {
    layer.fourierOptions = mState.fourierOptions;
    mState.layers.push_back(std::move(layer));
}

void RcwaSolver::addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2D layer) {
    PrecomputedPeriodicLayer2DPtr immutableLayer =
        std::make_shared<const PrecomputedPeriodicLayer2D>(std::move(layer));
    mState.layers.emplace_back(std::move(immutableLayer));
}

void RcwaSolver::addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2DPtr layer) {
    mState.layers.emplace_back(std::move(layer));
}

void RcwaSolver::clearLayers() {
    mState.layers.clear();
}

SpectrumResult RcwaSolver::solve() const {
    return solveSpectrumOnly();
}

SpectrumResult RcwaSolver::solveSpectrumOnly() const {
    auto results = solveCurrentSpectra(mState, {mState.pol});
    return std::move(results.front());
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumOnlyForPolarizations(
    const std::vector<Polarization>& polarizations) const {
    return solveCurrentSpectra(mState, polarizations);
}

SParameterResult RcwaSolver::solveSParameters() const {
    const SolverState& s = mState;
    const StackEngine engine(s);
    return makeSParameterResult(
        s,
        engine.prepare(),
        s.thetaDeg,
        s.phiDeg);
}

SParameterResult RcwaSolver::solveSParametersAnalyticContinuationAtGamma() const {
    return solveSParametersAnalyticContinuation(Real{0}, Real{0});
}

SParameterResult RcwaSolver::solveSParametersAnalyticContinuation(
    Real blochKxReduced,
    Real blochKyReduced) const {
    const SolverState& s = mState;
    const StackBuilder builder = makeStackBuilder(s);
    return makeSParameterResult(
        s,
        prepareStackFromState(
            s,
            builder,
            s.wavelengthUm,
            Real{0},
            Real{0},
            nullptr,
            true,
            blochKxReduced,
            blochKyReduced),
        Real{0},
        Real{0});
}

SpectrumResult RcwaSolver::findAbsorptivityPeak(
    Real wavelengthMinUm,
    Real wavelengthMaxUm,
    Polarization polarization,
    int gridPoints,
    int refinements,
    int workers) const {
    if (!std::isfinite(wavelengthMinUm) || !std::isfinite(wavelengthMaxUm) ||
        wavelengthMinUm <= Real{0.0} || wavelengthMaxUm <= wavelengthMinUm) {
        throw std::invalid_argument(
            "absorptivity peak wavelength bracket must be finite, positive, and increasing");
    }
    if (polarization != Polarization::TE && polarization != Polarization::TM) {
        throw std::invalid_argument(
            "absorptivity peak search requires TE or TM polarization");
    }
    if (gridPoints < 5 || gridPoints > 1001) {
        throw std::invalid_argument(
            "absorptivity peak grid points must lie in [5, 1001]");
    }
    if (refinements < 1 || refinements > 10) {
        throw std::invalid_argument(
            "absorptivity peak refinements must lie in [1, 10]");
    }

    Real lower = wavelengthMinUm;
    Real upper = wavelengthMaxUm;
    std::vector<Real> wavelengths;
    std::vector<const SpectrumResult*> spectra;
    std::map<Real, SpectrumResult> evaluated;
    Real previousPeakWavelength{};
    std::size_t peakIndex{};
    for (int refinement = 0; refinement < refinements; ++refinement) {
        wavelengths.resize(static_cast<std::size_t>(gridPoints));
        const Real step = (upper - lower) / static_cast<Real>(gridPoints - 1);
        for (int point = 0; point < gridPoints; ++point) {
            wavelengths[static_cast<std::size_t>(point)] =
                lower + static_cast<Real>(point) * step;
        }
        wavelengths.front() = lower;
        wavelengths.back() = upper;
        if (refinement > 0 && gridPoints % 2 == 1) {
            wavelengths[static_cast<std::size_t>(gridPoints / 2)] =
                previousPeakWavelength;
        }

        std::vector<Real> missingWavelengths;
        missingWavelengths.reserve(wavelengths.size());
        for (const Real wavelengthUm : wavelengths) {
            if (evaluated.find(wavelengthUm) == evaluated.end()) {
                missingWavelengths.push_back(wavelengthUm);
            }
        }
        if (!missingWavelengths.empty()) {
            auto missingSpectra = solveSpectrumBatchOnly(
                missingWavelengths,
                std::vector<Polarization>{polarization},
                workers);
            for (std::size_t index = 0; index < missingWavelengths.size(); ++index) {
                evaluated.emplace(
                    missingWavelengths[index],
                    std::move(missingSpectra[index]));
            }
        }

        spectra.clear();
        spectra.reserve(wavelengths.size());
        for (const Real wavelengthUm : wavelengths) {
            spectra.push_back(&evaluated.at(wavelengthUm));
        }
        peakIndex = 0;
        Real peakValue = -std::numeric_limits<Real>::infinity();
        for (std::size_t index = 0; index < spectra.size(); ++index) {
            const Real value = absorptivity(*spectra[index]);
            if (value > peakValue) {
                peakValue = value;
                peakIndex = index;
            }
        }
        if (peakIndex == 0 || peakIndex + 1 == spectra.size()) {
            throw std::runtime_error(
                "absorptivity maximum lies on the wavelength bracket boundary");
        }
        lower = wavelengths[peakIndex - 1];
        upper = wavelengths[peakIndex + 1];
        previousPeakWavelength = wavelengths[peakIndex];
    }

    const Real yMinus = absorptivity(*spectra[peakIndex - 1]);
    const Real yZero = absorptivity(*spectra[peakIndex]);
    const Real yPlus = absorptivity(*spectra[peakIndex + 1]);
    const Real denominator = yMinus - Real{2.0} * yZero + yPlus;
    Real offset{};
    if (denominator != Real{0.0}) {
        offset = Real{0.5} * (yMinus - yPlus) / denominator;
        offset = std::clamp(offset, Real{-1.0}, Real{1.0});
    }
    const Real step = wavelengths[1] - wavelengths[0];
    const Real peakWavelength = wavelengths[peakIndex] + offset * step;
    const auto cachedPeak = evaluated.find(peakWavelength);
    if (cachedPeak != evaluated.end()) {
        return cachedPeak->second;
    }
    RcwaSolver peakSolver = *this;
    peakSolver.setWavelength(peakWavelength);
    peakSolver.setPolarization(polarization);
    return peakSolver.solveSpectrumOnly();
}

ZeroOrderAmplitudeResult RcwaSolver::solveZeroOrderAmplitudes() const {
    const SolverState& s = mState;
    PreparedStack stack = prepareStackFromState(s);
    return makeZeroOrderAmplitudeResult(
        stack,
        computeZeroOrderAmplitudes(
            stack.media,
            stack.totalHarmonics,
            stack.centerIdx,
            s.stackingAlgorithm),
        s.wavelengthUm,
        s.thetaDeg,
        s.phiDeg);
}

std::vector<ZeroOrderAmplitudeResult> RcwaSolver::solveZeroOrderAmplitudesBatchOnly(
    const std::vector<Real>& wavelengthsUm) const {
    return solveZeroOrderAmplitudesBatchOnly(wavelengthsUm, 1);
}

std::vector<ZeroOrderAmplitudeResult> RcwaSolver::solveZeroOrderAmplitudesBatchOnly(
    const std::vector<Real>& wavelengthsUm,
    int workers) const {
    const SolverState& s = mState;
    return runPreparedWavelengthBatch<ZeroOrderAmplitudeResult>(
        s,
        wavelengthsUm,
        workers,
        [&](const PreparedStack& stack, Real wavelengthUm) {
            return makeZeroOrderAmplitudeResult(
                stack,
                computeZeroOrderAmplitudes(
                    stack.media,
                    stack.totalHarmonics,
                    stack.centerIdx,
                    s.stackingAlgorithm),
                wavelengthUm,
                s.thetaDeg,
                s.phiDeg);
        });
}

DirectionalThermalChannelResult RcwaSolver::solveDirectionalThermalChannels() const {
    const SolverState& s = mState;
    const StackBuilder builder = makeStackBuilder(s);
    builder.validateThermalPassivity(
        s.wavelengthUm, s.superstrate, s.substrate);
    PreparedStack stack = prepareStackFromState(
        s,
        builder,
        s.wavelengthUm,
        s.thetaDeg,
        s.phiDeg);
    TopDirectionalThermalChannels channel =
        computeStackTotalsAndTopDirectionalThermalChannels(
            stack.media,
            stack.totalHarmonics,
            stack.centerIdx,
            {},
            s.stackingAlgorithm).thermal;
    return makeDirectionalThermalChannelResult(
        channel,
        stack,
        s.wavelengthUm,
        s.thetaDeg,
        s.phiDeg);
}

std::vector<DirectionalThermalChannelResult>
RcwaSolver::solveDirectionalThermalChannelsBatchOnly(
    const std::vector<Real>& wavelengthsUm) const {
    return solveDirectionalThermalChannelsBatchOnly(wavelengthsUm, 1);
}

std::vector<DirectionalThermalChannelResult>
RcwaSolver::solveDirectionalThermalChannelsBatchOnly(
    const std::vector<Real>& wavelengthsUm,
    int workers) const {
    const SolverState& s = mState;
    const StackBuilder passivityBuilder = makeStackBuilder(s);
    for (const Real wavelengthUm : wavelengthsUm) {
        passivityBuilder.validateThermalPassivity(
            wavelengthUm, s.superstrate, s.substrate);
    }
    return runPreparedWavelengthBatch<DirectionalThermalChannelResult>(
        s,
        wavelengthsUm,
        workers,
        [&](const PreparedStack& stack, Real wavelengthUm) {
            TopDirectionalThermalChannels channel =
                computeStackTotalsAndTopDirectionalThermalChannels(
                    stack.media,
                    stack.totalHarmonics,
                    stack.centerIdx,
                    {},
                    s.stackingAlgorithm).thermal;
            return makeDirectionalThermalChannelResult(
                channel,
                stack,
                wavelengthUm,
                s.thetaDeg,
                s.phiDeg);
        });
}

} // namespace rcwa
