#include "rcwa/solver.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "rcwa/scalar_stack.hpp"

namespace rcwa {
namespace {

template <bool TotalsOnly>
struct SpectrumBatchTraits;

template <>
struct SpectrumBatchTraits<false> {
    using Result = SpectrumResult;
    using Diffraction = DiffractionResult;

    [[nodiscard]] static std::vector<Diffraction> compute(
        const PreparedStack& stack,
        const std::vector<Polarization>& pols,
        StackingAlgorithm algorithm) {
        return computeStackEfficienciesForPolarizations(
            stack.media,
            stack.totalHarmonics,
            stack.centerIdx,
            pols,
            algorithm);
    }

    [[nodiscard]] static Result make(const PreparedStack& stack,
                                     Diffraction d,
                                     Real wavelengthUm,
                                     Real thetaDeg,
                                     Real phiDeg) {
        return makeSpectrumResult(
            stack,
            std::move(d),
            wavelengthUm,
            thetaDeg,
            phiDeg);
    }

    [[nodiscard]] static std::optional<std::vector<Diffraction>> computeScalar(
        const SolverState& state,
        const StackGeometry& geometry,
        Real wavelengthUm,
        Real thetaDeg,
        Real phiDeg,
        const std::vector<Polarization>& pols) {
        return solveUniformScalarSpectra(
            state, geometry, wavelengthUm, thetaDeg, phiDeg, pols);
    }

    [[nodiscard]] static Result make(const StackGeometry& geometry,
                                     Diffraction d,
                                     Real wavelengthUm,
                                     Real thetaDeg,
                                     Real phiDeg) {
        return makeSpectrumResult(
            geometry,
            std::move(d),
            wavelengthUm,
            thetaDeg,
            phiDeg);
    }
};

template <>
struct SpectrumBatchTraits<true> {
    using Result = SpectrumTotalsResult;
    using Diffraction = DiffractionTotals;

    [[nodiscard]] static std::vector<Diffraction> compute(
        const PreparedStack& stack,
        const std::vector<Polarization>& pols,
        StackingAlgorithm algorithm) {
        return computeStackTotalsForPolarizations(
            stack.media,
            stack.totalHarmonics,
            stack.centerIdx,
            pols,
            algorithm);
    }

    [[nodiscard]] static Result make(const PreparedStack& stack,
                                     Diffraction d,
                                     Real wavelengthUm,
                                     Real thetaDeg,
                                     Real phiDeg) {
        return makeSpectrumTotalsResult(
            stack,
            std::move(d),
            wavelengthUm,
            thetaDeg,
            phiDeg);
    }

    [[nodiscard]] static std::optional<std::vector<Diffraction>> computeScalar(
        const SolverState& state,
        const StackGeometry&,
        Real wavelengthUm,
        Real thetaDeg,
        Real phiDeg,
        const std::vector<Polarization>& pols) {
        return solveUniformScalarTotals(
            state, wavelengthUm, thetaDeg, phiDeg, pols);
    }

    [[nodiscard]] static Result make(const StackGeometry& geometry,
                                     Diffraction d,
                                     Real wavelengthUm,
                                     Real thetaDeg,
                                     Real phiDeg) {
        return makeSpectrumTotalsResult(
            geometry,
            std::move(d),
            wavelengthUm,
            thetaDeg,
            phiDeg);
    }
};

class WavelengthCacheScheduler {
public:
    WavelengthCacheScheduler(const StackBuilder& builder,
                             const std::vector<Real>& wavelengthsUm,
                             std::size_t groupedAxisSize,
                             std::size_t jobCount,
                             bool hasDispersiveMaterials,
                             bool hasAnglePairs)
        : mBuilder(builder),
          mWavelengthsUm(wavelengthsUm),
          mGroupedAxisSize(groupedAxisSize),
          mJobCount(jobCount),
          mHasPeriodicLayers(builder.hasPeriodicLayers()),
          mHasDynamicPeriodicLayers(builder.hasDynamicPeriodicLayers()),
          mHasDispersiveMaterials(hasDispersiveMaterials),
          mHasAnglePairs(hasAnglePairs) {}

    [[nodiscard]] PeriodicTensorCache makeStaticCache() const {
        if (!mHasPeriodicLayers || mWavelengthsUm.empty()) {
            return {};
        }
        return mBuilder.makeStaticPeriodicCache(mWavelengthsUm.front());
    }

    [[nodiscard]] bool shouldGroupByWavelength() const {
        return mGroupedAxisSize > 1 &&
               (mHasDynamicPeriodicLayers || mHasDispersiveMaterials ||
                (mHasPeriodicLayers && mHasAnglePairs));
    }

    [[nodiscard]] bool shouldPrefetchCaches(int workers) const {
        return mHasDynamicPeriodicLayers &&
               boundedWorkerCount(workers, mJobCount) >
               boundedWorkerCount(workers, mWavelengthsUm.size());
    }

    [[nodiscard]] bool hasDynamicPeriodicLayers() const noexcept {
        return mHasDynamicPeriodicLayers;
    }

    [[nodiscard]] PeriodicTensorCache makeCacheForWavelength(
        std::size_t wavelengthIndex,
        const PeriodicTensorCache* staticCache) const {
        if (!mHasPeriodicLayers) {
            return {};
        }
        if (!mHasDynamicPeriodicLayers && staticCache != nullptr) {
            return *staticCache;
        }
        return mBuilder.makePeriodicTensorCache(
            mWavelengthsUm[wavelengthIndex],
            staticCache);
    }

private:
    const StackBuilder& mBuilder;
    const std::vector<Real>& mWavelengthsUm;
    std::size_t mGroupedAxisSize{};
    std::size_t mJobCount{};
    bool mHasPeriodicLayers{};
    bool mHasDynamicPeriodicLayers{};
    bool mHasDispersiveMaterials{};
    bool mHasAnglePairs{};
};

[[nodiscard]] const PeriodicTensorCache* cachePtrOrNull(
    const PeriodicTensorCache& cache) noexcept {
    return cache.empty() ? nullptr : &cache;
}

template <typename Plan>
void runCachedJobs(const Plan& plan,
                     int workers,
                     const WavelengthCacheScheduler& scheduler,
                     const PeriodicTensorCache* staticCache) {
    if (scheduler.shouldGroupByWavelength()) {
        const std::size_t solveGroupCount = plan.wavelengthSolveGroupCount();
        if (scheduler.shouldPrefetchCaches(workers)) {
            const std::size_t workerCount = static_cast<std::size_t>(
                boundedWorkerCount(workers, plan.jobCount()));
            const std::size_t cacheTileSize = std::min(
                plan.wavelengthCount(),
                std::size_t{1} +
                    (workerCount - 1) / std::max<std::size_t>(solveGroupCount, 1));
            for (std::size_t tileBegin = 0;
                 tileBegin < plan.wavelengthCount();
                 tileBegin += cacheTileSize) {
                const std::size_t tileCount = std::min(
                    cacheTileSize,
                    plan.wavelengthCount() - tileBegin);
                std::vector<PeriodicTensorCache> caches(tileCount);
                runIndexedJobs(tileCount, workers, [&](std::size_t tileIndex) {
                    caches[tileIndex] = scheduler.makeCacheForWavelength(
                        tileBegin + tileIndex,
                        staticCache);
                });

                runIndexedJobs(
                    checkedProduct(
                        tileCount,
                        solveGroupCount,
                        "cached wavelength-group solve"),
                    workers,
                    [&](std::size_t taskIndex) {
                        const std::size_t tileIndex = taskIndex / solveGroupCount;
                        plan.solveWavelengthGroupItem(
                            tileBegin + tileIndex,
                            taskIndex % solveGroupCount,
                            cachePtrOrNull(caches[tileIndex]));
                    });
            }
            return;
        }

        if (!scheduler.hasDynamicPeriodicLayers()) {
            runIndexedJobs(
                checkedProduct(
                    plan.wavelengthCount(),
                    solveGroupCount,
                    "wavelength-group solve"),
                workers,
                [&](std::size_t taskIndex) {
                    plan.solveWavelengthGroupItem(
                        taskIndex / solveGroupCount,
                        taskIndex % solveGroupCount,
                        staticCache);
                });
            return;
        }

        runIndexedJobs(plan.wavelengthCount(), workers, [&](std::size_t wavelengthIndex) {
            PeriodicTensorCache cache =
                scheduler.makeCacheForWavelength(wavelengthIndex, staticCache);
            plan.solveWavelengthGroup(wavelengthIndex, cachePtrOrNull(cache));
        });
        return;
    }

    runIndexedJobs(plan.jobCount(), workers, [&](std::size_t jobIndex) {
        plan.solveFlat(jobIndex, staticCache);
    });
}

template <typename Plan>
[[nodiscard]] auto runSpectrumPlan(const Plan& plan, int workers) {
    using Result = typename Plan::Result;
    if (plan.empty()) {
        return std::vector<Result>{};
    }

    std::vector<Result> out(plan.outputCount());
    plan.bindOutput(out);

    const WavelengthCacheScheduler scheduler(
        plan.builder(),
        plan.wavelengths(),
        plan.groupedAxisSize(),
        plan.jobCount(),
        plan.hasDispersiveMaterials(),
        plan.hasAnglePairs());
    const PeriodicTensorCache staticCache = scheduler.makeStaticCache();
    runCachedJobs(plan, workers, scheduler, cachePtrOrNull(staticCache));
    return out;
}

template <typename Items, typename Predicate>
[[nodiscard]] std::size_t findIndex(const Items& items, Predicate&& predicate) {
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (predicate(items[i])) {
            return i;
        }
    }
    return items.size();
}

struct AngleSolveGroup {
    std::size_t primaryIndex{};
    std::optional<std::size_t> mirrorIndex;
};

[[nodiscard]] std::vector<AngleSolveGroup> makeAngleSolveGroups(
    const std::vector<Real>& angles) {
    std::vector<AngleSolveGroup> groups;
    groups.reserve(angles.size());
    std::unordered_map<Real, std::vector<std::size_t>> unmatchedGroups;
    unmatchedGroups.reserve(angles.size());
    for (std::size_t index = 0; index < angles.size(); ++index) {
        const Real angle = angles[index];
        if (angle != Real{0.0}) {
            auto mirrorIt = unmatchedGroups.find(-angle);
            if (mirrorIt != unmatchedGroups.end() && !mirrorIt->second.empty()) {
                groups[mirrorIt->second.back()].mirrorIndex = index;
                mirrorIt->second.pop_back();
                continue;
            }
        }
        groups.push_back({index, std::nullopt});
        if (angle != Real{0.0}) {
            unmatchedGroups[angle].push_back(groups.size() - 1);
        }
    }
    return groups;
}

[[nodiscard]] bool hasAnglePairs(
    const std::vector<AngleSolveGroup>& groups) noexcept {
    for (const AngleSolveGroup& group : groups) {
        if (group.mirrorIndex) {
            return true;
        }
    }
    return false;
}

template <typename ThetaAt, typename TryScalar, typename ConsumePrepared>
void executeAngleSolveGroup(
    const AngleSolveGroup& group,
    Real wavelengthUm,
    Real phiDeg,
    const SolverState& state,
    const StackBuilder& builder,
    const PeriodicTensorCache* periodicCache,
    ThetaAt&& thetaAt,
    TryScalar&& tryScalar,
    ConsumePrepared&& consumePrepared) {
    const auto prepare = [&](std::size_t index, const PreparedStack* mirrorSource) {
        return builder.prepare(
            wavelengthUm,
            thetaAt(index),
            phiDeg,
            state.superstrate,
            state.substrate,
            periodicCache,
            false,
            mirrorSource);
    };
    const auto solveWithoutMirror = [&](std::size_t index) {
        if (tryScalar(index)) {
            return;
        }
        PreparedStack stack = prepare(index, nullptr);
        consumePrepared(index, stack);
    };

    if (tryScalar(group.primaryIndex)) {
        if (group.mirrorIndex) {
            solveWithoutMirror(*group.mirrorIndex);
        }
        return;
    }

    PreparedStack primary = prepare(group.primaryIndex, nullptr);
    consumePrepared(group.primaryIndex, primary);
    if (group.mirrorIndex && !tryScalar(*group.mirrorIndex)) {
        PreparedStack mirrored = prepare(*group.mirrorIndex, &primary);
        consumePrepared(*group.mirrorIndex, mirrored);
    }
}

class AngleGridPlanBase {
public:
    AngleGridPlanBase(const SolverState& state,
                      const StackBuilder& builder,
                      const std::vector<Real>& thetaDegs,
                      Real phiDeg,
                      const std::vector<Real>& wavelengthsUm,
                      const char* jobLabel)
        : mState(state),
          mBuilder(builder),
          mThetaDegs(thetaDegs),
          mPhiDeg(phiDeg),
          mWavelengthsUm(wavelengthsUm),
          mAngleSolveGroups(makeAngleSolveGroups(thetaDegs)),
          mJobLabel(jobLabel) {}

    virtual ~AngleGridPlanBase() = default;

    [[nodiscard]] bool empty() const {
        return mThetaDegs.empty() || mWavelengthsUm.empty();
    }

    [[nodiscard]] const StackBuilder& builder() const { return mBuilder; }
    [[nodiscard]] const std::vector<Real>& wavelengths() const {
        return mWavelengthsUm;
    }
    [[nodiscard]] std::size_t wavelengthCount() const {
        return mWavelengthsUm.size();
    }
    [[nodiscard]] std::size_t groupedAxisSize() const {
        return mThetaDegs.size();
    }
    [[nodiscard]] std::size_t wavelengthSolveGroupCount() const {
        return mAngleSolveGroups.size();
    }
    [[nodiscard]] std::size_t jobCount() const {
        return checkedProduct(mThetaDegs.size(), mWavelengthsUm.size(), mJobLabel);
    }
    [[nodiscard]] bool hasDispersiveMaterials() const {
        return mBuilder.hasDispersiveMaterials() ||
               mState.superstrate.isDispersive() ||
               mState.substrate.isDispersive();
    }
    [[nodiscard]] bool hasAnglePairs() const noexcept {
        return rcwa::hasAnglePairs(mAngleSolveGroups);
    }

    void solveFlat(std::size_t jobIndex,
                   const PeriodicTensorCache* periodicCache) const {
        executeGroup(
            AngleSolveGroup{
                jobIndex / mWavelengthsUm.size(),
                std::nullopt},
            jobIndex % mWavelengthsUm.size(),
            periodicCache);
    }

    void solveWavelengthGroup(
        std::size_t wavelengthIndex,
        const PeriodicTensorCache* periodicCache) const {
        for (const AngleSolveGroup& solveGroup : mAngleSolveGroups) {
            executeGroup(solveGroup, wavelengthIndex, periodicCache);
        }
    }

    void solveWavelengthGroupItem(
        std::size_t wavelengthIndex,
        std::size_t solveGroupIndex,
        const PeriodicTensorCache* periodicCache) const {
        if (solveGroupIndex >= mAngleSolveGroups.size()) {
            throw std::out_of_range("wavelength solve group index is outside the angle plan");
        }
        executeGroup(
            mAngleSolveGroups[solveGroupIndex],
            wavelengthIndex,
            periodicCache);
    }

protected:
    virtual void executeGroup(const AngleSolveGroup& group,
                              std::size_t wavelengthIndex,
                              const PeriodicTensorCache* periodicCache) const = 0;

    const SolverState& mState;
    const StackBuilder& mBuilder;
    const std::vector<Real>& mThetaDegs;
    Real mPhiDeg{};
    const std::vector<Real>& mWavelengthsUm;
    std::vector<AngleSolveGroup> mAngleSolveGroups;

private:
    const char* mJobLabel{};
};

template <bool TotalsOnly>
class AngleSweepPlan final : public AngleGridPlanBase {
public:
    using Traits = SpectrumBatchTraits<TotalsOnly>;
    using Result = typename Traits::Result;

    AngleSweepPlan(const SolverState& state,
                   const StackBuilder& builder,
                   const std::vector<Real>& thetaDegs,
                   Real phiDeg,
                   const std::vector<Real>& wavelengthsUm,
                   std::vector<Polarization> polarizations)
        : AngleGridPlanBase(
              state,
              builder,
              thetaDegs,
              phiDeg,
              wavelengthsUm,
              "spectrum job"),
          mPolarizations(std::move(polarizations)) {}

    [[nodiscard]] std::size_t outputCount() const {
        return checkedProduct(jobCount(), mPolarizations.size(), "spectrum result");
    }

    void bindOutput(std::vector<Result>& out) const {
        mOut = &out;
    }

private:
    void executeGroup(const AngleSolveGroup& group,
                      std::size_t wavelengthIndex,
                      const PeriodicTensorCache* periodicCache) const override {
        const Real wavelengthUm = mWavelengthsUm[wavelengthIndex];
        const auto jobIndex = [&](std::size_t angleIndex) {
            return angleIndex * mWavelengthsUm.size() + wavelengthIndex;
        };
        executeAngleSolveGroup(
            group,
            wavelengthUm,
            mPhiDeg,
            mState,
            mBuilder,
            periodicCache,
            [&](std::size_t angleIndex) { return mThetaDegs[angleIndex]; },
            [&](std::size_t angleIndex) {
                return solveScalar(
                    wavelengthUm,
                    mThetaDegs[angleIndex],
                    jobIndex(angleIndex));
            },
            [&](std::size_t angleIndex, const PreparedStack& stack) {
                solvePrepared(
                    stack,
                    wavelengthUm,
                    mThetaDegs[angleIndex],
                    jobIndex(angleIndex));
            });
    }

    void solvePrepared(const PreparedStack& stack,
                       Real wavelengthUm,
                       Real thetaDeg,
                       std::size_t jobIndex) const {
        auto diffraction = Traits::compute(
            stack,
            mPolarizations,
            mState.stackingAlgorithm);
        storeResults(stack, std::move(diffraction), wavelengthUm, thetaDeg, jobIndex);
    }

    template <typename Geometry>
    void storeResults(const Geometry& geometry,
                      std::vector<typename Traits::Diffraction> diffraction,
                      Real wavelengthUm,
                      Real thetaDeg,
                      std::size_t jobIndex) const {
        const std::size_t offset = jobIndex * mPolarizations.size();
        for (std::size_t polIndex = 0; polIndex < diffraction.size(); ++polIndex) {
            (*mOut)[offset + polIndex] = Traits::make(
                geometry,
                std::move(diffraction[polIndex]),
                wavelengthUm,
                thetaDeg,
                mPhiDeg);
        }
    }

    [[nodiscard]] bool solveScalar(Real wavelengthUm,
                                   Real thetaDeg,
                                   std::size_t jobIndex) const {
        auto diffraction = Traits::computeScalar(
            mState,
            mBuilder.geometry(),
            wavelengthUm,
            thetaDeg,
            mPhiDeg,
            mPolarizations);
        if (!diffraction) {
            return false;
        }
        storeResults(
            mBuilder.geometry(),
            std::move(*diffraction),
            wavelengthUm,
            thetaDeg,
            jobIndex);
        return true;
    }

    std::vector<Polarization> mPolarizations;
    mutable std::vector<Result>* mOut{};
};

class AngleSweepThermalPlan final : public AngleGridPlanBase {
public:
    AngleSweepThermalPlan(const SolverState& state,
                          const StackBuilder& builder,
                          const std::vector<Real>& thetaDegs,
                          Real phiDeg,
                          const std::vector<Real>& wavelengthsUm,
                          std::vector<Polarization> polarizations)
        : AngleGridPlanBase(
              state,
              builder,
              thetaDegs,
              phiDeg,
              wavelengthsUm,
              "combined spectrum/thermal job"),
          mPolarizations(std::move(polarizations)) {}

    void bindOutput(SpectrumThermalBatchResult& out) const { mOut = &out; }

private:
    void executeGroup(const AngleSolveGroup& group,
                      std::size_t wavelengthIndex,
                      const PeriodicTensorCache* periodicCache) const override {
        const Real wavelengthUm = mWavelengthsUm[wavelengthIndex];
        const auto jobIndex = [&](std::size_t angleIndex) {
            return angleIndex * mWavelengthsUm.size() + wavelengthIndex;
        };
        executeAngleSolveGroup(
            group,
            wavelengthUm,
            mPhiDeg,
            mState,
            mBuilder,
            periodicCache,
            [&](std::size_t angleIndex) { return mThetaDegs[angleIndex]; },
            [](std::size_t) { return false; },
            [&](std::size_t angleIndex, const PreparedStack& stack) {
                solvePrepared(
                    stack,
                    wavelengthUm,
                    mThetaDegs[angleIndex],
                    jobIndex(angleIndex));
            });
    }

    void solvePrepared(const PreparedStack& stack,
                       Real wavelengthUm,
                       Real thetaDeg,
                       std::size_t jobIndex) const {
        auto combined = computeStackTotalsAndTopDirectionalThermalChannels(
            stack.media,
            stack.totalHarmonics,
            stack.centerIdx,
            mPolarizations,
            mState.stackingAlgorithm);
        const std::size_t offset = jobIndex * mPolarizations.size();
        for (std::size_t polIndex = 0; polIndex < combined.spectra.size(); ++polIndex) {
            mOut->spectra[offset + polIndex] = makeSpectrumTotalsResult(
                stack,
                std::move(combined.spectra[polIndex]),
                wavelengthUm,
                thetaDeg,
                mPhiDeg);
        }
        mOut->thermalChannels[jobIndex] = makeDirectionalThermalChannelResult(
            std::move(combined.thermal),
            stack,
            wavelengthUm,
            thetaDeg,
            mPhiDeg);
    }

    std::vector<Polarization> mPolarizations;
    mutable SpectrumThermalBatchResult* mOut{};
};

template <bool TotalsOnly>
class AnglePolarizationPlan {
public:
    using Traits = SpectrumBatchTraits<TotalsOnly>;
    using Result = typename Traits::Result;

    struct AngleGroup {
        Real thetaDeg{};
        std::vector<Polarization> polarizations;
        std::vector<std::vector<std::size_t>> requestIndicesByPol;
    };

    AnglePolarizationPlan(const SolverState& state,
                          const StackBuilder& builder,
                          const std::vector<SpectrumAnglePolarization>& requests,
                          Real phiDeg,
                          const std::vector<Real>& wavelengthsUm)
        : mState(state),
          mBuilder(builder),
          mRequestCount(requests.size()),
          mGroups(makeGroups(requests)),
          mAngleSolveGroups(makeAngleSolveGroups(groupAngles(mGroups))),
          mPhiDeg(phiDeg),
          mWavelengthsUm(wavelengthsUm) {}

    [[nodiscard]] bool empty() const {
        return mRequestCount == 0 || mWavelengthsUm.empty();
    }

    [[nodiscard]] const StackBuilder& builder() const { return mBuilder; }
    [[nodiscard]] const std::vector<Real>& wavelengths() const { return mWavelengthsUm; }
    [[nodiscard]] std::size_t wavelengthCount() const { return mWavelengthsUm.size(); }
    [[nodiscard]] std::size_t groupedAxisSize() const { return mGroups.size(); }
    [[nodiscard]] std::size_t wavelengthSolveGroupCount() const {
        return mAngleSolveGroups.size();
    }
    [[nodiscard]] std::size_t jobCount() const {
        return checkedProduct(mGroups.size(), mWavelengthsUm.size(), "spectrum pair job");
    }
    [[nodiscard]] std::size_t outputCount() const {
        return checkedProduct(
            mRequestCount,
            mWavelengthsUm.size(),
            "spectrum pair result");
    }
    [[nodiscard]] bool hasDispersiveMaterials() const {
        return mBuilder.hasDispersiveMaterials() ||
               mState.superstrate.isDispersive() ||
               mState.substrate.isDispersive();
    }

    [[nodiscard]] bool hasAnglePairs() const noexcept {
        return rcwa::hasAnglePairs(mAngleSolveGroups);
    }

    void bindOutput(std::vector<Result>& out) const {
        mOut = &out;
    }

    void solveFlat(std::size_t jobIndex, const PeriodicTensorCache* periodicCache) const {
        executeGroup(
            AngleSolveGroup{
                jobIndex / mWavelengthsUm.size(),
                std::nullopt},
            jobIndex % mWavelengthsUm.size(),
            periodicCache);
    }

    void solveWavelengthGroup(std::size_t wavelengthIndex,
                                const PeriodicTensorCache* periodicCache) const {
        for (const AngleSolveGroup& solveGroup : mAngleSolveGroups) {
            executeGroup(solveGroup, wavelengthIndex, periodicCache);
        }
    }

    void solveWavelengthGroupItem(
        std::size_t wavelengthIndex,
        std::size_t solveGroupIndex,
        const PeriodicTensorCache* periodicCache) const {
        if (solveGroupIndex >= mAngleSolveGroups.size()) {
            throw std::out_of_range("wavelength solve group index is outside the angle/polarization plan");
        }
        executeGroup(
            mAngleSolveGroups[solveGroupIndex],
            wavelengthIndex,
            periodicCache);
    }

private:
    void executeGroup(const AngleSolveGroup& group,
                      std::size_t wavelengthIndex,
                      const PeriodicTensorCache* periodicCache) const {
        const Real wavelengthUm = mWavelengthsUm[wavelengthIndex];
        executeAngleSolveGroup(
            group,
            wavelengthUm,
            mPhiDeg,
            mState,
            mBuilder,
            periodicCache,
            [&](std::size_t groupIndex) { return mGroups[groupIndex].thetaDeg; },
            [&](std::size_t groupIndex) {
                return solveScalar(groupIndex, wavelengthIndex);
            },
            [&](std::size_t groupIndex, const PreparedStack& stack) {
                solvePrepared(groupIndex, wavelengthIndex, stack);
            });
    }

    [[nodiscard]] static std::vector<AngleGroup> makeGroups(
        const std::vector<SpectrumAnglePolarization>& requests) {
        std::vector<AngleGroup> groups;
        groups.reserve(requests.size());
        // Angle requests are commonly generated from dense sweeps.  Keep the
        // insertion order (which defines the public result ordering) while
        // using an index for O(1) angle lookup instead of rescanning every
        // previously seen group for each request.
        std::unordered_map<Real, std::size_t> groupByTheta;
        groupByTheta.reserve(requests.size());
        for (std::size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
            const auto& request = requests[requestIndex];
            const auto [groupIt, inserted] = groupByTheta.emplace(
                request.thetaDeg,
                groups.size());
            const std::size_t groupIndex = groupIt->second;
            if (inserted) {
                groups.push_back(AngleGroup{request.thetaDeg, {}, {}});
            }

            AngleGroup& group = groups[groupIndex];
            const std::size_t polIndex = findIndex(
                group.polarizations,
                [&](Polarization pol) { return pol == request.polarization; });
            if (polIndex == group.polarizations.size()) {
                group.polarizations.push_back(request.polarization);
                group.requestIndicesByPol.emplace_back();
            }
            group.requestIndicesByPol[polIndex].push_back(requestIndex);
        }
        return groups;
    }

    [[nodiscard]] static std::vector<Real> groupAngles(
        const std::vector<AngleGroup>& groups) {
        std::vector<Real> angles;
        angles.reserve(groups.size());
        for (const AngleGroup& group : groups) {
            angles.push_back(group.thetaDeg);
        }
        return angles;
    }

    void solvePrepared(std::size_t groupIndex,
                       std::size_t wavelengthIndex,
                       const PreparedStack& stack) const {
        const AngleGroup& group = mGroups[groupIndex];
        const Real wavelengthUm = mWavelengthsUm[wavelengthIndex];
        auto diffraction = Traits::compute(
            stack,
            group.polarizations,
            mState.stackingAlgorithm);
        storeResults(groupIndex, wavelengthIndex, stack, std::move(diffraction));
    }

    template <typename Geometry>
    void storeResults(
        std::size_t groupIndex,
        std::size_t wavelengthIndex,
        const Geometry& geometry,
        std::vector<typename Traits::Diffraction> diffraction) const {
        const AngleGroup& group = mGroups[groupIndex];
        const Real wavelengthUm = mWavelengthsUm[wavelengthIndex];
        for (std::size_t polIndex = 0; polIndex < diffraction.size(); ++polIndex) {
            auto result = Traits::make(
                geometry,
                std::move(diffraction[polIndex]),
                wavelengthUm,
                group.thetaDeg,
                mPhiDeg);
            for (std::size_t requestIndex : group.requestIndicesByPol[polIndex]) {
                (*mOut)[requestIndex * mWavelengthsUm.size() + wavelengthIndex] = result;
            }
        }
    }

    [[nodiscard]] bool solveScalar(std::size_t groupIndex,
                                   std::size_t wavelengthIndex) const {
        const AngleGroup& group = mGroups[groupIndex];
        auto diffraction = Traits::computeScalar(
            mState,
            mBuilder.geometry(),
            mWavelengthsUm[wavelengthIndex],
            group.thetaDeg,
            mPhiDeg,
            group.polarizations);
        if (!diffraction) {
            return false;
        }
        storeResults(
            groupIndex,
            wavelengthIndex,
            mBuilder.geometry(),
            std::move(*diffraction));
        return true;
    }

    const SolverState& mState;
    const StackBuilder& mBuilder;
    std::size_t mRequestCount{};
    std::vector<AngleGroup> mGroups;
    std::vector<AngleSolveGroup> mAngleSolveGroups;
    Real mPhiDeg{};
    const std::vector<Real>& mWavelengthsUm;
    mutable std::vector<Result>* mOut{};
};

template <bool TotalsOnly>
using BatchResult = typename SpectrumBatchTraits<TotalsOnly>::Result;

template <bool TotalsOnly>
std::vector<BatchResult<TotalsOnly>> runAngleSweep(
    const SolverState& state,
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) {
    if (thetaDegs.empty() || wavelengthsUm.empty()) {
        return {};
    }
    std::vector<Polarization> pols = polarizations.empty()
        ? std::vector<Polarization>{state.pol}
        : polarizations;
    const StackBuilder builder = makeStackBuilder(state);
    return runSpectrumPlan(
        AngleSweepPlan<TotalsOnly>(
            state,
            builder,
            thetaDegs,
            phiDeg,
            wavelengthsUm,
            std::move(pols)),
        workers);
}

template <bool TotalsOnly>
std::vector<BatchResult<TotalsOnly>> runAnglePolarizationSweep(
    const SolverState& state,
    const std::vector<SpectrumAnglePolarization>& requests,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    int workers) {
    if (requests.empty() || wavelengthsUm.empty()) {
        return {};
    }
    const StackBuilder builder = makeStackBuilder(state);
    return runSpectrumPlan(
        AnglePolarizationPlan<TotalsOnly>(
            state,
            builder,
            requests,
            phiDeg,
            wavelengthsUm),
        workers);
}

SpectrumThermalBatchResult runThermalSweep(
    const SolverState& state,
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) {
    if (thetaDegs.empty() || wavelengthsUm.empty()) {
        return {};
    }
    std::vector<Polarization> pols = polarizations.empty()
        ? std::vector<Polarization>{state.pol}
        : polarizations;
    const StackBuilder builder = makeStackBuilder(state);
    runIndexedJobs(wavelengthsUm.size(), workers, [&](std::size_t index) {
        builder.validateThermalPassivity(
            wavelengthsUm[index], state.superstrate, state.substrate);
    });
    const std::size_t polarizationCount = pols.size();
    AngleSweepThermalPlan plan(
        state,
        builder,
        thetaDegs,
        phiDeg,
        wavelengthsUm,
        std::move(pols));
    SpectrumThermalBatchResult out;
    out.spectra.resize(checkedProduct(
        plan.jobCount(),
        polarizationCount,
        "combined spectrum result"));
    out.thermalChannels.resize(plan.jobCount());
    plan.bindOutput(out);

    const WavelengthCacheScheduler scheduler(
        plan.builder(),
        plan.wavelengths(),
        plan.groupedAxisSize(),
        plan.jobCount(),
        plan.hasDispersiveMaterials(),
        plan.hasAnglePairs());
    const PeriodicTensorCache staticCache = scheduler.makeStaticCache();
    runCachedJobs(plan, workers, scheduler, cachePtrOrNull(staticCache));
    return out;
}

} // namespace

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchOnly(
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations) const {
    return solveSpectrumBatchOnly(wavelengthsUm, polarizations, 1);
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchOnly(
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) const {
    return runAngleSweep<false>(
        mState,
        {mState.thetaDeg},
        mState.phiDeg,
        wavelengthsUm,
        polarizations,
        workers);
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchForAngles(
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations) const {
    return solveSpectrumBatchForAngles(thetaDegs, phiDeg, wavelengthsUm, polarizations, 1);
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchForAngles(
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) const {
    return runAngleSweep<false>(
        mState, thetaDegs, phiDeg, wavelengthsUm, polarizations, workers);
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchForAnglePolarizations(
    const std::vector<SpectrumAnglePolarization>& requests,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm) const {
    return solveSpectrumBatchForAnglePolarizations(requests, phiDeg, wavelengthsUm, 1);
}

std::vector<SpectrumResult> RcwaSolver::solveSpectrumBatchForAnglePolarizations(
    const std::vector<SpectrumAnglePolarization>& requests,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    int workers) const {
    return runAnglePolarizationSweep<false>(
        mState, requests, phiDeg, wavelengthsUm, workers);
}

std::vector<SpectrumTotalsResult> RcwaSolver::solveSpectrumBatchTotalsOnly(
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations) const {
    return solveSpectrumBatchTotalsOnly(wavelengthsUm, polarizations, 1);
}

std::vector<SpectrumTotalsResult> RcwaSolver::solveSpectrumBatchTotalsOnly(
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) const {
    return runAngleSweep<true>(
        mState,
        {mState.thetaDeg},
        mState.phiDeg,
        wavelengthsUm,
        polarizations,
        workers);
}

std::vector<SpectrumTotalsResult> RcwaSolver::solveSpectrumBatchTotalsForAngles(
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations) const {
    return solveSpectrumBatchTotalsForAngles(
        thetaDegs, phiDeg, wavelengthsUm, polarizations, 1);
}

std::vector<SpectrumTotalsResult> RcwaSolver::solveSpectrumBatchTotalsForAngles(
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) const {
    return runAngleSweep<true>(
        mState, thetaDegs, phiDeg, wavelengthsUm, polarizations, workers);
}

std::vector<SpectrumTotalsResult>
RcwaSolver::solveSpectrumBatchTotalsForAnglePolarizations(
    const std::vector<SpectrumAnglePolarization>& requests,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm) const {
    return solveSpectrumBatchTotalsForAnglePolarizations(
        requests, phiDeg, wavelengthsUm, 1);
}

std::vector<SpectrumTotalsResult>
RcwaSolver::solveSpectrumBatchTotalsForAnglePolarizations(
    const std::vector<SpectrumAnglePolarization>& requests,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    int workers) const {
    return runAnglePolarizationSweep<true>(
        mState, requests, phiDeg, wavelengthsUm, workers);
}

SpectrumThermalBatchResult RcwaSolver::solveSpectrumThermalBatchForAngles(
    const std::vector<Real>& thetaDegs,
    Real phiDeg,
    const std::vector<Real>& wavelengthsUm,
    const std::vector<Polarization>& polarizations,
    int workers) const {
    return runThermalSweep(
        mState, thetaDegs, phiDeg, wavelengthsUm, polarizations, workers);
}

} // namespace rcwa
