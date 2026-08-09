#pragma once

#include <vector>

#include "rcwa/geometry.hpp"
#include "rcwa/scalar_stack.hpp"
#include "rcwa/smatrix.hpp"

namespace rcwa {

struct SolverState;

// Single orchestration boundary for stack preparation and scattering.  The
// public RcwaSolver remains an API facade; wavelength batches and bindings can
// reuse this engine without duplicating cache, basis, or polarization logic.
class StackEngine {
public:
    explicit StackEngine(const SolverState& state);

    [[nodiscard]] const StackBuilder& builder() const noexcept { return mBuilder; }
    [[nodiscard]] PreparedStack prepare() const;
    [[nodiscard]] PreparedStack prepare(
        Real wavelengthUm,
        Real thetaDeg,
        Real phiDeg,
        const PeriodicTensorCache* cache = nullptr,
        bool analyticContinuation = false,
        Real blochKxReduced = 0.0,
        Real blochKyReduced = 0.0) const;

    [[nodiscard]] std::vector<DiffractionResult> efficiencies(
        const PreparedStack& stack,
        const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] std::optional<std::vector<DiffractionResult>>
    uniformScalarSpectra(const std::vector<Polarization>& polarizations) const;
    [[nodiscard]] SMatrix scattering(const PreparedStack& stack) const;

private:
    const SolverState& mState;
    StackBuilder mBuilder;
};

} // namespace rcwa
