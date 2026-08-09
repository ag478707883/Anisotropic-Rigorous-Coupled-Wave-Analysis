#pragma once

#include <optional>
#include <vector>

#include "rcwa/geometry.hpp"
#include "rcwa/smatrix.hpp"

namespace rcwa {

// Exact characteristic-matrix route for a stack that is physically uniform
// in x/y.  It returns nullopt for patterned or tensor-coupled stacks so the
// general RCWA engine can handle those cases.
[[nodiscard]] std::optional<std::vector<DiffractionResult>>
solveUniformScalarSpectra(
    const SolverState& state,
    const StackGeometry& geometry,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const std::vector<Polarization>& polarizations);
[[nodiscard]] std::optional<std::vector<DiffractionTotals>>
solveUniformScalarTotals(
    const SolverState& state,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const std::vector<Polarization>& polarizations);

} // namespace rcwa
