#pragma once

#include <vector>

#include "rcwa/layer.hpp"
#include "rcwa/matrix.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

enum class StackingAlgorithm {
    ScatteringMatrix,
    EnhancedTransmittanceMatrix,
};

struct SMatrix {
    Matrix S11;
    Matrix S12;
    Matrix S21;
    Matrix S22;
};

struct DiffractionResult {
    std::vector<Real> R;
    std::vector<Real> T;
    Real R0{};
    Real T0{};
    Real rTotal{};
    Real tTotal{};
    Real conservation{};
};

struct DiffractionTotals {
    Real R0{};
    Real T0{};
    Real rTotal{};
    Real tTotal{};
    Real conservation{};
};

struct ZeroOrderAmplitudes {
    Matrix R;
    Matrix T;
};

// Power-normalized emissivity into every upward channel of the superstrate.
// The entries are the row deficiencies of the complete two-port scattering
// matrix, including incidence from both exterior half spaces.
struct TopDirectionalThermalChannels {
    std::vector<Real> absorptivity;
    std::vector<Real> emissivity;
    std::vector<Real> incidentScattering;
    std::vector<Real> outgoingScattering;
    // Off-diagonal TE/TM entries of I-S^H S and I-S S^H for equal
    // diffraction-order indices.  These retain the coherence required to
    // project a thermal channel onto LCP/RCP instead of averaging TE/TM.
    std::vector<Complex> teTmAbsorptivityCoherence;
    std::vector<Complex> teTmEmissivityCoherence;
};

struct SpectrumThermalChannels {
    std::vector<DiffractionTotals> spectra;
    TopDirectionalThermalChannels thermal;
};

std::vector<DiffractionResult> computeStackEfficienciesForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm = StackingAlgorithm::ScatteringMatrix
);

std::vector<DiffractionTotals> computeStackTotalsForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm = StackingAlgorithm::ScatteringMatrix
);

std::vector<DiffractionTotals> computeStackTotalsForPolarizations(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    const SMatrix& smatrix
);

[[nodiscard]] SpectrumThermalChannels
computeStackTotalsAndTopDirectionalThermalChannels(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    const std::vector<Polarization>& polarizations,
    StackingAlgorithm algorithm = StackingAlgorithm::ScatteringMatrix);

[[nodiscard]] SMatrix computeStackSmatrix(
    const std::vector<LayerModes>& media,
    StackingAlgorithm algorithm = StackingAlgorithm::ScatteringMatrix);
[[nodiscard]] ZeroOrderAmplitudes computeZeroOrderAmplitudes(
    const std::vector<LayerModes>& media,
    int harmonicCount,
    int centerIdx,
    StackingAlgorithm algorithm = StackingAlgorithm::ScatteringMatrix);
[[nodiscard]] TopDirectionalThermalChannels computeTopDirectionalThermalChannels(
    const std::vector<LayerModes>& media,
    const SMatrix& smatrix);

} // namespace rcwa
