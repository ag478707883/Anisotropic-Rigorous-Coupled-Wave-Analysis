#include "rcwaBindingUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/complex.h>

namespace py = pybind11;

namespace rcwaPy {

namespace {

using rcwa::Complex;
using rcwa::Real;
using rcwa::BlackPhosphorusModelParameters;
using rcwa::CdTeLorentzDrudeModelParameters;
using rcwa::DrudeModelParameters;
using rcwa::GrapheneModelParameters;
using rcwa::InAsMagnetoModelParameters;
using rcwa::InSbLorentzDrudeModelParameters;
using rcwa::PhononPolaritonModelParameters;
using rcwa::WsmModelParameters;
using rcwa::YigModelParameters;
using rcwa::builtinMaterialModel;
using rcwa::defaultBlackPhosphorusModelParameters;
using rcwa::defaultCdteLorentzDrudeModelParameters;
using rcwa::defaultDrudeModelParameters;
using rcwa::defaultGrapheneModelParameters;
using rcwa::defaultInasMagnetoModelParameters;
using rcwa::defaultInsbLorentzDrudeModelParameters;
using rcwa::defaultPhononPolaritonModelParameters;
using rcwa::defaultWsmModelParameters;
using rcwa::defaultYigModelParameters;
using rcwa::isBlackPhosphorusMaterialModel;
using rcwa::isCdteLorentzDrudeMaterialModel;
using rcwa::isDrudeMaterialModel;
using rcwa::isGrapheneMaterialModel;
using rcwa::isInasMagnetoMaterialModel;
using rcwa::isInsbLorentzDrudeMaterialModel;
using rcwa::isPhononPolaritonMaterialModel;
using rcwa::isWsmMaterialModel;
using rcwa::isYigMaterialModel;

template <typename T>
T getOr(const py::dict& dict, const char* key, T defaultValue) {
    if (!dict.contains(key)) {
        return defaultValue;
    }
    return dict[py::str(key)].cast<T>();
}

bool tensorsIdentical(const rcwa::Tensor3& a, const rcwa::Tensor3& b) {
    for (std::size_t i = 0; i < a.v.size(); ++i) {
        if (!(std::abs(a.v[i] - b.v[i]) == Real{0.0})) {
            return false;
        }
    }
    return true;
}

Real getRealAny(const py::dict& dict,
                  std::initializer_list<const char*> keys,
                  Real defaultValue) {
    const char* selectedKey = nullptr;
    Real selectedValue = defaultValue;
    for (const char* key : keys) {
        if (dict.contains(py::str(key))) {
            if (selectedKey != nullptr) {
                throw std::invalid_argument(
                    std::string("conflicting aliases '") + selectedKey + "' and '" + key + "'");
            }
            selectedKey = key;
            selectedValue = dict[py::str(key)].cast<Real>();
            if (!std::isfinite(selectedValue)) {
                throw std::invalid_argument(std::string(key) + " must be finite");
            }
        }
    }
    return selectedValue;
}

Complex getComplexAny(const py::dict& dict,
                      std::initializer_list<const char*> keys,
                      Complex defaultValue) {
    const char* selectedKey = nullptr;
    Complex selectedValue = defaultValue;
    for (const char* key : keys) {
        if (dict.contains(py::str(key))) {
            if (selectedKey != nullptr) {
                throw std::invalid_argument(
                    std::string("conflicting aliases '") + selectedKey + "' and '" + key + "'");
            }
            selectedKey = key;
            selectedValue = dict[py::str(key)].cast<Complex>();
            if (!std::isfinite(std::real(selectedValue)) ||
                !std::isfinite(std::imag(selectedValue))) {
                throw std::invalid_argument(std::string(key) + " must be finite");
            }
        }
    }
    return selectedValue;
}

std::array<Real, 3> getRealTripleAny(
    const py::dict& dict,
    std::initializer_list<const char*> keys,
    std::array<Real, 3> defaultValue) {
    const char* selectedKey = nullptr;
    std::array<Real, 3> selectedValue = defaultValue;
    for (const char* key : keys) {
        if (!dict.contains(py::str(key))) {
            continue;
        }
        if (selectedKey != nullptr) {
            throw std::invalid_argument(
                std::string("conflicting aliases '") + selectedKey + "' and '" + key + "'");
        }
        selectedKey = key;
        const py::sequence values =
            py::reinterpret_borrow<py::sequence>(dict[py::str(key)]);
        if (py::len(values) != 3) {
            throw std::invalid_argument(
                std::string(key) + " must contain exactly three x/y/z values");
        }
        selectedValue = {
            values[0].cast<Real>(),
            values[1].cast<Real>(),
            values[2].cast<Real>(),
        };
        if (!std::all_of(
                selectedValue.begin(), selectedValue.end(),
                [](Real value) { return std::isfinite(value); })) {
            throw std::invalid_argument(std::string(key) + " must contain finite values");
        }
    }
    return selectedValue;
}

bool isCommonMaterialDefinitionKey(const std::string& key) {
    return key == "kind" || key == "name" || key == "model";
}

void validateParameterKeys(const py::dict& dict,
                           const char* model,
                           std::initializer_list<const char*> allowedKeys) {
    for (const auto item : dict) {
        std::string key;
        try {
            key = py::cast<std::string>(item.first);
        } catch (const py::cast_error&) {
            throw std::invalid_argument(std::string(model) + " parameter keys must be strings");
        }
        if (isCommonMaterialDefinitionKey(key)) {
            continue;
        }
        const bool known = std::any_of(
            allowedKeys.begin(), allowedKeys.end(),
            [&](const char* allowed) { return key == allowed; });
        if (!known) {
            throw std::invalid_argument(
                std::string(model) + " has unknown parameter '" + key + "'");
        }
    }
}

void requireAtMostOneParameter(const py::dict& dict,
                               const char* quantity,
                               std::initializer_list<const char*> keys) {
    const char* selectedKey = nullptr;
    for (const char* key : keys) {
        if (!dict.contains(py::str(key))) {
            continue;
        }
        if (selectedKey != nullptr) {
            throw std::invalid_argument(
                std::string(quantity) + " was specified more than once ('" +
                selectedKey + "' and '" + key + "')");
        }
        selectedKey = key;
    }
}

GrapheneModelParameters grapheneParametersFromDict(const std::string& model,
                                                      const py::dict& dict) {
    if (dict.contains("parameter")) {
        throw std::invalid_argument(
            "Graphene parameters use 'fermi_energy_ev'; the old 'parameter' key is not accepted");
    }
    if (dict.contains("include_intraband") || dict.contains("intraband") ||
        dict.contains("include_interband") || dict.contains("interband")) {
        throw std::invalid_argument(
            "GrapheneKubo is the only graphene material model; intraband/interband switches "
            "are not accepted");
    }
    validateParameterKeys(
        dict,
        "GrapheneKubo",
        {"fermi_energy_ev", "fermi_ev", "ef_ev", "chemical_potential_ev", "mu_c_ev",
         "chemical_potential", "mu_c", "temperature_k", "temperature",
         "relaxation_time_s", "tau_s", "tau", "relaxation_time_ps", "tau_ps",
         "relaxation_time_fs", "tau_fs", "thickness_m", "graphene_thickness_m",
         "thickness_nm", "graphene_thickness_nm", "thickness_um",
         "graphene_thickness_um"});
    requireAtMostOneParameter(
        dict,
        "Graphene relaxation time",
        {"relaxation_time_s", "tau_s", "tau", "relaxation_time_ps", "tau_ps",
         "relaxation_time_fs", "tau_fs"});
    requireAtMostOneParameter(
        dict,
        "Graphene thickness",
        {"thickness_m", "graphene_thickness_m", "thickness_nm",
         "graphene_thickness_nm", "thickness_um", "graphene_thickness_um"});
    GrapheneModelParameters parameters = defaultGrapheneModelParameters(model);
    parameters.fermiEnergyEv = getRealAny(
        dict,
        {"fermi_energy_ev", "fermi_ev", "ef_ev", "chemical_potential_ev", "mu_c_ev",
         "chemical_potential", "mu_c"},
        parameters.fermiEnergyEv);
    parameters.temperatureK = getRealAny(
        dict,
        {"temperature_k", "temperature"},
        parameters.temperatureK);

    parameters.relaxationTimeS = getRealAny(
        dict,
        {"relaxation_time_s", "tau_s", "tau"},
        parameters.relaxationTimeS);
    const Real relaxationTimePs = getRealAny(
        dict,
        {"relaxation_time_ps", "tau_ps"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(relaxationTimePs)) {
        parameters.relaxationTimeS = relaxationTimePs * Real{1e-12};
    }
    const Real relaxationTimeFs = getRealAny(
        dict,
        {"relaxation_time_fs", "tau_fs"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(relaxationTimeFs)) {
        parameters.relaxationTimeS = relaxationTimeFs * Real{1e-15};
    }

    parameters.thicknessM = getRealAny(
        dict,
        {"thickness_m", "graphene_thickness_m"},
        parameters.thicknessM);
    const Real thicknessNm = getRealAny(
        dict,
        {"thickness_nm", "graphene_thickness_nm"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(thicknessNm)) {
        parameters.thicknessM = thicknessNm * Real{1e-9};
    }
    const Real thicknessUm = getRealAny(
        dict,
        {"thickness_um", "graphene_thickness_um"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(thicknessUm)) {
        parameters.thicknessM = thicknessUm * Real{1e-6};
    }

    return parameters;
}

DrudeModelParameters drudeParametersFromDict(const std::string& model,
                                                const py::dict& dict) {
    validateParameterKeys(
        dict,
        "Drude",
        {"eps_inf", "epsilon_inf", "eps_infinity", "plasma_frequency_rad_s",
         "omega_p_rad_s", "wp_rad_s", "wp", "damping_rate_rad_s", "gamma_rad_s",
         "gamma", "relaxation_rate_rad_s"});
    DrudeModelParameters parameters = defaultDrudeModelParameters(model);
    parameters.epsInf = getRealAny(
        dict,
        {"eps_inf", "epsilon_inf", "eps_infinity"},
        parameters.epsInf);
    parameters.plasmaFrequencyRadS = getRealAny(
        dict,
        {"plasma_frequency_rad_s", "omega_p_rad_s", "wp_rad_s", "wp"},
        parameters.plasmaFrequencyRadS);
    parameters.dampingRateRadS = getRealAny(
        dict,
        {"damping_rate_rad_s", "gamma_rad_s", "gamma", "relaxation_rate_rad_s"},
        parameters.dampingRateRadS);
    return parameters;
}

PhononPolaritonModelParameters phononPolaritonParametersFromDict(
    const std::string& model,
    const py::dict& dict) {
    validateParameterKeys(
        dict,
        "PhononPolariton",
        {"epsilon_inf", "eps_inf", "omega_to_cm_inv", "transverse_wavenumber_cm_inv",
         "omega_lo_cm_inv", "longitudinal_wavenumber_cm_inv", "damping_cm_inv",
         "gamma_cm_inv", "twist_deg", "rotation_deg"});
    PhononPolaritonModelParameters parameters =
        defaultPhononPolaritonModelParameters(model);
    parameters.epsilonInf = getRealTripleAny(
        dict,
        {"epsilon_inf", "eps_inf"},
        parameters.epsilonInf);
    parameters.transverseWavenumberCmInv = getRealTripleAny(
        dict,
        {"omega_to_cm_inv", "transverse_wavenumber_cm_inv"},
        parameters.transverseWavenumberCmInv);
    parameters.longitudinalWavenumberCmInv = getRealTripleAny(
        dict,
        {"omega_lo_cm_inv", "longitudinal_wavenumber_cm_inv"},
        parameters.longitudinalWavenumberCmInv);
    parameters.dampingWavenumberCmInv = getRealTripleAny(
        dict,
        {"damping_cm_inv", "gamma_cm_inv"},
        parameters.dampingWavenumberCmInv);
    parameters.twistDeg = getRealAny(
        dict,
        {"twist_deg", "rotation_deg"},
        parameters.twistDeg);
    return parameters;
}

YigModelParameters yigParametersFromDict(const std::string& model,
                                         const py::dict& dict) {
    validateParameterKeys(
        dict,
        "YIG",
        {"diagonal_epsilon", "epsilon_diagonal", "epsilon", "eps",
         "offdiagonal_gyration", "off_diagonal_gyration", "offdiagonal_b",
         "off_diagonal_b", "gyration", "b", "magnetization_direction",
         "magnetic_field_direction", "bias_direction"});
    YigModelParameters parameters = defaultYigModelParameters(model);
    parameters.diagonalEpsilon = getComplexAny(
        dict,
        {"diagonal_epsilon", "epsilon_diagonal", "epsilon", "eps"},
        parameters.diagonalEpsilon);
    parameters.offDiagonalGyration = getComplexAny(
        dict,
        {"offdiagonal_gyration", "off_diagonal_gyration", "offdiagonal_b",
         "off_diagonal_b", "gyration", "b"},
        parameters.offDiagonalGyration);
    parameters.magnetizationDirection = getRealTripleAny(
        dict,
        {"magnetization_direction", "magnetic_field_direction", "bias_direction"},
        parameters.magnetizationDirection);
    return parameters;
}

WsmModelParameters wsmParametersFromDict(const std::string& model,
                                            const py::dict& dict) {
    validateParameterKeys(
        dict,
        "WSM",
        {"node_separation_m_inv", "b_m_inv", "b", "node_separation_direction",
         "b_direction", "node_direction", "fermi_velocity_m_s", "v_f_m_s", "vf_m_s",
         "v_f", "cutoff_xi", "xi_c", "xi", "background_epsilon", "eps_b",
         "epsilon_b", "temperature_k", "temperature", "degeneracy", "g",
         "relaxation_time_s", "tau_s", "tau", "relaxation_time_fs", "tau_fs",
         "fermi_energy_ev", "fermi_ev", "ef_ev"});
    requireAtMostOneParameter(
        dict,
        "WSM relaxation time",
        {"relaxation_time_s", "tau_s", "tau", "relaxation_time_fs", "tau_fs"});
    WsmModelParameters parameters = defaultWsmModelParameters(model);
    parameters.nodeSeparationMInv = getRealAny(
        dict,
        {"node_separation_m_inv", "b_m_inv", "b"},
        parameters.nodeSeparationMInv);
    parameters.nodeSeparationDirection = getRealTripleAny(
        dict,
        {"node_separation_direction", "b_direction", "node_direction"},
        parameters.nodeSeparationDirection);
    parameters.fermiVelocityMS = getRealAny(
        dict,
        {"fermi_velocity_m_s", "v_f_m_s", "vf_m_s", "v_f"},
        parameters.fermiVelocityMS);
    parameters.cutoffXi = getRealAny(
        dict,
        {"cutoff_xi", "xi_c", "xi"},
        parameters.cutoffXi);
    parameters.backgroundEpsilon = getRealAny(
        dict,
        {"background_epsilon", "eps_b", "epsilon_b"},
        parameters.backgroundEpsilon);
    parameters.temperatureK = getRealAny(
        dict,
        {"temperature_k", "temperature"},
        parameters.temperatureK);
    parameters.degeneracy = getRealAny(
        dict,
        {"degeneracy", "g"},
        parameters.degeneracy);
    parameters.relaxationTimeS = getRealAny(
        dict,
        {"relaxation_time_s", "tau_s", "tau"},
        parameters.relaxationTimeS);
    const Real relaxationTimeFs = getRealAny(
        dict,
        {"relaxation_time_fs", "tau_fs"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(relaxationTimeFs)) {
        parameters.relaxationTimeS = relaxationTimeFs * Real{1e-15};
    }
    parameters.fermiEnergyEv = getRealAny(
        dict,
        {"fermi_energy_ev", "fermi_ev", "ef_ev"},
        parameters.fermiEnergyEv);
    return parameters;
}

BlackPhosphorusModelParameters blackPhosphorusParametersFromDict(
    const std::string& model,
    const py::dict& dict) {
    validateParameterKeys(
        dict,
        "BlackPhosphorusDrude",
        {"carrier_density_cm2", "density_cm2", "n_cm2", "N_cm2", "N",
         "carrier_density_m2", "density_m2", "n_m2", "scattering_rate_ev", "eta_ev",
         "damping_ev", "scattering_rate_mev", "eta_mev", "damping_mev", "thickness_m",
         "bp_thickness_m", "thickness_nm", "bp_thickness_nm", "thickness_um",
         "bp_thickness_um", "bandgap_ev", "delta_ev", "Delta_ev", "lattice_constant_m",
         "a_m", "lattice_constant_nm", "a_nm", "twistDeg", "twist_deg",
         "rotation_deg"});
    requireAtMostOneParameter(
        dict,
        "Black-phosphorus carrier density",
        {"carrier_density_cm2", "density_cm2", "n_cm2", "N_cm2", "N",
         "carrier_density_m2", "density_m2", "n_m2"});
    requireAtMostOneParameter(
        dict,
        "Black-phosphorus scattering rate",
        {"scattering_rate_ev", "eta_ev", "damping_ev", "scattering_rate_mev", "eta_mev",
         "damping_mev"});
    requireAtMostOneParameter(
        dict,
        "Black-phosphorus thickness",
        {"thickness_m", "bp_thickness_m", "thickness_nm", "bp_thickness_nm",
         "thickness_um", "bp_thickness_um"});
    requireAtMostOneParameter(
        dict,
        "Black-phosphorus lattice constant",
        {"lattice_constant_m", "a_m", "lattice_constant_nm", "a_nm"});
    BlackPhosphorusModelParameters parameters =
        defaultBlackPhosphorusModelParameters(model);
    parameters.carrierDensityCm2 = getRealAny(
        dict,
        {"carrier_density_cm2", "density_cm2", "n_cm2", "N_cm2", "N"},
        parameters.carrierDensityCm2);
    const Real carrierDensityM2 = getRealAny(
        dict,
        {"carrier_density_m2", "density_m2", "n_m2"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(carrierDensityM2)) {
        parameters.carrierDensityCm2 = carrierDensityM2 / Real{1.0e4};
    }

    parameters.scatteringRateEv = getRealAny(
        dict,
        {"scattering_rate_ev", "eta_ev", "damping_ev"},
        parameters.scatteringRateEv);
    const Real scatteringRateMev = getRealAny(
        dict,
        {"scattering_rate_mev", "eta_mev", "damping_mev"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(scatteringRateMev)) {
        parameters.scatteringRateEv = scatteringRateMev * Real{1.0e-3};
    }

    parameters.thicknessM = getRealAny(
        dict,
        {"thickness_m", "bp_thickness_m"},
        parameters.thicknessM);
    const Real thicknessNm = getRealAny(
        dict,
        {"thickness_nm", "bp_thickness_nm"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(thicknessNm)) {
        parameters.thicknessM = thicknessNm * Real{1e-9};
    }
    const Real thicknessUm = getRealAny(
        dict,
        {"thickness_um", "bp_thickness_um"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(thicknessUm)) {
        parameters.thicknessM = thicknessUm * Real{1e-6};
    }

    parameters.bandgapEv = getRealAny(
        dict,
        {"bandgap_ev", "delta_ev", "Delta_ev"},
        parameters.bandgapEv);
    parameters.latticeConstantM = getRealAny(
        dict,
        {"lattice_constant_m", "a_m"},
        parameters.latticeConstantM);
    const Real latticeConstantNm = getRealAny(
        dict,
        {"lattice_constant_nm", "a_nm"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(latticeConstantNm)) {
        parameters.latticeConstantM = latticeConstantNm * Real{1e-9};
    }
    parameters.twistDeg = getRealAny(
        dict,
        {"twistDeg", "twist_deg", "rotation_deg"},
        parameters.twistDeg);
    return parameters;
}

template <typename Parameters>
Parameters carrierMagnetoParametersFromDict(
    Parameters parameters,
    const py::dict& dict,
    bool acceptCollisionFrequency) {
    parameters.magneticFieldT = getRealAny(
        dict,
        {"magnetic_field_t", "b_t", "B_t", "B"},
        parameters.magneticFieldT);
    parameters.epsInf = getRealAny(
        dict,
        {"eps_inf", "epsilon_inf", "eps_infinity"},
        parameters.epsInf);
    parameters.carrierDensityCm3 = getRealAny(
        dict,
        {"carrier_density_cm3", "density_cm3", "n_cm3"},
        parameters.carrierDensityCm3);
    const Real densityM3 = getRealAny(
        dict,
        {"carrier_density_m3", "density_m3", "n_m3"},
        std::numeric_limits<Real>::quiet_NaN());
    if (std::isfinite(densityM3)) {
        parameters.carrierDensityCm3 = densityM3 / Real{1.0e6};
    }
    parameters.effectiveMassRatio = getRealAny(
        dict,
        {"effective_mass_ratio", "m_eff_over_me", "mstar"},
        parameters.effectiveMassRatio);
    parameters.dampingRateRadS = acceptCollisionFrequency
        ? getRealAny(
            dict,
            {"damping_rate_rad_s", "collision_frequency_rad_s", "gamma_rad_s",
             "gamma", "relaxation_rate_rad_s"},
            parameters.dampingRateRadS)
        : getRealAny(
            dict,
            {"damping_rate_rad_s", "gamma_rad_s", "gamma", "relaxation_rate_rad_s"},
            parameters.dampingRateRadS);
    return parameters;
}

InAsMagnetoModelParameters inasMagnetoParametersFromDict(
    const std::string& model,
    const py::dict& dict) {
    validateParameterKeys(
        dict,
        "InAsMagneto",
        {"magnetic_field_t", "b_t", "B_t", "B", "eps_inf", "epsilon_inf",
         "eps_infinity", "carrier_density_cm3", "density_cm3", "n_cm3",
         "carrier_density_m3", "density_m3", "n_m3", "effective_mass_ratio",
         "m_eff_over_me", "mstar", "damping_rate_rad_s", "gamma_rad_s", "gamma",
         "relaxation_rate_rad_s"});
    requireAtMostOneParameter(
        dict,
        "InAs carrier density",
        {"carrier_density_cm3", "density_cm3", "n_cm3", "carrier_density_m3",
         "density_m3", "n_m3"});
    return carrierMagnetoParametersFromDict(
        defaultInasMagnetoModelParameters(model),
        dict,
        false);
}

template <typename Parameters>
Parameters lorentzDrudeParametersFromDict(Parameters parameters, const py::dict& dict) {
    parameters = carrierMagnetoParametersFromDict(
        std::move(parameters),
        dict,
        true);
    parameters.transverseResonanceRadS = getRealAny(
        dict,
        {"transverse_resonance_rad_s", "omega_t_rad_s", "omega_t"},
        parameters.transverseResonanceRadS);
    return parameters;
}

CdTeLorentzDrudeModelParameters cdteLorentzDrudeParametersFromDict(
    const std::string& model,
    const py::dict& dict) {
    validateParameterKeys(
        dict,
        "CdTeLorentzDrude",
        {"magnetic_field_t", "b_t", "B_t", "B", "eps_inf", "epsilon_inf",
         "eps_infinity", "carrier_density_cm3", "density_cm3", "n_cm3",
         "carrier_density_m3", "density_m3", "n_m3", "effective_mass_ratio",
         "m_eff_over_me", "mstar", "damping_rate_rad_s", "collision_frequency_rad_s",
         "gamma_rad_s", "gamma", "relaxation_rate_rad_s", "transverse_resonance_rad_s",
         "omega_t_rad_s", "omega_t"});
    requireAtMostOneParameter(
        dict,
        "CdTe carrier density",
        {"carrier_density_cm3", "density_cm3", "n_cm3", "carrier_density_m3",
         "density_m3", "n_m3"});
    return lorentzDrudeParametersFromDict(
        defaultCdteLorentzDrudeModelParameters(model),
        dict);
}

InSbLorentzDrudeModelParameters insbLorentzDrudeParametersFromDict(
    const std::string& model,
    const py::dict& dict) {
    validateParameterKeys(
        dict,
        "InSbLorentzDrude",
        {"magnetic_field_t", "b_t", "B_t", "B", "eps_inf", "epsilon_inf",
         "eps_infinity", "carrier_density_cm3", "density_cm3", "n_cm3",
         "carrier_density_m3", "density_m3", "n_m3", "effective_mass_ratio",
         "m_eff_over_me", "mstar", "damping_rate_rad_s", "collision_frequency_rad_s",
         "gamma_rad_s", "gamma", "relaxation_rate_rad_s", "transverse_resonance_rad_s",
         "omega_t_rad_s", "omega_t"});
    requireAtMostOneParameter(
        dict,
        "InSb carrier density",
        {"carrier_density_cm3", "density_cm3", "n_cm3", "carrier_density_m3",
         "density_m3", "n_m3"});
    return lorentzDrudeParametersFromDict(
        defaultInsbLorentzDrudeModelParameters(model),
        dict);
}

rcwa::Material tabulatedIndexMaterialFromDict(const std::string& name,
                                              const py::dict& dict) {
    if (!dict.contains(py::str("wavelength_um")) ||
        !dict.contains(py::str("n")) ||
        !dict.contains(py::str("k"))) {
        throw std::invalid_argument(
            "tabulated_index material requires wavelength_um, n, and k arrays");
    }

    py::array_t<Real, py::array::c_style | py::array::forcecast> wavelengths(
        dict[py::str("wavelength_um")]);
    py::array_t<Real, py::array::c_style | py::array::forcecast> realIndex(
        dict[py::str("n")]);
    py::array_t<Real, py::array::c_style | py::array::forcecast> extinction(
        dict[py::str("k")]);
    if (wavelengths.ndim() != 1 || realIndex.ndim() != 1 || extinction.ndim() != 1 ||
        wavelengths.size() < 2 || wavelengths.size() != realIndex.size() ||
        wavelengths.size() != extinction.size()) {
        throw std::invalid_argument(
            "tabulated_index arrays must be one-dimensional, have equal lengths, "
            "and contain at least two samples");
    }

    std::vector<Real> lambdaValues(static_cast<std::size_t>(wavelengths.size()));
    std::vector<Complex> indexValues(static_cast<std::size_t>(wavelengths.size()));
    const auto lambdaView = wavelengths.unchecked<1>();
    const auto nView = realIndex.unchecked<1>();
    const auto kView = extinction.unchecked<1>();
    for (py::ssize_t i = 0; i < wavelengths.size(); ++i) {
        lambdaValues[static_cast<std::size_t>(i)] = lambdaView(i);
        indexValues[static_cast<std::size_t>(i)] = Complex{nView(i), kView(i)};
    }

    const std::string extrapolation = lowercase(
        getOr(dict, "extrapolation", std::string{"error"}));
    rcwa::MaterialExtrapolation extrapolationMode;
    if (extrapolation == "error") {
        extrapolationMode = rcwa::MaterialExtrapolation::Error;
    } else if (extrapolation == "clamp" || extrapolation == "hold") {
        extrapolationMode = rcwa::MaterialExtrapolation::Clamp;
    } else {
        throw std::invalid_argument(
            "tabulated_index extrapolation must be 'error' or 'clamp'");
    }

    return rcwa::Material::tabulatedIndex(
        name,
        std::move(lambdaValues),
        std::move(indexValues),
        extrapolationMode);
}

} // namespace

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

int normalizedWorkerCount(int requested, std::size_t jobCount) {
    return rcwa::boundedWorkerCount(requested, jobCount);
}

py::dtype makeDtype(std::initializer_list<std::pair<const char*, const char*>> fields) {
    py::list spec;
    for (const auto& field : fields) {
        spec.append(py::make_tuple(field.first, field.second));
    }
    py::object dtype = py::module_::import("numpy").attr("dtype")(spec);
    return py::reinterpret_borrow<py::dtype>(dtype);
}

py::array makeStructuredArray(const py::dtype& dtype, std::size_t rows) {
    return py::array(dtype, std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows)});
}

rcwa::Tensor3 tensorFromArray(py::handle obj) {
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr(
        py::reinterpret_borrow<py::object>(obj));
    if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3) {
        throw std::invalid_argument("tensor arrays must have shape (3, 3)");
    }
    const auto view = arr.unchecked<2>();
    rcwa::Tensor3 tensor;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const auto value =
                view(static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c));
            tensor(r, c) =
                Complex{static_cast<Real>(value.real()), static_cast<Real>(value.imag())};
        }
    }
    return tensor;
}

rcwa::Material materialFromObject(py::handle obj, std::string defaultName) {
    if (py::isinstance<py::dict>(obj)) {
        const py::dict dict = py::reinterpret_borrow<py::dict>(obj);
        const std::string kind = lowercase(getOr(dict, "kind", std::string{"constant"}));
        const std::string name = getOr(dict, "name", defaultName);
        if (kind == "vacuum") {
            return rcwa::Material::vacuum();
        }
        if (kind == "constant") {
            return rcwa::Material::constant(
                name,
                dict[py::str("epsilon")].cast<Complex>(),
                getOr(dict, "mu", Complex{1.0, 0.0}));
        }
        if (kind == "index") {
            return rcwa::Material::fromIndex(name, dict[py::str("n")].cast<Complex>());
        }
        if (kind == "tabulated_index" || kind == "tabulated-index" ||
            kind == "tabulated") {
            return tabulatedIndexMaterialFromDict(name, dict);
        }
        if (kind == "tensor") {
            const rcwa::Tensor3 eps = tensorFromArray(dict[py::str("epsilon")]);
            const rcwa::Tensor3 mu = dict.contains("mu")
                ? tensorFromArray(dict[py::str("mu")])
                : rcwa::Tensor3::isotropic({1.0, 0.0});
            return rcwa::Material::anisotropic(name, eps, mu);
        }
        if (kind == "builtin" || kind == "built-in" || kind == "model") {
            const std::string model = getOr(dict, "model", std::string{});
            if (dict.contains("parameters")) {
                return builtinMaterialModelFromObject(name, model, dict[py::str("parameters")]);
            }
            if (dict.contains("params")) {
                return builtinMaterialModelFromObject(name, model, dict[py::str("params")]);
            }
            if (dict.contains("parameter")) {
                throw std::invalid_argument(
                    "built-in material dictionaries use 'parameters', not the old scalar "
                    "'parameter' key");
            }
            if (isGrapheneMaterialModel(model) || isDrudeMaterialModel(model) ||
                isPhononPolaritonMaterialModel(model) ||
                isYigMaterialModel(model) ||
                isWsmMaterialModel(model) || isBlackPhosphorusMaterialModel(model) ||
                isInasMagnetoMaterialModel(model) ||
                isCdteLorentzDrudeMaterialModel(model) ||
                isInsbLorentzDrudeMaterialModel(model)) {
                return builtinMaterialModelFromObject(name, model, dict);
            }
            return builtinMaterialModel(name, model);
        }
        throw std::invalid_argument(
            "material kind must be vacuum, constant, index, tabulated_index, "
            "tensor, or builtin");
    }

    if (py::isinstance<py::array>(obj)) {
        return rcwa::Material::anisotropic(defaultName, tensorFromArray(obj));
    }
    return rcwa::Material::constant(defaultName, obj.cast<Complex>());
}

rcwa::Material builtinMaterialModelFromObject(const std::string& name,
                                                  const std::string& model,
                                                  py::handle parameters) {
    if (parameters.is_none()) {
        return builtinMaterialModel(name, model);
    }
    if (py::isinstance<py::dict>(parameters)) {
        const py::dict dict = py::reinterpret_borrow<py::dict>(parameters);
        if (isGrapheneMaterialModel(model)) {
            return builtinMaterialModel(name, model, grapheneParametersFromDict(model, dict));
        }
        if (isDrudeMaterialModel(model)) {
            return builtinMaterialModel(name, model, drudeParametersFromDict(model, dict));
        }
        if (isPhononPolaritonMaterialModel(model)) {
            return builtinMaterialModel(
                name,
                model,
                phononPolaritonParametersFromDict(model, dict));
        }
        if (isYigMaterialModel(model)) {
            return builtinMaterialModel(name, model, yigParametersFromDict(model, dict));
        }
        if (isWsmMaterialModel(model)) {
            return builtinMaterialModel(name, model, wsmParametersFromDict(model, dict));
        }
        if (isBlackPhosphorusMaterialModel(model)) {
            return builtinMaterialModel(
                name,
                model,
                blackPhosphorusParametersFromDict(model, dict));
        }
        if (isInasMagnetoMaterialModel(model)) {
            return builtinMaterialModel(
                name,
                model,
                inasMagnetoParametersFromDict(model, dict));
        }
        if (isCdteLorentzDrudeMaterialModel(model)) {
            return builtinMaterialModel(
                name,
                model,
                cdteLorentzDrudeParametersFromDict(model, dict));
        }
        if (isInsbLorentzDrudeMaterialModel(model)) {
            return builtinMaterialModel(
                name,
                model,
                insbLorentzDrudeParametersFromDict(model, dict));
        }
        throw std::invalid_argument(
            "this built-in material model does not accept parameter dictionaries");
    }
    throw std::invalid_argument(
        "built-in material model parameters must be a dictionary; scalar parameters were removed");
}

bool materialsIdenticalAtReferenceWavelength(const rcwa::Material& a,
                                                 const rcwa::Material& b) {
    return tensorsIdentical(a.epsilonTensor(Real{1.0}), b.epsilonTensor(Real{1.0})) &&
           tensorsIdentical(a.muTensor(Real{1.0}), b.muTensor(Real{1.0}));
}

rcwa::Polarization parsePolarization(std::string_view text) {
    const std::string value = lowercase(std::string{text});
    if (value == "te" || value == "s") {
        return rcwa::Polarization::TE;
    }
    if (value == "tm" || value == "p") {
        return rcwa::Polarization::TM;
    }
    if (value == "both" || value == "linear45") {
        return rcwa::Polarization::Both;
    }
    if (value == "lcp" || value == "lhcp" || value == "left" ||
        value == "leftcircular" || value == "left-circular") {
        return rcwa::Polarization::LCP;
    }
    if (value == "rcp" || value == "rhcp" || value == "right" ||
        value == "rightcircular" || value == "right-circular") {
        return rcwa::Polarization::RCP;
    }
    throw std::invalid_argument("polarization must be TE, TM, Both, LCP, or RCP");
}

const char* polarizationName(rcwa::Polarization pol) {
    switch (pol) {
    case rcwa::Polarization::TE:
        return "TE";
    case rcwa::Polarization::TM:
        return "TM";
    case rcwa::Polarization::Both:
        return "Both";
    case rcwa::Polarization::LCP:
        return "LCP";
    case rcwa::Polarization::RCP:
        return "RCP";
    }
    return "Unknown";
}

rcwa::FieldComponent parseFieldComponent(std::string_view text) {
    if (text == "Ex") return rcwa::FieldComponent::Ex;
    if (text == "Ey") return rcwa::FieldComponent::Ey;
    if (text == "Ez") return rcwa::FieldComponent::Ez;
    if (text == "E") return rcwa::FieldComponent::E;
    if (text == "Hx") return rcwa::FieldComponent::Hx;
    if (text == "Hy") return rcwa::FieldComponent::Hy;
    if (text == "Hz") return rcwa::FieldComponent::Hz;
    if (text == "H") return rcwa::FieldComponent::H;
    throw std::invalid_argument(
        "field component must be E, Ex, Ey, Ez, H, Hx, Hy, or Hz");
}

rcwa::FieldPlane parseFieldPlane(std::string_view text) {
    if (text == "xy" || text == "XY") return rcwa::FieldPlane::XY;
    if (text == "xz" || text == "XZ") return rcwa::FieldPlane::XZ;
    if (text == "yz" || text == "YZ") return rcwa::FieldPlane::YZ;
    throw std::invalid_argument("field plane must be xy, xz, or yz");
}

} // namespace rcwa_py
