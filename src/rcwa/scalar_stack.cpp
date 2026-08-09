#include "rcwa/scalar_stack.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "rcwa/physics.hpp"
#include "rcwa/solver.hpp"

namespace rcwa {

namespace {

bool scalarMaterial(const Material& material,
                    Real wavelengthUm,
                    Complex& epsilon,
                    Complex& mu) {
    const auto tensors = material.tensors(wavelengthUm);
    return tensorIsScalar(tensors.first, &epsilon) &&
           tensorIsScalar(tensors.second, &mu);
}

bool appendUniformScalarLayer(const LayerSpec& spec,
                              Real wavelengthUm,
                              std::vector<IsotropicFilm>& films) {
    const auto* layer = std::get_if<UniformLayer>(&spec);
    if (layer == nullptr || layer->thicknessUm < Real{0} ||
        !std::isfinite(layer->thicknessUm)) {
        return false;
    }
    Complex epsilon{};
    Complex mu{};
    if (!scalarMaterial(layer->material, wavelengthUm, epsilon, mu)) {
        return false;
    }
    films.push_back({epsilon, mu, layer->thicknessUm});
    return true;
}

struct ScalarStackInput {
    Complex superstrateEpsilon{};
    Complex superstrateMu{};
    Complex substrateEpsilon{};
    Complex substrateMu{};
    Complex kxNormalized{};
    Complex kyNormalized{};
    std::vector<IsotropicFilm> films;
};

std::optional<ScalarStackInput> makeScalarStackInput(
    const SolverState& state,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg) {
    requirePositive(wavelengthUm, "wavelength");
    requireFinite(thetaDeg, "theta_deg");
    requireFinite(phiDeg, "phi_deg");
    if (std::abs(thetaDeg) >= Real{90}) {
        throw std::invalid_argument(
            "theta_deg must be strictly between -90 and 90 degrees");
    }

    ScalarStackInput input;
    if (!scalarMaterial(
            state.superstrate,
            wavelengthUm,
            input.superstrateEpsilon,
            input.superstrateMu) ||
        !scalarMaterial(
            state.substrate,
            wavelengthUm,
            input.substrateEpsilon,
            input.substrateMu)) {
        return std::nullopt;
    }

    input.films.reserve(state.layers.size());
    for (const LayerSpec& layer : state.layers) {
        if (!appendUniformScalarLayer(layer, wavelengthUm, input.films)) {
            return std::nullopt;
        }
    }

    const Real theta = degToRad(thetaDeg);
    const Real phi = degToRad(std::remainder(phiDeg, Real{360}));
    const Real incidentIndex = std::real(
        validatedIncidentIndex(state.superstrate, wavelengthUm));
    input.kxNormalized = {
        incidentIndex * std::sin(theta) * std::cos(phi), Real{0}};
    input.kyNormalized = {
        incidentIndex * std::sin(theta) * std::sin(phi), Real{0}};
    return input;
}

std::optional<std::vector<FresnelAmplitudes>> solveScalarResponses(
    const SolverState& state,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const std::vector<Polarization>& polarizations) {
    auto input = makeScalarStackInput(
        state, wavelengthUm, thetaDeg, phiDeg);
    if (!input) {
        return std::nullopt;
    }

    std::optional<FresnelAmplitudes> te;
    std::optional<FresnelAmplitudes> tm;
    const auto solveChannel = [&](Polarization channel) {
        auto& cached = channel == Polarization::TE ? te : tm;
        if (!cached) {
            cached = isotropicFilmStack(
                input->superstrateEpsilon,
                input->superstrateMu,
                input->substrateEpsilon,
                input->substrateMu,
                input->films,
                input->kxNormalized,
                input->kyNormalized,
                wavelengthUm,
                channel);
        }
        return *cached;
    };

    std::vector<FresnelAmplitudes> responses;
    responses.reserve(polarizations.size());
    for (const Polarization polarization : polarizations) {
        if (polarization == Polarization::TE || polarization == Polarization::TM) {
            responses.push_back(solveChannel(polarization));
            continue;
        }
        const FresnelAmplitudes teResponse = solveChannel(Polarization::TE);
        const FresnelAmplitudes tmResponse = solveChannel(Polarization::TM);
        FresnelAmplitudes response;
        response.reflectance =
            Real{0.5} * (teResponse.reflectance + tmResponse.reflectance);
        response.transmittance =
            Real{0.5} * (teResponse.transmittance + tmResponse.transmittance);
        responses.push_back(response);
    }
    return responses;
}

DiffractionTotals makeScalarTotals(const FresnelAmplitudes& response) {
    return {
        response.reflectance,
        response.transmittance,
        response.reflectance,
        response.transmittance,
        response.reflectance + response.transmittance,
    };
}

} // namespace

std::optional<std::vector<DiffractionResult>> solveUniformScalarSpectra(
    const SolverState& state,
    const StackGeometry& geometry,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const std::vector<Polarization>& polarizations) {
    auto responses = solveScalarResponses(
        state, wavelengthUm, thetaDeg, phiDeg, polarizations);
    if (!responses) {
        return std::nullopt;
    }

    std::vector<DiffractionResult> results;
    results.reserve(responses->size());
    for (const FresnelAmplitudes& response : *responses) {
        DiffractionResult result;
        result.R.assign(static_cast<std::size_t>(geometry.totalHarmonics), Real{0});
        result.T.assign(static_cast<std::size_t>(geometry.totalHarmonics), Real{0});
        result.R[static_cast<std::size_t>(geometry.centerIdx)] = response.reflectance;
        result.T[static_cast<std::size_t>(geometry.centerIdx)] = response.transmittance;
        const DiffractionTotals totals = makeScalarTotals(response);
        result.R0 = totals.R0;
        result.T0 = totals.T0;
        result.rTotal = totals.rTotal;
        result.tTotal = totals.tTotal;
        result.conservation = totals.conservation;
        results.push_back(std::move(result));
    }
    return results;
}

std::optional<std::vector<DiffractionTotals>> solveUniformScalarTotals(
    const SolverState& state,
    Real wavelengthUm,
    Real thetaDeg,
    Real phiDeg,
    const std::vector<Polarization>& polarizations) {
    auto responses = solveScalarResponses(
        state, wavelengthUm, thetaDeg, phiDeg, polarizations);
    if (!responses) {
        return std::nullopt;
    }
    std::vector<DiffractionTotals> results;
    results.reserve(responses->size());
    for (const FresnelAmplitudes& response : *responses) {
        results.push_back(makeScalarTotals(response));
    }
    return results;
}

} // namespace rcwa
