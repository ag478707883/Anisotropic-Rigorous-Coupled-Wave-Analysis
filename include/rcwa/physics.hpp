#pragma once

#include <vector>

#include "rcwa/material.hpp"
#include "rcwa/types.hpp"
#include "rcwa/wavevector.hpp"

namespace rcwa {

// Normalized longitudinal wave number q = kz/k0 for an isotropic medium:
// q^2 = epsilon*mu - kx^2 - ky^2.
[[nodiscard]] Complex isotropicLongitudinalWaveNumber(
    Complex epsilon,
    Complex mu,
    Complex kxNormalized,
    Complex kyNormalized);

// Validate the incident half-space required for real-power normalization and
// return its positive real refractive index.
[[nodiscard]] Complex validatedIncidentIndex(const Material& superstrate,
                                              Real wavelengthUm);

struct IsotropicWaveProperties {
    Complex q{};
    Complex teAdmittance{}; // H_y / E_x
    Complex tmAdmittance{}; // H_x / (-E_y)
};

// TE/TM modal admittances follow directly from Maxwell's curl equations.
// Singular q is rejected instead of silently manufacturing a nonphysical mode.
[[nodiscard]] IsotropicWaveProperties isotropicWaveProperties(
    Complex epsilon,
    Complex mu,
    Complex kxNormalized,
    Complex kyNormalized);

struct FresnelAmplitudes {
    Complex reflection{};
    Complex transmission{};
    Real reflectance{};
    Real transmittance{};
};

// Interface amplitudes for a single TE or TM incident diffraction order.  The
// admittance convention is Y_TE=q/mu and Y_TM=epsilon/q, so
// r=(Y1-Y2)/(Y1+Y2), t=2Y1/(Y1+Y2).
[[nodiscard]] FresnelAmplitudes isotropicInterface(
    Complex epsilon1,
    Complex mu1,
    Complex epsilon2,
    Complex mu2,
    Complex kxNormalized,
    Complex kyNormalized,
    Polarization polarization);

struct IsotropicFilm {
    Complex epsilon{};
    Complex mu{1.0, 0.0};
    Real thicknessUm{};
};

// Stable scalar scattering recurrence for a finite isotropic stack and one
// TE/TM incident channel.  It cascades interfaces from substrate to incidence
// using exp(i*kz*d), avoiding the exponentially large intermediate entries of
// a characteristic matrix in thick absorbing films.
[[nodiscard]] FresnelAmplitudes isotropicFilmStack(
    Complex superstrateEpsilon,
    Complex superstrateMu,
    Complex substrateEpsilon,
    Complex substrateMu,
    const std::vector<IsotropicFilm>& films,
    Complex kxNormalized,
    Complex kyNormalized,
    Real wavelengthUm,
    Polarization polarization);

} // namespace rcwa
