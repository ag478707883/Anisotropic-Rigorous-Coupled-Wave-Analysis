#include "rcwa/physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace rcwa {

namespace {

void requireFiniteComplex(Complex value, const char* name) {
    if (!isFinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

Complex checkedRatio(Complex numerator, Complex denominator, const char* name) {
    const Real scale = std::max({Real{1}, std::abs(numerator), std::abs(denominator)});
    if (std::abs(denominator) <= std::numeric_limits<Real>::epsilon() * scale) {
        throw std::domain_error(std::string(name) + " is singular at a grazing mode");
    }
    const Complex result = numerator / denominator;
    requireFiniteComplex(result, name);
    return result;
}

bool interfaceIsSingular(Complex left, Complex right) {
    const Real scale = std::max({Real{1}, std::abs(left), std::abs(right)});
    return std::abs(left + right) <=
           std::numeric_limits<Real>::epsilon() * scale;
}

std::pair<Complex, Complex> characteristicFilmAmplitudes(
    const std::vector<Complex>& admittances,
    const std::vector<Complex>& longitudinalWaveNumbers,
    const std::vector<IsotropicFilm>& films,
    Real wavelengthUm) {
    Complex a{1.0, 0.0};
    Complex b{};
    Complex c{};
    Complex d{1.0, 0.0};
    for (std::size_t index = 0; index < films.size(); ++index) {
        const Complex delta = twoPi * films[index].thicknessUm *
                              longitudinalWaveNumbers[index] / wavelengthUm;
        const Complex cosine = std::cos(delta);
        const Complex sine = std::sin(delta);
        const Complex y = admittances[index + 1];
        const Complex na = a * cosine - b * iu * y * sine;
        const Complex nb = -a * iu * sine / y + b * cosine;
        const Complex nc = c * cosine - d * iu * y * sine;
        const Complex nd = -c * iu * sine / y + d * cosine;
        a = na;
        b = nb;
        c = nc;
        d = nd;
    }

    const Complex yi = admittances.front();
    const Complex ys = admittances.back();
    const Complex denominator = yi * (a + b * ys) + c + d * ys;
    return {
        checkedRatio(
            yi * (a + b * ys) - c - d * ys,
            denominator,
            "film-stack characteristic reflection"),
        checkedRatio(
            Real{2} * yi,
            denominator,
            "film-stack characteristic transmission"),
    };
}

} // namespace

Complex isotropicLongitudinalWaveNumber(Complex epsilon,
                                        Complex mu,
                                        Complex kxNormalized,
                                        Complex kyNormalized) {
    requireFiniteComplex(epsilon, "epsilon");
    requireFiniteComplex(mu, "mu");
    requireFiniteComplex(kxNormalized, "normalized kx");
    requireFiniteComplex(kyNormalized, "normalized ky");
    return outgoingSqrt(epsilon * mu -
                        kxNormalized * kxNormalized -
                        kyNormalized * kyNormalized);
}

Complex validatedIncidentIndex(const Material& superstrate, Real wavelengthUm) {
    const auto [epsilon, mu] = superstrate.tensors(wavelengthUm);
    Complex epsilonScalar{};
    Complex muScalar{};
    if (!tensorIsScalar(epsilon, &epsilonScalar) ||
        !tensorIsScalar(mu, &muScalar)) {
        throw std::invalid_argument(
            "incident superstrate must be an isotropic scalar material");
    }
    const Real materialScale = std::max({Real{1}, std::abs(epsilonScalar), std::abs(muScalar)});
#ifdef RCWA_SINGLE_PRECISION
    const Real lossTolerance = Real{1e-5f};
#else
    const Real lossTolerance = Real{1e-12};
#endif
    if (std::abs(std::imag(epsilonScalar)) > lossTolerance * materialScale ||
        std::abs(std::imag(muScalar)) > lossTolerance * materialScale) {
        throw std::invalid_argument(
            "incident superstrate must be lossless; injection from a lossy or active "
            "half-space has no unambiguous far-field power normalization");
    }
    const Complex index = outgoingSqrt(epsilonScalar * muScalar);
    if (!isFinite(index) ||
        std::abs(std::imag(index)) > lossTolerance * std::max(Real{1}, std::abs(index)) ||
        !(std::real(index) > Real{0})) {
        throw std::invalid_argument(
            "incident superstrate must have a positive real refractive index");
    }
    return {std::real(index), Real{0}};
}

IsotropicWaveProperties isotropicWaveProperties(Complex epsilon,
                                                Complex mu,
                                                Complex kxNormalized,
                                                Complex kyNormalized) {
    const Complex q = isotropicLongitudinalWaveNumber(
        epsilon, mu, kxNormalized, kyNormalized);
    if (std::abs(q) <= std::numeric_limits<Real>::epsilon()) {
        throw std::domain_error("isotropic modal admittance is singular at q=0");
    }
    return {q, checkedRatio(q, mu, "TE modal admittance"),
            checkedRatio(epsilon, q, "TM modal admittance")};
}

FresnelAmplitudes isotropicInterface(Complex epsilon1,
                                     Complex mu1,
                                     Complex epsilon2,
                                     Complex mu2,
                                     Complex kxNormalized,
                                     Complex kyNormalized,
                                     Polarization polarization) {
    const auto first = isotropicWaveProperties(
        epsilon1, mu1, kxNormalized, kyNormalized);
    const auto second = isotropicWaveProperties(
        epsilon2, mu2, kxNormalized, kyNormalized);

    const auto one = [&](Complex y1, Complex y2) {
        const Complex denominator = y1 + y2;
        const Complex r = checkedRatio(y1 - y2, denominator, "Fresnel reflection");
        const Complex t = checkedRatio(Real{2} * y1, denominator, "Fresnel transmission");
        // Evanescent orders carry no net far-field power.  For propagating
        // lossless exteriors, Poynting normalization is Re(Y2)/Re(Y1).
        const Real y1Real = std::real(y1);
        const Real y2Real = std::real(y2);
        const Real scale = std::max({Real{1}, std::abs(y1), std::abs(y2)});
        const Real transmittance =
            y1Real > std::numeric_limits<Real>::epsilon() * scale &&
            y2Real > std::numeric_limits<Real>::epsilon() * scale
                ? (y2Real / y1Real) * std::norm(t)
                : Real{0};
        return FresnelAmplitudes{r, t, std::norm(r), transmittance};
    };

    switch (polarization) {
    case Polarization::TE:
        return one(first.teAdmittance, second.teAdmittance);
    case Polarization::TM:
        return one(first.tmAdmittance, second.tmAdmittance);
    case Polarization::Both:
    case Polarization::LCP:
    case Polarization::RCP:
        throw std::invalid_argument(
            "isotropicInterface expects one coherent TE or TM channel");
    }
    throw std::invalid_argument("unsupported polarization");
}

FresnelAmplitudes isotropicFilmStack(Complex superstrateEpsilon,
                                     Complex superstrateMu,
                                     Complex substrateEpsilon,
                                     Complex substrateMu,
                                     const std::vector<IsotropicFilm>& films,
                                     Complex kxNormalized,
                                     Complex kyNormalized,
                                     Real wavelengthUm,
                                     Polarization polarization) {
    requirePositive(wavelengthUm, "wavelength");
    if (polarization == Polarization::Both ||
        polarization == Polarization::LCP ||
        polarization == Polarization::RCP) {
        throw std::invalid_argument(
            "isotropicFilmStack expects one coherent TE or TM channel");
    }
    const auto incident = isotropicWaveProperties(
        superstrateEpsilon, superstrateMu, kxNormalized, kyNormalized);
    const auto substrate = isotropicWaveProperties(
        substrateEpsilon, substrateMu, kxNormalized, kyNormalized);
    const auto admittance = [&](const IsotropicWaveProperties& value) {
        return polarization == Polarization::TM ? value.tmAdmittance
                                                : value.teAdmittance;
    };

    std::vector<Complex> admittances;
    std::vector<Complex> longitudinalWaveNumbers;
    admittances.reserve(films.size() + 2);
    longitudinalWaveNumbers.reserve(films.size());
    admittances.push_back(admittance(incident));
    for (const IsotropicFilm& film : films) {
        if (film.thicknessUm < Real{0} || !std::isfinite(film.thicknessUm)) {
            throw std::invalid_argument("film thickness must be finite and non-negative");
        }
        const auto mode = isotropicWaveProperties(
            film.epsilon, film.mu, kxNormalized, kyNormalized);
        admittances.push_back(admittance(mode));
        longitudinalWaveNumbers.push_back(mode.q);
    }
    admittances.push_back(admittance(substrate));

    const auto interface = [&](std::size_t left, std::size_t right) {
        const Complex sum = admittances[left] + admittances[right];
        return std::array<Complex, 4>{
            checkedRatio(
                admittances[left] - admittances[right],
                sum,
                "film-stack interface reflection"),
            checkedRatio(
                Real{2} * admittances[left],
                sum,
                "film-stack interface transmission"),
            checkedRatio(
                admittances[right] - admittances[left],
                sum,
                "film-stack reverse reflection"),
            checkedRatio(
                Real{2} * admittances[right],
                sum,
                "film-stack reverse transmission"),
        };
    };

    Complex reflection{};
    Complex transmission{};
    const bool singularInterface = [&] {
        for (std::size_t right = 1; right < admittances.size(); ++right) {
            if (interfaceIsSingular(admittances[right - 1], admittances[right])) {
                return true;
            }
        }
        return false;
    }();
    if (singularInterface) {
        std::tie(reflection, transmission) = characteristicFilmAmplitudes(
            admittances, longitudinalWaveNumbers, films, wavelengthUm);
    } else {
        const std::size_t substrateIndex = admittances.size() - 1;
        const auto lastInterface = interface(substrateIndex - 1, substrateIndex);
        reflection = lastInterface[0];
        transmission = lastInterface[1];
        for (std::size_t right = films.size(); right > 0; --right) {
            const auto boundary = interface(right - 1, right);
            const Complex phase = boundedExp(
                iu * twoPi * films[right - 1].thicknessUm *
                longitudinalWaveNumbers[right - 1] / wavelengthUm);
            const Complex roundTrip = phase * phase * reflection;
            const Complex denominator = Complex{1.0, 0.0} - boundary[2] * roundTrip;
            reflection = boundary[0] + checkedRatio(
                boundary[3] * boundary[1] * roundTrip,
                denominator,
                "film-stack recursive reflection");
            transmission = checkedRatio(
                transmission * phase * boundary[1],
                denominator,
                "film-stack recursive transmission");
        }
    }

    const Complex yi = admittances.front();
    const Complex ys = admittances.back();
    const Real yiReal = std::real(yi);
    const Real ysReal = std::real(ys);
    const Real scale = std::max({Real{1}, std::abs(yi), std::abs(ys)});
    const Real transmittance =
        yiReal > std::numeric_limits<Real>::epsilon() * scale &&
        ysReal > std::numeric_limits<Real>::epsilon() * scale
            ? (ysReal / yiReal) * std::norm(transmission)
            : Real{0};
    return {reflection, transmission, std::norm(reflection), transmittance};
}

} // namespace rcwa
