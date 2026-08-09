#include "rcwa/material.hpp"
#include "rcwa/wavevector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rcwa {

namespace {

Real minimumHermitianEigenvalue(const Tensor3& tensor) {
    const Real a00 = std::real(tensor(0, 0));
    const Real a11 = std::real(tensor(1, 1));
    const Real a22 = std::real(tensor(2, 2));
    const Real offDiagonalNormSquared =
        std::norm(tensor(0, 1)) +
        std::norm(tensor(0, 2)) +
        std::norm(tensor(1, 2));
    if (offDiagonalNormSquared == Real{0}) {
        return std::min({a00, a11, a22});
    }

    // Stable closed form for the eigenvalues of a 3x3 Hermitian matrix.
    // See Kopp, Int. J. Mod. Phys. C 19, 523-548 (2008), Sec. 3.
    const Real q = (a00 + a11 + a22) / Real{3};
    const Real pSquared = (
        (a00 - q) * (a00 - q) +
        (a11 - q) * (a11 - q) +
        (a22 - q) * (a22 - q) +
        Real{2} * offDiagonalNormSquared) / Real{6};
    if (!(pSquared > Real{0})) {
        return q;
    }
    const Real p = std::sqrt(pSquared);

    Tensor3 normalized = tensor;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            normalized(row, column) /= p;
        }
        normalized(row, row) -= q / p;
    }
    const Complex determinant =
        normalized(0, 0) * (
            normalized(1, 1) * normalized(2, 2) -
            normalized(1, 2) * normalized(2, 1)) -
        normalized(0, 1) * (
            normalized(1, 0) * normalized(2, 2) -
            normalized(1, 2) * normalized(2, 0)) +
        normalized(0, 2) * (
            normalized(1, 0) * normalized(2, 1) -
            normalized(1, 1) * normalized(2, 0));
    const Real r = std::clamp(std::real(determinant) / Real{2}, Real{-1}, Real{1});
    const Real phi = std::acos(r) / Real{3};
    return q + Real{2} * p *
        std::cos(phi + Real{2} * std::numbers::pi_v<Real> / Real{3});
}

Tensor3 hermitianLossOperator(const Tensor3& tensor) {
    Tensor3 loss;
    const Complex inverseTwoI{0.0, -0.5};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            loss(row, column) = inverseTwoI * (
                tensor(row, column) - std::conj(tensor(column, row)));
        }
    }
    return loss;
}

Real passivityScale(const Tensor3& epsilonLoss, const Tensor3& muLoss) {
    Real scale{1};
    for (const Complex value : epsilonLoss.v) {
        scale = std::max(scale, std::abs(value));
    }
    for (const Complex value : muLoss.v) {
        scale = std::max(scale, std::abs(value));
    }
    return scale;
}

void validateComplexMaterialValue(Complex value,
                                  const std::string& materialName,
                                  const char* quantity) {
    if (!isFinite(value)) {
        throw std::invalid_argument(
            "material '" + materialName + "' " + quantity + " must be finite");
    }
}

void validateMaterialTensor(const Tensor3& tensor,
                            const std::string& materialName,
                            const char* quantity) {
    for (const Complex value : tensor.v) {
        if (!isFinite(value)) {
            throw std::invalid_argument(
                "material '" + materialName + "' " + quantity +
                " tensor must contain only finite values");
        }
    }
}

void validateMaterialTensors(const std::pair<Tensor3, Tensor3>& tensors,
                             const std::string& materialName) {
    validateMaterialTensor(tensors.first, materialName, "epsilon");
    validateMaterialTensor(tensors.second, materialName, "mu");
}

} // namespace

MaterialPassivityResult materialPassivity(const Tensor3& epsilon,
                                           const Tensor3& mu) {
    validateMaterialTensor(epsilon, "passivity input", "epsilon");
    validateMaterialTensor(mu, "passivity input", "mu");
    const Tensor3 epsilonLoss = hermitianLossOperator(epsilon);
    const Tensor3 muLoss = hermitianLossOperator(mu);
    const Real tolerance = passivityScale(epsilonLoss, muLoss) *
#ifdef RCWA_SINGLE_PRECISION
        Real{5e-5f};
#else
        Real{1e-11};
#endif
    const Real epsilonMinimum = minimumHermitianEigenvalue(epsilonLoss);
    const Real muMinimum = minimumHermitianEigenvalue(muLoss);
    return {
        epsilonMinimum,
        muMinimum,
        tolerance,
        epsilonMinimum >= -tolerance && muMinimum >= -tolerance,
    };
}

struct Material::TensorCache {
    mutable std::mutex mutex;
    mutable std::unordered_map<Real, std::pair<Tensor3, Tensor3>> values;
};

Material::Material(std::string n, Complex e, Complex m)
    : mName(std::move(n)), mEps(e), mMu(m) {}

Material::Material(std::string n, DispersionFn fn)
    : mName(std::move(n)),
      mDispersion(std::move(fn)),
      mTensorCache(std::make_shared<TensorCache>()) {}

Material::Material(std::string n, Tensor3 e, Tensor3 m)
    : mName(std::move(n)), mEpsTensor(e), mMuTensor(m), mTensorMaterial(true) {}

Material::Material(std::string n, TensorDispersionFn fn)
    : mName(std::move(n)),
      mTensorDispersion(std::move(fn)),
      mTensorCache(std::make_shared<TensorCache>()),
      mTensorMaterial(true) {}

Material Material::constant(std::string name, Complex epsilon, Complex mu) {
    validateComplexMaterialValue(epsilon, name, "epsilon");
    validateComplexMaterialValue(mu, name, "mu");
    return Material(std::move(name), epsilon, mu);
}

Material Material::fromIndex(std::string name, Complex nComplex) {
    validateComplexMaterialValue(nComplex, name, "refractive index");
    return constant(std::move(name), nComplex * nComplex, {1.0, 0.0});
}

Material Material::tabulatedIndex(
    std::string name,
    std::vector<Real> wavelengthsUm,
    std::vector<Complex> refractiveIndices,
    MaterialExtrapolation extrapolation) {
    if (wavelengthsUm.size() < 2 ||
        wavelengthsUm.size() != refractiveIndices.size()) {
        throw std::invalid_argument(
            "tabulated material '" + name +
            "' requires equal wavelength/index arrays with at least two samples");
    }

    for (std::size_t index = 0; index < wavelengthsUm.size(); ++index) {
        const Real wavelengthUm = wavelengthsUm[index];
        const Complex refractiveIndex = refractiveIndices[index];
        if (!std::isfinite(wavelengthUm) || wavelengthUm <= Real{0} ||
            !isFinite(refractiveIndex) || std::real(refractiveIndex) < Real{0} ||
            std::imag(refractiveIndex) < Real{0}) {
            throw std::invalid_argument(
                "tabulated material '" + name + "' sample " +
                std::to_string(index + 1) +
                " must have a positive wavelength and finite non-negative n/k values");
        }
        if (index > 0 && wavelengthUm <= wavelengthsUm[index - 1]) {
            throw std::invalid_argument(
                "tabulated material '" + name +
                "' wavelengths must be strictly increasing");
        }
    }

    const bool clamp = extrapolation == MaterialExtrapolation::Clamp;
    const std::string materialName = name;
    return Material::dispersive(
        std::move(name),
        [materialName,
         wavelengthsUm = std::move(wavelengthsUm),
         refractiveIndices = std::move(refractiveIndices),
         clamp](Real wavelengthUm) {
            if (wavelengthUm < wavelengthsUm.front() ||
                wavelengthUm > wavelengthsUm.back()) {
                if (!clamp) {
                    std::ostringstream message;
                    message << "material '" << materialName << "' wavelength "
                            << wavelengthUm << " um is outside the nk data range ["
                            << wavelengthsUm.front() << ", " << wavelengthsUm.back()
                            << "] um";
                    throw std::invalid_argument(message.str());
                }
                wavelengthUm = std::clamp(
                    wavelengthUm, wavelengthsUm.front(), wavelengthsUm.back());
            }

            const auto upper = std::upper_bound(
                wavelengthsUm.begin(), wavelengthsUm.end(), wavelengthUm);
            const std::size_t upperIndex = static_cast<std::size_t>(
                std::distance(wavelengthsUm.begin(), upper));
            Complex refractiveIndex;
            if (upperIndex == 0) {
                refractiveIndex = refractiveIndices.front();
            } else if (upperIndex >= wavelengthsUm.size()) {
                refractiveIndex = refractiveIndices.back();
            } else {
                const std::size_t lowerIndex = upperIndex - 1;
                const Real fraction =
                    (wavelengthUm - wavelengthsUm[lowerIndex]) /
                    (wavelengthsUm[upperIndex] - wavelengthsUm[lowerIndex]);
                refractiveIndex =
                    refractiveIndices[lowerIndex] +
                    fraction * (refractiveIndices[upperIndex] -
                                refractiveIndices[lowerIndex]);
            }
            return std::make_pair(
                refractiveIndex * refractiveIndex, Complex{1.0, 0.0});
        });
}

Material Material::fromNkFile(
    std::string name,
    const std::filesystem::path& path,
    Real wavelengthScaleToUm,
    MaterialExtrapolation extrapolation) {
    if (!std::isfinite(wavelengthScaleToUm) || wavelengthScaleToUm <= Real{0}) {
        throw std::invalid_argument("nk wavelength scale must be positive and finite");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open nk material file: " + path.string());
    }

    std::vector<Real> wavelengthsUm;
    std::vector<Complex> refractiveIndices;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        std::istringstream row(line);
        Real wavelength{};
        Real n{};
        Real k{};
        if (!(row >> wavelength >> n >> k)) {
            throw std::invalid_argument(
                "nk file '" + path.string() + "' line " +
                std::to_string(lineNumber) +
                " must contain exactly three numeric columns: wavelength, n, k");
        }
        row >> std::ws;
        if (!row.eof()) {
            throw std::invalid_argument(
                "nk file '" + path.string() + "' line " +
                std::to_string(lineNumber) +
                " must contain exactly three numeric columns: wavelength, n, k");
        }
        wavelengthsUm.push_back(wavelength * wavelengthScaleToUm);
        refractiveIndices.emplace_back(n, k);
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading nk material file: " + path.string());
    }

    return tabulatedIndex(
        std::move(name),
        std::move(wavelengthsUm),
        std::move(refractiveIndices),
        extrapolation);
}

Material Material::dispersive(std::string name, DispersionFn fn) {
    if (!fn) {
        throw std::invalid_argument(
            "material '" + name + "' dispersion callback must not be empty");
    }
    return Material(std::move(name), std::move(fn));
}

Material Material::anisotropic(std::string name, Tensor3 epsilon, Tensor3 mu) {
    validateMaterialTensor(epsilon, name, "epsilon");
    validateMaterialTensor(mu, name, "mu");
    return Material(std::move(name), epsilon, mu);
}

Material Material::anisotropicDispersive(std::string name, TensorDispersionFn fn) {
    if (!fn) {
        throw std::invalid_argument(
            "material '" + name + "' tensor dispersion callback must not be empty");
    }
    return Material(std::move(name), std::move(fn));
}

Complex Material::epsilon(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    if (mTensorMaterial) {
        const Tensor3 eps = epsilonTensor(wavelengthUm);
        return (eps(0, 0) + eps(1, 1) + eps(2, 2)) / Complex{3.0, 0.0};
    }
    if (mDispersion) {
        const Tensor3 eps = tensors(wavelengthUm).first;
        return (eps(0, 0) + eps(1, 1) + eps(2, 2)) / Complex{3.0, 0.0};
    }
    return mEps;
}

Complex Material::mu(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    if (mTensorMaterial) {
        const Tensor3 muv = muTensor(wavelengthUm);
        return (muv(0, 0) + muv(1, 1) + muv(2, 2)) / Complex{3.0, 0.0};
    }
    if (mDispersion) {
        const Tensor3 muv = tensors(wavelengthUm).second;
        return (muv(0, 0) + muv(1, 1) + muv(2, 2)) / Complex{3.0, 0.0};
    }
    return mMu;
}

std::pair<Tensor3, Tensor3> Material::tensors(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    if (mTensorDispersion) {
        {
            std::lock_guard<std::mutex> lock(mTensorCache->mutex);
            const auto found = mTensorCache->values.find(wavelengthUm);
            if (found != mTensorCache->values.end()) {
                return found->second;
            }
        }
        const auto tensors = mTensorDispersion(wavelengthUm);
        validateMaterialTensors(tensors, mName);
        {
            std::lock_guard<std::mutex> lock(mTensorCache->mutex);
            const auto [it, inserted] = mTensorCache->values.emplace(wavelengthUm, tensors);
            (void)inserted;
            return it->second;
        }
    }
    if (mTensorMaterial) {
        return {mEpsTensor, mMuTensor};
    }
    if (mDispersion) {
        {
            std::lock_guard<std::mutex> lock(mTensorCache->mutex);
            const auto found = mTensorCache->values.find(wavelengthUm);
            if (found != mTensorCache->values.end()) {
                return found->second;
            }
        }
        const auto dispersionValues = mDispersion(wavelengthUm);
        const auto tensors = std::make_pair(
            Tensor3::isotropic(dispersionValues.first),
            Tensor3::isotropic(dispersionValues.second));
        validateMaterialTensors(tensors, mName);
        {
            std::lock_guard<std::mutex> lock(mTensorCache->mutex);
            const auto [it, inserted] = mTensorCache->values.emplace(wavelengthUm, tensors);
            (void)inserted;
            return it->second;
        }
    }
    return {Tensor3::isotropic(mEps), Tensor3::isotropic(mMu)};
}

Tensor3 Material::epsilonTensor(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    if (mTensorDispersion || mDispersion) {
        return tensors(wavelengthUm).first;
    }
    if (mTensorMaterial) {
        return mEpsTensor;
    }
    return Tensor3::isotropic(epsilon(wavelengthUm));
}

Tensor3 Material::muTensor(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    if (mTensorDispersion || mDispersion) {
        return tensors(wavelengthUm).second;
    }
    if (mTensorMaterial) {
        return mMuTensor;
    }
    return Tensor3::isotropic(mu(wavelengthUm));
}

MaterialPassivityResult Material::passivity(Real wavelengthUm) const {
    const auto [eps, muv] = tensors(wavelengthUm);
    return materialPassivity(eps, muv);
}

Complex Material::refractiveIndex(Real wavelengthUm) const {
    requirePositive(wavelengthUm, "material wavelength");
    const auto [eps, muv] = tensors(wavelengthUm);
    const Complex epsScalar = (eps(0, 0) + eps(1, 1) + eps(2, 2)) / Complex{3.0, 0.0};
    const Complex muScalar = (muv(0, 0) + muv(1, 1) + muv(2, 2)) / Complex{3.0, 0.0};
    return outgoingSqrt(epsScalar * muScalar);
}

} // namespace rcwa

namespace rcwa {

// Built-in model formulas stay in material.cpp so Material has one
// implementation unit while bindings only translate user input.
namespace {

enum class MaterialModelKind {
    Air,
    Silicon,
    Germanium,
    GrapheneKubo,
    Drude,
    PhononPolariton,
    Yig,
    Wsm,
    BlackPhosphorusDrude,
    InAsMagneto,
    CdTeLorentzDrude,
    InSbLorentzDrude,
    MoS2,
    Unknown,
};

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

MaterialModelKind materialModelKindFromKey(const std::string& key) {
    if (key == "vacuum" || key == "air") {
        return MaterialModelKind::Air;
    }
    if (key == "si" || key == "silicon") {
        return MaterialModelKind::Silicon;
    }
    if (key == "ge" || key == "germanium") {
        return MaterialModelKind::Germanium;
    }
    if (key == "graphene_kubo" || key == "graphenekubo") {
        return MaterialModelKind::GrapheneKubo;
    }
    if (key == "drude" || key == "metal_drude" || key == "metaldrude") {
        return MaterialModelKind::Drude;
    }
    if (key == "phonon_polariton" || key == "phononpolariton" ||
        key == "lorentz_phonon" || key == "lorentzphonon") {
        return MaterialModelKind::PhononPolariton;
    }
    if (key == "yig" || key == "ceyig" || key == "ce:yig" || key == "ce-yig" ||
        key == "magneto_optical_yig" || key == "magnetoopticalyig" ||
        key == "magneto_optic_yig" || key == "magnetoopticyig") {
        return MaterialModelKind::Yig;
    }
    if (key == "wsm" || key == "weyl_semimetal" || key == "weyl-semimetal") {
        return MaterialModelKind::Wsm;
    }
    if (key == "bp" || key == "black_phosphorus" || key == "black-phosphorus" ||
        key == "blackphosphorus" || key == "black_phosphorus_drude" ||
        key == "black-phosphorus-drude" || key == "blackphosphorusdrude" ||
        key == "phosphorene" || key == "phosphorene_drude") {
        return MaterialModelKind::BlackPhosphorusDrude;
    }
    if (key == "inas_magneto" || key == "inasmagneto" || key == "inas") {
        return MaterialModelKind::InAsMagneto;
    }
    if (key == "cdte_lorentz_drude" || key == "cdtelorentzdrude" ||
        key == "cdte_magneto" || key == "cdtemagneto" || key == "cdte") {
        return MaterialModelKind::CdTeLorentzDrude;
    }
    if (key == "insb_lorentz_drude" || key == "insblorentzdrude" ||
        key == "insb_magneto" || key == "insbmagneto" || key == "insb") {
        return MaterialModelKind::InSbLorentzDrude;
    }
    if (key == "mos2" || key == "mo_s2" || key == "molybdenum_disulfide") {
        return MaterialModelKind::MoS2;
    }
    return MaterialModelKind::Unknown;
}

MaterialModelKind materialModelKind(const std::string& model) {
    return materialModelKindFromKey(lowercase(model));
}

Real dawsonFunction(Real x) {
    if (x == Real{0}) {
        return Real{0};
    }

    const Real ax = std::abs(x);
    if (ax < Real{4.5}) {
        Real term = x;
        Real sum = term;
        for (int n = 1; n < 1024; ++n) {
            term *= -Real{2} * x * x / static_cast<Real>(2 * n + 1);
            sum += term;
            if (std::abs(term) < Real{1e-18} * std::max(Real{1}, std::abs(sum))) {
                break;
            }
        }
        return sum;
    }

    Real term = Real{1} / (Real{2} * x);
    Real sum = term;
    Real previousAbs = std::abs(term);
    for (int n = 1; n < 256; ++n) {
        term *= static_cast<Real>(2 * n - 1) / (Real{2} * x * x);
        const Real currentAbs = std::abs(term);
        if (currentAbs > previousAbs) {
            break;
        }
        sum += term;
        previousAbs = currentAbs;
    }
    return sum;
}

Complex mos2Epsilon(Real wavelengthUm) {
    constexpr Real epsilonInf = 4.44;
    constexpr Real plasmaEnergyEv = 0.0046075;
    constexpr Real temperatureK = 300.0;
    constexpr Real boltzmannEvPerK = 8.621738e-05;
    constexpr Real gaussianAlpha = 23.224;
    constexpr Real gaussianCenterEv = 2.7982;
    constexpr Real gaussianSigmaEv = 0.30893;

    constexpr std::array<Real, 6> oscillatorEnergiesEv{
        0.0,
        1.895,
        2.06532481,
        2.97493578,
        4.10199274,
        4.34,
    };
    std::array<Real, 6> oscillatorStrength{
        200890.0,
        57534.0,
        81496.0,
        82293.0,
        331300.0,
        4390600.0,
    };
    std::array<Real, 6> oscillatorGammaEv{
        0.010853,
        0.059099,
        0.11302,
        0.11957,
        0.28322,
        0.78515,
    };

    const Real kbT = boltzmannEvPerK * temperatureK;
    const Real ddd = std::exp(
        -((Real{2} * Real{0} - kbT) * (Real{2} * Real{0} - kbT)) /
        (boltzmannEvPerK * temperatureK));
    oscillatorStrength[1] *= ddd * ddd * ddd * ddd;
    oscillatorStrength[2] *= ddd;
    oscillatorStrength[3] *= ddd;
    oscillatorStrength[4] *= ddd;
    oscillatorGammaEv[1] /= ddd;
    oscillatorGammaEv[2] /= ddd;
    oscillatorGammaEv[3] /= ddd;
    oscillatorGammaEv[4] /= ddd;

    const Real photonEnergyEv = Real{1.24} / wavelengthUm;
    const Real gaussianCenterShiftedEv = gaussianCenterEv - kbT;
    const Real gaussianImag = gaussianAlpha * std::exp(
        -((photonEnergyEv - gaussianCenterShiftedEv) *
          (photonEnergyEv - gaussianCenterShiftedEv)) /
        (Real{2} * gaussianSigmaEv * gaussianSigmaEv));
    const Real gaussianReal = -gaussianAlpha / std::sqrt(std::numbers::pi_v<Real>) *
        (
            dawsonFunction(
                (photonEnergyEv - gaussianCenterShiftedEv) /
                (std::sqrt(Real{2}) * gaussianSigmaEv)) +
            dawsonFunction(
                (-photonEnergyEv - gaussianCenterShiftedEv) /
                (std::sqrt(Real{2}) * gaussianSigmaEv)));

    Complex epsilon{epsilonInf, 0.0};
    const Real plasmaEnergySquaredEv2 = plasmaEnergyEv * plasmaEnergyEv;
    for (std::size_t i = 0; i < oscillatorEnergiesEv.size(); ++i) {
        const Real resonanceEv = oscillatorEnergiesEv[i];
        const Real gammaEv = oscillatorGammaEv[i];
        epsilon += oscillatorStrength[i] * plasmaEnergySquaredEv2 /
            (resonanceEv * resonanceEv -
             photonEnergyEv * photonEnergyEv -
             Complex{0.0, 1.0} * gammaEv * photonEnergyEv);
    }
    epsilon += Complex{gaussianReal, gaussianImag};
    return epsilon;
}

Real angularFrequencyFromWavelengthUm(Real wavelengthUm) {
    return Real{2.0} * rcwa::pi * Real{299792458.0} / (wavelengthUm * Real{1e-6});
}

Complex fermiDirac(Complex energyJoule, Real efJoule, Real temperatureK) {
    const Real kB = Real{1.380649e-23};
    return Complex{1.0, 0.0} /
        (std::exp((energyJoule - Complex{efJoule, 0.0}) / (kB * temperatureK)) +
         Complex{1.0, 0.0});
}

struct WsmQuadratureNode {
    Real xi{};
    Real fourXiSquared{};
    Real weight{};
    Complex g2{};
};

struct WsmQuadrature {
    Real dx{};
    std::vector<WsmQuadratureNode> nodes;
};

struct WsmModelCache {
    Real eps0{};
    Real hbar{};
    Real ef{};
    Real tauInv{};
    Real temperatureK{};
    Real epsB{};
    Real f1Scale{};
    Complex f2{};
    Real f3Scale{};
    Real epsAPrefactor{};
    std::array<Real, 3> nodeSeparationDirection{};
    WsmQuadrature quadrature;
};

WsmQuadrature makeWsmQuadrature(WsmModelParameters parameters, Real efJoule) {
    constexpr int xiCount = 1500;

    WsmQuadrature out;
    out.dx = parameters.cutoffXi / static_cast<Real>(xiCount - 1);
    out.nodes.reserve(xiCount);

    for (int i = 0; i < xiCount; ++i) {
        const Real xi = out.dx * static_cast<Real>(i);
        out.nodes.push_back({
            xi,
            Real{4.0} * xi * xi,
            (i == 0 || i == xiCount - 1) ? Real{0.5} : Real{1.0},
            fermiDirac(Complex{-efJoule * xi, 0.0}, efJoule, parameters.temperatureK) -
                fermiDirac(Complex{efJoule * xi, 0.0}, efJoule, parameters.temperatureK),
        });
    }
    return out;
}

WsmModelCache makeWsmModelCache(WsmModelParameters parameters) {
    const Real eps0 = Real{8.8541878128e-12};
    const Real hbar = Real{1.054571817e-34};
    const Real electronCharge = Real{1.602176634e-19};
    const Real kB = Real{1.380649e-23};
    const Real b = parameters.nodeSeparationMInv;
    const Real vF = parameters.fermiVelocityMS;
    const Real g = parameters.degeneracy;
    const Real tau = parameters.relaxationTimeS;
    const Real ef = parameters.fermiEnergyEv * electronCharge;
    const Real rs =
        electronCharge * electronCharge / (Real{4.0} * rcwa::pi * eps0 * hbar * vF);

    WsmModelCache cache;
    cache.eps0 = eps0;
    cache.hbar = hbar;
    cache.ef = ef;
    cache.tauInv = Real{1.0} / tau;
    cache.temperatureK = parameters.temperatureK;
    cache.epsB = parameters.backgroundEpsilon;
    cache.f1Scale = eps0 * rs * g * ef / (Real{6.0} * hbar);
    cache.f2 = Complex{0.0, 1.0} * eps0 * rs * g * ef /
        (Real{6.0} * rcwa::pi * hbar);
    cache.f3Scale =
        (Real{1.0} + (rcwa::pi * rcwa::pi / Real{3.0}) *
                        std::pow(kB * parameters.temperatureK / ef, Real{2.0})) *
        Real{4.0};
    cache.epsAPrefactor =
        b * electronCharge * electronCharge /
        (Real{2.0} * rcwa::pi * rcwa::pi * hbar * eps0);
    const Real directionNorm = std::sqrt(
        parameters.nodeSeparationDirection[0] * parameters.nodeSeparationDirection[0] +
        parameters.nodeSeparationDirection[1] * parameters.nodeSeparationDirection[1] +
        parameters.nodeSeparationDirection[2] * parameters.nodeSeparationDirection[2]);
    for (std::size_t axis = 0; axis < cache.nodeSeparationDirection.size(); ++axis) {
        cache.nodeSeparationDirection[axis] =
            parameters.nodeSeparationDirection[axis] / directionNorm;
    }
    cache.quadrature = makeWsmQuadrature(parameters, ef);
    return cache;
}

rcwa::Tensor3 wsmEpsilonTensor(Real wavelengthUm, const WsmModelCache& cache) {
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    const Complex omegaBar =
        cache.hbar * (Complex{omega, 0.0} + Complex{0.0, cache.tauInv}) / cache.ef;
    const Complex g1 =
        fermiDirac(-cache.ef * omegaBar / Real{2.0}, cache.ef, cache.temperatureK) -
        fermiDirac(cache.ef * omegaBar / Real{2.0}, cache.ef, cache.temperatureK);
    const Complex f1 = cache.f1Scale * omegaBar * g1;
    const Complex f3 = cache.f3Scale / omegaBar;
    const Complex omegaBarSquared = omegaBar * omegaBar;

    Complex integral{};
    const auto& q = cache.quadrature;
    for (const auto& node : q.nodes) {
        const Complex integrand =
            ((node.g2 - g1) / (omegaBarSquared - node.fourXiSquared)) * node.xi;
        integral += node.weight * integrand;
    }
    const Complex f4 = Real{8.0} * omegaBar * q.dx * integral;
    const Complex sigma = f1 + cache.f2 * (f3 + f4);
    const Complex epsD =
        cache.epsB + Complex{0.0, 1.0} * sigma / (omega * cache.eps0);
    const Real epsA = cache.epsAPrefactor / omega;

    auto eps = rcwa::Tensor3::isotropic(epsD);
    const Real bx = cache.nodeSeparationDirection[0];
    const Real by = cache.nodeSeparationDirection[1];
    const Real bz = cache.nodeSeparationDirection[2];
    eps(0, 1) = Complex{0.0, -epsA * bz};
    eps(0, 2) = Complex{0.0, epsA * by};
    eps(1, 0) = Complex{0.0, epsA * bz};
    eps(1, 2) = Complex{0.0, -epsA * bx};
    eps(2, 0) = Complex{0.0, -epsA * by};
    eps(2, 1) = Complex{0.0, epsA * bx};
    return eps;
}

Complex drudeEpsilon(Real wavelengthUm, DrudeModelParameters parameters) {
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    return parameters.epsInf -
        (parameters.plasmaFrequencyRadS * parameters.plasmaFrequencyRadS) /
            Complex{
                omega * omega,
                parameters.dampingRateRadS * omega,
            };
}

rcwa::Tensor3 phononPolaritonEpsilonTensor(
    Real wavelengthUm,
    PhononPolaritonModelParameters parameters) {
    const Real wavenumberCmInv = Real{1.0e4} / wavelengthUm;
    std::array<Complex, 3> epsilon{};
    for (std::size_t axis = 0; axis < epsilon.size(); ++axis) {
        const Real omegaTo = parameters.transverseWavenumberCmInv[axis];
        const Real omegaLo = parameters.longitudinalWavenumberCmInv[axis];
        const Real gamma = parameters.dampingWavenumberCmInv[axis];
        const Complex denominator{
            omegaTo * omegaTo - wavenumberCmInv * wavenumberCmInv,
            -wavenumberCmInv * gamma};
        epsilon[axis] = parameters.epsilonInf[axis] *
            (Complex{1.0, 0.0} +
             (omegaLo * omegaLo - omegaTo * omegaTo) / denominator);
    }

    const Real angleRad = rcwa::degToRad(parameters.twistDeg);
    const Real c = std::cos(angleRad);
    const Real s = std::sin(angleRad);
    rcwa::Tensor3 tensor =
        rcwa::Tensor3::diagonal(epsilon[0], epsilon[1], epsilon[2]);
    tensor(0, 0) = c * c * epsilon[0] + s * s * epsilon[1];
    tensor(1, 1) = s * s * epsilon[0] + c * c * epsilon[1];
    tensor(0, 1) = c * s * (epsilon[0] - epsilon[1]);
    tensor(1, 0) = tensor(0, 1);
    return tensor;
}

Complex blackPhosphorusSurfaceConductivityFromOmega(
    Real omega,
    BlackPhosphorusModelParameters parameters,
    Real effectiveMassKg) {
    const Real hbar = Real{1.054571817e-34};
    const Real electronCharge = Real{1.602176634e-19};
    const Real carrierDensityM2 = parameters.carrierDensityCm2 * Real{1.0e4};
    const Real scatteringRateJoule = parameters.scatteringRateEv * electronCharge;
    const Real drudeWeight =
        rcwa::pi * carrierDensityM2 * electronCharge * electronCharge /
        effectiveMassKg;
    return Complex{0.0, 1.0} * hbar * drudeWeight /
        (rcwa::pi * Complex{hbar * omega, scatteringRateJoule});
}

rcwa::Tensor3 blackPhosphorusEpsilonTensor(
    Real wavelengthUm,
    BlackPhosphorusModelParameters parameters) {
    const Real eps0 = Real{8.8541878128e-12};
    const Real hbar = Real{1.054571817e-34};
    const Real electronCharge = Real{1.602176634e-19};
    const Real electronMass = Real{9.1093837015e-31};
    const Real delta = parameters.bandgapEv * electronCharge;
    const Real gamma = Real{4.0} * parameters.latticeConstantM / rcwa::pi * electronCharge;
    const Real etaC = hbar * hbar / electronMass;
    const Real nuC = hbar * hbar / (Real{1.4} * electronMass);
    const Real massX = hbar * hbar /
        (Real{2.0} * gamma * gamma / delta + etaC);
    const Real massY = hbar * hbar / (Real{2.0} * nuC);
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    const Complex sigmaX =
        blackPhosphorusSurfaceConductivityFromOmega(omega, parameters, massX);
    const Complex sigmaY =
        blackPhosphorusSurfaceConductivityFromOmega(omega, parameters, massY);
    const Complex epsX =
        Complex{1.0, 0.0} +
        Complex{0.0, 1.0} * sigmaX / (eps0 * omega * parameters.thicknessM);
    const Complex epsY =
        Complex{1.0, 0.0} +
        Complex{0.0, 1.0} * sigmaY / (eps0 * omega * parameters.thicknessM);
    const Real angleRad = rcwa::degToRad(parameters.twistDeg);
    const Real c = std::cos(angleRad);
    const Real s = std::sin(angleRad);
    rcwa::Tensor3 eps = rcwa::Tensor3::diagonal(epsX, epsY, Complex{1.0, 0.0});
    eps(0, 0) = c * c * epsX + s * s * epsY;
    eps(1, 1) = s * s * epsX + c * c * epsY;
    eps(0, 1) = c * s * (epsX - epsY);
    eps(1, 0) = eps(0, 1);
    return eps;
}

rcwa::Tensor3 yigEpsilonTensor(const YigModelParameters& parameters) {
    const Real directionNorm = std::sqrt(
        parameters.magnetizationDirection[0] * parameters.magnetizationDirection[0] +
        parameters.magnetizationDirection[1] * parameters.magnetizationDirection[1] +
        parameters.magnetizationDirection[2] * parameters.magnetizationDirection[2]);
    const Real mx = parameters.magnetizationDirection[0] / directionNorm;
    const Real my = parameters.magnetizationDirection[1] / directionNorm;
    const Real mz = parameters.magnetizationDirection[2] / directionNorm;
    const Complex ig = Complex{0.0, 1.0} * parameters.offDiagonalGyration;

    auto eps = rcwa::Tensor3::isotropic(parameters.diagonalEpsilon);
    eps(0, 1) = ig * mz;
    eps(1, 0) = -ig * mz;
    eps(0, 2) = -ig * my;
    eps(2, 0) = ig * my;
    eps(1, 2) = ig * mx;
    eps(2, 1) = -ig * mx;
    return eps;
}

void validateWsmParameters(const WsmModelParameters& parameters) {
    if (!std::isfinite(parameters.nodeSeparationMInv)) {
        throw std::invalid_argument("WSM node_separation_m_inv must be finite");
    }
    Real directionNormSquared{};
    for (const Real component : parameters.nodeSeparationDirection) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "WSM node_separation_direction must contain finite values");
        }
        directionNormSquared += component * component;
    }
    if (!(directionNormSquared > Real{0.0})) {
        throw std::invalid_argument(
            "WSM node_separation_direction must be nonzero");
    }
    if (!(parameters.fermiVelocityMS > Real{0.0}) ||
        !std::isfinite(parameters.fermiVelocityMS)) {
        throw std::invalid_argument("WSM fermi_velocity_m_s must be positive and finite");
    }
    if (!(parameters.cutoffXi > Real{0.0}) || !std::isfinite(parameters.cutoffXi)) {
        throw std::invalid_argument("WSM cutoff_xi must be positive and finite");
    }
    if (!std::isfinite(parameters.backgroundEpsilon)) {
        throw std::invalid_argument("WSM background_epsilon must be finite");
    }
    if (!(parameters.temperatureK > Real{0.0}) || !std::isfinite(parameters.temperatureK)) {
        throw std::invalid_argument("WSM temperature_k must be positive and finite");
    }
    if (!(parameters.degeneracy > Real{0.0}) || !std::isfinite(parameters.degeneracy)) {
        throw std::invalid_argument("WSM degeneracy must be positive and finite");
    }
    if (!(parameters.relaxationTimeS > Real{0.0}) ||
        !std::isfinite(parameters.relaxationTimeS)) {
        throw std::invalid_argument("WSM relaxation_time_s must be positive and finite");
    }
    if (!(parameters.fermiEnergyEv > Real{0.0}) ||
        !std::isfinite(parameters.fermiEnergyEv)) {
        throw std::invalid_argument("WSM fermi_energy_ev must be positive and finite");
    }
}

void requireFiniteParameter(Real value, const char* model, const char* field) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(model) + " " + field + " must be finite");
    }
}

void requirePositiveFiniteParameter(Real value, const char* model, const char* field) {
    if (!(value > Real{0.0}) || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(model) + " " + field + " must be positive and finite");
    }
}

void validatePhononPolaritonParameters(
    const PhononPolaritonModelParameters& parameters) {
    for (std::size_t axis = 0; axis < parameters.epsilonInf.size(); ++axis) {
        requirePositiveFiniteParameter(
            parameters.epsilonInf[axis], "PhononPolariton", "epsilon_inf");
        requirePositiveFiniteParameter(
            parameters.transverseWavenumberCmInv[axis],
            "PhononPolariton",
            "omega_to_cm_inv");
        requirePositiveFiniteParameter(
            parameters.longitudinalWavenumberCmInv[axis],
            "PhononPolariton",
            "omega_lo_cm_inv");
        requirePositiveFiniteParameter(
            parameters.dampingWavenumberCmInv[axis],
            "PhononPolariton",
            "damping_cm_inv");
        if (!(parameters.longitudinalWavenumberCmInv[axis] >
              parameters.transverseWavenumberCmInv[axis])) {
            throw std::invalid_argument(
                "PhononPolariton omega_lo_cm_inv must exceed omega_to_cm_inv "
                "on every axis");
        }
    }
    requireFiniteParameter(parameters.twistDeg, "PhononPolariton", "twist_deg");
}

void validateYigParameters(const YigModelParameters& parameters) {
    if (!isFinite(parameters.diagonalEpsilon)) {
        throw std::invalid_argument("YIG diagonal_epsilon must be finite");
    }
    if (!isFinite(parameters.offDiagonalGyration)) {
        throw std::invalid_argument("YIG offdiagonal_gyration must be finite");
    }
    Real directionNormSquared{};
    for (const Real component : parameters.magnetizationDirection) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "YIG magnetization_direction must contain finite values");
        }
        directionNormSquared += component * component;
    }
    if (!(directionNormSquared > Real{0.0})) {
        throw std::invalid_argument("YIG magnetization_direction must be nonzero");
    }
}

void validateBlackPhosphorusParameters(
    const BlackPhosphorusModelParameters& parameters) {
    requirePositiveFiniteParameter(
        parameters.carrierDensityCm2,
        "BlackPhosphorusDrude",
        "carrier_density_cm2");
    requirePositiveFiniteParameter(
        parameters.scatteringRateEv,
        "BlackPhosphorusDrude",
        "scattering_rate_ev");
    requirePositiveFiniteParameter(
        parameters.thicknessM,
        "BlackPhosphorusDrude",
        "thickness_m");
    requirePositiveFiniteParameter(
        parameters.bandgapEv,
        "BlackPhosphorusDrude",
        "bandgap_ev");
    requirePositiveFiniteParameter(
        parameters.latticeConstantM,
        "BlackPhosphorusDrude",
        "lattice_constant_m");
    requireFiniteParameter(
        parameters.twistDeg,
        "BlackPhosphorusDrude",
        "twistDeg");
}

template <typename Parameters>
void validateCarrierMagnetoParameters(const Parameters& parameters, const char* model) {
    requireFiniteParameter(parameters.magneticFieldT, model, "magnetic_field_t");
    requireFiniteParameter(parameters.epsInf, model, "eps_inf");
    requirePositiveFiniteParameter(
        parameters.carrierDensityCm3,
        model,
        "carrier_density_cm3");
    requirePositiveFiniteParameter(
        parameters.effectiveMassRatio,
        model,
        "effective_mass_ratio");
    requirePositiveFiniteParameter(
        parameters.dampingRateRadS,
        model,
        "damping_rate_rad_s");
}

void validateInasParameters(const InAsMagnetoModelParameters& parameters) {
    validateCarrierMagnetoParameters(parameters, "InAsMagneto");
}

template <typename Parameters>
void validateLorentzDrudeParameters(const Parameters& parameters, const char* model) {
    validateCarrierMagnetoParameters(parameters, model);
    requirePositiveFiniteParameter(
        parameters.transverseResonanceRadS,
        model,
        "transverse_resonance_rad_s");
}

void validateCdteLorentzDrudeParameters(
    const CdTeLorentzDrudeModelParameters& parameters) {
    validateLorentzDrudeParameters(parameters, "CdTeLorentzDrude");
}

void validateInsbLorentzDrudeParameters(
    const InSbLorentzDrudeModelParameters& parameters) {
    validateLorentzDrudeParameters(parameters, "InSbLorentzDrude");
}

void validateGrapheneParameters(const GrapheneModelParameters& parameters) {
    if (!std::isnan(parameters.fermiEnergyEv) &&
        !std::isfinite(parameters.fermiEnergyEv)) {
        throw std::invalid_argument("graphene fermi_energy_ev must be finite");
    }
    if (!(parameters.temperatureK > Real{0.0}) || !std::isfinite(parameters.temperatureK)) {
        throw std::invalid_argument("graphene temperature_k must be positive and finite");
    }
    if (!(parameters.relaxationTimeS > Real{0.0}) ||
        !std::isfinite(parameters.relaxationTimeS)) {
        throw std::invalid_argument("graphene relaxation_time_s must be positive and finite");
    }
    if (!(parameters.thicknessM > Real{0.0}) || !std::isfinite(parameters.thicknessM)) {
        throw std::invalid_argument("graphene thickness_m must be positive and finite");
    }
}

Complex grapheneKuboEpsilon(Real wavelengthUm, GrapheneModelParameters parameters) {
    validateGrapheneParameters(parameters);
    const Real eps0 = Real{8.8541878128e-12};
    const Real hbar = Real{1.054571817e-34};
    const Real electronCharge = Real{1.602176634e-19};
    const Real kB = Real{1.380649e-23};
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    const Real efEv = std::isfinite(parameters.fermiEnergyEv)
        ? parameters.fermiEnergyEv
        : Real{1.0};
    const Real muC = efEv * electronCharge;
    const Complex omegaTilde = Complex{omega, Real{1.0} / parameters.relaxationTimeS};

    const Complex photonEnergy = hbar * omegaTilde;
    Complex interbandLog =
        std::log((Real{2.0} * std::abs(muC) - photonEnergy) /
                 (Real{2.0} * std::abs(muC) + photonEnergy));
    // Select the branch continuous with Ef>0 at the logarithm branch cut.
    if (std::imag(interbandLog) > Real{0.0}) {
        interbandLog -= Complex{0.0, Real{2.0} * rcwa::pi};
    }
    Complex sigma =
        Complex{0.0, 1.0} * electronCharge * electronCharge /
        (Real{4.0} * rcwa::pi * hbar) * interbandLog;

    const Real thermalEnergy = kB * parameters.temperatureK;
    // 2*log(2*cosh(mu/(2*kT))) written in an overflow-safe, explicitly
    // electron/hole symmetric form.
    const Real absMuC = std::abs(muC);
    const Real occupationFactor =
        absMuC / thermalEnergy +
        Real{2.0} * std::log1p(std::exp(-absMuC / thermalEnergy));
    sigma +=
        Complex{0.0, 1.0} * electronCharge * electronCharge * thermalEnergy /
        (rcwa::pi * hbar * hbar * omegaTilde) * occupationFactor;

    return Complex{1.0, 0.0} +
        Complex{0.0, 1.0} * sigma / (eps0 * omega * parameters.thicknessM);
}

rcwa::Tensor3 inasEpsilonTensor(Real wavelengthUm,
                                  InAsMagnetoModelParameters parameters) {
    const Real electronCharge = Real{1.602176634e-19};
    const Real electronMass = Real{9.1093837015e-31};
    const Real effectiveMass = parameters.effectiveMassRatio * electronMass;
    const Real eps0 = Real{8.8541878128e-12};
    const Real densityM3 = parameters.carrierDensityCm3 * Real{1.0e6};
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    const Real omegaP = std::sqrt(densityM3 * electronCharge * electronCharge /
                                   (eps0 * effectiveMass));
    const Real omegaC = electronCharge * parameters.magneticFieldT / effectiveMass;
    const Complex omegaTilde = Complex{omega, parameters.dampingRateRadS};
    const Complex denominator =
        omega * (omegaTilde * omegaTilde - omegaC * omegaC);
    const Complex epsXx =
        parameters.epsInf - omegaP * omegaP * omegaTilde / denominator;
    const Complex epsParallel =
        parameters.epsInf - omegaP * omegaP / (omega * omegaTilde);
    const Complex offdiag =
        Complex{0.0, 1.0} * omegaP * omegaP * omegaC / denominator;

    auto eps = rcwa::Tensor3::isotropic(epsXx);
    eps(1, 1) = epsParallel;
    eps(0, 2) = -offdiag;
    eps(2, 0) = offdiag;
    return eps;
}

template <typename Parameters>
rcwa::Tensor3 lorentzDrudeEpsilonTensor(Real wavelengthUm, Parameters parameters) {
    const Real electronCharge = Real{1.602176634e-19};
    const Real electronMass = Real{9.1093837015e-31};
    const Real effectiveMass = parameters.effectiveMassRatio * electronMass;
    const Real eps0 = Real{8.8541878128e-12};
    const Real densityM3 = parameters.carrierDensityCm3 * Real{1.0e6};
    const Real omega = angularFrequencyFromWavelengthUm(wavelengthUm);
    const Real gamma = parameters.dampingRateRadS;
    const Real omegaT = parameters.transverseResonanceRadS;
    const Real omegaP = std::sqrt(densityM3 * electronCharge * electronCharge /
                                   (eps0 * effectiveMass));
    const Real omegaC = electronCharge * parameters.magneticFieldT / effectiveMass;
    const Complex omegaTilde = Complex{omega, gamma};
    const Complex tensorDenominator =
        omega * (omegaTilde * omegaTilde - omegaC * omegaC) -
        omegaT * omegaT * omegaTilde;
    const Complex epsXx =
        parameters.epsInf *
        (Complex{1.0, 0.0} - omegaP * omegaP * omegaTilde / tensorDenominator);
    const Complex offdiag =
        Complex{0.0, 1.0} * omegaP * omegaP * omegaC / tensorDenominator;
    const Complex epsZz =
        parameters.epsInf *
        (Complex{1.0, 0.0} +
         omegaP * omegaP /
             Complex{omegaT * omegaT - omega * omega, -gamma * omega});

    auto eps = rcwa::Tensor3::isotropic(epsXx);
    eps(1, 1) = epsZz;
    // Project convention: exp(-i*omega*t), layers along +z, and positive
    // magneticFieldT means +B_y.  This is the same xz Hall-tensor orientation
    // used by InAsMagneto.
    eps(0, 2) = -offdiag;
    eps(2, 0) = offdiag;
    return eps;
}

rcwa::Tensor3 cdteLorentzDrudeEpsilonTensor(
    Real wavelengthUm,
    CdTeLorentzDrudeModelParameters parameters) {
    return lorentzDrudeEpsilonTensor(wavelengthUm, parameters);
}

rcwa::Tensor3 insbLorentzDrudeEpsilonTensor(
    Real wavelengthUm,
    InSbLorentzDrudeModelParameters parameters) {
    return lorentzDrudeEpsilonTensor(wavelengthUm, parameters);
}

} // namespace

bool isGrapheneMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::GrapheneKubo;
}

bool isDrudeMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::Drude;
}

bool isPhononPolaritonMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::PhononPolariton;
}

bool isYigMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::Yig;
}

bool isWsmMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::Wsm;
}

bool isBlackPhosphorusMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::BlackPhosphorusDrude;
}

bool isInasMagnetoMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::InAsMagneto;
}

bool isCdteLorentzDrudeMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::CdTeLorentzDrude;
}

bool isInsbLorentzDrudeMaterialModel(const std::string& model) {
    return materialModelKind(model) == MaterialModelKind::InSbLorentzDrude;
}

GrapheneModelParameters defaultGrapheneModelParameters(const std::string& model) {
    (void)model;
    return {};
}

DrudeModelParameters defaultDrudeModelParameters(const std::string& model) {
    (void)model;
    return {};
}

PhononPolaritonModelParameters defaultPhononPolaritonModelParameters(
    const std::string&) {
    return {};
}

YigModelParameters defaultYigModelParameters(const std::string&) {
    return {};
}

WsmModelParameters defaultWsmModelParameters(const std::string&) {
    return {};
}

BlackPhosphorusModelParameters defaultBlackPhosphorusModelParameters(
    const std::string&) {
    return {};
}

InAsMagnetoModelParameters defaultInasMagnetoModelParameters(const std::string&) {
    return {};
}

CdTeLorentzDrudeModelParameters defaultCdteLorentzDrudeModelParameters(
    const std::string&) {
    return {};
}

InSbLorentzDrudeModelParameters defaultInsbLorentzDrudeModelParameters(
    const std::string&) {
    return {};
}

rcwa::Material builtinMaterialModel(const std::string& name, const std::string& model) {
    switch (materialModelKind(model)) {
    case MaterialModelKind::Air:
        return rcwa::Material::constant(name, Complex{1.0, 0.0});
    case MaterialModelKind::Silicon:
        return rcwa::Material::fromIndex(name, Complex{3.48, 0.0});
    case MaterialModelKind::Germanium:
        return rcwa::Material::fromIndex(name, Complex{4.0, 0.0});
    case MaterialModelKind::GrapheneKubo:
        return builtinMaterialModel(name, model, defaultGrapheneModelParameters(model));
    case MaterialModelKind::Drude:
        return builtinMaterialModel(name, model, defaultDrudeModelParameters(model));
    case MaterialModelKind::PhononPolariton:
        return builtinMaterialModel(
            name,
            model,
            defaultPhononPolaritonModelParameters(model));
    case MaterialModelKind::Yig:
        return builtinMaterialModel(name, model, defaultYigModelParameters(model));
    case MaterialModelKind::Wsm:
        return builtinMaterialModel(name, model, defaultWsmModelParameters(model));
    case MaterialModelKind::BlackPhosphorusDrude:
        return builtinMaterialModel(
            name,
            model,
            defaultBlackPhosphorusModelParameters(model));
    case MaterialModelKind::InAsMagneto:
        return builtinMaterialModel(name, model, defaultInasMagnetoModelParameters(model));
    case MaterialModelKind::CdTeLorentzDrude:
        return builtinMaterialModel(
            name,
            model,
            defaultCdteLorentzDrudeModelParameters(model));
    case MaterialModelKind::InSbLorentzDrude:
        return builtinMaterialModel(
            name,
            model,
            defaultInsbLorentzDrudeModelParameters(model));
    case MaterialModelKind::MoS2:
        return rcwa::Material::dispersive(
            name,
            [](Real wavelengthUm) {
                return std::make_pair(mos2Epsilon(wavelengthUm), Complex{1.0, 0.0});
            });
    case MaterialModelKind::Unknown:
        break;
    }
    throw std::invalid_argument(
        "built-in material model must be Air, Si, Ge, YIG, WSM, BP, Drude, "
        "GrapheneKubo, PhononPolariton, InAsMagneto, CdTeLorentzDrude, "
        "InSbLorentzDrude, or MoS2");
}

rcwa::Material builtinMaterialModel(const std::string& name,
                                      const std::string& model,
                                      const GrapheneModelParameters& parameters) {
    if (isGrapheneMaterialModel(model)) {
        return rcwa::Material::dispersive(
            name,
            [parameters](Real wavelengthUm) {
                return std::make_pair(
                    grapheneKuboEpsilon(wavelengthUm, parameters),
                    Complex{1.0, 0.0});
            });
    }
    throw std::invalid_argument(
        "Graphene parameters can only be used with the GrapheneKubo material model");
}

rcwa::Material builtinMaterialModel(const std::string& name,
                                      const std::string& model,
                                      const DrudeModelParameters& parameters) {
    if (!isDrudeMaterialModel(model)) {
        throw std::invalid_argument("Drude parameters can only be used with Drude models");
    }
    if (!std::isfinite(parameters.epsInf)) {
        throw std::invalid_argument("Drude eps_inf must be finite");
    }
    if (!(parameters.plasmaFrequencyRadS > Real{0.0}) ||
        !std::isfinite(parameters.plasmaFrequencyRadS)) {
        throw std::invalid_argument("Drude plasma_frequency_rad_s must be positive and finite");
    }
    if (parameters.dampingRateRadS < Real{0.0} ||
        !std::isfinite(parameters.dampingRateRadS)) {
        throw std::invalid_argument(
            "Drude damping_rate_rad_s must be non-negative and finite");
    }
    return rcwa::Material::dispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                drudeEpsilon(wavelengthUm, parameters),
                Complex{1.0, 0.0});
        });
}

rcwa::Material builtinMaterialModel(
    const std::string& name,
    const std::string& model,
    const PhononPolaritonModelParameters& parameters) {
    if (!isPhononPolaritonMaterialModel(model)) {
        throw std::invalid_argument(
            "PhononPolariton parameters can only be used with PhononPolariton models");
    }
    validatePhononPolaritonParameters(parameters);
    return rcwa::Material::anisotropicDispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                phononPolaritonEpsilonTensor(wavelengthUm, parameters),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

rcwa::Material builtinMaterialModel(
    const std::string& name,
    const std::string& model,
    const YigModelParameters& parameters) {
    if (!isYigMaterialModel(model)) {
        throw std::invalid_argument("YIG parameters can only be used with YIG models");
    }
    validateYigParameters(parameters);
    return rcwa::Material::anisotropic(
        name,
        yigEpsilonTensor(parameters),
        rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
}

rcwa::Material builtinMaterialModel(const std::string& name,
                                      const std::string& model,
                                      const WsmModelParameters& parameters) {
    if (!isWsmMaterialModel(model)) {
        throw std::invalid_argument("WSM parameters can only be used with WSM models");
    }
    validateWsmParameters(parameters);
    const auto cache = std::make_shared<const WsmModelCache>(makeWsmModelCache(parameters));
    return rcwa::Material::anisotropicDispersive(
        name,
        [cache](Real wavelengthUm) {
            return std::make_pair(
                wsmEpsilonTensor(wavelengthUm, *cache),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

rcwa::Material builtinMaterialModel(
    const std::string& name,
    const std::string& model,
    const BlackPhosphorusModelParameters& parameters) {
    if (!isBlackPhosphorusMaterialModel(model)) {
        throw std::invalid_argument(
            "BlackPhosphorus parameters can only be used with BP models");
    }
    validateBlackPhosphorusParameters(parameters);
    return rcwa::Material::anisotropicDispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                blackPhosphorusEpsilonTensor(wavelengthUm, parameters),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

rcwa::Material builtinMaterialModel(const std::string& name,
                                      const std::string& model,
                                      const InAsMagnetoModelParameters& parameters) {
    if (!isInasMagnetoMaterialModel(model)) {
        throw std::invalid_argument(
            "InAsMagneto parameters can only be used with InAsMagneto models");
    }
    validateInasParameters(parameters);
    return rcwa::Material::anisotropicDispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                inasEpsilonTensor(wavelengthUm, parameters),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

rcwa::Material builtinMaterialModel(
    const std::string& name,
    const std::string& model,
    const CdTeLorentzDrudeModelParameters& parameters) {
    if (!isCdteLorentzDrudeMaterialModel(model)) {
        throw std::invalid_argument(
            "CdTeLorentzDrude parameters can only be used with CdTeLorentzDrude models");
    }
    validateCdteLorentzDrudeParameters(parameters);
    return rcwa::Material::anisotropicDispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                cdteLorentzDrudeEpsilonTensor(wavelengthUm, parameters),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

rcwa::Material builtinMaterialModel(
    const std::string& name,
    const std::string& model,
    const InSbLorentzDrudeModelParameters& parameters) {
    if (!isInsbLorentzDrudeMaterialModel(model)) {
        throw std::invalid_argument(
            "InSbLorentzDrude parameters can only be used with InSbLorentzDrude models");
    }
    validateInsbLorentzDrudeParameters(parameters);
    return rcwa::Material::anisotropicDispersive(
        name,
        [parameters](Real wavelengthUm) {
            return std::make_pair(
                insbLorentzDrudeEpsilonTensor(wavelengthUm, parameters),
                rcwa::Tensor3::isotropic(Complex{1.0, 0.0}));
        });
}

} // namespace rcwa
