#pragma once

#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rcwa/types.hpp"

namespace rcwa {

struct Tensor3 {
    std::array<Complex, 9> v{
        Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}
    };

    Complex& operator()(std::size_t r, std::size_t c) {
        return v[r * 3 + c];
    }

    const Complex& operator()(std::size_t r, std::size_t c) const {
        return v[r * 3 + c];
    }

    static Tensor3 diagonal(Complex xx, Complex yy, Complex zz) {
        Tensor3 t;
        t(0, 0) = xx; t(0, 1) = {}; t(0, 2) = {};
        t(1, 0) = {}; t(1, 1) = yy; t(1, 2) = {};
        t(2, 0) = {}; t(2, 1) = {}; t(2, 2) = zz;
        return t;
    }

    static Tensor3 isotropic(Complex value) {
        return diagonal(value, value, value);
    }
};

struct MaterialPassivityResult {
    Real epsilonMinimumLossEigenvalue{};
    Real muMinimumLossEigenvalue{};
    Real tolerance{};
    bool passive{};
};

// For the project-wide exp(-i*omega*t) convention, a local material without
// magnetoelectric coupling is passive when the Hermitian loss operators
// (epsilon-epsilon^H)/(2i) and (mu-mu^H)/(2i) are positive semidefinite.
[[nodiscard]] MaterialPassivityResult materialPassivity(
    const Tensor3& epsilon,
    const Tensor3& mu);

inline bool tensorIsScalar(const Tensor3& tensor,
                             Complex* value = nullptr,
                             Real tol =
#ifdef RCWA_SINGLE_PRECISION
                                 Real{1e-5}
#else
                                 Real{1e-12}
#endif
) {
    const Complex first = tensor(0, 0);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const Complex expected = (r == c) ? first : Complex{};
            if (std::abs(tensor(r, c) - expected) > tol) {
                return false;
            }
        }
    }
    if (value != nullptr) {
        *value = first;
    }
    return true;
}

using DispersionFn = std::function<std::pair<Complex, Complex>(Real wavelengthUm)>;
using TensorDispersionFn = std::function<std::pair<Tensor3, Tensor3>(Real wavelengthUm)>;

enum class MaterialExtrapolation {
    Error,
    Clamp,
};

class Material {
public:
    Material() = default;

    static Material constant(std::string name,
                             Complex epsilon,
                             Complex mu = {1.0, 0.0});

    static Material fromIndex(std::string name, Complex nComplex);

    static Material tabulatedIndex(
        std::string name,
        std::vector<Real> wavelengthsUm,
        std::vector<Complex> refractiveIndices,
        MaterialExtrapolation extrapolation = MaterialExtrapolation::Error);

    static Material fromNkFile(
        std::string name,
        const std::filesystem::path& path,
        Real wavelengthScaleToUm = Real{1},
        MaterialExtrapolation extrapolation = MaterialExtrapolation::Error);

    static Material dispersive(std::string name, DispersionFn fn);

    static Material anisotropic(std::string name,
                                Tensor3 epsilon,
                                Tensor3 mu = Tensor3::isotropic({1.0, 0.0}));

    static Material anisotropicDispersive(std::string name, TensorDispersionFn fn);

    static Material vacuum() { return constant("vacuum", {1.0, 0.0}); }

    static Material air() { return constant("Air", {1.0, 0.0}); }

    [[nodiscard]] const std::string& name() const noexcept { return mName; }
    [[nodiscard]] Complex epsilon(Real wavelengthUm = 1.0) const;
    [[nodiscard]] Complex mu(Real wavelengthUm = 1.0) const;
    [[nodiscard]] std::pair<Tensor3, Tensor3> tensors(Real wavelengthUm = 1.0) const;
    [[nodiscard]] Tensor3 epsilonTensor(Real wavelengthUm = 1.0) const;
    [[nodiscard]] Tensor3 muTensor(Real wavelengthUm = 1.0) const;
    [[nodiscard]] MaterialPassivityResult passivity(Real wavelengthUm = 1.0) const;
    [[nodiscard]] Complex refractiveIndex(Real wavelengthUm = 1.0) const;
    [[nodiscard]] bool isDispersive() const noexcept {
        return static_cast<bool>(mDispersion) || static_cast<bool>(mTensorDispersion);
    }

private:
    struct TensorCache;

    Material(std::string n, Complex e, Complex m);
    Material(std::string n, DispersionFn fn);
    Material(std::string n, Tensor3 e, Tensor3 m);
    Material(std::string n, TensorDispersionFn fn);

    std::string mName{"unnamed"};
    Complex mEps{1.0, 0.0};
    Complex mMu{1.0, 0.0};
    DispersionFn mDispersion{};
    Tensor3 mEpsTensor{Tensor3::isotropic({1.0, 0.0})};
    Tensor3 mMuTensor{Tensor3::isotropic({1.0, 0.0})};
    TensorDispersionFn mTensorDispersion{};
    std::shared_ptr<TensorCache> mTensorCache{};
    bool mTensorMaterial{false};
};

struct GrapheneModelParameters {
    Real fermiEnergyEv = std::numeric_limits<Real>::quiet_NaN();
    Real temperatureK = 300.0;
    Real relaxationTimeS = 0.5e-12;
    Real thicknessM = 0.34e-9;
};

struct DrudeModelParameters {
    Real epsInf = 1.0;
    Real plasmaFrequencyRadS = std::numeric_limits<Real>::quiet_NaN();
    Real dampingRateRadS = std::numeric_limits<Real>::quiet_NaN();
};

struct PhononPolaritonModelParameters {
    std::array<Real, 3> epsilonInf{
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN()};
    std::array<Real, 3> transverseWavenumberCmInv{
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN()};
    std::array<Real, 3> longitudinalWavenumberCmInv{
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN()};
    std::array<Real, 3> dampingWavenumberCmInv{
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN(),
        std::numeric_limits<Real>::quiet_NaN()};
    Real twistDeg = 0.0;
};

struct YigModelParameters {
    // epsilon = epsilon_d I - i*g*[m_hat]_x in the project-wide
    // exp(-i*omega*t) coordinate convention.
    Complex diagonalEpsilon{4.0, 0.0};
    Complex offDiagonalGyration{0.0, 0.0};
    std::array<Real, 3> magnetizationDirection{0.0, 1.0, 0.0};
};

struct WsmModelParameters {
    Real nodeSeparationMInv = 2.0e9;
    std::array<Real, 3> nodeSeparationDirection{0.0, 1.0, 0.0};
    Real fermiVelocityMS = 0.83e5;
    Real cutoffXi = 3.0;
    Real backgroundEpsilon = 6.2;
    Real temperatureK = 300.0;
    Real degeneracy = 2.0;
    Real relaxationTimeS = 1000.0e-15;
    Real fermiEnergyEv = 0.15;
};

struct BlackPhosphorusModelParameters {
    Real carrierDensityCm2 = 1.0e13;
    Real scatteringRateEv = 0.010;
    Real thicknessM = 1.0e-9;
    Real bandgapEv = 2.0;
    Real latticeConstantM = 0.223e-9;
    Real twistDeg = 0.0;
};

struct InAsMagnetoModelParameters {
    // Signed +y field in the project-wide exp(-i*omega*t) convention.
    Real magneticFieldT = 0.0;
    Real epsInf = 12.37;
    Real carrierDensityCm3 = 7.8e17;
    Real effectiveMassRatio = 0.033;
    Real dampingRateRadS = 1.55e11;
};

struct CdTeLorentzDrudeModelParameters {
    // Signed +y field in the project-wide exp(-i*omega*t) convention.
    Real magneticFieldT = 0.0;
    Real epsInf = 7.1;
    Real carrierDensityCm3 = 3.0e17;
    Real effectiveMassRatio = 0.09;
    Real transverseResonanceRadS = 2.652e13;
    Real dampingRateRadS = 1.24e12;
};

struct InSbLorentzDrudeModelParameters {
    // Signed +y field in the project-wide exp(-i*omega*t) convention.
    Real magneticFieldT = 0.0;
    Real epsInf = 15.68;
    Real carrierDensityCm3 = 5.8e17;
    Real effectiveMassRatio = 0.014;
    Real transverseResonanceRadS = 3.376e13;
    Real dampingRateRadS = 2.017e12;
};

bool isGrapheneMaterialModel(const std::string& model);
bool isDrudeMaterialModel(const std::string& model);
bool isPhononPolaritonMaterialModel(const std::string& model);
bool isYigMaterialModel(const std::string& model);
bool isWsmMaterialModel(const std::string& model);
bool isBlackPhosphorusMaterialModel(const std::string& model);
bool isInasMagnetoMaterialModel(const std::string& model);
bool isCdteLorentzDrudeMaterialModel(const std::string& model);
bool isInsbLorentzDrudeMaterialModel(const std::string& model);

GrapheneModelParameters defaultGrapheneModelParameters(const std::string& model);
DrudeModelParameters defaultDrudeModelParameters(const std::string& model);
PhononPolaritonModelParameters defaultPhononPolaritonModelParameters(
    const std::string& model);
YigModelParameters defaultYigModelParameters(const std::string& model);
WsmModelParameters defaultWsmModelParameters(const std::string& model);
BlackPhosphorusModelParameters defaultBlackPhosphorusModelParameters(
    const std::string& model);
InAsMagnetoModelParameters defaultInasMagnetoModelParameters(const std::string& model);
CdTeLorentzDrudeModelParameters defaultCdteLorentzDrudeModelParameters(
    const std::string& model);
InSbLorentzDrudeModelParameters defaultInsbLorentzDrudeModelParameters(
    const std::string& model);

Material builtinMaterialModel(const std::string& name, const std::string& model);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const GrapheneModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const DrudeModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const PhononPolaritonModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const YigModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const WsmModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const BlackPhosphorusModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const InAsMagnetoModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const CdTeLorentzDrudeModelParameters& parameters);
Material builtinMaterialModel(const std::string& name,
                                const std::string& model,
                                const InSbLorentzDrudeModelParameters& parameters);

} // namespace rcwa
