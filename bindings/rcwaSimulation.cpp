#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "rcwaBindingUtils.hpp"
#include "rcwaSimulation.hpp"

#include "rcwa/fourier.hpp"
#include "rcwa/geometry.hpp"
#include "rcwa/layer.hpp"
#include "rcwa/material.hpp"
#include "rcwa/pattern.hpp"
#include "rcwa/solver.hpp"
#include "rcwa/types.hpp"

namespace py = pybind11;

namespace {

using rcwa::Complex;
using rcwa::DirectionalThermalChannelResult;
using rcwa::FieldComponent;
using rcwa::FieldGridResult;
using rcwa::FieldPlane;
using rcwa::FieldPlaneRequest;
using rcwa::FieldPlaneResult;
using rcwa::LatticeTruncation;
using rcwa::Polarization;
using rcwa::Real;
using rcwa::SParameterResult;
using rcwa::checkedProduct;
using rcwa::SpectrumResult;
using rcwa::SpectrumAnglePolarization;
using rcwa::ZeroOrderAmplitudeResult;

using namespace rcwaPy;

template <typename T>
void writeUnaligned(char* row, std::size_t offset, const T& value) {
    std::memcpy(row + offset, &value, sizeof(T));
}

void writeNumpyText(char* row, std::size_t offset, std::string_view text, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        const std::uint32_t codepoint =
            i < text.size() ? static_cast<std::uint8_t>(text[i]) : 0;
        writeUnaligned(row, offset + i * sizeof(std::uint32_t), codepoint);
    }
}

std::vector<Real> realVectorFromObject(
    py::handle values,
    const char* label,
    bool requirePositiveValues = false) {
    py::array_t<Real, py::array::c_style | py::array::forcecast> arr(
        py::reinterpret_borrow<py::object>(values));
    const py::buffer_info buffer = arr.request();
    if (buffer.ndim != 1) {
        throw std::invalid_argument(
            std::string(label) + " must be a one-dimensional real array");
    }
    std::vector<Real> out(static_cast<std::size_t>(buffer.shape[0]));
    const auto view = arr.unchecked<1>();
    for (py::ssize_t index = 0; index < arr.size(); ++index) {
        const Real value = view(index);
        if (!std::isfinite(value) ||
            (requirePositiveValues && !(value > Real{0.0}))) {
            throw std::invalid_argument(
                std::string(label) + " must contain only finite" +
                (requirePositiveValues ? " positive" : "") + " values");
        }
        out[static_cast<std::size_t>(index)] = value;
    }
    return out;
}

const char* fourierFormulationName(rcwa::FourierFormulation formulation) {
    switch (formulation) {
    case rcwa::FourierFormulation::Default:
        return "Default";
    case rcwa::FourierFormulation::PolBasisVL:
        return "PolBasisVL";
    case rcwa::FourierFormulation::PolBasisNV:
        return "PolBasisNV";
    case rcwa::FourierFormulation::PolBasisJones:
        return "PolBasisJones";
    case rcwa::FourierFormulation::Kottke:
        return "Kottke";
    }
    return "Default";
}

const char* tensorFactorizationApplicabilityName(
    rcwa::TensorFactorizationApplicability applicability) {
    switch (applicability) {
    case rcwa::TensorFactorizationApplicability::Unknown:
        return "Unknown";
    case rcwa::TensorFactorizationApplicability::NotApplicable:
        return "NotApplicable";
    case rcwa::TensorFactorizationApplicability::CoordinateAligned:
        return "CoordinateAligned";
    case rcwa::TensorFactorizationApplicability::CurvedOrOblique:
        return "CurvedOrOblique";
    }
    return "Unknown";
}

struct Region {
    enum class Shape {
        Circle,
        Rectangle,
        Polygon,
    };

    Shape shape{Shape::Circle};
    std::string material;
    Real x{};
    Real y{};
    Real angleDeg{};
    Real radius{};
    Real halfwidthX{};
    Real halfwidthY{};
    int reticoloRectangleCount{10};
    std::vector<rcwa::Vec2> vertices;
};

struct Layer {
    std::string name;
    Real thicknessUm{};
    std::string background;
    std::vector<Region> regions;
    mutable std::shared_ptr<const rcwa::PrecomputedPeriodicLayer2D> cachedPrecomputed;
    mutable std::shared_ptr<const rcwa::PatternedLayer2DGeometryCache> cachedGeometry;
    mutable std::set<std::string> cachedMaterialDependencies;
    mutable std::mutex cacheMutex;

    Layer() = default;

    Layer(std::string layerName, Real thickness, std::string material)
        : name(std::move(layerName)), thicknessUm(thickness), background(std::move(material)) {}

    Layer(const Layer& other)
        : name(other.name),
          thicknessUm(other.thicknessUm),
          background(other.background),
          regions(other.regions),
          cachedPrecomputed(other.cachedPrecomputed),
          cachedGeometry(other.cachedGeometry),
          cachedMaterialDependencies(other.cachedMaterialDependencies) {}

    Layer& operator=(const Layer& other) {
        if (this == &other) {
            return *this;
        }
        std::scoped_lock lock(cacheMutex, other.cacheMutex);
        name = other.name;
        thicknessUm = other.thicknessUm;
        background = other.background;
        regions = other.regions;
        cachedPrecomputed = other.cachedPrecomputed;
        cachedGeometry = other.cachedGeometry;
        cachedMaterialDependencies = other.cachedMaterialDependencies;
        return *this;
    }

    Layer(Layer&& other) noexcept
        : name(std::move(other.name)),
          thicknessUm(other.thicknessUm),
          background(std::move(other.background)),
          regions(std::move(other.regions)),
          cachedPrecomputed(std::move(other.cachedPrecomputed)),
          cachedGeometry(std::move(other.cachedGeometry)),
          cachedMaterialDependencies(std::move(other.cachedMaterialDependencies)) {}

    Layer& operator=(Layer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        std::scoped_lock lock(cacheMutex, other.cacheMutex);
        name = std::move(other.name);
        thicknessUm = other.thicknessUm;
        background = std::move(other.background);
        regions = std::move(other.regions);
        cachedPrecomputed = std::move(other.cachedPrecomputed);
        cachedGeometry = std::move(other.cachedGeometry);
        cachedMaterialDependencies = std::move(other.cachedMaterialDependencies);
        return *this;
    }
};

class Simulation {
public:
    Simulation(py::object lattice, py::object orders) {
        setLattice(lattice);
        configureOrders(std::move(orders));
    }

    void setMaterial(const std::string& name, py::handle epsilon, py::object mu) {
        rcwa::Material material;
        if (!mu.is_none()) {
            const bool epsTensor = py::isinstance<py::array>(epsilon);
            const bool muTensor = py::isinstance<py::array>(mu);
            if (epsTensor || muTensor) {
                const rcwa::Tensor3 eps = epsTensor
                    ? tensorFromArray(epsilon)
                    : rcwa::Tensor3::isotropic(epsilon.cast<Complex>());
                const rcwa::Tensor3 muv = muTensor
                    ? tensorFromArray(mu)
                    : rcwa::Tensor3::isotropic(mu.cast<Complex>());
                material = rcwa::Material::anisotropic(name, eps, muv);
            } else {
                material =
                    rcwa::Material::constant(name, epsilon.cast<Complex>(), mu.cast<Complex>());
            }
            assignMaterial(name, std::move(material));
            return;
        }
        assignMaterial(name, materialFromObject(epsilon, name));
    }

    void setMaterialModel(const std::string& name,
                          const std::string& model,
                          py::handle parameters) {
        assignMaterial(name, builtinMaterialModelFromObject(name, model, parameters));
    }

    void setMaterialNk(const std::string& name,
                       const std::filesystem::path& path,
                       std::string wavelengthUnit,
                       std::string extrapolation) {
        wavelengthUnit = lowercase(std::move(wavelengthUnit));
        Real wavelengthScaleToUm{};
        if (wavelengthUnit == "um" || wavelengthUnit == "micrometer" ||
            wavelengthUnit == "micrometers" || wavelengthUnit == "micrometre" ||
            wavelengthUnit == "micrometres") {
            wavelengthScaleToUm = Real{1};
        } else if (wavelengthUnit == "nm" || wavelengthUnit == "nanometer" ||
                   wavelengthUnit == "nanometers" || wavelengthUnit == "nanometre" ||
                   wavelengthUnit == "nanometres") {
            wavelengthScaleToUm = Real{1e-3};
        } else {
            throw std::invalid_argument("WavelengthUnit must be 'um' or 'nm'");
        }

        extrapolation = lowercase(std::move(extrapolation));
        rcwa::MaterialExtrapolation extrapolationMode;
        if (extrapolation == "error") {
            extrapolationMode = rcwa::MaterialExtrapolation::Error;
        } else if (extrapolation == "clamp" || extrapolation == "hold") {
            extrapolationMode = rcwa::MaterialExtrapolation::Clamp;
        } else {
            throw std::invalid_argument("Extrapolation must be 'error' or 'clamp'");
        }

        assignMaterial(
            name,
            rcwa::Material::fromNkFile(
                name, path, wavelengthScaleToUm, extrapolationMode));
    }

    void setSuperstrate(const std::string& material) {
        requireMaterial(material);
        mSuperstrateName = material;
    }

    void setSubstrate(const std::string& material) {
        requireMaterial(material);
        mSubstrateName = material;
    }

    void setStackingAlgorithm(rcwa::StackingAlgorithm algorithm) {
        switch (algorithm) {
        case rcwa::StackingAlgorithm::ScatteringMatrix:
        case rcwa::StackingAlgorithm::EnhancedTransmittanceMatrix:
            mStackingAlgorithm = algorithm;
            return;
        }
        throw std::invalid_argument("unsupported RCWA stacking algorithm");
    }

    void addLayer(const std::string& name, Real thicknessUm, const std::string& material) {
        requireMaterial(material);
        mLayers.emplace_back(name, thicknessUm, material);
    }

    void setRegionCircle(const std::string& layer,
                         const std::string& material,
                         py::object center,
                         Real radius,
                         int rectangles = 10) {
        requireMaterial(material);
        if (rectangles < 2 || rectangles > 4096) {
            throw std::invalid_argument("Rectangles must be in [2, 4096]");
        }
        auto& target = layerByName(layer);
        const auto c = pairFromObject(center, "Center");
        Region region;
        region.shape = Region::Shape::Circle;
        region.material = material;
        region.x = c.first;
        region.y = c.second;
        region.radius = radius;
        region.reticoloRectangleCount = rectangles;
        target.regions.push_back(std::move(region));
        target.cachedPrecomputed.reset();
        target.cachedGeometry.reset();
        target.cachedMaterialDependencies.clear();
    }

    void setRegionRectangle(const std::string& layer,
                            const std::string& material,
                            py::object center,
                            Real angle,
                            py::object halfwidths) {
        requireMaterial(material);
        auto& target = layerByName(layer);
        const auto c = pairFromObject(center, "Center");
        const auto hw = pairFromObject(halfwidths, "Halfwidths");
        Region region;
        region.shape = Region::Shape::Rectangle;
        region.material = material;
        region.x = c.first;
        region.y = c.second;
        region.angleDeg = angle;
        region.halfwidthX = hw.first;
        region.halfwidthY = hw.second;
        target.regions.push_back(std::move(region));
        target.cachedPrecomputed.reset();
        target.cachedGeometry.reset();
        target.cachedMaterialDependencies.clear();
    }

    void setRegionPolygon(const std::string& layer,
                          const std::string& material,
                          py::object center,
                          Real angle,
                          py::object vertices) {
        requireMaterial(material);
        auto& target = layerByName(layer);
        const auto c = pairFromObject(center, "Center");
        const Real theta = angle * rcwa::pi / 180.0;
        Region region;
        region.shape = Region::Shape::Polygon;
        region.material = material;
        for (py::handle item : vertices) {
            const auto v = pairFromObject(py::reinterpret_borrow<py::object>(item), "Vertex");
            const Real xr = v.first * std::cos(theta) - v.second * std::sin(theta);
            const Real yr = v.first * std::sin(theta) + v.second * std::cos(theta);
            region.vertices.push_back({c.first + xr, c.second + yr});
        }
        target.regions.push_back(std::move(region));
        target.cachedPrecomputed.reset();
        target.cachedGeometry.reset();
        target.cachedMaterialDependencies.clear();
    }

    void addPolygonRegion(const std::string& layer,
                            const std::string& material,
                            const std::vector<rcwa::Vec2>& vertices) {
        requireMaterial(material);
        if (vertices.size() < 3) {
            throw std::invalid_argument("polygon region must contain at least three vertices");
        }
        auto& target = layerByName(layer);
        Region region;
        region.shape = Region::Shape::Polygon;
        region.material = material;
        region.vertices = vertices;
        target.regions.push_back(std::move(region));
        target.cachedPrecomputed.reset();
        target.cachedGeometry.reset();
        target.cachedMaterialDependencies.clear();
    }

    void rotateLastRegionAbout(const std::string& layer,
                                  Real centerX,
                                  Real centerY,
                                  Real angleDeg) {
        auto& target = layerByName(layer);
        if (target.regions.empty()) {
            throw std::logic_error("cannot transform an empty region list");
        }
        Region& region = target.regions.back();
        if (region.shape != Region::Shape::Polygon) {
            throw std::logic_error("last region is not a polygon");
        }
        const Real theta = angleDeg * rcwa::pi / Real{180.0};
        for (auto& vertex : region.vertices) {
            const Real xr = vertex.x * std::cos(theta) - vertex.y * std::sin(theta);
            const Real yr = vertex.x * std::sin(theta) + vertex.y * std::cos(theta);
            vertex.x = centerX + xr;
            vertex.y = centerY + yr;
        }
    }

    void setRegionEllipse(const std::string& layer,
                          const std::string& material,
                          py::object center,
                          Real angle,
                          py::object radii,
                          int vertices = 96) {
        const auto c = pairFromObject(center, "Center");
        const auto r = pairFromObject(radii, "Radii");
        addPolygonRegion(
            layer,
            material,
            rcwa::regularEllipseVertices(r.first, r.second, vertices));
        rotateLastRegionAbout(layer, c.first, c.second, angle);
    }

    void setRegionRegularPolygon(const std::string& layer,
                                 const std::string& material,
                                 py::object center,
                                 Real angle,
                                 int sides,
                                 Real radius) {
        const auto c = pairFromObject(center, "Center");
        addPolygonRegion(
            layer,
            material,
            rcwa::regularPolygonVertices(sides, radius));
        rotateLastRegionAbout(layer, c.first, c.second, angle);
    }

    void setRegionTriangle(const std::string& layer,
                           const std::string& material,
                           py::object center,
                           Real angle,
                           Real radius) {
        setRegionRegularPolygon(layer, material, center, angle, 3, radius);
    }

    void setRegionCross(const std::string& layer,
                        const std::string& material,
                        py::object center,
                        Real angle,
                        Real armLength,
                        Real armWidth) {
        rcwa::requirePositive(armLength, "cross arm_length");
        rcwa::requirePositive(armWidth, "cross arm_width");
        if (armWidth > armLength) {
            throw std::invalid_argument("cross arm_width must be <= arm_length");
        }
        const Real h = Real{0.5} * armLength;
        const Real w = Real{0.5} * armWidth;
        const std::vector<rcwa::Vec2> vertices = {
            {-w, -h}, {w, -h}, {w, -w}, {h, -w},
            {h, w}, {w, w}, {w, h}, {-w, h},
            {-w, w}, {-h, w}, {-h, -w}, {-w, -w},
        };
        const auto c = pairFromObject(center, "Center");
        addPolygonRegion(layer, material, vertices);
        rotateLastRegionAbout(layer, c.first, c.second, angle);
    }

    void setRegionRing(const std::string& layer,
                       const std::string& material,
                       const std::string& holeMaterial,
                       py::object center,
                       Real outerRadius,
                       Real innerRadius) {
        requireMaterial(holeMaterial);
        if (!(outerRadius > innerRadius) || !(innerRadius >= Real{0.0})) {
            throw std::invalid_argument("ring requires outer_radius > inner_radius >= 0");
        }
        setRegionCircle(layer, material, center, outerRadius);
        setRegionCircle(layer, holeMaterial, center, innerRadius);
    }

    void setRegionRoundedRectangle(const std::string& layer,
                                   const std::string& material,
                                   py::object center,
                                   Real angle,
                                   py::object halfwidths,
                                   Real cornerRadius,
                                   int verticesPerCorner = 16) {
        const auto c = pairFromObject(center, "Center");
        const auto hw = pairFromObject(halfwidths, "Halfwidths");
        addPolygonRegion(
            layer,
            material,
            rcwa::roundedRectangleVertices(
                hw.first,
                hw.second,
                cornerRadius,
                verticesPerCorner));
        rotateLastRegionAbout(layer, c.first, c.second, angle);
    }

    void setRegionCapsule(const std::string& layer,
                          const std::string& material,
                          py::object center,
                          Real angle,
                          Real length,
                          Real radius,
                          int verticesPerCap = 32) {
        const auto c = pairFromObject(center, "Center");
        addPolygonRegion(
            layer,
            material,
            rcwa::capsuleVertices(length, radius, verticesPerCap));
        rotateLastRegionAbout(layer, c.first, c.second, angle);
    }

    void setRegionStripe(const std::string& layer,
                         const std::string& material,
                         py::object center,
                         Real angle,
                         Real width) {
        rcwa::requirePositive(width, "stripe width");
        setRegionRectangle(
            layer,
            material,
            center,
            angle,
            py::make_tuple(Real{0.5} * std::hypot(mPeriodXUm, mPeriodYUm) + width,
                           Real{0.5} * width));
    }

    void setExcitationPlanewave(py::object angles,
                                Complex sAmplitude,
                                Complex pAmplitude,
                                int order) {
        if (order != 0) {
            throw std::invalid_argument("only zero-order incident planewaves are supported");
        }
        const auto pair = pairFromObject(angles, "Angles");
        mThetaDeg = pair.first;
        mPhiDeg = pair.second;
        if (isCircularAmplitudePair(sAmplitude, pAmplitude, {0.0, 1.0})) {
            mPolarization = Polarization::LCP;
        } else if (isCircularAmplitudePair(sAmplitude, pAmplitude, {0.0, -1.0})) {
            mPolarization = Polarization::RCP;
        } else if (std::abs(sAmplitude) > 0.0 && std::abs(pAmplitude) > 0.0) {
            mPolarization = Polarization::Both;
        } else if (std::abs(pAmplitude) > 0.0) {
            mPolarization = Polarization::TM;
        } else {
            mPolarization = Polarization::TE;
        }
    }

    void setFrequency(Real frequency) {
        if (!(frequency > 0.0)) {
            throw std::invalid_argument("frequency must be positive");
        }
        mWavelengthUm = Real{1.0} / frequency;
    }

    void setWavelength(Real wavelengthUm) {
        if (!(wavelengthUm > 0.0)) {
            throw std::invalid_argument("wavelength must be positive");
        }
        mWavelengthUm = wavelengthUm;
    }

    void setHarmonicOrders(int orderX, int orderY) {
        if (orderX < 0 || orderY < 0) {
            throw std::invalid_argument("harmonic orders must be non-negative");
        }
        mOrderX = orderX;
        mOrderY = orderY;
        clearCachedLayers();
    }

    py::array getPowerFlux(py::object layerName, Real zOffset) const {
        (void)layerName;
        (void)zOffset;
        const auto result = solveOne(mPolarization);
        py::array_t<Real> out({2});
        auto view = out.mutable_unchecked<1>();
        view(0) = result.rTotal;
        view(1) = result.tTotal;
        return out;
    }

    py::array getHarmonicBasis() const {
        return harmonicBasisArray(harmonicBasis());
    }

    py::dict getWavevectorMatrices(py::object wavelength, py::object theta, py::object phi) const {
        const Real wavelengthUm = optionalReal(wavelength, mWavelengthUm);
        const Real thetaDeg = optionalReal(theta, mThetaDeg);
        const Real phiDeg = optionalReal(phi, mPhiDeg);
        const auto basis = harmonicBasis();
        const auto [Kx, Ky] = wavevectorOperators(wavelengthUm, thetaDeg, phiDeg, basis);

        py::dict out;
        out[py::str("wavelength_um")] = wavelengthUm;
        out[py::str("theta_deg")] = thetaDeg;
        out[py::str("phi_deg")] = phiDeg;
        out[py::str("period_x_um")] = mPeriodXUm;
        out[py::str("period_y_um")] = mPeriodYUm;
        out[py::str("harmonics")] = static_cast<int>(basis.size());
        out[py::str("orders")] = harmonicBasisArray(basis);
        out[py::str("Kx")] = matrixToArray(Kx.toDense());
        out[py::str("Ky")] = matrixToArray(Ky.toDense());
        out[py::str("kx_diag")] = complexVectorToArray(Kx.values());
        out[py::str("ky_diag")] = complexVectorToArray(Ky.values());
        return out;
    }

    py::dict getLayerFourierMatrices(const std::string& layer, py::object wavelength) const {
        const Real wavelengthUm = optionalReal(wavelength, mWavelengthUm);
        const auto basis = harmonicBasis();
        auto tensors = layerFourierMatrices(layerByName(layer), wavelengthUm);
        rcwa::annotateTensorFourierMetadata(tensors, basis.size());

        py::dict out;
        out[py::str("layer")] = layer;
        out[py::str("wavelength_um")] = wavelengthUm;
        out[py::str("harmonics")] = static_cast<int>(basis.size());
        out[py::str("orders")] = harmonicBasisArray(basis);
        out[py::str("eps")] = tensorMatrixBlocksToDict(tensors.eps);
        out[py::str("mu")] = tensorMatrixBlocksToDict(tensors.mu);
        out[py::str("li_eps_q")] = tensorMatrixBlocksToDict(tensors.liEpsQ);
        out[py::str("li_mu_q")] = tensorMatrixBlocksToDict(tensors.liMuQ);
        out[py::str("li_eps_q_zz_inverse")] = matrixToArray(tensors.liEpsQZzInverse);
        out[py::str("li_mu_q_zz_inverse")] = matrixToArray(tensors.liMuQZzInverse);
        out[py::str("metadata")] = tensorMetadataDict(tensors.metadata);
        return out;
    }

    py::dict getLayerModes(const std::string& layer,
                           py::object wavelength,
                           py::object theta,
                           py::object phi) const {
        const Real wavelengthUm = optionalReal(wavelength, mWavelengthUm);
        const Real thetaDeg = optionalReal(theta, mThetaDeg);
        const Real phiDeg = optionalReal(phi, mPhiDeg);
        const auto basis = harmonicBasis();
        const auto [Kx, Ky] = wavevectorOperators(wavelengthUm, thetaDeg, phiDeg, basis);
        const Layer& target = layerByName(layer);
        rcwa::LayerModes modes = layerModes(target, wavelengthUm, Kx, Ky, basis);

        py::dict out;
        out[py::str("layer")] = layer;
        out[py::str("wavelength_um")] = wavelengthUm;
        out[py::str("theta_deg")] = thetaDeg;
        out[py::str("phi_deg")] = phiDeg;
        out[py::str("harmonics")] = static_cast<int>(basis.size());
        out[py::str("kind")] = layerModeKindName(modes.kind);
        // Backward-compatible diagnostic key.  Scalar and tensor homogeneous
        // layers now share the same 4x4 TMM; this flag only says that the
        // scalar-degenerate Berreman eigenbasis was built analytically.
        out[py::str("is_isotropic_uniform")] = modes.usesAnalyticScalarBasis;
        out[py::str("uses_analytic_scalar_basis")] =
            modes.usesAnalyticScalarBasis;
        out[py::str("used_y_mirror_symmetry_split")] =
            modes.usedYMirrorSymmetrySplit;
        out[py::str("W")] = matrixToArray(rcwa::materializeModeMatrix(modes));
        out[py::str("gamma")] = complexVectorToArray(modes.gamma);
        out[py::str("phase")] = complexVectorToArray(modes.phase);
        return out;
    }

    py::dict findAbsorptivityPeak(
        py::object wavelengthBracket,
        const std::string& polarization,
        py::object theta,
        py::object phi,
        int gridPoints,
        int refinements,
        int workers) const {
        const auto bracket = pairFromObject(wavelengthBracket, "WavelengthBracket");
        const Polarization pol = parsePolarization(polarization);
        const Real thetaDeg = optionalReal(theta, mThetaDeg);
        const Real phiDeg = optionalReal(phi, mPhiDeg);
        auto solver = makeSolver(Real{0.5} * (bracket.first + bracket.second), pol);
        solver.setIncidence(thetaDeg, phiDeg);
        SpectrumResult result;
        {
            py::gil_scoped_release release;
            result = solver.findAbsorptivityPeak(
                bracket.first,
                bracket.second,
                pol,
                gridPoints,
                refinements,
                workers);
        }
        py::dict out;
        out[py::str("wavelength_um")] = result.wavelengthUm;
        out[py::str("polarization")] = rcwa::toString(pol);
        out[py::str("theta_deg")] = result.thetaDeg;
        out[py::str("phi_deg")] = result.phiDeg;
        out[py::str("harmonics")] = result.N;
        out[py::str("R")] = result.rTotal;
        out[py::str("T")] = result.tTotal;
        out[py::str("A")] = Real{1.0} - result.rTotal - result.tTotal;
        out[py::str("conservation")] = result.conservation;
        return out;
    }

    py::list precomputeLayerCache(py::object layerName, py::object wavelength) const {
        const Real wavelengthUm = optionalReal(wavelength, mWavelengthUm);
        if (layerName.is_none()) {
            for (const auto& layer : mLayers) {
                warmLayerCache(layer, wavelengthUm);
            }
            return layerCacheInfoList(py::none());
        }
        const std::string name = layerName.cast<std::string>();
        warmLayerCache(layerByName(name), wavelengthUm);
        return layerCacheInfoList(layerName);
    }

    py::list getLayerCacheInfo(py::object layerName) const {
        return layerCacheInfoList(layerName);
    }

    py::array getSpectrum(py::object wavelengths, py::object polarizations, int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || pols.empty()) {
            return spectrumTotalsArray({}, pols);
        }
        const int workerCount = normalizedWorkerCount(workers, wavelengthValues.size());
        std::vector<rcwa::SpectrumTotalsResult> results;

        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumBatchTotalsOnly(
                wavelengthValues,
                pols,
                workerCount);
        }

        return spectrumTotalsArray(results, pols);
    }

    py::array getDiffractionOrders(py::object wavelengths,
                                   py::object polarizations,
                                   int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || pols.empty()) {
            return diffractionOrdersArray({}, pols, {});
        }
        const int workerCount = normalizedWorkerCount(workers, wavelengthValues.size());
        std::vector<SpectrumResult> results;

        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumBatchOnly(
                wavelengthValues,
                pols,
                workerCount);
        }

        return diffractionOrdersArray(results, pols, harmonicBasis().orders);
    }

    py::array getSpectrumForAngles(py::object wavelengths,
                                   py::object angles,
                                   py::object polarizations,
                                   Real phiDeg,
                                   int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto thetaValues = vectorFromArray(angles);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || thetaValues.empty() || pols.empty()) {
            return spectrumTotalsArray({}, pols);
        }
        const std::size_t jobCount = checkedProduct(
            wavelengthValues.size(),
            thetaValues.size(),
            "angle spectrum job");
        const int workerCount = normalizedWorkerCount(workers, jobCount);
        std::vector<rcwa::SpectrumTotalsResult> results;

        {
            py::gil_scoped_release release;
            const Real referenceWavelength = wavelengthValues.front();
            auto solver = makeBatchSolver(referenceWavelength, mPolarization);
            results = solver.solveSpectrumBatchTotalsForAngles(
                thetaValues,
                phiDeg,
                wavelengthValues,
                pols,
                workerCount);
        }

        return spectrumTotalsArray(results, pols);
    }

    py::tuple getSpectrumAndDirectionalThermalChannelsForAngles(
        py::object wavelengths,
        py::object angles,
        py::object polarizations,
        Real phiDeg,
        int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto thetaValues = vectorFromArray(angles);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        const auto channels = thermalChannelPolarizations(polarizations);
        if (wavelengthValues.empty() || thetaValues.empty() || pols.empty()) {
            return py::make_tuple(
                spectrumTotalsArray({}, pols),
                directionalThermalChannelArray({}, channels));
        }
        const std::size_t jobCount = checkedProduct(
            wavelengthValues.size(),
            thetaValues.size(),
            "combined angle spectrum/thermal job");
        const int workerCount = normalizedWorkerCount(workers, jobCount);
        rcwa::SpectrumThermalBatchResult results;
        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumThermalBatchForAngles(
                thetaValues,
                phiDeg,
                wavelengthValues,
                pols,
                workerCount);
        }
        return py::make_tuple(
            spectrumTotalsArray(results.spectra, pols),
            directionalThermalChannelArray(results.thermalChannels, channels));
    }

    py::array getDiffractionOrdersForAngles(py::object wavelengths,
                                            py::object angles,
                                            py::object polarizations,
                                            Real phiDeg,
                                            int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto thetaValues = vectorFromArray(angles);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || thetaValues.empty() || pols.empty()) {
            return diffractionOrdersArray({}, pols, {});
        }
        const std::size_t jobCount = checkedProduct(
            wavelengthValues.size(),
            thetaValues.size(),
            "angle diffraction-order job");
        const int workerCount = normalizedWorkerCount(workers, jobCount);
        std::vector<SpectrumResult> results;

        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumBatchForAngles(
                thetaValues,
                phiDeg,
                wavelengthValues,
                pols,
                workerCount);
        }

        return diffractionOrdersArray(results, pols, harmonicBasis().orders);
    }

    py::array getSpectrumForAnglePolarizationPairs(py::object wavelengths,
                                                   py::object angles,
                                                   py::object polarizations,
                                                   Real phiDeg,
                                                   int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto thetaValues = vectorFromArray(angles);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || thetaValues.empty() || pols.empty()) {
            return spectrumTotalsArray({}, pols);
        }
        auto [requests, requestPolarizations] =
            anglePolarizationRequests(thetaValues, pols, "GetSpectrumForAnglePolarizationPairs");

        const std::size_t jobCount = checkedProduct(
            wavelengthValues.size(),
            thetaValues.size(),
            "angle/polarization spectrum job");
        const int workerCount = normalizedWorkerCount(workers, jobCount);
        std::vector<rcwa::SpectrumTotalsResult> results;

        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumBatchTotalsForAnglePolarizations(
                requests,
                phiDeg,
                wavelengthValues,
                workerCount);
        }

        return spectrumTotalsArrayForAnglePolarizations(
            results,
            requestPolarizations,
            wavelengthValues.size());
    }

    py::array getDiffractionOrdersForAnglePolarizationPairs(py::object wavelengths,
                                                            py::object angles,
                                                            py::object polarizations,
                                                            Real phiDeg,
                                                            int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        const auto thetaValues = vectorFromArray(angles);
        const auto pols = polarizationsFromObject(polarizations, mPolarization);
        if (wavelengthValues.empty() || thetaValues.empty() || pols.empty()) {
            return diffractionOrdersArray({}, pols, {});
        }
        auto [requests, requestPolarizations] = anglePolarizationRequests(
            thetaValues,
            pols,
            "GetDiffractionOrdersForAnglePolarizationPairs");

        const std::size_t jobCount = checkedProduct(
            wavelengthValues.size(),
            thetaValues.size(),
            "angle/polarization diffraction-order job");
        const int workerCount = normalizedWorkerCount(workers, jobCount);
        std::vector<SpectrumResult> results;

        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            results = solver.solveSpectrumBatchForAnglePolarizations(
                requests,
                phiDeg,
                wavelengthValues,
                workerCount);
        }

        return diffractionOrdersArrayForAnglePolarizations(
            results,
            requestPolarizations,
            wavelengthValues.size(),
            harmonicBasis().orders);
    }

    py::array getSParameters(Real wavelengthUm, py::object theta, py::object phi) const {
        const Real thetaDeg = theta.is_none() ? mThetaDeg : theta.cast<Real>();
        const Real phiDeg = phi.is_none() ? mPhiDeg : phi.cast<Real>();
        auto solver = makeSolver(wavelengthUm, mPolarization);
        solver.setIncidence(thetaDeg, phiDeg);
        SParameterResult result;
        {
            py::gil_scoped_release release;
            result = solver.solveSParameters();
        }
        return sparameterArray(result);
    }

    py::array getSParametersAnalyticContinuationAtGamma(Real wavelengthUm) const {
        auto solver = makeSolver(wavelengthUm, mPolarization);
        solver.setIncidence(0.0, 0.0);
        SParameterResult result;
        {
            py::gil_scoped_release release;
            result = solver.solveSParametersAnalyticContinuationAtGamma();
        }
        return sparameterArray(result);
    }

    py::array getSParametersAnalyticContinuation(
        Real wavelengthUm,
        py::object blochWavevector) const {
        const auto [blochKxReduced, blochKyReduced] =
            pairFromObject(std::move(blochWavevector), "BlochWavevector");
        auto solver = makeSolver(wavelengthUm, mPolarization);
        SParameterResult result;
        {
            py::gil_scoped_release release;
            result = solver.solveSParametersAnalyticContinuation(
                blochKxReduced,
                blochKyReduced);
        }
        return sparameterArray(result);
    }

    py::array getZeroOrderAmplitudes(py::object wavelengths,
                                     py::object theta,
                                     py::object phi,
                                     int workers) const {
        const auto wavelengthValues = vectorFromArray(wavelengths);
        if (wavelengthValues.empty()) {
            return zeroOrderAmplitudeArray({});
        }
        const Real thetaDeg = theta.is_none() ? mThetaDeg : theta.cast<Real>();
        const Real phiDeg = phi.is_none() ? mPhiDeg : phi.cast<Real>();
        const int workerCount = normalizedWorkerCount(workers, wavelengthValues.size());
        std::vector<ZeroOrderAmplitudeResult> results;
        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            solver.setIncidence(thetaDeg, phiDeg);
            results = solver.solveZeroOrderAmplitudesBatchOnly(
                wavelengthValues,
                workerCount);
        }
        return zeroOrderAmplitudeArray(results);
    }

    py::array getDirectionalThermalChannels(py::object wavelengths,
                                            py::object theta,
                                            py::object phi,
                                            int workers,
                                            py::object polarizations) const {
        const auto channels = thermalChannelPolarizations(polarizations);
        const auto wavelengthValues = vectorFromArray(wavelengths);
        if (wavelengthValues.empty()) {
            return directionalThermalChannelArray({}, channels);
        }
        const Real thetaDeg = theta.is_none() ? mThetaDeg : theta.cast<Real>();
        const Real phiDeg = phi.is_none() ? mPhiDeg : phi.cast<Real>();
        const int workerCount = normalizedWorkerCount(workers, wavelengthValues.size());
        std::vector<DirectionalThermalChannelResult> results;
        {
            py::gil_scoped_release release;
            auto solver = makeBatchSolver(wavelengthValues.front(), mPolarization);
            solver.setIncidence(thetaDeg, phiDeg);
            results = solver.solveDirectionalThermalChannelsBatchOnly(
                wavelengthValues,
                workerCount);
        }
        return directionalThermalChannelArray(results, channels);
    }

    py::array getFieldPlane(Real wavelengthUm,
                            const std::string& polarization,
                            Real thetaDeg,
                            Real phiDeg,
                            py::object components,
                            const std::string& plane,
                            Real uMinUm,
                            Real uMaxUm,
                            int uPoints,
                            Real vMinUm,
                            Real vMaxUm,
                            int vPoints,
                            Real fixedUm,
                            int workers) const {
        auto solver = makeSolver(wavelengthUm, parsePolarization(polarization));
        solver.setIncidence(thetaDeg, phiDeg);
        const FieldPlaneRequest request = makeFieldPlaneRequest(
            components,
            plane,
            uMinUm,
            uMaxUm,
            uPoints,
            vMinUm,
            vMaxUm,
            vPoints,
            fixedUm,
            workers);
        FieldPlaneResult result;
        {
            py::gil_scoped_release release;
            result = solver.solveFieldPlane(request);
        }
        return fieldPlaneArray(polarization, result);
    }

    py::dict getFieldGrid(Real wavelengthUm,
                          const std::string& polarization,
                          Real thetaDeg,
                          Real phiDeg,
                          py::object components,
                          const std::string& plane,
                          Real uMinUm,
                          Real uMaxUm,
                          int uPoints,
                          Real vMinUm,
                          Real vMaxUm,
                          int vPoints,
                          Real fixedUm,
                          int workers) const {
        auto solver = makeSolver(wavelengthUm, parsePolarization(polarization));
        solver.setIncidence(thetaDeg, phiDeg);
        const FieldPlaneRequest request = makeFieldPlaneRequest(
            components,
            plane,
            uMinUm,
            uMaxUm,
            uPoints,
            vMinUm,
            vMaxUm,
            vPoints,
            fixedUm,
            workers);
        FieldGridResult result;
        {
            py::gil_scoped_release release;
            result = solver.solveFieldGrid(request);
        }
        return fieldGridDict(polarization, std::move(result));
    }

private:
    static FieldPlaneRequest makeFieldPlaneRequest(
        py::object components,
        const std::string& plane,
        Real uMinUm,
        Real uMaxUm,
        int uPoints,
        Real vMinUm,
        Real vMaxUm,
        int vPoints,
        Real fixedUm,
        int workers) {
        FieldPlaneRequest request;
        request.plane = parseFieldPlane(plane);
        request.uMinUm = uMinUm;
        request.uMaxUm = uMaxUm;
        request.uPoints = uPoints;
        request.vMinUm = vMinUm;
        request.vMaxUm = vMaxUm;
        request.vPoints = vPoints;
        request.fixedUm = fixedUm;
        request.workers = normalizedWorkerCount(
            workers,
            checkedProduct(
                static_cast<std::size_t>(std::max(uPoints, 1)),
                static_cast<std::size_t>(std::max(vPoints, 1)),
                "field plane point"));
        request.components.clear();
        for (py::handle component : components) {
            request.components.push_back(
                parseFieldComponent(component.cast<std::string>()));
        }
        return request;
    }

    void setLattice(py::object lattice) {
        if (py::isinstance<py::float_>(lattice) || py::isinstance<py::int_>(lattice)) {
            mPeriodXUm = lattice.cast<Real>();
            mPeriodYUm = mPeriodXUm;
            return;
        }
        const auto pair = pairFromObject(lattice, "Lattice");
        mPeriodXUm = pair.first;
        mPeriodYUm = pair.second;
    }

    void configureOrders(py::object orders) {
        if (orders.is_none()) {
            setHarmonicOrders(0, 0);
            return;
        }
        const auto pair = pairFromObject(std::move(orders), "Orders");
        setHarmonicOrders(
            exactNonnegativeInteger(pair.first, "Orders[0]"),
            exactNonnegativeInteger(pair.second, "Orders[1]"));
    }

    rcwa::HarmonicBasis harmonicBasis() const {
        return rcwa::makeHarmonicBasisOrders(
            mOrderX,
            mOrderY,
            LatticeTruncation::Parallelogramic,
            mPeriodXUm,
            mPeriodYUm);
    }

    std::pair<rcwa::DiagonalOperator, rcwa::DiagonalOperator> wavevectorOperators(
        Real wavelengthUm,
        Real thetaDeg,
        Real phiDeg,
        const rcwa::HarmonicBasis& basis) const {
        rcwa::requirePositive(wavelengthUm, "wavelength");
        rcwa::requirePositive(mPeriodXUm, "period_x");
        rcwa::requirePositive(mPeriodYUm, "period_y");
        const Real theta = rcwa::degToRad(thetaDeg);
        const Real phi = rcwa::degToRad(phiDeg);
        const Complex nInc = requireMaterial(mSuperstrateName).refractiveIndex(wavelengthUm);
        const Real kx0 = std::real(nInc) * std::sin(theta) * std::cos(phi);
        const Real ky0 = std::real(nInc) * std::sin(theta) * std::sin(phi);
        return rcwa::makeTransverseWavevectorOperators(
            kx0,
            ky0,
            wavelengthUm / mPeriodXUm,
            wavelengthUm / mPeriodYUm,
            basis);
    }

    static Real optionalReal(py::object value, Real fallback) {
        return value.is_none() ? fallback : value.cast<Real>();
    }

    static int exactNonnegativeInteger(Real value, const char* name) {
        if (!std::isfinite(value) || value < Real{0} ||
            value > static_cast<Real>(std::numeric_limits<int>::max()) ||
            std::floor(value) != value) {
            throw std::invalid_argument(
                std::string(name) + " must be an exact non-negative integer");
        }
        return static_cast<int>(value);
    }

    static int exactInteger(Real value, const char* name) {
        if (!std::isfinite(value) ||
            value < static_cast<Real>(std::numeric_limits<int>::min()) ||
            value > static_cast<Real>(std::numeric_limits<int>::max()) ||
            std::floor(value) != value) {
            throw std::invalid_argument(
                std::string(name) + " must be an exact integer");
        }
        return static_cast<int>(value);
    }

    static std::pair<Real, Real> pairFromObject(py::object obj, const char* name) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        if (seq.size() != 2) {
            throw std::invalid_argument(std::string(name) + " must contain two values");
        }
        return {seq[0].cast<Real>(), seq[1].cast<Real>()};
    }

    static std::vector<Real> vectorFromArray(py::handle values) {
        return realVectorFromObject(values, "real input array");
    }

    static std::vector<Polarization> polarizationsFromObject(py::handle obj,
                                                              Polarization defaultPolarization) {
        if (obj.is_none()) {
            return {defaultPolarization};
        }
        if (py::isinstance<py::str>(obj)) {
            return {parsePolarization(obj.cast<std::string>())};
        }
        std::vector<Polarization> out;
        for (py::handle item : obj) {
            out.push_back(parsePolarization(item.cast<std::string>()));
        }
        return out;
    }

    static std::vector<Polarization> thermalChannelPolarizations(py::handle obj) {
        const std::vector<Polarization> requested = obj.is_none()
            ? std::vector<Polarization>{Polarization::TE, Polarization::TM}
            : polarizationsFromObject(obj, Polarization::TE);
        std::vector<Polarization> channels;
        channels.reserve(requested.size() + 1);
        for (const Polarization polarization : requested) {
            if (polarization == Polarization::Both) {
                channels.push_back(Polarization::TE);
                channels.push_back(Polarization::TM);
            } else {
                channels.push_back(polarization);
            }
        }
        return channels;
    }

    static py::array_t<std::complex<double>> matrixToArray(const rcwa::Matrix& matrix) {
        py::array_t<std::complex<double>> out({
            static_cast<py::ssize_t>(matrix.rows()),
            static_cast<py::ssize_t>(matrix.cols())});
        auto view = out.mutable_unchecked<2>();
        for (std::size_t r = 0; r < matrix.rows(); ++r) {
            for (std::size_t c = 0; c < matrix.cols(); ++c) {
                const Complex value = matrix(r, c);
                view(static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c)) =
                    std::complex<double>{
                        static_cast<double>(std::real(value)),
                        static_cast<double>(std::imag(value))};
            }
        }
        return out;
    }

    static py::array_t<std::complex<double>> complexVectorToArray(
        const std::vector<Complex>& values) {
        py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(values.size()));
        auto view = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            view(static_cast<py::ssize_t>(i)) =
                std::complex<double>{
                    static_cast<double>(std::real(values[i])),
                    static_cast<double>(std::imag(values[i]))};
        }
        return out;
    }

    static py::array harmonicBasisArray(const rcwa::HarmonicBasis& basis) {
        py::dtype dtype = makeDtype({
            {"order_index", "i4"},
            {"m", "i4"},
            {"n", "i4"},
        });
        py::array out = makeStructuredArray(dtype, basis.orders.size());
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);
        for (std::size_t i = 0; i < basis.orders.size(); ++i) {
            char* row = data + i * stride;
            writeUnaligned(row, 0, static_cast<std::int32_t>(i));
            writeUnaligned(row, 4, static_cast<std::int32_t>(basis.orders[i].m));
            writeUnaligned(row, 8, static_cast<std::int32_t>(basis.orders[i].n));
        }
        return out;
    }

    static py::dict tensorMatrixBlocksToDict(
        const std::array<std::array<rcwa::Matrix, 3>, 3>& blocks) {
        py::dict out;
        static constexpr std::array<const char*, 3> names{"x", "y", "z"};
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 3; ++c) {
                out[py::str(std::string{names[r]} + names[c])] = matrixToArray(blocks[r][c]);
            }
        }
        return out;
    }

    static py::dict tensorMetadataDict(const rcwa::TensorFourierMatrices::Metadata& metadata) {
        py::dict out;
        out[py::str("valid")] = metadata.valid;
        out[py::str("harmonic_count")] = static_cast<int>(metadata.harmonicCount);
        out[py::str("factorization_complete")] = metadata.factorizationComplete;
        out[py::str("scalar_epsilon_blocks")] = metadata.scalarEpsilonBlocks;
        out[py::str("common_scalar_mu_identity")] = metadata.commonScalarMuIdentity;
        out[py::str("common_scalar_mu")] =
            std::complex<double>{
                static_cast<double>(std::real(metadata.commonScalarMu)),
                static_cast<double>(std::imag(metadata.commonScalarMu))};
        out[py::str("diagonal_epsilon_blocks")] = metadata.diagonalEpsilonBlocks;
        out[py::str("tensor_factorization_applicability")] =
            tensorFactorizationApplicabilityName(
                metadata.tensorFactorizationApplicability);
        out[py::str("factorization_uses_sampled_geometry")] =
            metadata.factorizationUsesSampledGeometry;
        out[py::str("requires_joint_convergence")] =
            metadata.requiresJointConvergence;
        out[py::str("fourier_formulation")] =
            fourierFormulationName(metadata.formulation);
        out[py::str("lanczos_smoothing_applied")] =
            metadata.lanczosSmoothingApplied;
        out[py::str("formulation_resolution")] = metadata.formulationResolution;
        return out;
    }

    static const char* layerModeKindName(rcwa::LayerComputationKind kind) {
        switch (kind) {
        case rcwa::LayerComputationKind::UniformTmm:
            return "UniformTMM";
        case rcwa::LayerComputationKind::PatternedRcwa:
            return "PatternedRcwa";
        }
        return "Unknown";
    }

    static std::pair<std::vector<SpectrumAnglePolarization>, std::vector<Polarization>>
    anglePolarizationRequests(const std::vector<Real>& thetaValues,
                                const std::vector<Polarization>& polarizations,
                                const char* methodName) {
        if (polarizations.size() != 1 && polarizations.size() != thetaValues.size()) {
            throw std::invalid_argument(
                std::string(methodName) +
                " expects one polarization or one polarization per angle");
        }

        std::vector<SpectrumAnglePolarization> requests;
        std::vector<Polarization> requestPolarizations;
        requests.reserve(thetaValues.size());
        requestPolarizations.reserve(thetaValues.size());
        for (std::size_t i = 0; i < thetaValues.size(); ++i) {
            const Polarization pol =
                polarizations.size() == 1 ? polarizations.front() : polarizations[i];
            requests.push_back({thetaValues[i], pol});
            requestPolarizations.push_back(pol);
        }
        return {std::move(requests), std::move(requestPolarizations)};
    }

    static bool realClose(Real a, Real b) {
        const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
        return std::abs(a - b) <= Real{1e-10} * scale;
    }

    static bool complexClose(Complex a, Complex b) {
        const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
        return std::abs(a - b) <= Real{1e-10} * scale;
    }

    static bool isCircularAmplitudePair(Complex sAmplitude,
                                           Complex pAmplitude,
                                           Complex handedPhase) {
        const Real sAbs = std::abs(sAmplitude);
        const Real pAbs = std::abs(pAmplitude);
        if (sAbs == Real{0.0} || pAbs == Real{0.0} || !realClose(sAbs, pAbs)) {
            return false;
        }
        return complexClose(pAmplitude / sAmplitude, handedPhase);
    }

    template <typename SpectrumRow, typename PolarizationAt>
    static py::array spectrumArrayImplWith(const std::vector<SpectrumRow>& results,
                                              PolarizationAt polarizationAt) {
        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"pol", "U4"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"harmonics", "i4"},
            {"R0", "f8"},
            {"T0", "f8"},
            {"R", "f8"},
            {"T", "f8"},
            {"A", "f8"},
            {"conservation", "f8"},
        });
        py::array out = makeStructuredArray(dtype, results.size());
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const Polarization pol = polarizationAt(i);
            char* row = data + i * stride;
            writeUnaligned(row, 0, static_cast<double>(result.wavelengthUm));
            writeNumpyText(row, 8, rcwa::toString(pol), 4);
            writeUnaligned(row, 24, static_cast<double>(result.thetaDeg));
            writeUnaligned(row, 32, static_cast<double>(result.phiDeg));
            writeUnaligned(row, 40, static_cast<std::int32_t>(result.N));
            writeUnaligned(row, 44, static_cast<double>(result.R0));
            writeUnaligned(row, 52, static_cast<double>(result.T0));
            writeUnaligned(row, 60, static_cast<double>(result.rTotal));
            writeUnaligned(row, 68, static_cast<double>(result.tTotal));
            writeUnaligned(row, 76, static_cast<double>(
                1.0 - result.rTotal - result.tTotal));
            writeUnaligned(row, 84, static_cast<double>(result.conservation));
        }
        return out;
    }

    template <typename SpectrumRow>
    static py::array spectrumArrayImpl(const std::vector<SpectrumRow>& results,
                                         const std::vector<Polarization>& polarizations) {
        const std::size_t polCount = polarizations.empty() ? std::size_t{1} : polarizations.size();
        return spectrumArrayImplWith(results, [&](std::size_t i) {
            return polarizations.empty() ? Polarization::TE : polarizations[i % polCount];
        });
    }

    static py::array spectrumTotalsArray(
        const std::vector<rcwa::SpectrumTotalsResult>& results,
        const std::vector<Polarization>& polarizations) {
        return spectrumArrayImpl(results, polarizations);
    }

    static py::array spectrumTotalsArrayForAnglePolarizations(
        const std::vector<rcwa::SpectrumTotalsResult>& results,
        const std::vector<Polarization>& requestPolarizations,
        std::size_t wavelengthCount) {
        if (results.empty()) {
            return spectrumArrayImplWith(results, [](std::size_t) {
                return Polarization::TE;
            });
        }
        if (wavelengthCount == 0 ||
            results.size() != requestPolarizations.size() * wavelengthCount) {
            throw std::invalid_argument("angle/polarization spectrum result shape is inconsistent");
        }
        return spectrumArrayImplWith(results, [&](std::size_t i) {
            return requestPolarizations[i / wavelengthCount];
        });
    }

    template <typename PolarizationAt>
    static py::array diffractionOrdersArrayWith(
        const std::vector<SpectrumResult>& results,
        const std::vector<rcwa::HarmonicIndex>& orders,
        PolarizationAt polarizationAt) {
        std::size_t rowCount = 0;
        for (const auto& result : results) {
            if (result.rOrders.size() != result.tOrders.size()) {
                throw std::invalid_argument(
                    "diffraction-order R/T result sizes are inconsistent");
            }
            if (result.rOrders.size() != orders.size()) {
                throw std::invalid_argument(
                    "diffraction-order result size does not match harmonic basis");
            }
            rowCount += result.rOrders.size();
        }

        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"pol", "U4"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"harmonics", "i4"},
            {"order_index", "i4"},
            {"m", "i4"},
            {"n", "i4"},
            {"R", "f8"},
            {"T", "f8"},
        });
        py::array out = makeStructuredArray(dtype, rowCount);
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);

        std::size_t row = 0;
        for (std::size_t resultIndex = 0; resultIndex < results.size(); ++resultIndex) {
            const auto& result = results[resultIndex];
            const Polarization pol = polarizationAt(resultIndex);
            for (std::size_t orderIndex = 0; orderIndex < result.rOrders.size();
                 ++orderIndex) {
                const rcwa::HarmonicIndex order = orders[orderIndex];
                char* dst = data + row++ * stride;
                writeUnaligned(dst, 0, static_cast<double>(result.wavelengthUm));
                writeNumpyText(dst, 8, rcwa::toString(pol), 4);
                writeUnaligned(dst, 24, static_cast<double>(result.thetaDeg));
                writeUnaligned(dst, 32, static_cast<double>(result.phiDeg));
                writeUnaligned(dst, 40, static_cast<std::int32_t>(result.N));
                writeUnaligned(dst, 44, static_cast<std::int32_t>(orderIndex));
                writeUnaligned(dst, 48, static_cast<std::int32_t>(order.m));
                writeUnaligned(dst, 52, static_cast<std::int32_t>(order.n));
                writeUnaligned(dst, 56, static_cast<double>(result.rOrders[orderIndex]));
                writeUnaligned(dst, 64, static_cast<double>(result.tOrders[orderIndex]));
            }
        }
        return out;
    }

    static py::array diffractionOrdersArray(
        const std::vector<SpectrumResult>& results,
        const std::vector<Polarization>& polarizations,
        const std::vector<rcwa::HarmonicIndex>& orders) {
        const std::size_t polCount = polarizations.empty() ? std::size_t{1} : polarizations.size();
        return diffractionOrdersArrayWith(results, orders, [&](std::size_t i) {
            return polarizations.empty() ? Polarization::TE : polarizations[i % polCount];
        });
    }

    static py::array diffractionOrdersArrayForAnglePolarizations(
        const std::vector<SpectrumResult>& results,
        const std::vector<Polarization>& requestPolarizations,
        std::size_t wavelengthCount,
        const std::vector<rcwa::HarmonicIndex>& orders) {
        if (results.empty()) {
            return diffractionOrdersArrayWith(results, orders, [](std::size_t) {
                return Polarization::TE;
            });
        }
        if (wavelengthCount == 0 ||
            results.size() != requestPolarizations.size() * wavelengthCount) {
            throw std::invalid_argument(
                "angle/polarization diffraction-order result shape is inconsistent");
        }
        return diffractionOrdersArrayWith(results, orders, [&](std::size_t i) {
            return requestPolarizations[i / wavelengthCount];
        });
    }

    static const rcwa::Matrix& zeroOrderBlock(const ZeroOrderAmplitudeResult& result,
                                                int block) {
        switch (block) {
        case 0:
            return result.R;
        case 1:
            return result.T;
        }
        throw std::invalid_argument("unsupported zero-order amplitude block");
    }

    static const char* zeroOrderBlockName(int block) {
        switch (block) {
        case 0:
            return "R";
        case 1:
            return "T";
        }
        throw std::invalid_argument("unsupported zero-order amplitude block");
    }

    static py::array zeroOrderAmplitudeArray(
        const std::vector<ZeroOrderAmplitudeResult>& results) {
        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"harmonics", "i4"},
            {"block", "U1"},
            {"out_pol", "U2"},
            {"in_pol", "U2"},
            {"value_re", "f8"},
            {"value_im", "f8"},
            {"magnitude", "f8"},
            {"phase_rad", "f8"},
        });
        py::array out = makeStructuredArray(
            dtype,
            checkedProduct(results.size(), std::size_t{8}, "zero-order amplitude row"));
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);
        const std::array<Polarization, 2> pols{Polarization::TE, Polarization::TM};

        std::size_t row = 0;
        for (const auto& result : results) {
            for (int block = 0; block < 2; ++block) {
                const rcwa::Matrix& matrix = zeroOrderBlock(result, block);
                if (matrix.rows() != 2 || matrix.cols() != 2) {
                    throw std::invalid_argument("zero-order amplitude blocks must be 2x2");
                }
                for (std::size_t outPol = 0; outPol < pols.size(); ++outPol) {
                    for (std::size_t inPol = 0; inPol < pols.size(); ++inPol) {
                        const Complex value = matrix(outPol, inPol);
                        char* dst = data + row++ * stride;
                        writeUnaligned(dst, 0, static_cast<double>(result.wavelengthUm));
                        writeUnaligned(dst, 8, static_cast<double>(result.thetaDeg));
                        writeUnaligned(dst, 16, static_cast<double>(result.phiDeg));
                        writeUnaligned(dst, 24, static_cast<std::int32_t>(result.N));
                        writeNumpyText(dst, 28, zeroOrderBlockName(block), 1);
                        writeNumpyText(dst, 32, polarizationName(pols[outPol]), 2);
                        writeNumpyText(dst, 40, polarizationName(pols[inPol]), 2);
                        writeUnaligned(dst, 48, static_cast<double>(std::real(value)));
                        writeUnaligned(dst, 56, static_cast<double>(std::imag(value)));
                        writeUnaligned(dst, 64, static_cast<double>(std::abs(value)));
                        writeUnaligned(dst, 72, static_cast<double>(std::arg(value)));
                    }
                }
            }
        }
        return out;
    }

    static py::array directionalThermalChannelArray(
        const std::vector<DirectionalThermalChannelResult>& results,
        const std::vector<Polarization>& polarizations) {
        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"pol", "U4"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"harmonics", "i4"},
            {"absorptivity", "f8"},
            {"emissivity", "f8"},
            {"incident_scattering", "f8"},
            {"outgoing_scattering", "f8"},
        });
        py::array out = makeStructuredArray(
            dtype,
            checkedProduct(results.size(), polarizations.size(),
                           "directional thermal-channel row"));
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);

        std::size_t row = 0;
        for (const auto& result : results) {
            for (const Polarization polarization : polarizations) {
                Real absorptivity{};
                Real emissivity{};
                Real incidentScattering{};
                Real outgoingScattering{};
                switch (polarization) {
                case Polarization::TE:
                    absorptivity = result.teAbsorptivity;
                    emissivity = result.teEmissivity;
                    incidentScattering = result.teIncidentScattering;
                    outgoingScattering = result.teOutgoingScattering;
                    break;
                case Polarization::TM:
                    absorptivity = result.tmAbsorptivity;
                    emissivity = result.tmEmissivity;
                    incidentScattering = result.tmIncidentScattering;
                    outgoingScattering = result.tmOutgoingScattering;
                    break;
                case Polarization::LCP:
                    absorptivity = result.lcpAbsorptivity;
                    emissivity = result.lcpEmissivity;
                    incidentScattering = result.lcpIncidentScattering;
                    outgoingScattering = result.lcpOutgoingScattering;
                    break;
                case Polarization::RCP:
                    absorptivity = result.rcpAbsorptivity;
                    emissivity = result.rcpEmissivity;
                    incidentScattering = result.rcpIncidentScattering;
                    outgoingScattering = result.rcpOutgoingScattering;
                    break;
                case Polarization::Both:
                    throw std::logic_error("Both must be expanded before thermal shaping");
                }
                char* dst = data + row++ * stride;
                writeUnaligned(dst, 0, static_cast<double>(result.wavelengthUm));
                writeNumpyText(dst, 8, rcwa::toString(polarization), 4);
                writeUnaligned(dst, 24, static_cast<double>(result.thetaDeg));
                writeUnaligned(dst, 32, static_cast<double>(result.phiDeg));
                writeUnaligned(dst, 40, static_cast<std::int32_t>(result.N));
                writeUnaligned(dst, 44, static_cast<double>(absorptivity));
                writeUnaligned(dst, 52, static_cast<double>(emissivity));
                writeUnaligned(dst, 60, static_cast<double>(incidentScattering));
                writeUnaligned(dst, 68, static_cast<double>(outgoingScattering));
            }
        }
        return out;
    }

    static const rcwa::Matrix& sparameterBlock(const SParameterResult& result, int block) {
        switch (block) {
        case 0:
            return result.S11;
        case 1:
            return result.S12;
        case 2:
            return result.S21;
        case 3:
            return result.S22;
        }
        throw std::invalid_argument("unsupported S-parameter block");
    }

    static const char* sparameterBlockName(int block) {
        switch (block) {
        case 0:
            return "S11";
        case 1:
            return "S12";
        case 2:
            return "S21";
        case 3:
            return "S22";
        }
        throw std::invalid_argument("unsupported S-parameter block");
    }

    static int sparameterOutPort(int block) {
        return (block == 0 || block == 1) ? 1 : 2;
    }

    static int sparameterInPort(int block) {
        return (block == 0 || block == 2) ? 1 : 2;
    }

    static py::array sparameterArray(const SParameterResult& result) {
        const std::size_t harmonicCount = result.orders.size();
        const std::size_t channelCount =
            checkedProduct(std::size_t{2}, harmonicCount, "S-parameter channel");
        for (int block = 0; block < 4; ++block) {
            const rcwa::Matrix& matrix = sparameterBlock(result, block);
            if (matrix.rows() != channelCount || matrix.cols() != channelCount) {
                throw std::invalid_argument("S-parameter matrix dimensions do not match basis");
            }
        }

        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"harmonics", "i4"},
            {"block", "U3"},
            {"out_port", "i4"},
            {"in_port", "i4"},
            {"out_m", "i4"},
            {"out_n", "i4"},
            {"out_pol", "U2"},
            {"in_m", "i4"},
            {"in_n", "i4"},
            {"in_pol", "U2"},
            {"value_re", "f8"},
            {"value_im", "f8"},
            {"magnitude", "f8"},
            {"phase_rad", "f8"},
        });
        const std::size_t blockRows =
            checkedProduct(channelCount, channelCount, "S-parameter block row");
        py::array out = makeStructuredArray(
            dtype,
            checkedProduct(std::size_t{4}, blockRows, "S-parameter row"));
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);

        std::size_t row = 0;
        for (int block = 0; block < 4; ++block) {
            const rcwa::Matrix& matrix = sparameterBlock(result, block);
            for (std::size_t outChannel = 0; outChannel < channelCount; ++outChannel) {
                const bool outIsTe = outChannel < harmonicCount;
                const std::size_t outOrderIndex =
                    outIsTe ? outChannel : outChannel - harmonicCount;
                const rcwa::HarmonicIndex outOrder = result.orders[outOrderIndex];
                const Polarization outPol = outIsTe ? Polarization::TE : Polarization::TM;

                for (std::size_t inChannel = 0; inChannel < channelCount; ++inChannel) {
                    const bool inIsTe = inChannel < harmonicCount;
                    const std::size_t inOrderIndex =
                        inIsTe ? inChannel : inChannel - harmonicCount;
                    const rcwa::HarmonicIndex inOrder = result.orders[inOrderIndex];
                    const Polarization inPol = inIsTe ? Polarization::TE : Polarization::TM;
                    const Complex value = matrix(outChannel, inChannel);
                    char* dst = data + row++ * stride;
                    writeUnaligned(dst, 0, static_cast<double>(result.wavelengthUm));
                    writeUnaligned(dst, 8, static_cast<double>(result.thetaDeg));
                    writeUnaligned(dst, 16, static_cast<double>(result.phiDeg));
                    writeUnaligned(dst, 24, static_cast<std::int32_t>(result.N));
                    writeNumpyText(dst, 28, sparameterBlockName(block), 3);
                    writeUnaligned(dst, 40, static_cast<std::int32_t>(sparameterOutPort(block)));
                    writeUnaligned(dst, 44, static_cast<std::int32_t>(sparameterInPort(block)));
                    writeUnaligned(dst, 48, static_cast<std::int32_t>(outOrder.m));
                    writeUnaligned(dst, 52, static_cast<std::int32_t>(outOrder.n));
                    writeNumpyText(dst, 56, polarizationName(outPol), 2);
                    writeUnaligned(dst, 64, static_cast<std::int32_t>(inOrder.m));
                    writeUnaligned(dst, 68, static_cast<std::int32_t>(inOrder.n));
                    writeNumpyText(dst, 72, polarizationName(inPol), 2);
                    writeUnaligned(dst, 80, static_cast<double>(std::real(value)));
                    writeUnaligned(dst, 88, static_cast<double>(std::imag(value)));
                    writeUnaligned(dst, 96, static_cast<double>(std::abs(value)));
                    writeUnaligned(dst, 104, static_cast<double>(std::arg(value)));
                }
            }
        }
        return out;
    }

    static py::array fieldPlaneArray(const std::string& pol, const FieldPlaneResult& result) {
        py::dtype dtype = makeDtype({
            {"lambda_um", "f8"},
            {"pol", "U4"},
            {"theta_deg", "f8"},
            {"phi_deg", "f8"},
            {"plane", "U2"},
            {"component", "U4"},
            {"iu", "i4"},
            {"iv", "i4"},
            {"layer_index", "i4"},
            {"x_um", "f8"},
            {"y_um", "f8"},
            {"z_um", "f8"},
            {"z_local_um", "f8"},
            {"value_re", "f8"},
            {"value_im", "f8"},
            {"magnitude", "f8"},
        });
        py::array out = makeStructuredArray(dtype, result.samples.size());
        py::buffer_info buffer = out.request();
        const auto stride = static_cast<std::size_t>(buffer.strides[0]);
        char* data = static_cast<char*>(buffer.ptr);
        for (std::size_t i = 0; i < result.samples.size(); ++i) {
            const auto& sample = result.samples[i];
            char* row = data + i * stride;
            writeUnaligned(row, 0, static_cast<double>(result.wavelengthUm));
            writeNumpyText(row, 8, pol, 4);
            writeUnaligned(row, 24, static_cast<double>(result.thetaDeg));
            writeUnaligned(row, 32, static_cast<double>(result.phiDeg));
            writeNumpyText(row, 40, rcwa::fieldPlaneName(result.plane), 2);
            writeNumpyText(row, 48, rcwa::fieldComponentName(sample.component), 4);
            writeUnaligned(row, 64, static_cast<std::int32_t>(sample.iu));
            writeUnaligned(row, 68, static_cast<std::int32_t>(sample.iv));
            writeUnaligned(row, 72, static_cast<std::int32_t>(sample.layerIndex));
            writeUnaligned(row, 76, static_cast<double>(sample.xUm));
            writeUnaligned(row, 84, static_cast<double>(sample.yUm));
            writeUnaligned(row, 92, static_cast<double>(sample.zUm));
            writeUnaligned(row, 100, static_cast<double>(sample.zLocalUm));
            writeUnaligned(row, 108, static_cast<double>(std::real(sample.value)));
            writeUnaligned(row, 116, static_cast<double>(std::imag(sample.value)));
            writeUnaligned(row, 124, static_cast<double>(sample.magnitude));
        }
        return out;
    }

    static py::dict fieldGridDict(const std::string& pol, FieldGridResult result) {
        const auto componentCount = static_cast<py::ssize_t>(result.components.size());
        const auto vPoints = static_cast<py::ssize_t>(result.vPoints);
        const auto uPoints = static_cast<py::ssize_t>(result.uPoints);
        auto values = std::make_unique<std::vector<Complex>>(std::move(result.values));
        Complex* valueData = values->data();
        py::capsule valueOwner(values.release(), [](void* pointer) {
            delete static_cast<std::vector<Complex>*>(pointer);
        });
        py::array valueArray(
            py::dtype::of<Complex>(),
            std::vector<py::ssize_t>{componentCount, vPoints, uPoints},
            std::vector<py::ssize_t>{
                vPoints * uPoints * static_cast<py::ssize_t>(sizeof(Complex)),
                uPoints * static_cast<py::ssize_t>(sizeof(Complex)),
                static_cast<py::ssize_t>(sizeof(Complex))},
            valueData,
            valueOwner);

        const auto realArray = [](const std::vector<Real>& values) {
            py::array_t<Real> out(values.size());
            std::copy(values.begin(), values.end(), out.mutable_data());
            return out;
        };
        py::array_t<std::int32_t> layerIndex(result.layerIndexByV.size());
        auto* layerIndexData = layerIndex.mutable_data();
        for (std::size_t i = 0; i < result.layerIndexByV.size(); ++i) {
            layerIndexData[i] = static_cast<std::int32_t>(result.layerIndexByV[i]);
        }
        py::tuple components(result.components.size());
        for (std::size_t i = 0; i < result.components.size(); ++i) {
            components[i] = py::str(rcwa::fieldComponentName(result.components[i]));
        }

        const char* uAxis = "x";
        const char* vAxis = "y";
        const char* fixedAxis = "z";
        if (result.plane == FieldPlane::XZ) {
            vAxis = "z";
            fixedAxis = "y";
        } else if (result.plane == FieldPlane::YZ) {
            uAxis = "y";
            vAxis = "z";
            fixedAxis = "x";
        }

        py::dict out;
        out[py::str("values")] = std::move(valueArray);
        out[py::str("u_um")] = realArray(result.uUm);
        out[py::str("v_um")] = realArray(result.vUm);
        out[py::str("layer_index")] = std::move(layerIndex);
        out[py::str("z_local_um")] = realArray(result.zLocalUmByV);
        out[py::str("components")] = std::move(components);
        out[py::str("u_axis")] = uAxis;
        out[py::str("v_axis")] = vAxis;
        out[py::str("fixed_axis")] = fixedAxis;
        out[py::str("fixed_um")] = result.fixedUm;
        out[py::str("plane")] = rcwa::fieldPlaneName(result.plane);
        out[py::str("wavelength_um")] = result.wavelengthUm;
        out[py::str("polarization")] = pol;
        out[py::str("theta_deg")] = result.thetaDeg;
        out[py::str("phi_deg")] = result.phiDeg;
        out[py::str("harmonics")] = result.N;
        return out;
    }

    const rcwa::Material& requireMaterial(const std::string& name) const {
        const auto it = mMaterials.find(name);
        if (it == mMaterials.end()) {
            throw std::invalid_argument("unknown material: " + name);
        }
        return it->second;
    }

    Layer& layerByName(const std::string& name) {
        for (auto& layer : mLayers) {
            if (layer.name == name) {
                return layer;
            }
        }
        throw std::invalid_argument("unknown layer: " + name);
    }

    const Layer& layerByName(const std::string& name) const {
        for (const auto& layer : mLayers) {
            if (layer.name == name) {
                return layer;
            }
        }
        throw std::invalid_argument("unknown layer: " + name);
    }

    std::size_t finiteLayerIndexByName(const std::string& name) const {
        for (std::size_t index = 0; index < mLayers.size(); ++index) {
            if (mLayers[index].name == name) {
                return index;
            }
        }
        throw std::invalid_argument("unknown layer: " + name);
    }

    void clearCachedLayers() {
        for (auto& layer : mLayers) {
            layer.cachedPrecomputed.reset();
            layer.cachedGeometry.reset();
            layer.cachedMaterialDependencies.clear();
        }
    }

    bool materialIsDispersive(const std::string& name) const {
        return requireMaterial(name).isDispersive();
    }

    bool layerDependsOnDispersiveMaterial(const Layer& layer) const {
        if (materialIsDispersive(layer.background)) {
            return true;
        }
        for (const auto& region : layer.regions) {
            if (materialIsDispersive(region.material)) {
                return true;
            }
        }
        return false;
    }

    void assignMaterial(const std::string& name, rcwa::Material material) {
        const auto old = mMaterials.find(name);
        if (old != mMaterials.end() &&
            old->second.isDispersive() == material.isDispersive() &&
            materialsIdenticalAtReferenceWavelength(old->second, material)) {
            old->second = std::move(material);
            return;
        }
        mMaterials[name] = std::move(material);
        invalidateLayersUsingMaterial(name);
    }

    void invalidateLayersUsingMaterial(const std::string& material) {
        for (auto& layer : mLayers) {
            if (layer.cachedPrecomputed &&
                layer.cachedMaterialDependencies.find(material) ==
                    layer.cachedMaterialDependencies.end()) {
                continue;
            }
            if (layer.background == material) {
                layer.cachedPrecomputed.reset();
                layer.cachedMaterialDependencies.clear();
                continue;
            }
            for (const auto& region : layer.regions) {
                if (region.material == material) {
                    layer.cachedPrecomputed.reset();
                    layer.cachedMaterialDependencies.clear();
                    break;
                }
            }
        }
    }

    rcwa::LayerSpec layerSpecForAnalysis(
        const Layer& layer,
        Real wavelengthUm,
        bool useCachedPatternGeometry) const {
        if (layer.regions.empty()) {
            return rcwa::UniformLayer{requireMaterial(layer.background), layer.thicknessUm};
        }
        return patternedLayer(layer, useCachedPatternGeometry, wavelengthUm);
    }

    rcwa::TensorFourierMatrices layerFourierMatrices(const Layer& layer,
                                                       Real wavelengthUm) const {
        const auto basis = harmonicBasis();
        if (!layer.regions.empty()) {
            return rcwa::tensorMatricesForLayer(
                layerSpecForAnalysis(layer, wavelengthUm, true),
                basis,
                wavelengthUm);
        }
        return rcwa::tensorMatricesForLayer(
            layerSpecForAnalysis(layer, wavelengthUm, false),
            basis,
            wavelengthUm);
    }

    rcwa::LayerModes layerModes(const Layer& layer,
                                 Real wavelengthUm,
                                 const rcwa::DiagonalOperator& Kx,
                                 const rcwa::DiagonalOperator& Ky,
                                 const rcwa::HarmonicBasis& basis) const {
        return rcwa::computeLayerModes(
            layerSpecForAnalysis(layer, wavelengthUm, true),
            Kx,
            Ky,
            basis,
            wavelengthUm);
    }

    void warmLayerCache(const Layer& layer, Real wavelengthUm) const {
        if (layer.regions.empty()) {
            return;
        }
        const rcwa::PatternedLayer2D patterned = patternedLayer(layer, false, wavelengthUm);
        if (rcwa::patternedLayerHasTensorBoundary(patterned, wavelengthUm) ||
            patterned.regions.size() > 1) {
            (void)patternedGeometryCache(layer, patterned);
        }
        (void)precomputeLayer(layer, wavelengthUm);
    }

    py::dict layerCacheInfo(const Layer& layer) const {
        std::lock_guard<std::mutex> lock(layer.cacheMutex);
        py::dict out;
        out[py::str("layer")] = layer.name;
        out[py::str("kind")] = layer.regions.empty() ? "Uniform" : "Patterned";
        out[py::str("region_count")] = static_cast<int>(layer.regions.size());
        out[py::str("has_precomputed_tensor_cache")] =
            static_cast<bool>(layer.cachedPrecomputed);
        out[py::str("has_geometry_cache")] = static_cast<bool>(layer.cachedGeometry);
        out[py::str("sample_count_x")] = layer.cachedGeometry
            ? layer.cachedGeometry->sampleCountX
            : 0;
        out[py::str("sample_count_y")] = layer.cachedGeometry
            ? layer.cachedGeometry->sampleCountY
            : 0;
        out[py::str("material_slot_count")] = layer.cachedGeometry
            ? static_cast<int>(layer.cachedGeometry->materialSlotCount)
            : 0;
        out[py::str("cached_material_dependencies")] =
            py::cast(std::vector<std::string>(
                layer.cachedMaterialDependencies.begin(),
                layer.cachedMaterialDependencies.end()));
        return out;
    }

    py::list layerCacheInfoList(py::object layerName) const {
        py::list out;
        if (layerName.is_none()) {
            for (const auto& layer : mLayers) {
                out.append(layerCacheInfo(layer));
            }
            return out;
        }
        out.append(layerCacheInfo(layerByName(layerName.cast<std::string>())));
        return out;
    }

    rcwa::PatternRegion patternRegionFromRegion(const Region& region) const {
        switch (region.shape) {
        case Region::Shape::Circle:
            return rcwa::PatternRegion::circle(
                requireMaterial(region.material),
                {region.x, region.y},
                region.radius,
                region.reticoloRectangleCount);
        case Region::Shape::Rectangle:
            return rcwa::PatternRegion::rectangle(
                requireMaterial(region.material),
                {region.x, region.y},
                region.angleDeg,
                {region.halfwidthX, region.halfwidthY});
        case Region::Shape::Polygon: {
            rcwa::PatternRegion out;
            out.shape = rcwa::PatternRegionShape::Polygon;
            out.material = requireMaterial(region.material);
            out.center = {0.0, 0.0};
            out.vertices = region.vertices;
            return out;
        }
        }
        throw std::invalid_argument("unsupported pattern region shape");
    }

    std::shared_ptr<const rcwa::PatternedLayer2DGeometryCache> patternedGeometryCache(
        const Layer& layer,
        const rcwa::PatternedLayer2D& patterned) const {
        std::lock_guard<std::mutex> lock(layer.cacheMutex);
        if (!layer.cachedGeometry) {
            layer.cachedGeometry =
                std::make_shared<const rcwa::PatternedLayer2DGeometryCache>(
                    rcwa::precomputePatternedLayer2dGeometry(
                        patterned,
                        harmonicBasis()));
        }
        return layer.cachedGeometry;
    }

    rcwa::PatternedLayer2D patternedLayer(
        const Layer& layer,
        bool attachGeometryCache = false,
        Real wavelengthUm = Real{1.0}) const {
        rcwa::PatternedLayer2D out;
        out.background = requireMaterial(layer.background);
        out.periodXUm = mPeriodXUm;
        out.periodYUm = mPeriodYUm;
        out.thicknessUm = layer.thicknessUm;
        const int defaultSampleCount = rcwa::recommendedPatternSampleCount(harmonicBasis());
        out.sampleCountX = defaultSampleCount;
        out.sampleCountY = defaultSampleCount;
        out.preferAnalytic = true;
        out.fourierOptions = mFourierOptions;
        out.regions.reserve(layer.regions.size());
        for (const auto& region : layer.regions) {
            out.regions.push_back(patternRegionFromRegion(region));
        }
        if (attachGeometryCache &&
            (out.regions.size() > 1 ||
             rcwa::patternedLayerHasTensorBoundary(out, wavelengthUm))) {
            out.geometryCache = patternedGeometryCache(layer, out);
        }
        return out;
    }

    std::shared_ptr<const rcwa::PrecomputedPeriodicLayer2D> precomputeLayer(
        const Layer& layer,
        Real wavelengthUm) const {
        const bool wavelengthIndependent = !layerDependsOnDispersiveMaterial(layer);
        if (wavelengthIndependent) {
            std::lock_guard<std::mutex> lock(layer.cacheMutex);
            if (!layer.cachedPrecomputed) {
                layer.cachedPrecomputed = std::make_shared<const rcwa::PrecomputedPeriodicLayer2D>(
                    rcwa::precomputePatternedLayer2d(
                        patternedLayer(layer, false, wavelengthUm),
                        harmonicBasis(),
                        wavelengthUm));
                layer.cachedMaterialDependencies.clear();
                layer.cachedMaterialDependencies.insert(layer.background);
                for (const auto& region : layer.regions) {
                    layer.cachedMaterialDependencies.insert(region.material);
                }
            }
            return layer.cachedPrecomputed;
        }

        return std::make_shared<const rcwa::PrecomputedPeriodicLayer2D>(
            rcwa::precomputePatternedLayer2d(
                patternedLayer(layer, false, wavelengthUm),
                harmonicBasis(),
                wavelengthUm));
    }

    rcwa::RcwaSolver makeSolver(Real wavelengthUm, Polarization pol) const {
        return makeSolverImpl(wavelengthUm, pol, false);
    }

    rcwa::RcwaSolver makeBatchSolver(Real referenceWavelengthUm, Polarization pol) const {
        return makeSolverImpl(referenceWavelengthUm, pol, true);
    }

    rcwa::RcwaSolver makeSolverImpl(
        Real wavelengthUm,
        Polarization pol,
        bool deferDispersivePatternedLayers) const {
        rcwa::RcwaSolver solver;
        solver.setWavelength(wavelengthUm);
        solver.setIncidence(mThetaDeg, mPhiDeg);
        solver.setHarmonicOrders(
            mOrderX,
            mOrderY,
            LatticeTruncation::Parallelogramic);
        solver.setPolarization(pol);
        solver.setStackingAlgorithm(mStackingAlgorithm);
        solver.setFourierConvergenceOptions(mFourierOptions);
        solver.setSuperstrate(requireMaterial(mSuperstrateName));
        solver.setSubstrate(requireMaterial(mSubstrateName));
        for (const auto& layer : mLayers) {
            if (layer.regions.empty()) {
                solver.addUniformLayer(requireMaterial(layer.background), layer.thicknessUm);
            } else if (deferDispersivePatternedLayers &&
                       layerDependsOnDispersiveMaterial(layer)) {
                solver.addPatternedLayer2d(patternedLayer(layer, true, wavelengthUm));
            } else {
                solver.addPrecomputedPeriodicLayer2d(precomputeLayer(layer, wavelengthUm));
            }
        }
        return solver;
    }

    rcwa::SpectrumResult solveOne(Polarization pol) const {
        auto solver = makeSolver(mWavelengthUm, pol);
        return solver.solveSpectrumOnly();
    }

    Real mPeriodXUm{1.0};
    Real mPeriodYUm{1.0};
    int mOrderX{0};
    int mOrderY{0};
    Real mWavelengthUm{1.0};
    Real mThetaDeg{0.0};
    Real mPhiDeg{0.0};
    Polarization mPolarization{Polarization::TE};
    rcwa::StackingAlgorithm mStackingAlgorithm{
        rcwa::StackingAlgorithm::ScatteringMatrix};
    rcwa::FourierConvergenceOptions mFourierOptions{};
    std::map<std::string, rcwa::Material> mMaterials{
        {"Air", rcwa::Material::air()},
        {"Vacuum", rcwa::Material::vacuum()},
    };
    std::string mSuperstrateName{"Air"};
    std::string mSubstrateName{"Air"};
    std::vector<Layer> mLayers;
};

py::array wavelengthGrid(Real startUm, Real stopUm, int points) {
    if (points <= 0) {
        throw std::invalid_argument("points must be positive");
    }
    if (!std::isfinite(startUm) || !std::isfinite(stopUm)) {
        throw std::invalid_argument("wavelength grid bounds must be finite");
    }
    py::array_t<Real> out(static_cast<py::ssize_t>(points));
    auto view = out.mutable_unchecked<1>();
    if (points == 1) {
        view(0) = startUm;
        return out;
    }
    for (int i = 0; i < points; ++i) {
        const Real t = static_cast<Real>(i) / static_cast<Real>(points - 1);
        view(i) = startUm + (stopUm - startUm) * t;
    }
    return out;
}

py::array_t<std::complex<double>> tensorToArray(const rcwa::Tensor3& tensor) {
    py::array_t<std::complex<double>> out({py::ssize_t{3}, py::ssize_t{3}});
    auto view = out.mutable_unchecked<2>();
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const Complex value = tensor(r, c);
            view(static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c)) =
                std::complex<double>{static_cast<double>(std::real(value)),
                                     static_cast<double>(std::imag(value))};
        }
    }
    return out;
}

py::dict materialModelTensors(const std::string& model,
                              Real wavelengthUm,
                              py::object parameters) {
    const rcwa::Material material =
        builtinMaterialModelFromObject("material", model, parameters);
    const auto [epsilon, mu] = material.tensors(wavelengthUm);
    py::dict out;
    out[py::str("epsilon")] = tensorToArray(epsilon);
    out[py::str("mu")] = tensorToArray(mu);
    return out;
}

py::dict materialModelTensorsBatch(const std::string& model,
                                   py::object wavelengths,
                                   py::object parameters) {
    const auto wavelengthValues = realVectorFromObject(
        wavelengths,
        "wavelengths_um",
        true);
    const rcwa::Material material =
        builtinMaterialModelFromObject("material", model, parameters);
    py::array_t<std::complex<double>> epsilon({
        static_cast<py::ssize_t>(wavelengthValues.size()),
        py::ssize_t{3},
        py::ssize_t{3}});
    py::array_t<std::complex<double>> mu({
        static_cast<py::ssize_t>(wavelengthValues.size()),
        py::ssize_t{3},
        py::ssize_t{3}});
    auto epsilonView = epsilon.mutable_unchecked<3>();
    auto muView = mu.mutable_unchecked<3>();
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < wavelengthValues.size(); ++index) {
            const auto [epsTensor, muTensor] =
                material.tensors(wavelengthValues[index]);
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    const Complex epsValue = epsTensor(row, column);
                    const Complex muValue = muTensor(row, column);
                    epsilonView(
                        static_cast<py::ssize_t>(index),
                        static_cast<py::ssize_t>(row),
                        static_cast<py::ssize_t>(column)) =
                        std::complex<double>{
                            static_cast<double>(std::real(epsValue)),
                            static_cast<double>(std::imag(epsValue))};
                    muView(
                        static_cast<py::ssize_t>(index),
                        static_cast<py::ssize_t>(row),
                        static_cast<py::ssize_t>(column)) =
                        std::complex<double>{
                            static_cast<double>(std::real(muValue)),
                            static_cast<double>(std::imag(muValue))};
                }
            }
        }
    }
    py::dict out;
    out[py::str("epsilon")] = std::move(epsilon);
    out[py::str("mu")] = std::move(mu);
    return out;
}

Simulation newSimulation(py::object lattice, py::object orders) {
    return Simulation(std::move(lattice), std::move(orders));
}

} // namespace

void bindRcwaCpp(py::module_& m) {
    py::class_<Simulation>(m, "Simulation")
        .def("SetMaterial",
             &Simulation::setMaterial,
             py::arg("Name"),
             py::arg("Epsilon"),
             py::arg("Mu") = py::none())
        .def("SetMaterialModel",
             [](Simulation& self,
                const std::string& name,
                const std::string& model,
                py::object parameters) {
                 self.setMaterialModel(name, model, parameters);
             },
             py::arg("Name"),
             py::arg("Model"),
             py::arg("Parameters") = py::none())
        .def("SetMaterialNK",
             &Simulation::setMaterialNk,
             py::arg("Name"),
             py::arg("Path"),
             py::arg("WavelengthUnit") = "um",
             py::arg("Extrapolation") = "error")
        .def("SetSuperstrate", &Simulation::setSuperstrate, py::arg("Material"))
        .def("SetSubstrate", &Simulation::setSubstrate, py::arg("Material"))
        .def("SetStackingAlgorithm",
             &Simulation::setStackingAlgorithm,
             py::arg("Algorithm"))
        .def("AddLayer",
             &Simulation::addLayer,
             py::arg("Name"),
             py::arg("Thickness"),
             py::arg("Material"))
        .def("SetRegionCircle",
             &Simulation::setRegionCircle,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Radius"),
             py::arg("Rectangles") = 10)
        .def("SetRegionRectangle",
             &Simulation::setRegionRectangle,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Halfwidths"))
        .def("SetRegionPolygon",
             &Simulation::setRegionPolygon,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Vertices"))
        .def("SetRegionEllipse",
             &Simulation::setRegionEllipse,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Radii"),
             py::arg("Vertices") = 96)
        .def("SetRegionRegularPolygon",
             &Simulation::setRegionRegularPolygon,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Sides"),
             py::arg("Radius"))
        .def("SetRegionTriangle",
             &Simulation::setRegionTriangle,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Radius"))
        .def("SetRegionCross",
             &Simulation::setRegionCross,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("ArmLength"),
             py::arg("ArmWidth"))
        .def("SetRegionRing",
             &Simulation::setRegionRing,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("HoleMaterial"),
             py::arg("Center"),
             py::arg("OuterRadius"),
             py::arg("InnerRadius"))
        .def("SetRegionRoundedRectangle",
             &Simulation::setRegionRoundedRectangle,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Halfwidths"),
             py::arg("CornerRadius"),
             py::arg("VerticesPerCorner") = 16)
        .def("SetRegionCapsule",
             &Simulation::setRegionCapsule,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Length"),
             py::arg("Radius"),
             py::arg("VerticesPerCap") = 32)
        .def("SetRegionStripe",
             &Simulation::setRegionStripe,
             py::arg("Layer"),
             py::arg("Material"),
             py::arg("Center"),
             py::arg("Angle"),
             py::arg("Width"))
        .def("SetExcitationPlanewave",
             &Simulation::setExcitationPlanewave,
             py::arg("Angles"),
             py::arg("sAmplitude") = Complex{1.0, 0.0},
             py::arg("pAmplitude") = Complex{0.0, 0.0},
             py::arg("Order") = 0)
        .def("SetFrequency", &Simulation::setFrequency, py::arg("Frequency"))
        .def("SetWavelength", &Simulation::setWavelength, py::arg("Wavelength"))
        .def("SetHarmonicOrders",
             &Simulation::setHarmonicOrders,
             py::arg("OrderX"),
             py::arg("OrderY"))
        .def("GetPowerFlux",
             &Simulation::getPowerFlux,
             py::arg("Layer") = py::none(),
             py::arg("zOffset") = 0.0)
        .def("GetHarmonicBasis",
             &Simulation::getHarmonicBasis)
        .def("GetWavevectorMatrices",
             &Simulation::getWavevectorMatrices,
             py::arg("Wavelength") = py::none(),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none())
        .def("GetLayerFourierMatrices",
             &Simulation::getLayerFourierMatrices,
             py::arg("Layer"),
             py::arg("Wavelength") = py::none())
        .def("GetLayerModes",
             &Simulation::getLayerModes,
             py::arg("Layer"),
             py::arg("Wavelength") = py::none(),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none())
        .def("FindAbsorptivityPeak",
             &Simulation::findAbsorptivityPeak,
             py::arg("WavelengthBracket"),
             py::arg("Polarization"),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none(),
             py::arg("GridPoints") = 17,
             py::arg("Refinements") = 3,
             py::arg("Workers") = 1)
        .def("PrecomputeLayerCache",
             &Simulation::precomputeLayerCache,
             py::arg("Layer") = py::none(),
             py::arg("Wavelength") = py::none())
        .def("GetLayerCacheInfo",
             &Simulation::getLayerCacheInfo,
             py::arg("Layer") = py::none())
        .def("GetSpectrum",
             &Simulation::getSpectrum,
             py::arg("Wavelengths"),
             py::arg("Polarizations") = py::none(),
             py::arg("Workers") = 1)
        .def("GetDiffractionOrders",
             &Simulation::getDiffractionOrders,
             py::arg("Wavelengths"),
             py::arg("Polarizations") = py::none(),
             py::arg("Workers") = 1)
        .def("GetSpectrumForAngles",
             &Simulation::getSpectrumForAngles,
             py::arg("Wavelengths"),
             py::arg("Angles"),
             py::arg("Polarizations") = py::none(),
             py::arg("Phi") = 0.0,
             py::arg("Workers") = 1)
        .def("GetSpectrumAndDirectionalThermalChannelsForAngles",
             &Simulation::getSpectrumAndDirectionalThermalChannelsForAngles,
             py::arg("Wavelengths"),
             py::arg("Angles"),
             py::arg("Polarizations") = py::none(),
             py::arg("Phi") = 0.0,
             py::arg("Workers") = 1)
        .def("GetDiffractionOrdersForAngles",
             &Simulation::getDiffractionOrdersForAngles,
             py::arg("Wavelengths"),
             py::arg("Angles"),
             py::arg("Polarizations") = py::none(),
             py::arg("Phi") = 0.0,
             py::arg("Workers") = 1)
        .def("GetSpectrumForAnglePolarizationPairs",
             &Simulation::getSpectrumForAnglePolarizationPairs,
             py::arg("Wavelengths"),
             py::arg("Angles"),
             py::arg("Polarizations"),
             py::arg("Phi") = 0.0,
             py::arg("Workers") = 1)
        .def("GetDiffractionOrdersForAnglePolarizationPairs",
             &Simulation::getDiffractionOrdersForAnglePolarizationPairs,
             py::arg("Wavelengths"),
             py::arg("Angles"),
             py::arg("Polarizations"),
             py::arg("Phi") = 0.0,
             py::arg("Workers") = 1)
        .def("GetSParameters",
             &Simulation::getSParameters,
             py::arg("Wavelength"),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none())
        .def("GetSParametersAnalyticContinuationAtGamma",
             &Simulation::getSParametersAnalyticContinuationAtGamma,
             py::arg("Wavelength"),
             "Return the complete amplitude S matrix at Gamma without real-power "
             "normalization. This is intended for complex-frequency analytic "
             "continuation represented through complex-scaled epsilon and mu.")
        .def("GetSParametersAnalyticContinuation",
             &Simulation::getSParametersAnalyticContinuation,
             py::arg("Wavelength"),
             py::arg("BlochWavevector"),
             "Return the complete unnormalized amplitude S matrix at a direct "
             "Bloch wavevector (kx*a_x/(2*pi), ky*a_y/(2*pi)).")
        .def("GetZeroOrderAmplitudes",
             &Simulation::getZeroOrderAmplitudes,
             py::arg("Wavelengths"),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none(),
             py::arg("Workers") = 1)
        .def("GetDirectionalThermalChannels",
             &Simulation::getDirectionalThermalChannels,
             py::arg("Wavelengths"),
             py::arg("Theta") = py::none(),
             py::arg("Phi") = py::none(),
             py::arg("Workers") = 1,
             py::arg("Polarizations") = py::none())
        .def("GetFieldPlane",
             &Simulation::getFieldPlane,
             py::arg("Wavelength"),
             py::arg("Polarization"),
             py::arg("Theta"),
             py::arg("Phi"),
             py::arg("Components"),
             py::arg("Plane"),
             py::arg("u_min_um"),
             py::arg("u_max_um"),
             py::arg("u_points"),
             py::arg("v_min_um"),
             py::arg("v_max_um"),
             py::arg("v_points"),
             py::arg("fixed_um"),
             py::arg("Workers") = 1)
        .def("GetFieldGrid",
             &Simulation::getFieldGrid,
             py::arg("Wavelength"),
             py::arg("Polarization"),
             py::arg("Theta"),
             py::arg("Phi"),
             py::arg("Components"),
             py::arg("Plane"),
             py::arg("u_min_um"),
             py::arg("u_max_um"),
             py::arg("u_points"),
             py::arg("v_min_um"),
             py::arg("v_max_um"),
             py::arg("v_points"),
             py::arg("fixed_um"),
             py::arg("Workers") = 1);

    py::enum_<rcwa::StackingAlgorithm>(m, "StackingAlgorithm")
        .value("ScatteringMatrix", rcwa::StackingAlgorithm::ScatteringMatrix)
        .value(
            "EnhancedTransmittanceMatrix",
            rcwa::StackingAlgorithm::EnhancedTransmittanceMatrix)
        .export_values();

    m.def(
        "New",
        &newSimulation,
        py::arg("Lattice") = py::float_(1.0),
        py::arg("Orders") = py::none(),
        "Create the single simulation interface. Add every finite layer with "
        "AddLayer; a layer remains uniform when it has no regions and becomes "
        "patterned when SetRegion... is called. Orders=(Nx, Ny) always uses "
        "the RETICOLO rectangular harmonic basis and ordered-Li formulation.");

    m.def(
        "MaterialModelTensors",
        &materialModelTensors,
        py::arg("Model"),
        py::arg("Wavelength_um"),
        py::arg("Parameters") = py::none(),
        "Evaluate a built-in material model at one wavelength and return epsilon/mu tensors.");

    m.def(
        "MaterialModelTensorsBatch",
        &materialModelTensorsBatch,
        py::arg("Model"),
        py::arg("Wavelengths_um"),
        py::arg("Parameters") = py::none(),
        "Evaluate a built-in material model for a wavelength array in one native call.");

    m.def("wavelength_grid", &wavelengthGrid, py::arg("start_um"), py::arg("stop_um"), py::arg("points"));
}
