#pragma once

#include <vector>

#include "rcwa/types.hpp"

namespace rcwa {

enum class FieldComponent {
    Ex,
    Ey,
    Ez,
    E,
    Hx,
    Hy,
    Hz,
    H,
};

enum class FieldPlane {
    XY,
    XZ,
    YZ,
};

struct FieldValue {
    Complex Ex{};
    Complex Ey{};
    Complex Ez{};
    Complex Hx{};
    Complex Hy{};
    Complex Hz{};
};

[[nodiscard]] Complex fieldComponentValue(const FieldValue& value, FieldComponent component);

struct FieldPlaneRequest {
    FieldPlane plane{FieldPlane::XY};
    Real uMinUm{};
    Real uMaxUm{};
    int uPoints{1};
    Real vMinUm{};
    Real vMaxUm{};
    int vPoints{1};
    // Fixed coordinate: z for XY, y for XZ, x for YZ.
    Real fixedUm{};
    std::vector<FieldComponent> components{FieldComponent::E};
    int workers{1};
};

struct FieldPlaneSample {
    int iu{};
    int iv{};
    int layerIndex{};
    Real xUm{};
    Real yUm{};
    Real zUm{};
    Real zLocalUm{};
    FieldComponent component{FieldComponent::E};
    Complex value{};
    Real magnitude{};
};

struct FieldPlaneResult {
    std::vector<FieldPlaneSample> samples;
    FieldPlane plane{FieldPlane::XY};
    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
    int uPoints{};
    int vPoints{};
};

// Compact component-major field storage. Values are contiguous in
// [component][v][u] order; axes and depth metadata are stored only once.
struct FieldGridResult {
    std::vector<Complex> values;
    std::vector<Real> uUm;
    std::vector<Real> vUm;
    std::vector<int> layerIndexByV;
    std::vector<Real> zLocalUmByV;
    std::vector<FieldComponent> components;
    FieldPlane plane{FieldPlane::XY};
    Real fixedUm{};
    Real wavelengthUm{};
    Real thetaDeg{};
    Real phiDeg{};
    int N{};
    int uPoints{};
    int vPoints{};
};

[[nodiscard]] const char* fieldComponentName(FieldComponent component);
[[nodiscard]] const char* fieldPlaneName(FieldPlane plane);

} // namespace rcwa
