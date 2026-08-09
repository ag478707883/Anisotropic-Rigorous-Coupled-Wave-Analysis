#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rcwa/fourier.hpp"
#include "rcwa/material.hpp"
#include "rcwa/types.hpp"

namespace rcwa {

struct PeriodicLayer2D {
    std::vector<Material> cells;
    int sampleCountX{};
    int sampleCountY{};
    Real periodXUm{1.0};
    Real periodYUm{1.0};
    Real thicknessUm{0.3};
    FourierConvergenceOptions fourierOptions{};
};

struct PrecomputedPeriodicLayer2D {
    TensorFourierMatrices tensors;
    // Retained so thermal-channel solves can still verify material passivity
    // after the geometry/Fourier blocks have been precomputed. Direct callers
    // that construct only opaque blocks leave this empty and cannot use the
    // thermodynamic thermal-channel API.
    std::vector<Material> sourceMaterials;
    Real periodXUm{1.0};
    Real periodYUm{1.0};
    Real thicknessUm{0.3};
};

struct PatternedLayer2DGeometryCache {
    std::vector<std::size_t> sampleMaterialIndices;
    std::size_t materialSlotCount{};
    int sampleCountX{};
    int sampleCountY{};
    int coeffCountX{};
    int coeffCountY{};
    std::uint64_t geometryFingerprint{};
};

enum class PatternRegionShape {
    Circle,
    Rectangle,
    Polygon,
};

struct PatternRegion {
    PatternRegionShape shape{PatternRegionShape::Circle};
    Material material;
    Vec2 center{};
    Real angleDeg{};
    Real radiusUm{};
    Real halfwidthXUm{};
    Real halfwidthYUm{};
    int reticoloRectangleCount{10};
    std::vector<Vec2> vertices;

    [[nodiscard]] static PatternRegion circle(Material material,
                                              Vec2 center,
                                              Real radiusUm,
                                              int reticoloRectangles = 10);
    [[nodiscard]] static PatternRegion rectangle(Material material,
                                                 Vec2 center,
                                                 Real angleDeg,
                                                 Vec2 halfwidthsUm);
    [[nodiscard]] static PatternRegion polygon(Material material,
                                               Vec2 center,
                                               Real angleDeg,
                                               std::vector<Vec2> vertices);
};

struct PatternedLayer2D {
    Material background;
    std::vector<PatternRegion> regions;
    Real periodXUm{1.0};
    Real periodYUm{1.0};
    Real thicknessUm{0.3};

    // If left non-positive, a conservative rectilinear cell grid is chosen
    // from the retained harmonic basis. Default factorization uses exact cell
    // integrals: aligned rectangles are exact, circles use RETICOLO's
    // area-normalized nested rectangles, and arbitrary shapes use this grid.
    int sampleCountX{};
    int sampleCountY{};
    bool preferAnalytic{true}; // Prefer exact RETICOLO rectangle partitions.
    FourierConvergenceOptions fourierOptions{};
    std::shared_ptr<const PatternedLayer2DGeometryCache> geometryCache;
};

using PrecomputedPeriodicLayer2DPtr = std::shared_ptr<const PrecomputedPeriodicLayer2D>;
using PatternedLayer2DGeometryCachePtr = std::shared_ptr<const PatternedLayer2DGeometryCache>;

[[nodiscard]] std::vector<Vec2> regularEllipseVertices(Real radiusXUm,
                                                         Real radiusYUm,
                                                         int vertices);
[[nodiscard]] std::vector<Vec2> regularPolygonVertices(int sides, Real radiusUm);
[[nodiscard]] std::vector<Vec2> roundedRectangleVertices(Real halfwidthXUm,
                                                           Real halfwidthYUm,
                                                           Real cornerRadiusUm,
                                                           int verticesPerCorner);
[[nodiscard]] std::vector<Vec2> capsuleVertices(Real lengthUm,
                                                 Real radiusUm,
                                                 int verticesPerCap);
[[nodiscard]] bool pointInsideRegion(const PatternRegion& region, Real xUm, Real yUm);
[[nodiscard]] int recommendedPatternSampleCount(const HarmonicBasis& basis);
[[nodiscard]] bool patternedLayerHasTensorBoundary(
    const PatternedLayer2D& layer,
    Real wavelengthUm);
[[nodiscard]] TensorFactorizationApplicability
patternedTensorFactorizationApplicability(
    const PatternedLayer2D& layer,
    Real wavelengthUm);
[[nodiscard]] PeriodicLayer2D samplePatternedLayer2d(const PatternedLayer2D& layer,
                                                        const HarmonicBasis& basis);
[[nodiscard]] TensorFourierMatrices tensorFourierMatricesPatternedLayer2d(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm);
[[nodiscard]] PatternedLayer2DGeometryCache precomputePatternedLayer2dGeometry(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis);
[[nodiscard]] PrecomputedPeriodicLayer2D precomputePatternedLayer2d(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm);

} // namespace rcwa
