#include "rcwa/pattern.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "rcwa/matrix.hpp"
#include "rcwa/li_factorization.hpp"

namespace rcwa {

namespace {

int checkedVertexCount(int count, int multiplier = 1) {
    if (count < 3 || count > 4096) {
        throw std::invalid_argument("polygon approximation vertex count must be in [3, 4096]");
    }
    if (multiplier > 1 && count > 4096 / multiplier) {
        throw std::invalid_argument("polygon approximation has too many vertices");
    }
    return count;
}

Real polygonSignedArea(const std::vector<Vec2>& vertices) {
    Real twiceArea{};
    const std::size_t count = vertices.size();
    for (std::size_t i = 0; i < count; ++i) {
        const Vec2& a = vertices[i];
        const Vec2& b = vertices[(i + 1) % count];
        twiceArea += a.x * b.y - b.x * a.y;
    }
    return Real{0.5} * twiceArea;
}

void validateRegion(const PatternRegion& region) {
    switch (region.shape) {
    case PatternRegionShape::Circle:
        if (!(region.radiusUm >= 0.0) || !std::isfinite(region.radiusUm)) {
            throw std::invalid_argument("pattern circle radius must be finite and non-negative");
        }
        if (region.reticoloRectangleCount < 2 ||
            region.reticoloRectangleCount > 4096) {
            throw std::invalid_argument(
                "RETICOLO circle rectangle count must be in [2, 4096]");
        }
        break;
    case PatternRegionShape::Rectangle:
        if (!(region.halfwidthXUm >= 0.0) || !(region.halfwidthYUm >= 0.0) ||
            !std::isfinite(region.halfwidthXUm) || !std::isfinite(region.halfwidthYUm)) {
            throw std::invalid_argument("pattern rectangle halfwidths must be finite and non-negative");
        }
        break;
    case PatternRegionShape::Polygon:
        if (region.vertices.size() < 3) {
            throw std::invalid_argument("pattern polygon must contain at least three vertices");
        }
        for (const auto& vertex : region.vertices) {
            if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
                throw std::invalid_argument("pattern polygon vertices must be finite");
            }
        }
        {
            Real minX = region.vertices.front().x;
            Real maxX = region.vertices.front().x;
            Real minY = region.vertices.front().y;
            Real maxY = region.vertices.front().y;
            for (const auto& vertex : region.vertices) {
                minX = std::min(minX, vertex.x);
                maxX = std::max(maxX, vertex.x);
                minY = std::min(minY, vertex.y);
                maxY = std::max(maxY, vertex.y);
            }
            const Real area = std::abs(polygonSignedArea(region.vertices));
            const Real areaScale = std::max(
                (maxX - minX) * (maxY - minY),
                std::numeric_limits<Real>::min());
            if (area <= Real{64.0} * std::numeric_limits<Real>::epsilon() * areaScale) {
                throw std::invalid_argument("pattern polygon area must be non-zero");
            }
        }
        break;
    }
}

bool insideCircle(const PatternRegion& region, Real x, Real y) {
    const Real dx = x - region.center.x;
    const Real dy = y - region.center.y;
    return dx * dx + dy * dy <= region.radiusUm * region.radiusUm;
}

bool insideRectangle(const PatternRegion& region, Real x, Real y) {
    const Real theta = -degToRad(region.angleDeg);
    const Real dx = x - region.center.x;
    const Real dy = y - region.center.y;
    const Real xr = dx * std::cos(theta) - dy * std::sin(theta);
    const Real yr = dx * std::sin(theta) + dy * std::cos(theta);
    return std::abs(xr) <= region.halfwidthXUm && std::abs(yr) <= region.halfwidthYUm;
}

bool insidePolygon(const PatternRegion& region, Real x, Real y) {
    bool inside = false;
    const std::size_t count = region.vertices.size();
    if (count < 3) {
        return false;
    }
    for (std::size_t i = 0, j = count - 1; i < count; j = i++) {
        const auto a = region.vertices[i];
        const auto b = region.vertices[j];
        const bool crosses = ((a.y > y) != (b.y > y)) &&
            (x < (b.x - a.x) * (y - a.y) / (b.y - a.y + Real{1e-300}) + a.x);
        if (crosses) {
            inside = !inside;
        }
    }
    return inside;
}

Real besselRatioJ1(Real x) {
    const Real ax = std::abs(x);
    if (ax < Real{1e-5}) {
        const Real x2 = x * x;
        return Real{1.0} - x2 / Real{8.0} + x2 * x2 / Real{192.0} - x2 * x2 * x2 / Real{9216.0};
    }
    return Real{2.0} * std::cyl_bessel_j(Real{1.0}, x) / x;
}

Real sincUnscaled(Real x) {
    const Real ax = std::abs(x);
    if (ax < Real{1e-6}) {
        const Real x2 = x * x;
        return Real{1.0} - x2 / Real{6.0} + x2 * x2 / Real{120.0} -
               x2 * x2 * x2 / Real{5040.0};
    }
    return std::sin(x) / x;
}

Complex diskIndicatorCoefficient(int dm,
                                   int dn,
                                   Real periodXUm,
                                   Real periodYUm,
                                   Real radiusUm,
                                   Real centerXUm,
                                   Real centerYUm) {
    const Real gx = twoPi * static_cast<Real>(dm) / periodXUm;
    const Real gy = twoPi * static_cast<Real>(dn) / periodYUm;
    const Real fill = pi * radiusUm * radiusUm / (periodXUm * periodYUm);
    const Real argument = std::sqrt(gx * gx + gy * gy) * radiusUm;
    const Real ratio = besselRatioJ1(argument);
    const Real phase = -(gx * centerXUm + gy * centerYUm);
    return Complex{fill * ratio * std::cos(phase), fill * ratio * std::sin(phase)};
}

Complex rectangleIndicatorCoefficient(int dm,
                                        int dn,
                                        Real periodXUm,
                                        Real periodYUm,
                                        Real centerXUm,
                                        Real centerYUm,
                                        Real angleDeg,
                                        Real halfwidthXUm,
                                        Real halfwidthYUm) {
    if (halfwidthXUm < Real{0.0} || halfwidthYUm < Real{0.0}) {
        throw std::invalid_argument("rectangle halfwidths must be non-negative");
    }
    const Real gx = twoPi * static_cast<Real>(dm) / periodXUm;
    const Real gy = twoPi * static_cast<Real>(dn) / periodYUm;
    const Real angle = degToRad(angleDeg);
    const Real ca = std::cos(angle);
    const Real sa = std::sin(angle);
    const Real gLocalX = gx * ca + gy * sa;
    const Real gLocalY = -gx * sa + gy * ca;
    const Real fill = Real{4.0} * halfwidthXUm * halfwidthYUm / (periodXUm * periodYUm);
    const Real window =
        sincUnscaled(gLocalX * halfwidthXUm) *
        sincUnscaled(gLocalY * halfwidthYUm);
    const Real phase = -(gx * centerXUm + gy * centerYUm);
    return Complex{fill * window * std::cos(phase), fill * window * std::sin(phase)};
}

Complex polygonIndicatorCoefficient(int dm,
                                      int dn,
                                      Real periodXUm,
                                      Real periodYUm,
                                      const std::vector<Vec2>& vertices) {
    if (vertices.size() < 3) {
        return {};
    }
    const Real cellArea = periodXUm * periodYUm;
    const Real signedArea = polygonSignedArea(vertices);
    if (std::abs(signedArea) <= Real{0.0}) {
        return {};
    }
    if (dm == 0 && dn == 0) {
        return {std::abs(signedArea) / cellArea, Real{0.0}};
    }

    const Real gx = twoPi * static_cast<Real>(dm) / periodXUm;
    const Real gy = twoPi * static_cast<Real>(dn) / periodYUm;
    const Real g2 = gx * gx + gy * gy;
    const bool reverse = signedArea < Real{0.0};
    const std::size_t count = vertices.size();
    const auto vertexAt = [&](std::size_t i) -> const Vec2& {
        return reverse ? vertices[count - 1 - i] : vertices[i];
    };

    Complex boundarySum{};
    for (std::size_t i = 0; i < count; ++i) {
        const Vec2& a = vertexAt(i);
        const Vec2& b = vertexAt((i + 1) % count);
        const Real dx = b.x - a.x;
        const Real dy = b.y - a.y;
        const Real phase0 = gx * a.x + gy * a.y;
        const Real phaseDelta = gx * dx + gy * dy;
        const Real phase = -(phase0 + Real{0.5} * phaseDelta);
        const Complex edgeAverage{
            std::cos(phase) * sincUnscaled(Real{0.5} * phaseDelta),
            std::sin(phase) * sincUnscaled(Real{0.5} * phaseDelta)};
        boundarySum += (gx * dy - gy * dx) * edgeAverage;
    }
    return (iu * boundarySum) / Complex{g2 * cellArea, Real{0.0}};
}

Complex regionIndicatorCoefficient(int dm,
                                     int dn,
                                     Real periodXUm,
                                     Real periodYUm,
                                     const PatternRegion& region) {
    switch (region.shape) {
    case PatternRegionShape::Circle:
        return diskIndicatorCoefficient(
            dm,
            dn,
            periodXUm,
            periodYUm,
            region.radiusUm,
            region.center.x,
            region.center.y);
    case PatternRegionShape::Rectangle:
        return rectangleIndicatorCoefficient(
            dm,
            dn,
            periodXUm,
            periodYUm,
            region.center.x,
            region.center.y,
            region.angleDeg,
            region.halfwidthXUm,
            region.halfwidthYUm);
    case PatternRegionShape::Polygon:
        return polygonIndicatorCoefficient(
            dm,
            dn,
            periodXUm,
            periodYUm,
            region.vertices);
    }
    return {};
}

Real cross(Vec2 a, Vec2 b, Vec2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool segmentsHaveProperIntersection(Vec2 a, Vec2 b, Vec2 c, Vec2 d) {
    const Real scale = std::max({
        Real{1.0},
        std::abs(a.x), std::abs(a.y),
        std::abs(b.x), std::abs(b.y),
        std::abs(c.x), std::abs(c.y),
        std::abs(d.x), std::abs(d.y)});
    const Real tol = Real{256.0} * std::numeric_limits<Real>::epsilon() * scale * scale;
    const Real o1 = cross(a, b, c);
    const Real o2 = cross(a, b, d);
    const Real o3 = cross(c, d, a);
    const Real o4 = cross(c, d, b);
    return ((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
           ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol));
}

bool pointIsOnSegment(Vec2 a, Vec2 b, Vec2 p) {
    const Real scale = std::max({
        Real{1.0},
        std::abs(a.x), std::abs(a.y),
        std::abs(b.x), std::abs(b.y),
        std::abs(p.x), std::abs(p.y)});
    const Real tol = Real{256.0} * std::numeric_limits<Real>::epsilon() * scale * scale;
    if (std::abs(cross(a, b, p)) > tol) {
        return false;
    }
    return p.x >= std::min(a.x, b.x) - tol && p.x <= std::max(a.x, b.x) + tol &&
           p.y >= std::min(a.y, b.y) - tol && p.y <= std::max(a.y, b.y) + tol;
}

bool pointInsideOrOnPolygon(const std::vector<Vec2>& vertices, Vec2 point) {
    const std::size_t count = vertices.size();
    bool inside = false;
    for (std::size_t i = 0, j = count - 1; i < count; j = i++) {
        const Vec2& a = vertices[i];
        const Vec2& b = vertices[j];
        if (pointIsOnSegment(a, b, point)) {
            return true;
        }
        const bool crosses = ((a.y > point.y) != (b.y > point.y)) &&
            (point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x);
        if (crosses) {
            inside = !inside;
        }
    }
    return inside;
}

bool polygonsTouchOrOverlap(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    for (std::size_t ia = 0; ia < a.size(); ++ia) {
        const Vec2& a0 = a[ia];
        const Vec2& a1 = a[(ia + 1) % a.size()];
        for (std::size_t ib = 0; ib < b.size(); ++ib) {
            const Vec2& b0 = b[ib];
            const Vec2& b1 = b[(ib + 1) % b.size()];
            if (segmentsHaveProperIntersection(a0, a1, b0, b1) ||
                pointIsOnSegment(a0, a1, b0) ||
                pointIsOnSegment(a0, a1, b1) ||
                pointIsOnSegment(b0, b1, a0) ||
                pointIsOnSegment(b0, b1, a1)) {
                return true;
            }
        }
    }
    for (const Vec2& vertex : a) {
        if (pointInsideOrOnPolygon(b, vertex)) {
            return true;
        }
    }
    for (const Vec2& vertex : b) {
        if (pointInsideOrOnPolygon(a, vertex)) {
            return true;
        }
    }
    return false;
}

std::size_t localCoeffIndex2d(int p, int q, int coeffCountX, int coeffCountY) {
    const int spanX = (coeffCountX - 1) / 2;
    const int spanY = (coeffCountY - 1) / 2;
    return static_cast<std::size_t>((q + spanY) * coeffCountX + (p + spanX));
}

bool complexClose(Complex a, Complex b, Real tol = Real{1e-12}) {
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tol * scale;
}

std::uint64_t patternGeometryFingerprint(const PatternedLayer2D& layer,
                                           int sampleCountX,
                                           int sampleCountY,
                                           int coeffCountX,
                                           int coeffCountY) {
    detail::Fingerprint64 fingerprint(UINT64_C(1469598103934665603));
    fingerprint.append(static_cast<std::uint64_t>(sampleCountX));
    fingerprint.append(static_cast<std::uint64_t>(sampleCountY));
    fingerprint.append(static_cast<std::uint64_t>(coeffCountX));
    fingerprint.append(static_cast<std::uint64_t>(coeffCountY));
    fingerprint.append(layer.periodXUm);
    fingerprint.append(layer.periodYUm);
    fingerprint.append(static_cast<std::uint64_t>(layer.regions.size()));
    for (const auto& region : layer.regions) {
        fingerprint.append(static_cast<std::uint64_t>(region.shape));
        fingerprint.append(region.center.x);
        fingerprint.append(region.center.y);
        fingerprint.append(region.angleDeg);
        fingerprint.append(region.radiusUm);
        fingerprint.append(region.halfwidthXUm);
        fingerprint.append(region.halfwidthYUm);
        fingerprint.append(static_cast<std::uint64_t>(region.reticoloRectangleCount));
        fingerprint.append(static_cast<std::uint64_t>(region.vertices.size()));
        for (const auto& vertex : region.vertices) {
            fingerprint.append(vertex.x);
            fingerprint.append(vertex.y);
        }
    }
    return fingerprint.value();
}

int patternedCoeffCountX(const HarmonicBasis& basis) {
    return 4 * basis.orderX + 1;
}

int patternedCoeffCountY(const HarmonicBasis& basis) {
    return 4 * basis.orderY + 1;
}

bool circleIsInsideCircle(const PatternRegion& inner,
                             const PatternRegion& outer) {
    if (inner.shape != PatternRegionShape::Circle ||
        outer.shape != PatternRegionShape::Circle) {
        return false;
    }
    const Real dx = inner.center.x - outer.center.x;
    const Real dy = inner.center.y - outer.center.y;
    const Real distance = std::hypot(dx, dy);
    const Real scale = std::max({
        Real{1.0},
        std::abs(inner.center.x),
        std::abs(inner.center.y),
        std::abs(outer.center.x),
        std::abs(outer.center.y),
        inner.radiusUm,
        outer.radiusUm});
    return distance + inner.radiusUm <= outer.radiusUm + Real{1e-12} * scale;
}

bool pointIsInsideRectangleWithTolerance(const PatternRegion& rectangle, Vec2 point) {
    const Real theta = -degToRad(rectangle.angleDeg);
    const Real dx = point.x - rectangle.center.x;
    const Real dy = point.y - rectangle.center.y;
    const Real xr = dx * std::cos(theta) - dy * std::sin(theta);
    const Real yr = dx * std::sin(theta) + dy * std::cos(theta);
    const Real scale = std::max({
        Real{1.0},
        std::abs(rectangle.center.x),
        std::abs(rectangle.center.y),
        rectangle.halfwidthXUm,
        rectangle.halfwidthYUm});
    const Real tol = Real{1e-12} * scale;
    return std::abs(xr) <= rectangle.halfwidthXUm + tol &&
           std::abs(yr) <= rectangle.halfwidthYUm + tol;
}

Vec2 rectangleCorner(const PatternRegion& rectangle, Real sx, Real sy) {
    const Real theta = degToRad(rectangle.angleDeg);
    const Real ca = std::cos(theta);
    const Real sa = std::sin(theta);
    const Real localX = sx * rectangle.halfwidthXUm;
    const Real localY = sy * rectangle.halfwidthYUm;
    return {
        rectangle.center.x + localX * ca - localY * sa,
        rectangle.center.y + localX * sa + localY * ca};
}

std::vector<Vec2> rectangleVertices(const PatternRegion& rectangle) {
    return {
        rectangleCorner(rectangle, Real{-1.0}, Real{-1.0}),
        rectangleCorner(rectangle, Real{1.0}, Real{-1.0}),
        rectangleCorner(rectangle, Real{1.0}, Real{1.0}),
        rectangleCorner(rectangle, Real{-1.0}, Real{1.0}),
    };
}

Real pointSegmentDistance(Vec2 a, Vec2 b, Vec2 p) {
    const Real dx = b.x - a.x;
    const Real dy = b.y - a.y;
    const Real length2 = dx * dx + dy * dy;
    if (length2 <= Real{0.0}) {
        return std::hypot(p.x - a.x, p.y - a.y);
    }
    const Real t = std::clamp(
        ((p.x - a.x) * dx + (p.y - a.y) * dy) / length2,
        Real{0.0},
        Real{1.0});
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

std::vector<Vec2> regionBoundaryVertices(const PatternRegion& region) {
    if (region.shape == PatternRegionShape::Rectangle) {
        return rectangleVertices(region);
    }
    if (region.shape == PatternRegionShape::Polygon) {
        return region.vertices;
    }
    return {};
}

bool circleHasInteriorOverlapWithPolygon(const PatternRegion& circle,
                                         const std::vector<Vec2>& vertices) {
    const Real scale = std::max({
        Real{1.0},
        std::abs(circle.center.x),
        std::abs(circle.center.y),
        circle.radiusUm});
    const Real tol = Real{1e-12} * scale;
    if (circle.radiusUm > tol && pointInsideOrOnPolygon(vertices, circle.center)) {
        return true;
    }
    for (const Vec2& vertex : vertices) {
        if (std::hypot(vertex.x - circle.center.x, vertex.y - circle.center.y) <
            circle.radiusUm - tol) {
            return true;
        }
    }
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        if (pointSegmentDistance(
                vertices[i],
                vertices[(i + 1) % vertices.size()],
                circle.center) < circle.radiusUm - tol) {
            return true;
        }
    }
    return false;
}

bool regionsHaveInteriorOverlap(const PatternRegion& a, const PatternRegion& b) {
    if (a.shape == PatternRegionShape::Circle && b.shape == PatternRegionShape::Circle) {
        const Real scale = std::max({
            Real{1.0},
            std::abs(a.center.x),
            std::abs(a.center.y),
            std::abs(b.center.x),
            std::abs(b.center.y),
            a.radiusUm,
            b.radiusUm});
        const Real tol = Real{1e-12} * scale;
        return std::hypot(a.center.x - b.center.x, a.center.y - b.center.y) <
               a.radiusUm + b.radiusUm - tol;
    }
    if (a.shape == PatternRegionShape::Rectangle &&
        b.shape == PatternRegionShape::Rectangle &&
        std::abs(a.angleDeg) <= Real{1e-12} &&
        std::abs(b.angleDeg) <= Real{1e-12}) {
        const Real scale = std::max({
            Real{1.0},
            std::abs(a.center.x),
            std::abs(a.center.y),
            std::abs(b.center.x),
            std::abs(b.center.y),
            a.halfwidthXUm,
            a.halfwidthYUm,
            b.halfwidthXUm,
            b.halfwidthYUm});
        const Real tol = Real{1e-12} * scale;
        const Real overlapX = std::min(
            a.center.x + a.halfwidthXUm,
            b.center.x + b.halfwidthXUm) - std::max(
                a.center.x - a.halfwidthXUm,
                b.center.x - b.halfwidthXUm);
        const Real overlapY = std::min(
            a.center.y + a.halfwidthYUm,
            b.center.y + b.halfwidthYUm) - std::max(
                a.center.y - a.halfwidthYUm,
                b.center.y - b.halfwidthYUm);
        return overlapX > tol && overlapY > tol;
    }
    if (a.shape == PatternRegionShape::Circle) {
        return circleHasInteriorOverlapWithPolygon(a, regionBoundaryVertices(b));
    }
    if (b.shape == PatternRegionShape::Circle) {
        return circleHasInteriorOverlapWithPolygon(b, regionBoundaryVertices(a));
    }
    return polygonsTouchOrOverlap(regionBoundaryVertices(a), regionBoundaryVertices(b));
}

bool circleIsInsideRectangle(const PatternRegion& circle, const PatternRegion& rectangle) {
    if (circle.shape != PatternRegionShape::Circle ||
        rectangle.shape != PatternRegionShape::Rectangle) {
        return false;
    }
    const Real theta = -degToRad(rectangle.angleDeg);
    const Real dx = circle.center.x - rectangle.center.x;
    const Real dy = circle.center.y - rectangle.center.y;
    const Real xr = dx * std::cos(theta) - dy * std::sin(theta);
    const Real yr = dx * std::sin(theta) + dy * std::cos(theta);
    const Real scale = std::max({
        Real{1.0},
        std::abs(rectangle.center.x),
        std::abs(rectangle.center.y),
        rectangle.halfwidthXUm,
        rectangle.halfwidthYUm,
        circle.radiusUm});
    const Real tol = Real{1e-12} * scale;
    return std::abs(xr) + circle.radiusUm <= rectangle.halfwidthXUm + tol &&
           std::abs(yr) + circle.radiusUm <= rectangle.halfwidthYUm + tol;
}

bool verticesAreInsideCircle(const std::vector<Vec2>& vertices, const PatternRegion& circle) {
    const Real scale = std::max({
        Real{1.0},
        std::abs(circle.center.x),
        std::abs(circle.center.y),
        circle.radiusUm});
    const Real tol = Real{1e-12} * scale;
    for (const Vec2& vertex : vertices) {
        if (std::hypot(vertex.x - circle.center.x, vertex.y - circle.center.y) >
            circle.radiusUm + tol) {
            return false;
        }
    }
    return true;
}

bool verticesAreInsideRectangle(const std::vector<Vec2>& vertices,
                                   const PatternRegion& rectangle) {
    for (const Vec2& vertex : vertices) {
        if (!pointIsInsideRectangleWithTolerance(rectangle, vertex)) {
            return false;
        }
    }
    return true;
}

bool polygonVerticesAreInsidePolygon(const std::vector<Vec2>& inner,
                                         const std::vector<Vec2>& outer) {
    for (const Vec2& vertex : inner) {
        if (!pointInsideOrOnPolygon(outer, vertex)) {
            return false;
        }
    }
    for (std::size_t i = 0; i < inner.size(); ++i) {
        const Vec2& a0 = inner[i];
        const Vec2& a1 = inner[(i + 1) % inner.size()];
        for (std::size_t j = 0; j < outer.size(); ++j) {
            const Vec2& b0 = outer[j];
            const Vec2& b1 = outer[(j + 1) % outer.size()];
            if (segmentsHaveProperIntersection(a0, a1, b0, b1)) {
                return false;
            }
        }
    }
    return true;
}

bool circleIsInsidePolygon(const PatternRegion& circle,
                              const std::vector<Vec2>& polygonVertices) {
    const Real scale = std::max({
        Real{1.0},
        std::abs(circle.center.x),
        std::abs(circle.center.y),
        circle.radiusUm});
    const Real tol = Real{1e-12} * scale;
    if (!pointInsideOrOnPolygon(polygonVertices, circle.center)) {
        return false;
    }
    for (std::size_t i = 0; i < polygonVertices.size(); ++i) {
        if (pointSegmentDistance(
                polygonVertices[i],
                polygonVertices[(i + 1) % polygonVertices.size()],
                circle.center) < circle.radiusUm - tol) {
            return false;
        }
    }
    return true;
}

bool regionContainsRegion(const PatternRegion& inner, const PatternRegion& outer) {
    if (outer.shape == PatternRegionShape::Circle) {
        if (inner.shape == PatternRegionShape::Circle) {
            return circleIsInsideCircle(inner, outer);
        }
        return verticesAreInsideCircle(regionBoundaryVertices(inner), outer);
    }

    if (outer.shape == PatternRegionShape::Rectangle) {
        if (inner.shape == PatternRegionShape::Circle) {
            return circleIsInsideRectangle(inner, outer);
        }
        return verticesAreInsideRectangle(regionBoundaryVertices(inner), outer);
    }

    const std::vector<Vec2>& outerVertices = outer.vertices;
    if (inner.shape == PatternRegionShape::Circle) {
        return circleIsInsidePolygon(inner, outerVertices);
    }
    return polygonVerticesAreInsidePolygon(regionBoundaryVertices(inner), outerVertices);
}

struct Interval1D {
    Real start{};
    Real end{};
};

enum class LamellarAxis {
    X,
    Y,
};

Complex centeredIntervalIndicatorCoefficient(int p,
                                                Real centerUm,
                                                Real widthUm,
                                                Real periodUm) {
    const Real fill = widthUm / periodUm;
    if (p == 0) {
        return {fill, 0.0};
    }
    const Real x = pi * static_cast<Real>(p) * fill;
    const Real amplitude = std::sin(x) / (pi * static_cast<Real>(p));
    const Real phase = -twoPi * static_cast<Real>(p) * centerUm / periodUm;
    return {amplitude * std::cos(phase), amplitude * std::sin(phase)};
}

std::vector<Complex> intervalCoefficients2d(const std::vector<Interval1D>& intervals,
                                              Real periodUm,
                                              int coeffCountX,
                                              int coeffCountY,
                                              Complex background,
                                              Complex inclusion,
                                              LamellarAxis axis) {
    const int span = ((axis == LamellarAxis::X) ? coeffCountX : coeffCountY) / 2;
    std::vector<Complex> coeffs(
        static_cast<std::size_t>(coeffCountX * coeffCountY),
        Complex{});
    for (int h = -span; h <= span; ++h) {
        Complex indicator{};
        for (const auto& interval : intervals) {
            const Real width = interval.end - interval.start;
            const Real center = Real{0.5} * (interval.start + interval.end);
            indicator += centeredIntervalIndicatorCoefficient(
                h, center, width, periodUm);
        }
        Complex value = h == 0 ? background : Complex{};
        value += (inclusion - background) * indicator;
        const int p = axis == LamellarAxis::X ? h : 0;
        const int q = axis == LamellarAxis::Y ? h : 0;
        coeffs[localCoeffIndex2d(p, q, coeffCountX, coeffCountY)] = value;
    }
    return coeffs;
}

bool tryFullCellLamellarRectangles(const PatternedLayer2D& layer,
                                   const HarmonicBasis& basis,
                                   Real wavelengthUm,
                                   LamellarAxis axis,
                                   TensorFourierMatrices* out) {
    if (layer.regions.empty()) {
        return false;
    }

    // A single tensor rectangle spanning the complete transverse cell is an
    // analytic lamellar problem. Keep it on the directional Li inverse-rule
    // path instead of sampling merely because it entered through the general
    // PatternedLayer2D facade.
    if (layer.regions.size() == 1) {
        constexpr Real tensorLamellarTolerance = Real{1e-12};
        const PatternRegion& region = layer.regions.front();
        const bool alongX = axis == LamellarAxis::X;
        const Real normalPeriod = alongX ? layer.periodXUm : layer.periodYUm;
        const Real transverseHalfCell = Real{0.5} *
            (alongX ? layer.periodYUm : layer.periodXUm);
        const Real normalCenter = alongX ? region.center.x : region.center.y;
        const Real normalHalfwidth =
            alongX ? region.halfwidthXUm : region.halfwidthYUm;
        const Real transverseHalfwidth =
            alongX ? region.halfwidthYUm : region.halfwidthXUm;
        const Real reducedAngle = std::remainder(region.angleDeg, Real{180});
        const Real intervalStart = normalCenter - normalHalfwidth;
        const Real intervalEnd = normalCenter + normalHalfwidth;
        if (region.shape == PatternRegionShape::Rectangle &&
            std::abs(reducedAngle) <= tensorLamellarTolerance &&
            std::abs(transverseHalfwidth - transverseHalfCell) <=
                tensorLamellarTolerance *
                    std::max(Real{1}, transverseHalfCell) &&
            normalHalfwidth > Real{0} &&
            intervalStart >= -Real{0.5} * normalPeriod -
                tensorLamellarTolerance &&
            intervalEnd <= Real{0.5} * normalPeriod +
                tensorLamellarTolerance) {
            const auto [backgroundEps, backgroundMu] =
                layer.background.tensors(wavelengthUm);
            const auto [regionEps, regionMu] =
                region.material.tensors(wavelengthUm);
            const bool tensorMaterial = !tensorIsScalar(backgroundEps) ||
                !tensorIsScalar(backgroundMu) || !tensorIsScalar(regionEps) ||
                !tensorIsScalar(regionMu);
            if (tensorMaterial) {
                const Real fillFactor = Real{2} * normalHalfwidth / normalPeriod;
                const Real centerOffset = normalCenter / normalPeriod;
                *out = alongX
                    ? tensorFourierMatricesBinaryGrating(
                        fillFactor,
                        regionEps,
                        backgroundEps,
                        regionMu,
                        backgroundMu,
                        basis,
                        centerOffset)
                    : tensorFourierMatricesBinaryGratingY(
                        fillFactor,
                        regionEps,
                        backgroundEps,
                        regionMu,
                        backgroundMu,
                        basis,
                        centerOffset);
                return true;
            }
        }
    }

    Complex epsBackground{};
    Complex muBackground{};
    const auto [backgroundEpsTensor, backgroundMuTensor] =
        layer.background.tensors(wavelengthUm);
    if (!tensorIsScalar(backgroundEpsTensor, &epsBackground) ||
        !tensorIsScalar(backgroundMuTensor, &muBackground)) {
        return false;
    }

    constexpr Real tol = Real{1e-12};
    const bool alongX = axis == LamellarAxis::X;
    const Real period = alongX ? layer.periodXUm : layer.periodYUm;
    const Real cellMin = -Real{0.5} * period;
    const Real cellMax = Real{0.5} * period;
    const Real transverseHalfCell = Real{0.5} *
        (alongX ? layer.periodYUm : layer.periodXUm);
    std::vector<Interval1D> intervals;
    Complex epsInclusion{};
    Complex muInclusion{};
    bool hasInclusion = false;

    for (const auto& region : layer.regions) {
        if (region.shape != PatternRegionShape::Rectangle ||
            std::abs(region.angleDeg) > tol ||
            std::abs((alongX ? region.halfwidthYUm : region.halfwidthXUm) -
                     transverseHalfCell) >
                tol * std::max(Real{1.0}, transverseHalfCell)) {
            return false;
        }
        const auto [epsTensor, muTensor] = region.material.tensors(wavelengthUm);
        Complex epsRegion{};
        Complex muRegion{};
        if (!tensorIsScalar(epsTensor, &epsRegion) ||
            !tensorIsScalar(muTensor, &muRegion)) {
            return false;
        }
        if (!hasInclusion) {
            epsInclusion = epsRegion;
            muInclusion = muRegion;
            hasInclusion = true;
        } else if (std::abs(epsRegion - epsInclusion) > tol ||
                   std::abs(muRegion - muInclusion) > tol) {
            return false;
        }
        const Real center = alongX ? region.center.x : region.center.y;
        const Real halfwidth = alongX ? region.halfwidthXUm : region.halfwidthYUm;
        if (halfwidth == Real{0.0}) {
            continue;
        }
        Interval1D interval{
            center - halfwidth,
            center + halfwidth,
        };
        if (interval.start < cellMin - tol || interval.end > cellMax + tol) {
            return false;
        }
        interval.start = std::max(interval.start, cellMin);
        interval.end = std::min(interval.end, cellMax);
        intervals.push_back(interval);
    }
    if (intervals.empty()) {
        return false;
    }
    std::sort(intervals.begin(), intervals.end(), [](const Interval1D& a, const Interval1D& b) {
        return a.start < b.start;
    });
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].start < intervals[i - 1].end - tol) {
            return false;
        }
    }

    const int coeffCountX = 4 * basis.orderX + 1;
    const int coeffCountY = 4 * basis.orderY + 1;
    const auto epsCoeffs = intervalCoefficients2d(
        intervals,
        period,
        coeffCountX,
        coeffCountY,
        epsBackground,
        epsInclusion,
        axis);
    const auto invEpsCoeffs = intervalCoefficients2d(
        intervals,
        period,
        coeffCountX,
        coeffCountY,
        Complex{1.0, 0.0} / epsBackground,
        Complex{1.0, 0.0} / epsInclusion,
        axis);

    const Matrix epsConv = convMatrix2d(epsCoeffs, coeffCountX, coeffCountY, basis);
    const Matrix invEpsConv =
        convMatrix2d(invEpsCoeffs, coeffCountX, coeffCountY, basis);

    detail::assignIsotropicTensorBlocks(out->eps, epsConv);
    auto epsLi = detail::liFactorizeScalarLamellar(
        epsConv,
        invEpsConv,
        axis == LamellarAxis::X
            ? detail::LiInterfaceAxis::X
            : detail::LiInterfaceAxis::Y,
        "epsilon");
    out->liEpsQ = std::move(epsLi.q);
    out->liEpsQZzInverse = std::move(epsLi.zzInverse);

    Matrix muConv;
    Matrix invMuConv;
    if (complexClose(muBackground, muInclusion)) {
        muConv = Matrix::scaledIdentity(basis.size(), muBackground);
        invMuConv =
            Matrix::scaledIdentity(basis.size(), Complex{1.0, 0.0} / muBackground);
    } else {
        const auto muCoeffs = intervalCoefficients2d(
            intervals,
            period,
            coeffCountX,
            coeffCountY,
            muBackground,
            muInclusion,
            axis);
        const auto invMuCoeffs = intervalCoefficients2d(
            intervals,
            period,
            coeffCountX,
            coeffCountY,
            Complex{1.0, 0.0} / muBackground,
            Complex{1.0, 0.0} / muInclusion,
            axis);
        muConv = convMatrix2d(muCoeffs, coeffCountX, coeffCountY, basis);
        invMuConv = convMatrix2d(invMuCoeffs, coeffCountX, coeffCountY, basis);
    }
    detail::assignIsotropicTensorBlocks(out->mu, muConv);
    auto muLi = detail::liFactorizeScalarLamellar(
        muConv,
        invMuConv,
        axis == LamellarAxis::X
            ? detail::LiInterfaceAxis::X
            : detail::LiInterfaceAxis::Y,
        "mu");
    out->liMuQ = std::move(muLi.q);
    out->liMuQZzInverse = std::move(muLi.zzInverse);
    return true;
}

} // namespace

bool patternedLayerHasTensorBoundary(const PatternedLayer2D& layer,
                                     Real wavelengthUm) {
    const auto [backgroundEps, backgroundMu] = layer.background.tensors(wavelengthUm);
    for (const auto& region : layer.regions) {
        const auto [eps, mu] = region.material.tensors(wavelengthUm);
        const bool materialChanges = eps.v != backgroundEps.v || mu.v != backgroundMu.v;
        if (materialChanges &&
            (!tensorIsScalar(backgroundEps) || !tensorIsScalar(backgroundMu) ||
             !tensorIsScalar(eps) || !tensorIsScalar(mu))) {
            return true;
        }
    }
    return false;
}

TensorFactorizationApplicability patternedTensorFactorizationApplicability(
    const PatternedLayer2D& layer,
    Real wavelengthUm) {
    if (!patternedLayerHasTensorBoundary(layer, wavelengthUm)) {
        return TensorFactorizationApplicability::NotApplicable;
    }

    const Real geometryScale = std::max({
        Real{1.0},
        std::abs(layer.periodXUm),
        std::abs(layer.periodYUm)});
    const Real tolerance = Real{64} * std::numeric_limits<Real>::epsilon() *
        geometryScale;
    const auto edgeIsCoordinateAligned = [&](Real dx, Real dy) {
        return std::abs(dx) <= tolerance || std::abs(dy) <= tolerance;
    };

    for (const PatternRegion& region : layer.regions) {
        if (region.shape == PatternRegionShape::Circle) {
            return TensorFactorizationApplicability::CurvedOrOblique;
        }
        const Real angle = degToRad(std::remainder(region.angleDeg, Real{360}));
        const Real cosine = std::cos(angle);
        const Real sine = std::sin(angle);
        if (region.shape == PatternRegionShape::Rectangle) {
            const Real edgeXx = Real{2} * region.halfwidthXUm * cosine;
            const Real edgeXy = Real{2} * region.halfwidthXUm * sine;
            const Real edgeYx = -Real{2} * region.halfwidthYUm * sine;
            const Real edgeYy = Real{2} * region.halfwidthYUm * cosine;
            if (!edgeIsCoordinateAligned(edgeXx, edgeXy) ||
                !edgeIsCoordinateAligned(edgeYx, edgeYy)) {
                return TensorFactorizationApplicability::CurvedOrOblique;
            }
            continue;
        }

        for (std::size_t index = 0; index < region.vertices.size(); ++index) {
            const Vec2& first = region.vertices[index];
            const Vec2& second = region.vertices[(index + 1) % region.vertices.size()];
            const Real localDx = second.x - first.x;
            const Real localDy = second.y - first.y;
            const Real globalDx = cosine * localDx - sine * localDy;
            const Real globalDy = sine * localDx + cosine * localDy;
            if (!edgeIsCoordinateAligned(globalDx, globalDy)) {
                return TensorFactorizationApplicability::CurvedOrOblique;
            }
        }
    }
    return TensorFactorizationApplicability::CoordinateAligned;
}

std::vector<Vec2> regularEllipseVertices(Real radiusXUm,
                                           Real radiusYUm,
                                           int vertices) {
    requirePositive(radiusXUm, "ellipse radius_x");
    requirePositive(radiusYUm, "ellipse radius_y");
    checkedVertexCount(vertices);
    std::vector<Vec2> out;
    out.reserve(static_cast<std::size_t>(vertices));
    for (int i = 0; i < vertices; ++i) {
        const Real t = twoPi * static_cast<Real>(i) / static_cast<Real>(vertices);
        out.push_back({radiusXUm * std::cos(t), radiusYUm * std::sin(t)});
    }
    return out;
}

std::vector<Vec2> regularPolygonVertices(int sides, Real radiusUm) {
    if (sides < 3 || sides > 4096) {
        throw std::invalid_argument("regular polygon sides must be in [3, 4096]");
    }
    requirePositive(radiusUm, "regular polygon radius");
    std::vector<Vec2> out;
    out.reserve(static_cast<std::size_t>(sides));
    for (int i = 0; i < sides; ++i) {
        const Real t = twoPi * static_cast<Real>(i) / static_cast<Real>(sides);
        out.push_back({radiusUm * std::cos(t), radiusUm * std::sin(t)});
    }
    return out;
}

std::vector<Vec2> roundedRectangleVertices(Real halfwidthXUm,
                                             Real halfwidthYUm,
                                             Real cornerRadiusUm,
                                             int verticesPerCorner) {
    requirePositive(halfwidthXUm, "rounded rectangle halfwidth_x");
    requirePositive(halfwidthYUm, "rounded rectangle halfwidth_y");
    if (!(cornerRadiusUm >= Real{0.0}) || !std::isfinite(cornerRadiusUm)) {
        throw std::invalid_argument(
            "rounded rectangle corner_radius must be finite and non-negative");
    }
    const Real maxRadius = std::min(halfwidthXUm, halfwidthYUm);
    if (cornerRadiusUm > maxRadius) {
        throw std::invalid_argument("rounded rectangle corner_radius exceeds the smaller halfwidth");
    }
    checkedVertexCount(verticesPerCorner, 4);
    if (cornerRadiusUm == Real{0.0}) {
        return {
            {halfwidthXUm, halfwidthYUm},
            {-halfwidthXUm, halfwidthYUm},
            {-halfwidthXUm, -halfwidthYUm},
            {halfwidthXUm, -halfwidthYUm},
        };
    }

    std::vector<Vec2> out;
    out.reserve(static_cast<std::size_t>(4 * verticesPerCorner));
    const std::array<Vec2, 4> centers = {{
        {halfwidthXUm - cornerRadiusUm, halfwidthYUm - cornerRadiusUm},
        {-halfwidthXUm + cornerRadiusUm, halfwidthYUm - cornerRadiusUm},
        {-halfwidthXUm + cornerRadiusUm, -halfwidthYUm + cornerRadiusUm},
        {halfwidthXUm - cornerRadiusUm, -halfwidthYUm + cornerRadiusUm},
    }};
    const std::array<Real, 4> startAngles = {{
        Real{0.0},
        Real{0.5} * pi,
        pi,
        Real{1.5} * pi,
    }};
    for (std::size_t corner = 0; corner < centers.size(); ++corner) {
        for (int i = 0; i < verticesPerCorner; ++i) {
            const Real t = startAngles[corner] +
                Real{0.5} * pi * static_cast<Real>(i) /
                    static_cast<Real>(std::max(verticesPerCorner - 1, 1));
            out.push_back({
                centers[corner].x + cornerRadiusUm * std::cos(t),
                centers[corner].y + cornerRadiusUm * std::sin(t),
            });
        }
    }
    return out;
}

std::vector<Vec2> capsuleVertices(Real lengthUm,
                                   Real radiusUm,
                                   int verticesPerCap) {
    requirePositive(lengthUm, "capsule length");
    requirePositive(radiusUm, "capsule radius");
    if (lengthUm < Real{2.0} * radiusUm) {
        throw std::invalid_argument("capsule length must be at least 2 * radius");
    }
    checkedVertexCount(verticesPerCap, 2);
    const Real center = Real{0.5} * lengthUm - radiusUm;
    std::vector<Vec2> out;
    out.reserve(static_cast<std::size_t>(2 * verticesPerCap));
    for (int i = 0; i < verticesPerCap; ++i) {
        const Real t = -Real{0.5} * pi +
            pi * static_cast<Real>(i) /
                static_cast<Real>(std::max(verticesPerCap - 1, 1));
        out.push_back({center + radiusUm * std::cos(t), radiusUm * std::sin(t)});
    }
    for (int i = 0; i < verticesPerCap; ++i) {
        const Real t = Real{0.5} * pi +
            pi * static_cast<Real>(i) /
                static_cast<Real>(std::max(verticesPerCap - 1, 1));
        out.push_back({-center + radiusUm * std::cos(t), radiusUm * std::sin(t)});
    }
    return out;
}

PatternRegion PatternRegion::circle(Material material,
                                    Vec2 center,
                                    Real radiusUm,
                                    int reticoloRectangles) {
    PatternRegion out;
    out.shape = PatternRegionShape::Circle;
    out.material = std::move(material);
    out.center = center;
    out.radiusUm = radiusUm;
    out.reticoloRectangleCount = reticoloRectangles;
    validateRegion(out);
    return out;
}

PatternRegion PatternRegion::rectangle(Material material,
                                       Vec2 center,
                                       Real angleDeg,
                                       Vec2 halfwidthsUm) {
    PatternRegion out;
    out.shape = PatternRegionShape::Rectangle;
    out.material = std::move(material);
    out.center = center;
    out.angleDeg = angleDeg;
    out.halfwidthXUm = halfwidthsUm.x;
    out.halfwidthYUm = halfwidthsUm.y;
    validateRegion(out);
    return out;
}

PatternRegion PatternRegion::polygon(Material material,
                                     Vec2 center,
                                     Real angleDeg,
                                     std::vector<Vec2> vertices) {
    PatternRegion out;
    out.shape = PatternRegionShape::Polygon;
    out.material = std::move(material);
    out.center = center;
    out.angleDeg = angleDeg;
    out.vertices.reserve(vertices.size());
    const Real theta = degToRad(angleDeg);
    for (const auto& v : vertices) {
        const Real xr = v.x * std::cos(theta) - v.y * std::sin(theta);
        const Real yr = v.x * std::sin(theta) + v.y * std::cos(theta);
        out.vertices.push_back({center.x + xr, center.y + yr});
    }
    validateRegion(out);
    return out;
}

bool pointInsideRegion(const PatternRegion& region, Real xUm, Real yUm) {
    switch (region.shape) {
    case PatternRegionShape::Circle:
        return insideCircle(region, xUm, yUm);
    case PatternRegionShape::Rectangle:
        return insideRectangle(region, xUm, yUm);
    case PatternRegionShape::Polygon:
        return insidePolygon(region, xUm, yUm);
    }
    return false;
}

int recommendedPatternSampleCount(const HarmonicBasis& basis) {
    const int requiredX = 4 * basis.orderX + 1;
    const int requiredY = 4 * basis.orderY + 1;
    const int recommendedX = 8 * basis.orderX + 1;
    const int recommendedY = 8 * basis.orderY + 1;
    return std::max({requiredX, requiredY, recommendedX, recommendedY, 65});
}

std::size_t patternedMaterialSlotAt(const PatternedLayer2D& layer, Real x, Real y) {
    for (std::size_t offset = 0; offset < layer.regions.size(); ++offset) {
        const std::size_t regionIndex = layer.regions.size() - 1 - offset;
        const PatternRegion& region = layer.regions[regionIndex];
        const Real nearestX = region.center.x +
            std::remainder(x - region.center.x, layer.periodXUm);
        const Real nearestY = region.center.y +
            std::remainder(y - region.center.y, layer.periodYUm);
        for (int shiftY = -1; shiftY <= 1; ++shiftY) {
            for (int shiftX = -1; shiftX <= 1; ++shiftX) {
                if (pointInsideRegion(
                        region,
                        nearestX + static_cast<Real>(shiftX) * layer.periodXUm,
                        nearestY + static_cast<Real>(shiftY) * layer.periodYUm)) {
                    return regionIndex + 1;
                }
            }
        }
    }
    return 0;
}

const Material& patternedMaterialForSlot(const PatternedLayer2D& layer,
                                         std::size_t slot) {
    return slot == 0 ? layer.background : layer.regions.at(slot - 1).material;
}

std::pair<int, int> patternedSampleCounts(const PatternedLayer2D& layer,
                                          const HarmonicBasis& basis) {
    const int recommended = recommendedPatternSampleCount(basis);
    const int minimumX = 4 * basis.orderX + 1;
    const int minimumY = 4 * basis.orderY + 1;
    return {
        layer.sampleCountX > 0
            ? std::max(layer.sampleCountX, minimumX)
            : recommended,
        layer.sampleCountY > 0
            ? std::max(layer.sampleCountY, minimumY)
            : recommended,
    };
}

template <typename Visitor>
void forEachPatternSample(const PatternedLayer2D& layer,
                          int samplesX,
                          int samplesY,
                          Visitor&& visitor) {
    std::size_t index = 0;
    for (int iy = 0; iy < samplesY; ++iy) {
        const Real y = ((iy + Real{0.5}) / samplesY - Real{0.5}) *
            layer.periodYUm;
        for (int ix = 0; ix < samplesX; ++ix) {
            const Real x = ((ix + Real{0.5}) / samplesX - Real{0.5}) *
                layer.periodXUm;
            visitor(index++, patternedMaterialSlotAt(layer, x, y));
        }
    }
}

std::pair<std::vector<Tensor3>, std::vector<Tensor3>> patternedSlotTensors(
    const PatternedLayer2D& layer,
    Real wavelengthUm) {
    std::vector<Tensor3> eps;
    std::vector<Tensor3> mu;
    eps.reserve(layer.regions.size() + 1);
    mu.reserve(layer.regions.size() + 1);
    const auto [backgroundEps, backgroundMu] = layer.background.tensors(wavelengthUm);
    eps.push_back(backgroundEps);
    mu.push_back(backgroundMu);
    for (const auto& region : layer.regions) {
        const auto [regionEps, regionMu] = region.material.tensors(wavelengthUm);
        eps.push_back(regionEps);
        mu.push_back(regionMu);
    }
    return {std::move(eps), std::move(mu)};
}

bool coordinateAlignedRectangleBounds(const PatternRegion& region,
                                      Real periodXUm,
                                      Real periodYUm,
                                      std::array<Real, 4>* bounds) {
    if (region.shape != PatternRegionShape::Rectangle) {
        return false;
    }
    const Real scale = std::max({Real{1}, periodXUm, periodYUm});
    const Real tolerance = Real{128} * std::numeric_limits<Real>::epsilon() * scale;
    const Real reduced = std::remainder(region.angleDeg, Real{180});
    Real halfwidthX{};
    Real halfwidthY{};
    if (std::abs(reduced) <= tolerance) {
        halfwidthX = region.halfwidthXUm;
        halfwidthY = region.halfwidthYUm;
    } else if (std::abs(std::abs(reduced) - Real{90}) <= tolerance) {
        halfwidthX = region.halfwidthYUm;
        halfwidthY = region.halfwidthXUm;
    } else {
        return false;
    }

    *bounds = {
        region.center.x - halfwidthX,
        region.center.x + halfwidthX,
        region.center.y - halfwidthY,
        region.center.y + halfwidthY,
    };
    return true;
}

using ReticoloRegionRectangles = std::vector<std::array<Real, 4>>;

std::vector<Interval1D> periodicIntervals(Real start, Real end, Real period) {
    const Real cellMin = -Real{0.5} * period;
    const Real cellMax = Real{0.5} * period;
    const Real width = end - start;
    const Real tolerance = Real{128} * std::numeric_limits<Real>::epsilon() *
        std::max(Real{1.0}, period);
    if (width <= tolerance) {
        return {};
    }
    if (width >= period - tolerance) {
        return {{cellMin, cellMax}};
    }

    Real wrappedStart = start -
        std::floor((start - cellMin) / period) * period;
    if (wrappedStart >= cellMax - tolerance) {
        wrappedStart = cellMin;
    }
    const Real wrappedEnd = wrappedStart + width;
    if (wrappedEnd <= cellMax + tolerance) {
        return {{
            std::clamp(wrappedStart, cellMin, cellMax),
            std::clamp(wrappedEnd, cellMin, cellMax)}};
    }
    return {
        {wrappedStart, cellMax},
        {cellMin, wrappedEnd - period},
    };
}

ReticoloRegionRectangles wrapReticoloRectangles(
    const ReticoloRegionRectangles& rectangles,
    Real periodXUm,
    Real periodYUm) {
    ReticoloRegionRectangles wrapped;
    for (const auto& rectangle : rectangles) {
        const auto xIntervals = periodicIntervals(
            rectangle[0], rectangle[1], periodXUm);
        const auto yIntervals = periodicIntervals(
            rectangle[2], rectangle[3], periodYUm);
        for (const Interval1D x : xIntervals) {
            for (const Interval1D y : yIntervals) {
                wrapped.push_back({x.start, x.end, y.start, y.end});
            }
        }
    }
    return wrapped;
}

bool reticoloCircleRectangles(const PatternRegion& region,
                              ReticoloRegionRectangles* rectangles) {
    if (region.shape != PatternRegionShape::Circle) {
        return false;
    }
    rectangles->clear();
    if (region.radiusUm == Real{0.0}) {
        return true;
    }

    const int count = region.reticoloRectangleCount;
    std::vector<Real> halfwidthX(static_cast<std::size_t>(count));
    std::vector<Real> halfwidthY(static_cast<std::size_t>(count));
    Real quadrantArea{};
    Real previousY{};
    for (int rectangle = 0; rectangle < count; ++rectangle) {
        // RETICOLO V9 res0 default sets res1.angles=1, which negates the
        // ellipse selector before retu.m and selects equal angular spacing.
        const Real angle = pi * static_cast<Real>(2 * rectangle + 1) /
            (Real{4.0} * static_cast<Real>(count));
        const Real ax = std::cos(angle);
        const Real ay = std::sin(angle);
        halfwidthX[static_cast<std::size_t>(rectangle)] = ax;
        halfwidthY[static_cast<std::size_t>(rectangle)] = ay;
        quadrantArea += ax * (ay - previousY);
        previousY = ay;
    }
    const Real scale = std::sqrt(pi / (Real{4.0} * quadrantArea));
    rectangles->reserve(static_cast<std::size_t>(count));
    for (int rectangle = 0; rectangle < count; ++rectangle) {
        const Real hx = region.radiusUm * scale *
            halfwidthX[static_cast<std::size_t>(rectangle)];
        const Real hy = region.radiusUm * scale *
            halfwidthY[static_cast<std::size_t>(rectangle)];
        std::array<Real, 4> bounds{
            region.center.x - hx,
            region.center.x + hx,
            region.center.y - hy,
            region.center.y + hy,
        };
        rectangles->push_back(bounds);
    }
    return true;
}

bool reticoloRegionRectangles(const PatternRegion& region,
                              Real periodXUm,
                              Real periodYUm,
                              ReticoloRegionRectangles* rectangles) {
    std::array<Real, 4> rectangle{};
    ReticoloRegionRectangles unwrapped;
    if (coordinateAlignedRectangleBounds(
            region, periodXUm, periodYUm, &rectangle)) {
        unwrapped = {rectangle};
    } else if (!reticoloCircleRectangles(region, &unwrapped)) {
        return false;
    }
    *rectangles = wrapReticoloRectangles(
        unwrapped, periodXUm, periodYUm);
    return true;
}

bool pointInsideReticoloRectangles(const ReticoloRegionRectangles& rectangles,
                                   Real x,
                                   Real y) {
    for (const auto& rectangle : rectangles) {
        if (x >= rectangle[0] && x <= rectangle[1] &&
            y >= rectangle[2] && y <= rectangle[3]) {
            return true;
        }
    }
    return false;
}

void sortUniqueBoundaries(std::vector<Real>& boundaries, Real scale) {
    std::sort(boundaries.begin(), boundaries.end());
    const Real tolerance = Real{128} * std::numeric_limits<Real>::epsilon() *
        std::max(Real{1}, scale);
    std::vector<Real> unique;
    unique.reserve(boundaries.size());
    for (const Real value : boundaries) {
        if (unique.empty() || std::abs(value - unique.back()) > tolerance) {
            unique.push_back(value);
        }
    }
    boundaries = std::move(unique);
}

Complex intervalFourierWeight(Real start, Real end, int order) {
    const Real width = end - start;
    const Real center = Real{0.5} * (start + end);
    const Real integral = order == 0
        ? width
        : std::sin(pi * static_cast<Real>(order) * width) /
            (pi * static_cast<Real>(order));
    return integral * std::exp(
        Complex{0.0, -twoPi * static_cast<Real>(order) * center});
}

detail::LiTensorBlocks rectilinearTensorConvolution(
    const std::vector<Tensor3>& cells,
    const std::vector<Real>& xBoundaries,
    const std::vector<Real>& yBoundaries,
    const HarmonicBasis& basis) {
    if (xBoundaries.size() < 2 || yBoundaries.size() < 2) {
        throw std::invalid_argument(
            "rectilinear tensor convolution requires x and y boundaries");
    }
    const int cellCountX = static_cast<int>(xBoundaries.size() - 1);
    const int cellCountY = static_cast<int>(yBoundaries.size() - 1);
    const auto expected = checkedGridSize(
        cellCountX, cellCountY, "rectilinear tensor cell");
    if (cells.size() != expected) {
        throw std::invalid_argument(
            "rectilinear tensor cell array size must equal cell_count_x * cell_count_y");
    }
    const int coefficientCountX = 4 * basis.orderX + 1;
    const int coefficientCountY = 4 * basis.orderY + 1;
    const int halfX = coefficientCountX / 2;
    const int halfY = coefficientCountY / 2;

    std::vector<std::vector<Complex>> xWeights(
        static_cast<std::size_t>(coefficientCountX),
        std::vector<Complex>(static_cast<std::size_t>(cellCountX)));
    std::vector<std::vector<Complex>> yWeights(
        static_cast<std::size_t>(coefficientCountY),
        std::vector<Complex>(static_cast<std::size_t>(cellCountY)));
    for (int orderX = -halfX; orderX <= halfX; ++orderX) {
        auto& weights = xWeights[static_cast<std::size_t>(orderX + halfX)];
        for (int cellX = 0; cellX < cellCountX; ++cellX) {
            weights[static_cast<std::size_t>(cellX)] = intervalFourierWeight(
                xBoundaries[static_cast<std::size_t>(cellX)],
                xBoundaries[static_cast<std::size_t>(cellX + 1)],
                orderX);
        }
    }
    for (int orderY = -halfY; orderY <= halfY; ++orderY) {
        auto& weights = yWeights[static_cast<std::size_t>(orderY + halfY)];
        for (int cellY = 0; cellY < cellCountY; ++cellY) {
            weights[static_cast<std::size_t>(cellY)] = intervalFourierWeight(
                yBoundaries[static_cast<std::size_t>(cellY)],
                yBoundaries[static_cast<std::size_t>(cellY + 1)],
                orderY);
        }
    }

    std::array<std::array<std::vector<Complex>, 3>, 3> coefficients;
    const auto coefficientTotal = static_cast<std::size_t>(
        coefficientCountX * coefficientCountY);
    for (auto& row : coefficients) {
        for (auto& component : row) {
            component.assign(coefficientTotal, Complex{});
        }
    }
    for (int orderY = -halfY; orderY <= halfY; ++orderY) {
        for (int orderX = -halfX; orderX <= halfX; ++orderX) {
            const std::size_t coefficientIndex = localCoeffIndex2d(
                orderX, orderY, coefficientCountX, coefficientCountY);
            for (int cellY = 0; cellY < cellCountY; ++cellY) {
                const Complex yWeight = yWeights[
                    static_cast<std::size_t>(orderY + halfY)][
                        static_cast<std::size_t>(cellY)];
                for (int cellX = 0; cellX < cellCountX; ++cellX) {
                    const Complex weight = yWeight * xWeights[
                        static_cast<std::size_t>(orderX + halfX)][
                            static_cast<std::size_t>(cellX)];
                    const Tensor3& cell = cells[
                        static_cast<std::size_t>(cellY * cellCountX + cellX)];
                    for (std::size_t row = 0; row < 3; ++row) {
                        for (std::size_t column = 0; column < 3; ++column) {
                            coefficients[row][column][coefficientIndex] +=
                                cell(row, column) * weight;
                        }
                    }
                }
            }
        }
    }

    detail::LiTensorBlocks direct;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            direct[row][column] = convMatrix2d(
                coefficients[row][column],
                coefficientCountX,
                coefficientCountY,
                basis);
        }
    }
    return direct;
}

TensorFourierMatrices reticoloCellTensors(
    const std::vector<Tensor3>& epsCells,
    const std::vector<Tensor3>& muCells,
    const std::vector<Real>& normalizedX,
    const std::vector<Real>& normalizedY,
    const HarmonicBasis& basis,
    bool approximatedGeometry) {
    detail::LiFactorization epsLi = detail::liFactorizeTensor2dReticoloCells(
        epsCells, normalizedX, normalizedY, basis, "epsilon");
    detail::LiFactorization muLi = detail::liFactorizeTensor2dReticoloCells(
        muCells, normalizedX, normalizedY, basis, "mu");

    TensorFourierMatrices out;
    out.eps = rectilinearTensorConvolution(
        epsCells, normalizedX, normalizedY, basis);
    out.mu = rectilinearTensorConvolution(
        muCells, normalizedX, normalizedY, basis);
    out.liEpsQ = std::move(epsLi.q);
    out.liEpsQZzInverse = std::move(epsLi.zzInverse);
    out.liMuQ = std::move(muLi.q);
    out.liMuQZzInverse = std::move(muLi.zzInverse);
    out.metadata.sequentialLiFactorization2d =
        basis.orderX > 0 && basis.orderY > 0;
    out.metadata.factorizationUsesSampledGeometry = approximatedGeometry;
    out.metadata.requiresJointConvergence = approximatedGeometry;
    return out;
}

bool tryRectilinearReticolo(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm,
    TensorFourierMatrices* out) {
    const auto [epsBySlot, muBySlot] = patternedSlotTensors(layer, wavelengthUm);

    std::vector<Real> xBoundaries{
        -Real{0.5} * layer.periodXUm,
        Real{0.5} * layer.periodXUm,
    };
    std::vector<Real> yBoundaries{
        -Real{0.5} * layer.periodYUm,
        Real{0.5} * layer.periodYUm,
    };
    std::vector<ReticoloRegionRectangles> regionRectangles;
    regionRectangles.reserve(layer.regions.size());
    for (const PatternRegion& region : layer.regions) {
        ReticoloRegionRectangles rectangles;
        if (!reticoloRegionRectangles(
                region, layer.periodXUm, layer.periodYUm, &rectangles)) {
            return false;
        }
        for (const auto& bounds : rectangles) {
            xBoundaries.push_back(bounds[0]);
            xBoundaries.push_back(bounds[1]);
            yBoundaries.push_back(bounds[2]);
            yBoundaries.push_back(bounds[3]);
        }
        regionRectangles.push_back(std::move(rectangles));
    }
    sortUniqueBoundaries(xBoundaries, layer.periodXUm);
    sortUniqueBoundaries(yBoundaries, layer.periodYUm);

    const int cellCountX = static_cast<int>(xBoundaries.size() - 1);
    const int cellCountY = static_cast<int>(yBoundaries.size() - 1);
    std::vector<Tensor3> epsCells;
    std::vector<Tensor3> muCells;
    epsCells.reserve(static_cast<std::size_t>(cellCountX * cellCountY));
    muCells.reserve(static_cast<std::size_t>(cellCountX * cellCountY));
    for (int cellY = 0; cellY < cellCountY; ++cellY) {
        const Real y = Real{0.5} * (
            yBoundaries[static_cast<std::size_t>(cellY)] +
            yBoundaries[static_cast<std::size_t>(cellY + 1)]);
        for (int cellX = 0; cellX < cellCountX; ++cellX) {
            const Real x = Real{0.5} * (
                xBoundaries[static_cast<std::size_t>(cellX)] +
                xBoundaries[static_cast<std::size_t>(cellX + 1)]);
            std::size_t slot = 0;
            for (std::size_t offset = 0; offset < regionRectangles.size(); ++offset) {
                const std::size_t regionIndex = regionRectangles.size() - 1 - offset;
                if (pointInsideReticoloRectangles(
                        regionRectangles[regionIndex], x, y)) {
                    slot = regionIndex + 1;
                    break;
                }
            }
            epsCells.push_back(epsBySlot[slot]);
            muCells.push_back(muBySlot[slot]);
        }
    }

    std::vector<Real> normalizedX = xBoundaries;
    std::vector<Real> normalizedY = yBoundaries;
    for (Real& value : normalizedX) {
        value /= layer.periodXUm;
    }
    for (Real& value : normalizedY) {
        value /= layer.periodYUm;
    }

    *out = reticoloCellTensors(
        epsCells, muCells, normalizedX, normalizedY, basis, false);
    return true;
}

std::vector<Real> uniformCellBoundaries(int cellCount) {
    if (cellCount <= 0) {
        throw std::invalid_argument("uniform RETICOLO cell count must be positive");
    }
    std::vector<Real> boundaries(static_cast<std::size_t>(cellCount + 1));
    for (int cell = 0; cell <= cellCount; ++cell) {
        boundaries[static_cast<std::size_t>(cell)] =
            static_cast<Real>(cell) / static_cast<Real>(cellCount) - Real{0.5};
    }
    boundaries.front() = -Real{0.5};
    boundaries.back() = Real{0.5};
    return boundaries;
}

std::pair<std::vector<Tensor3>, std::vector<Tensor3>> expandPatternCellTensors(
    const std::vector<Tensor3>& epsBySlot,
    const std::vector<Tensor3>& muBySlot,
    const std::vector<std::size_t>& materialSlots) {
    if (epsBySlot.size() != muBySlot.size() || epsBySlot.empty()) {
        throw std::invalid_argument("patterned material tensor slot tables are inconsistent");
    }
    std::vector<Tensor3> epsCells;
    std::vector<Tensor3> muCells;
    epsCells.reserve(materialSlots.size());
    muCells.reserve(materialSlots.size());
    for (const std::size_t slot : materialSlots) {
        if (slot >= epsBySlot.size()) {
            throw std::invalid_argument("patterned material cell slot is out of range");
        }
        epsCells.push_back(epsBySlot[slot]);
        muCells.push_back(muBySlot[slot]);
    }
    return {std::move(epsCells), std::move(muCells)};
}

bool geometryCacheMatchesLayer(const PatternedLayer2D& layer,
                               const HarmonicBasis& basis,
                               const PatternedLayer2DGeometryCache& cache) {
    const int coeffCountX = patternedCoeffCountX(basis);
    const int coeffCountY = patternedCoeffCountY(basis);
    const auto [samplesX, samplesY] = patternedSampleCounts(layer, basis);
    return cache.materialSlotCount == layer.regions.size() + 1 &&
           cache.sampleCountX == samplesX &&
           cache.sampleCountY == samplesY &&
           cache.coeffCountX == coeffCountX &&
           cache.coeffCountY == coeffCountY &&
           cache.sampleMaterialIndices.size() ==
               checkedGridSize(
                   cache.sampleCountX,
                   cache.sampleCountY,
                   "patterned layer sample grid") &&
           cache.geometryFingerprint == patternGeometryFingerprint(
               layer,
               cache.sampleCountX,
               cache.sampleCountY,
               coeffCountX,
               coeffCountY);
}

PeriodicLayer2D samplePatternedLayer2dWithCounts(const PatternedLayer2D& layer,
                                                      int samplesX,
                                                      int samplesY) {
    requirePositive(layer.periodXUm, "patterned layer period_x");
    requirePositive(layer.periodYUm, "patterned layer period_y");
    if (!std::isfinite(layer.thicknessUm) || layer.thicknessUm < 0.0) {
        throw std::invalid_argument("patterned layer thickness must be finite and non-negative");
    }
    const auto sampleTotal = checkedGridSize(
        samplesX, samplesY, "patterned layer sample grid");
    PeriodicLayer2D sampled;
    sampled.sampleCountX = samplesX;
    sampled.sampleCountY = samplesY;
    sampled.periodXUm = layer.periodXUm;
    sampled.periodYUm = layer.periodYUm;
    sampled.thicknessUm = layer.thicknessUm;
    sampled.cells.reserve(sampleTotal);
    for (const auto& region : layer.regions) {
        validateRegion(region);
    }

    forEachPatternSample(
        layer,
        samplesX,
        samplesY,
        [&](std::size_t, std::size_t slot) {
            sampled.cells.push_back(patternedMaterialForSlot(layer, slot));
        });
    return sampled;
}

PeriodicLayer2D samplePatternedLayer2d(const PatternedLayer2D& layer,
                                          const HarmonicBasis& basis) {
    const auto [samplesX, samplesY] = patternedSampleCounts(layer, basis);
    return samplePatternedLayer2dWithCounts(layer, samplesX, samplesY);
}

TensorFourierMatrices rectilinearPatternApproximationTensors(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm,
    const PatternedLayer2DGeometryCache* cache) {
    const auto [samplesX, samplesY] = patternedSampleCounts(layer, basis);
    const auto sampleTotal = checkedGridSize(
        samplesX, samplesY, "patterned layer RETICOLO cell grid");
    std::vector<std::size_t> materialSlots;
    if (cache != nullptr && geometryCacheMatchesLayer(layer, basis, *cache)) {
        materialSlots = cache->sampleMaterialIndices;
    } else {
        materialSlots.resize(sampleTotal);
        forEachPatternSample(
            layer,
            samplesX,
            samplesY,
            [&](std::size_t index, std::size_t slot) {
                materialSlots[index] = slot;
            });
    }

    const auto [epsBySlot, muBySlot] = patternedSlotTensors(layer, wavelengthUm);
    auto [epsCells, muCells] = expandPatternCellTensors(
        epsBySlot,
        muBySlot,
        materialSlots);
    return reticoloCellTensors(
        epsCells,
        muCells,
        uniformCellBoundaries(samplesX),
        uniformCellBoundaries(samplesY),
        basis,
        true);
}

PatternedLayer2DGeometryCache precomputePatternedLayer2dGeometry(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis) {
    requirePositive(layer.periodXUm, "patterned layer period_x");
    requirePositive(layer.periodYUm, "patterned layer period_y");
    if (!std::isfinite(layer.thicknessUm) || layer.thicknessUm < Real{0.0}) {
        throw std::invalid_argument("patterned layer thickness must be finite and non-negative");
    }
    for (const auto& region : layer.regions) {
        validateRegion(region);
    }

    const auto [samplesX, samplesY] = patternedSampleCounts(layer, basis);
    const auto sampleTotal = checkedGridSize(
        samplesX, samplesY, "patterned layer sample grid");
    const int coeffCountX = patternedCoeffCountX(basis);
    const int coeffCountY = patternedCoeffCountY(basis);
    PatternedLayer2DGeometryCache cache;
    cache.materialSlotCount = layer.regions.size() + 1;
    cache.sampleCountX = samplesX;
    cache.sampleCountY = samplesY;
    cache.coeffCountX = coeffCountX;
    cache.coeffCountY = coeffCountY;
    cache.geometryFingerprint =
        patternGeometryFingerprint(layer, samplesX, samplesY, coeffCountX, coeffCountY);
    cache.sampleMaterialIndices.reserve(sampleTotal);

    forEachPatternSample(
        layer,
        samplesX,
        samplesY,
        [&](std::size_t, std::size_t slot) {
            cache.sampleMaterialIndices.push_back(slot);
        });
    return cache;
}

TensorFourierMatrices tensorFourierMatricesPatternedLayer2dDefault(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm) {
    requirePositive(layer.periodXUm, "patterned layer period_x");
    requirePositive(layer.periodYUm, "patterned layer period_y");
    if (!std::isfinite(layer.thicknessUm) || layer.thicknessUm < 0.0) {
        throw std::invalid_argument("patterned layer thickness must be finite and non-negative");
    }

    for (const auto& region : layer.regions) {
        validateRegion(region);
    }
    if (layer.regions.empty()) {
        const auto [eps, mu] = layer.background.tensors(wavelengthUm);
        return tensorFourierMatricesUniform(eps, mu, basis);
    }

    if (layer.preferAnalytic) {
        TensorFourierMatrices lamellarTensors;
        if (tryFullCellLamellarRectangles(
                layer,
                basis,
                wavelengthUm,
                LamellarAxis::X,
                &lamellarTensors)) {
            return lamellarTensors;
        }
        if (tryFullCellLamellarRectangles(
                layer,
                basis,
                wavelengthUm,
                LamellarAxis::Y,
                &lamellarTensors)) {
            return lamellarTensors;
        }

        TensorFourierMatrices rectilinearTensors;
        if (tryRectilinearReticolo(
                layer,
                basis,
                wavelengthUm,
                &rectilinearTensors)) {
            return rectilinearTensors;
        }
    }

    return rectilinearPatternApproximationTensors(
        layer,
        basis,
        wavelengthUm,
        layer.geometryCache.get());
}

namespace {

struct FormulationGrid {
    int x{};
    int y{};
};

int nextFastFourierSize(int value) {
    auto isFast = [](int candidate) {
        for (const int factor : {2, 3, 5}) {
            while (candidate % factor == 0) {
                candidate /= factor;
            }
        }
        return candidate == 1;
    };
    while (!isFast(value)) {
        if (value == std::numeric_limits<int>::max()) {
            throw std::overflow_error("Fourier formulation grid is too large");
        }
        ++value;
    }
    return value;
}

FormulationGrid formulationGrid(const PatternedLayer2D& layer,
                                const HarmonicBasis& basis) {
    const auto& options = layer.fourierOptions;
    const FourierFormulation formulation = effectiveFourierFormulation(options);
    const bool keepInactiveAxis = formulation == FourierFormulation::Kottke ||
        formulation == FourierFormulation::PolBasisJones;
    const auto axisSize = [&](int order, int requested) {
        if (order == 0 && !keepInactiveAxis) {
            return 1;
        }
        const std::size_t scaledSize = checkedProduct(
            static_cast<std::size_t>(std::max(order, 1)),
            static_cast<std::size_t>(options.resolution),
            "Fourier formulation grid");
        if (scaledSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Fourier formulation grid is too large");
        }
        const int scaled = static_cast<int>(scaledSize);
        return nextFastFourierSize(std::max(scaled, requested));
    };
    return {
        axisSize(basis.orderX, layer.sampleCountX),
        axisSize(basis.orderY, layer.sampleCountY),
    };
}

void smoothBlocks(std::array<std::array<Matrix, 3>, 3>& blocks,
                  const HarmonicBasis& basis,
                  const PatternedLayer2D& layer) {
    for (auto& row : blocks) {
        for (Matrix& block : row) {
            applyLanczosSmoothing(
                block,
                basis,
                layer.periodXUm,
                layer.periodYUm,
                layer.fourierOptions);
        }
    }
}

std::pair<bool, bool> scalarConvolutionDependencies(const Matrix& convolution,
                                                    const HarmonicBasis& basis) {
    const Real tolerance = Real{1e-12} * std::max(Real{1.0}, maxAbs(convolution));
    bool dependsX = false;
    bool dependsY = false;
    for (std::size_t row = 0; row < basis.size(); ++row) {
        for (std::size_t col = 0; col < basis.size(); ++col) {
            if (row == col || std::abs(convolution(row, col)) <= tolerance) {
                continue;
            }
            dependsX = dependsX || basis.orders[row].m != basis.orders[col].m;
            dependsY = dependsY || basis.orders[row].n != basis.orders[col].n;
        }
    }
    return {dependsX, dependsY};
}

void fillScalarFactorization(std::array<std::array<Matrix, 3>, 3>& q,
                             Matrix& zzInverse,
                             const Matrix& direct,
                             const Matrix& reciprocal,
                             bool factorizeX,
                             bool factorizeY) {
    const std::size_t n = direct.rows();
    const Matrix zero(n, n);
    const Matrix inverseRule = (factorizeX || factorizeY)
        ? inverse(reciprocal)
        : direct;
    q[0][0] = factorizeX ? inverseRule : direct;
    q[0][1] = zero;
    q[0][2] = zero;
    q[1][0] = zero;
    q[1][1] = factorizeY ? inverseRule : direct;
    q[1][2] = zero;
    q[2][0] = zero;
    q[2][1] = zero;
    q[2][2] = direct;
    zzInverse = reciprocal;
}

bool makeReciprocalScalarPattern(const PatternedLayer2D& layer,
                                 Real wavelengthUm,
                                 PatternedLayer2D* reciprocal) {
    if (reciprocal == nullptr) {
        return false;
    }
    auto reciprocalMaterial = [&](const Material& material,
                                  const std::string& name,
                                  Material* output) {
        const auto [epsTensor, muTensor] = material.tensors(wavelengthUm);
        Complex eps{};
        Complex mu{};
        if (!tensorIsScalar(epsTensor, &eps) || !tensorIsScalar(muTensor, &mu) ||
            std::abs(eps) <= Real{1e-30} || std::abs(mu) <= Real{1e-30}) {
            return false;
        }
        *output = Material::constant(
            name,
            Complex{1.0, 0.0} / eps,
            Complex{1.0, 0.0} / mu);
        return true;
    };

    *reciprocal = layer;
    reciprocal->geometryCache.reset();
    reciprocal->fourierOptions = {};
    if (!reciprocalMaterial(
            layer.background, "__reciprocal_background", &reciprocal->background)) {
        return false;
    }
    for (std::size_t index = 0; index < layer.regions.size(); ++index) {
        if (!reciprocalMaterial(
                layer.regions[index].material,
                "__reciprocal_region_" + std::to_string(index),
                &reciprocal->regions[index].material)) {
            return false;
        }
    }
    return true;
}

std::size_t wrappedGridIndex(int x, int y, int countX, int countY) {
    x %= countX;
    y %= countY;
    if (x < 0) {
        x += countX;
    }
    if (y < 0) {
        y += countY;
    }
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(countX) +
        static_cast<std::size_t>(x);
}

struct TangentField {
    std::vector<Real> x;
    std::vector<Real> y;
    int countX{};
    int countY{};
};

struct TangentConstraint {
    Real value{};
    int intersections{};
};

TangentConstraint regionTangentConstraint(const PatternRegion& region,
                                           Vec2 start,
                                           Vec2 delta,
                                           bool xComponent) {
    TangentConstraint out;
    const Real segmentLength = std::hypot(delta.x, delta.y);
    if (segmentLength <= Real{0.0}) {
        return out;
    }
    const auto accumulate = [&](Real tx, Real ty) {
        out.value += segmentLength * (xComponent ? tx : ty);
        ++out.intersections;
    };

    if (region.shape == PatternRegionShape::Circle) {
        const Real ox = start.x - region.center.x;
        const Real oy = start.y - region.center.y;
        const Real a = delta.x * delta.x + delta.y * delta.y;
        const Real b = Real{2.0} * (ox * delta.x + oy * delta.y);
        const Real c = ox * ox + oy * oy - region.radiusUm * region.radiusUm;
        const Real discriminant = b * b - Real{4.0} * a * c;
        const Real scale = std::max({Real{1.0}, b * b, std::abs(Real{4.0} * a * c)});
        const Real tolerance = Real{128.0} * std::numeric_limits<Real>::epsilon() * scale;
        if (discriminant < -tolerance || region.radiusUm <= Real{0.0}) {
            return out;
        }
        const Real root = std::sqrt(std::max(discriminant, Real{0.0}));
        const std::array<Real, 2> parameters{
            (-b - root) / (Real{2.0} * a),
            (-b + root) / (Real{2.0} * a),
        };
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            const Real t = parameters[index];
            if (t < -Real{1e-12} || t > Real{1.0} + Real{1e-12} ||
                (index == 1 && std::abs(parameters[1] - parameters[0]) <= Real{1e-12})) {
                continue;
            }
            const Real nx = (ox + t * delta.x) / region.radiusUm;
            const Real ny = (oy + t * delta.y) / region.radiusUm;
            accumulate(-ny, nx);
        }
        return out;
    }

    const std::vector<Vec2> vertices = regionBoundaryVertices(region);
    if (vertices.size() < 3) {
        return out;
    }
    const Real orientation = polygonSignedArea(vertices) < Real{0.0}
        ? Real{-1.0}
        : Real{1.0};
    for (std::size_t edge = 0; edge < vertices.size(); ++edge) {
        const Vec2 a = vertices[edge];
        const Vec2 b = vertices[(edge + 1) % vertices.size()];
        const Vec2 side{b.x - a.x, b.y - a.y};
        const Real denominator = delta.x * side.y - delta.y * side.x;
        const Real scale = std::max({
            Real{1.0},
            std::abs(delta.x),
            std::abs(delta.y),
            std::abs(side.x),
            std::abs(side.y)});
        if (std::abs(denominator) <=
            Real{128.0} * std::numeric_limits<Real>::epsilon() * scale * scale) {
            continue;
        }
        const Real ax = a.x - start.x;
        const Real ay = a.y - start.y;
        const Real t = (ax * side.y - ay * side.x) / denominator;
        const Real u = (ax * delta.y - ay * delta.x) / denominator;
        if (t < -Real{1e-12} || t > Real{1.0} + Real{1e-12} ||
            u < -Real{1e-12} || u > Real{1.0} + Real{1e-12}) {
            continue;
        }
        const Real length = std::hypot(side.x, side.y);
        if (length > Real{0.0}) {
            accumulate(
                orientation * side.x / length,
                orientation * side.y / length);
        }
    }
    return out;
}

TangentConstraint periodicTangentConstraint(const PatternedLayer2D& layer,
                                             Vec2 start,
                                             Vec2 delta,
                                             bool xComponent) {
    TangentConstraint out;
    for (int shiftY = -1; shiftY <= 1; ++shiftY) {
        for (int shiftX = -1; shiftX <= 1; ++shiftX) {
            const Vec2 shifted{
                start.x + shiftX * layer.periodXUm,
                start.y + shiftY * layer.periodYUm,
            };
            for (const PatternRegion& region : layer.regions) {
                const TangentConstraint local =
                    regionTangentConstraint(region, shifted, delta, xComponent);
                out.value += local.value;
                out.intersections += local.intersections;
            }
        }
    }
    return out;
}

Real vectorDot(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real out{};
    for (std::size_t index = 0; index < a.size(); ++index) {
        out += a[index] * b[index];
    }
    return out;
}

std::vector<Real> solvePeriodicTangentComponent(
    int countX,
    int countY,
    bool xComponent,
    const std::vector<unsigned char>& constrained,
    const std::vector<Real>& rhs) {
    const std::size_t total = checkedGridSize(
        countX, countY, "polarization vector field");
    const Real horizontalWeight = xComponent
        ? static_cast<Real>(countX) * countX /
            (static_cast<Real>(countY) * countY)
        : Real{1.0};
    const Real verticalWeight = xComponent
        ? Real{1.0}
        : static_cast<Real>(countY) * countY /
            (static_cast<Real>(countX) * countX);
    const Real baseDiagonal = Real{2.0} * (horizontalWeight + verticalWeight);
    const Real damping = Real{1e-8};
    const auto apply = [&](const std::vector<Real>& input, std::vector<Real>* output) {
        for (int y = 0; y < countY; ++y) {
            for (int x = 0; x < countX; ++x) {
                const std::size_t index = wrappedGridIndex(x, y, countX, countY);
                (*output)[index] =
                    (baseDiagonal + damping + (constrained[index] != 0 ? Real{1.0} : Real{0.0})) *
                        input[index] -
                    horizontalWeight * (
                        input[wrappedGridIndex(x - 1, y, countX, countY)] +
                        input[wrappedGridIndex(x + 1, y, countX, countY)]) -
                    verticalWeight * (
                        input[wrappedGridIndex(x, y - 1, countX, countY)] +
                        input[wrappedGridIndex(x, y + 1, countX, countY)]);
            }
        }
    };

    std::vector<Real> solution(total, Real{0.0});
    std::vector<Real> residual = rhs;
    const Real rhsNorm = std::sqrt(vectorDot(rhs, rhs));
    if (rhsNorm <= Real{0.0}) {
        return solution;
    }
    std::vector<Real> preconditioned(total);
    std::vector<Real> direction(total);
    std::vector<Real> product(total);
    for (std::size_t index = 0; index < total; ++index) {
        const Real diagonal = baseDiagonal + damping +
            (constrained[index] != 0 ? Real{1.0} : Real{0.0});
        preconditioned[index] = residual[index] / diagonal;
        direction[index] = preconditioned[index];
    }
    Real rho = vectorDot(residual, preconditioned);
    const Real tolerance = Real{1e-10} * static_cast<Real>(total);
    const std::size_t maximumIterations = 2 * total;
    for (std::size_t iteration = 0; iteration < maximumIterations; ++iteration) {
        apply(direction, &product);
        const Real denominator = vectorDot(direction, product);
        if (std::abs(denominator) <= std::numeric_limits<Real>::min()) {
            break;
        }
        const Real alpha = rho / denominator;
        for (std::size_t index = 0; index < total; ++index) {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        if (std::sqrt(vectorDot(residual, residual)) / rhsNorm <= tolerance) {
            break;
        }
        for (std::size_t index = 0; index < total; ++index) {
            const Real diagonal = baseDiagonal + damping +
                (constrained[index] != 0 ? Real{1.0} : Real{0.0});
            preconditioned[index] = residual[index] / diagonal;
        }
        const Real nextRho = vectorDot(residual, preconditioned);
        const Real beta = nextRho / rho;
        for (std::size_t index = 0; index < total; ++index) {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        rho = nextRho;
    }
    return solution;
}

TangentField generateTangentField(const PatternedLayer2D& layer,
                                  const HarmonicBasis& basis) {
    const FormulationGrid grid = formulationGrid(layer, basis);
    const std::size_t total = checkedGridSize(
        grid.x, grid.y, "patterned formulation grid");
    std::vector<unsigned char> constrainedX(total, 0);
    std::vector<unsigned char> constrainedY(total, 0);
    std::vector<Real> rhsX(total, Real{0.0});
    std::vector<Real> rhsY(total, Real{0.0});
    const Real dx = layer.periodXUm / grid.x;
    const Real dy = layer.periodYUm / grid.y;
    for (int y = 0; y < grid.y; ++y) {
        for (int x = 0; x < grid.x; ++x) {
            const std::size_t index = wrappedGridIndex(x, y, grid.x, grid.y);
            const Real left = (static_cast<Real>(x) / grid.x - Real{0.5}) *
                layer.periodXUm;
            const Real bottom = (static_cast<Real>(y) / grid.y - Real{0.5}) *
                layer.periodYUm;
            const TangentConstraint xConstraint = periodicTangentConstraint(
                layer,
                {left + Real{0.5} * dx, bottom - Real{0.5} * dy},
                {Real{0.0}, dy},
                true);
            const TangentConstraint yConstraint = periodicTangentConstraint(
                layer,
                {left - Real{0.5} * dx, bottom + Real{0.5} * dy},
                {dx, Real{0.0}},
                false);
            if (xConstraint.intersections > 0) {
                constrainedX[index] = 1;
                rhsX[index] = xConstraint.value;
            }
            if (yConstraint.intersections > 0) {
                constrainedY[index] = 1;
                rhsY[index] = yConstraint.value;
            }
        }
    }

    // Independently follows S4 7fd00a2 pattern_generate_flow_field_rect:
    // minimize periodic Dirichlet 1-form energy with contour-intersection
    // least-squares constraints (Liu & Fan, CPC 183, 2233-2244, 2012).
    const std::vector<Real> edgeX = solvePeriodicTangentComponent(
        grid.x, grid.y, true, constrainedX, rhsX);
    const std::vector<Real> edgeY = solvePeriodicTangentComponent(
        grid.x, grid.y, false, constrainedY, rhsY);
    std::vector<Real> fieldX(total);
    std::vector<Real> fieldY(total);
    for (int y = 0; y < grid.y; ++y) {
        for (int x = 0; x < grid.x; ++x) {
            const std::size_t index = wrappedGridIndex(x, y, grid.x, grid.y);
            fieldX[index] = Real{0.5} * (
                edgeX[index] + edgeX[wrappedGridIndex(x, y + 1, grid.x, grid.y)]);
            fieldY[index] = Real{0.5} * (
                edgeY[index] + edgeY[wrappedGridIndex(x + 1, y, grid.x, grid.y)]);
        }
    }

    Real maximumNorm{};
    for (std::size_t index = 0; index < total; ++index) {
        maximumNorm = std::max(
            maximumNorm,
            std::hypot(fieldX[index], fieldY[index]));
    }
    if (maximumNorm <= Real{1e-14}) {
        std::fill(fieldX.begin(), fieldX.end(), Real{1.0});
        std::fill(fieldY.begin(), fieldY.end(), Real{0.0});
        maximumNorm = Real{1.0};
    }
    for (std::size_t index = 0; index < total; ++index) {
        fieldX[index] /= maximumNorm;
        fieldY[index] /= maximumNorm;
    }

    if (layer.fourierOptions.polarizationBasis == PolarizationBasis::Normal) {
        for (std::size_t index = 0; index < total; ++index) {
            const Real norm = std::hypot(fieldX[index], fieldY[index]);
            if (norm > Real{1e-14}) {
                fieldX[index] /= norm;
                fieldY[index] /= norm;
            } else {
                fieldX[index] = Real{1.0};
                fieldY[index] = Real{0.0};
            }
        }
    }
    return {std::move(fieldX), std::move(fieldY), grid.x, grid.y};
}

TangentField constantTangentField(const PatternedLayer2D& layer,
                                  const HarmonicBasis& basis,
                                  Real tangentX,
                                  Real tangentY) {
    const FormulationGrid grid = formulationGrid(layer, basis);
    const std::size_t total = checkedGridSize(
        grid.x, grid.y, "patterned formulation grid");
    return {
        std::vector<Real>(total, tangentX),
        std::vector<Real>(total, tangentY),
        grid.x,
        grid.y,
    };
}

Matrix sampleConvolution(const std::vector<Complex>& samples,
                         int sampleCountX,
                         int sampleCountY,
                         const HarmonicBasis& basis) {
    const int coefficientCountX = 4 * basis.orderX + 1;
    const int coefficientCountY = 4 * basis.orderY + 1;
    return convMatrix2d(
        fourierCoeffsFromSamples2d(
            samples,
            sampleCountX,
            sampleCountY,
            coefficientCountX,
            coefficientCountY),
        coefficientCountX,
        coefficientCountY,
        basis);
}

void setScalarPolarizationBlocks(std::array<std::array<Matrix, 3>, 3>& q,
                                 Matrix& zzInverse,
                                 const Matrix& direct,
                                 const Matrix& reciprocal,
                                 const TangentField& field,
                                 const HarmonicBasis& basis,
                                 PolarizationBasis polarizationBasis) {
    const std::size_t n = basis.size();
    Complex uniformValue{};
    const Real scalarIdentityTolerance = Real{1e-11} * std::max(
        Real{1.0},
        direct.empty() ? Real{} : std::abs(direct(0, 0)));
    if (detail::matrixIsScalarIdentity(
            direct,
            scalarIdentityTolerance,
            &uniformValue)) {
        const Matrix zero(n, n);
        q[0][0] = direct;
        q[0][1] = zero;
        q[0][2] = zero;
        q[1][0] = zero;
        q[1][1] = direct;
        q[1][2] = zero;
        q[2][0] = zero;
        q[2][1] = zero;
        q[2][2] = direct;
        zzInverse = reciprocal;
        return;
    }

    if (polarizationBasis == PolarizationBasis::Jones) {
        std::vector<Complex> ux(field.x.size());
        std::vector<Complex> uy(field.x.size());
        std::vector<Complex> uxConjugate(field.x.size());
        std::vector<Complex> uyConjugate(field.x.size());
        for (std::size_t index = 0; index < field.x.size(); ++index) {
            Real tx = field.x[index];
            Real ty = field.y[index];
            Real amplitude = std::hypot(tx, ty);
            Real theta{};
            Real eta{};
            if (amplitude <= Real{1e-14}) {
                tx = Real{1.0};
                ty = Real{0.0};
                theta = Real{0.0};
                eta = Real{0.25} * pi;
            } else {
                theta = std::atan2(ty, tx);
                tx /= amplitude;
                ty /= amplitude;
                amplitude = std::min(amplitude, Real{1.0});
                eta = Real{0.125} * pi * (Real{1.0} + std::cos(pi * amplitude));
            }
            const Complex phase{std::cos(theta), std::sin(theta)};
            ux[index] = phase * Complex{
                std::cos(theta) * std::cos(eta),
                -std::sin(theta) * std::sin(eta)};
            uy[index] = phase * Complex{
                std::sin(theta) * std::cos(eta),
                std::cos(theta) * std::sin(eta)};
            uxConjugate[index] = std::conj(ux[index]);
            uyConjugate[index] = std::conj(uy[index]);
        }

        const Matrix Ux = sampleConvolution(
            ux, field.countX, field.countY, basis);
        const Matrix Uy = sampleConvolution(
            uy, field.countX, field.countY, basis);
        const Matrix UxConjugate = sampleConvolution(
            uxConjugate, field.countX, field.countY, basis);
        const Matrix UyConjugate = sampleConvolution(
            uyConjugate, field.countX, field.countY, basis);
        Matrix transform(2 * n, 2 * n);
        transform.setBlock(0, 0, Uy);
        transform.setBlock(0, n, UxConjugate);
        transform.setBlock(n, 0, -Ux);
        transform.setBlock(n, n, UyConjugate);
        Matrix diagonalRule(2 * n, 2 * n);
        diagonalRule.setBlock(0, 0, direct);
        diagonalRule.setBlock(n, n, inverse(reciprocal));
        const Matrix rotated = transform * diagonalRule * inverse(transform);
        const Matrix zero(n, n);
        q[0][0] = rotated.block(n, n, n, n);
        q[0][1] = -rotated.block(n, 0, n, n);
        q[0][2] = zero;
        q[1][0] = -rotated.block(0, n, n, n);
        q[1][1] = rotated.block(0, 0, n, n);
        q[1][2] = zero;
        q[2][0] = zero;
        q[2][1] = zero;
        q[2][2] = direct;
        zzInverse = reciprocal;
        return;
    }

    std::vector<Complex> pxx(field.x.size());
    std::vector<Complex> pxy(field.x.size());
    std::vector<Complex> pyy(field.x.size());
    for (std::size_t index = 0; index < field.x.size(); ++index) {
        pxx[index] = Complex{field.x[index] * field.x[index], 0.0};
        pxy[index] = Complex{field.x[index] * field.y[index], 0.0};
        pyy[index] = Complex{field.y[index] * field.y[index], 0.0};
    }
    const Matrix Pxx = sampleConvolution(
        pxx, field.countX, field.countY, basis);
    const Matrix Pxy = sampleConvolution(
        pxy, field.countX, field.countY, basis);
    const Matrix Pyy = sampleConvolution(
        pyy, field.countX, field.countY, basis);
    const Matrix delta = inverse(reciprocal) - direct;
    const Matrix eXx = direct + delta * Pxx;
    const Matrix eXy = delta * Pxy;
    const Matrix eYx = delta * Pxy;
    const Matrix eYy = direct + delta * Pyy;
    const Matrix zero(n, n);
    q[0][0] = eYy;
    q[0][1] = -eYx;
    q[0][2] = zero;
    q[1][0] = -eXy;
    q[1][1] = eXx;
    q[1][2] = zero;
    q[2][0] = zero;
    q[2][1] = zero;
    q[2][2] = direct;
    zzInverse = reciprocal;
}

std::array<Complex, 4> rotateInPlaneToNormalTangent(
    const Tensor3& tensor,
    Real nx,
    Real ny) {
    const Real tx = -ny;
    const Real ty = nx;
    return {
        nx * (tensor(0, 0) * nx + tensor(0, 1) * ny) +
            ny * (tensor(1, 0) * nx + tensor(1, 1) * ny),
        nx * (tensor(0, 0) * tx + tensor(0, 1) * ty) +
            ny * (tensor(1, 0) * tx + tensor(1, 1) * ty),
        tx * (tensor(0, 0) * nx + tensor(0, 1) * ny) +
            ty * (tensor(1, 0) * nx + tensor(1, 1) * ny),
        tx * (tensor(0, 0) * tx + tensor(0, 1) * ty) +
            ty * (tensor(1, 0) * tx + tensor(1, 1) * ty),
    };
}

void unrotateNormalTangentToInPlane(const std::array<Complex, 4>& rotated,
                                    Real nx,
                                    Real ny,
                                    Tensor3* tensor) {
    const Real tx = -ny;
    const Real ty = nx;
    (*tensor)(0, 0) = nx * (rotated[0] * nx + rotated[1] * tx) +
        tx * (rotated[2] * nx + rotated[3] * tx);
    (*tensor)(0, 1) = nx * (rotated[0] * ny + rotated[1] * ty) +
        tx * (rotated[2] * ny + rotated[3] * ty);
    (*tensor)(1, 0) = ny * (rotated[0] * nx + rotated[1] * tx) +
        ty * (rotated[2] * nx + rotated[3] * tx);
    (*tensor)(1, 1) = ny * (rotated[0] * ny + rotated[1] * ty) +
        ty * (rotated[2] * ny + rotated[3] * ty);
}

Tensor3 weightedTensor(const Tensor3& first,
                       const Tensor3& second,
                       Real firstFraction) {
    Tensor3 out;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            out(row, col) = firstFraction * first(row, col) +
                (Real{1.0} - firstFraction) * second(row, col);
        }
    }
    return out;
}

Tensor3 kottkeAverage(const Tensor3& first,
                      const Tensor3& second,
                      Real firstFraction,
                      Real nx,
                      Real ny) {
    Tensor3 out = weightedTensor(first, second, firstFraction);
    const auto a = rotateInPlaneToNormalTangent(first, nx, ny);
    const auto b = rotateInPlaneToNormalTangent(second, nx, ny);
    if (std::abs(a[0]) <= Real{1e-30} || std::abs(b[0]) <= Real{1e-30}) {
        return out;
    }
    const Real secondFraction = Real{1.0} - firstFraction;
    const Complex tau00 = firstFraction * (-Complex{1.0, 0.0} / a[0]) +
        secondFraction * (-Complex{1.0, 0.0} / b[0]);
    const Complex tau01 = firstFraction * (a[1] / a[0]) +
        secondFraction * (b[1] / b[0]);
    const Complex tau10 = firstFraction * (a[2] / a[0]) +
        secondFraction * (b[2] / b[0]);
    const Complex tau11 = firstFraction * (a[3] - a[2] * a[1] / a[0]) +
        secondFraction * (b[3] - b[2] * b[1] / b[0]);
    if (std::abs(tau00) <= Real{1e-30}) {
        return out;
    }
    const std::array<Complex, 4> effective{
        -Complex{1.0, 0.0} / tau00,
        -tau01 / tau00,
        -tau10 / tau00,
        tau11 - tau10 * tau01 / tau00,
    };
    unrotateNormalTangentToInPlane(effective, nx, ny, &out);
    return out;
}

Vec2 boundaryNormal(const PatternRegion& region, Real x, Real y) {
    if (region.shape == PatternRegionShape::Circle) {
        const Real dx = x - region.center.x;
        const Real dy = y - region.center.y;
        const Real norm = std::hypot(dx, dy);
        return norm > Real{1e-14} ? Vec2{dx / norm, dy / norm} : Vec2{1.0, 0.0};
    }
    const std::vector<Vec2> vertices = regionBoundaryVertices(region);
    Real bestDistance = std::numeric_limits<Real>::infinity();
    Vec2 best{1.0, 0.0};
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        const Vec2 a = vertices[index];
        const Vec2 b = vertices[(index + 1) % vertices.size()];
        const Real dx = b.x - a.x;
        const Real dy = b.y - a.y;
        const Real lengthSquared = dx * dx + dy * dy;
        if (lengthSquared <= Real{1e-30}) {
            continue;
        }
        const Real t = std::clamp(
            ((x - a.x) * dx + (y - a.y) * dy) / lengthSquared,
            Real{0.0},
            Real{1.0});
        const Real qx = a.x + t * dx;
        const Real qy = a.y + t * dy;
        const Real distance = std::hypot(x - qx, y - qy);
        if (distance < bestDistance) {
            const Real length = std::sqrt(lengthSquared);
            best = {dy / length, -dx / length};
            bestDistance = distance;
        }
    }
    return best;
}

std::vector<Vec2> clipPolygonToAxis(const std::vector<Vec2>& input,
                                    bool xAxis,
                                    Real boundary,
                                    bool keepGreater) {
    std::vector<Vec2> output;
    if (input.empty()) {
        return output;
    }
    const auto coordinate = [&](Vec2 point) { return xAxis ? point.x : point.y; };
    const auto inside = [&](Vec2 point) {
        return keepGreater
            ? coordinate(point) >= boundary
            : coordinate(point) <= boundary;
    };
    Vec2 previous = input.back();
    bool previousInside = inside(previous);
    for (const Vec2 current : input) {
        const bool currentInside = inside(current);
        if (previousInside != currentInside) {
            const Real denominator = coordinate(current) - coordinate(previous);
            if (std::abs(denominator) > Real{0.0}) {
                const Real t = (boundary - coordinate(previous)) / denominator;
                output.push_back({
                    previous.x + t * (current.x - previous.x),
                    previous.y + t * (current.y - previous.y),
                });
            }
        }
        if (currentInside) {
            output.push_back(current);
        }
        previous = current;
        previousInside = currentInside;
    }
    return output;
}

Real polygonPixelIntersectionArea(const std::vector<Vec2>& vertices,
                                  Real minX,
                                  Real maxX,
                                  Real minY,
                                  Real maxY) {
    std::vector<Vec2> clipped = clipPolygonToAxis(vertices, true, minX, true);
    clipped = clipPolygonToAxis(clipped, true, maxX, false);
    clipped = clipPolygonToAxis(clipped, false, minY, true);
    clipped = clipPolygonToAxis(clipped, false, maxY, false);
    return clipped.size() >= 3 ? std::abs(polygonSignedArea(clipped)) : Real{0.0};
}

bool pointInTriangle(Vec2 point, Vec2 a, Vec2 b, Vec2 c) {
    const Real ab = cross(a, b, point);
    const Real bc = cross(b, c, point);
    const Real ca = cross(c, a, point);
    const Real tolerance = Real{128.0} * std::numeric_limits<Real>::epsilon() *
        std::max({Real{1.0}, std::abs(ab), std::abs(bc), std::abs(ca)});
    return ab >= -tolerance && bc >= -tolerance && ca >= -tolerance;
}

std::vector<std::array<Vec2, 3>> triangulateSimplePolygon(
    const std::vector<Vec2>& input) {
    std::vector<Vec2> vertices = input;
    if (polygonSignedArea(vertices) < Real{0.0}) {
        std::reverse(vertices.begin(), vertices.end());
    }
    std::vector<std::size_t> indices(vertices.size());
    for (std::size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }
    std::vector<std::array<Vec2, 3>> triangles;
    triangles.reserve(vertices.size() >= 2 ? vertices.size() - 2 : 0);
    while (indices.size() > 3) {
        bool clipped = false;
        for (std::size_t position = 0; position < indices.size(); ++position) {
            const std::size_t previous =
                indices[(position + indices.size() - 1) % indices.size()];
            const std::size_t current = indices[position];
            const std::size_t next = indices[(position + 1) % indices.size()];
            if (cross(vertices[previous], vertices[current], vertices[next]) <= Real{0.0}) {
                continue;
            }
            bool containsVertex = false;
            for (const std::size_t candidate : indices) {
                if (candidate == previous || candidate == current || candidate == next) {
                    continue;
                }
                if (pointInTriangle(
                        vertices[candidate],
                        vertices[previous],
                        vertices[current],
                        vertices[next])) {
                    containsVertex = true;
                    break;
                }
            }
            if (containsVertex) {
                continue;
            }
            triangles.push_back({
                vertices[previous], vertices[current], vertices[next]});
            indices.erase(indices.begin() + static_cast<std::ptrdiff_t>(position));
            clipped = true;
            break;
        }
        if (!clipped) {
            throw std::invalid_argument("pattern polygon could not be triangulated");
        }
    }
    if (indices.size() == 3) {
        triangles.push_back({
            vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]});
    }
    return triangles;
}

Real circleVerticalOverlap(Real x, Real radius, Real minY, Real maxY) {
    const Real remaining = radius * radius - x * x;
    if (remaining <= Real{0.0}) {
        return Real{0.0};
    }
    const Real height = std::sqrt(remaining);
    return std::max(
        Real{0.0},
        std::min(maxY, height) - std::max(minY, -height));
}

Real integrateCirclePixel(Real radius,
                          Real minY,
                          Real maxY,
                          Real left,
                          Real right,
                          Real fLeft,
                          Real fMiddle,
                          Real fRight,
                          Real whole,
                          Real tolerance,
                          int depth) {
    const Real middle = Real{0.5} * (left + right);
    const Real leftMiddle = Real{0.5} * (left + middle);
    const Real rightMiddle = Real{0.5} * (middle + right);
    const Real fLeftMiddle = circleVerticalOverlap(leftMiddle, radius, minY, maxY);
    const Real fRightMiddle = circleVerticalOverlap(rightMiddle, radius, minY, maxY);
    const Real leftArea = (middle - left) *
        (fLeft + Real{4.0} * fLeftMiddle + fMiddle) / Real{6.0};
    const Real rightArea = (right - middle) *
        (fMiddle + Real{4.0} * fRightMiddle + fRight) / Real{6.0};
    const Real refined = leftArea + rightArea;
    if (depth == 0 || std::abs(refined - whole) <= Real{15.0} * tolerance) {
        return refined + (refined - whole) / Real{15.0};
    }
    return integrateCirclePixel(
               radius,
               minY,
               maxY,
               left,
               middle,
               fLeft,
               fLeftMiddle,
               fMiddle,
               leftArea,
               Real{0.5} * tolerance,
               depth - 1) +
        integrateCirclePixel(
               radius,
               minY,
               maxY,
               middle,
               right,
               fMiddle,
               fRightMiddle,
               fRight,
               rightArea,
               Real{0.5} * tolerance,
               depth - 1);
}

Real circlePixelIntersectionArea(const PatternRegion& circle,
                                 Real minX,
                                 Real maxX,
                                 Real minY,
                                 Real maxY) {
    minX -= circle.center.x;
    maxX -= circle.center.x;
    minY -= circle.center.y;
    maxY -= circle.center.y;
    const Real radius = circle.radiusUm;
    const Real left = std::max(minX, -radius);
    const Real right = std::min(maxX, radius);
    if (left >= right) {
        return Real{0.0};
    }
    const Real nearestX = std::clamp(Real{0.0}, minX, maxX);
    const Real nearestY = std::clamp(Real{0.0}, minY, maxY);
    if (nearestX * nearestX + nearestY * nearestY >= radius * radius &&
        !pointInsideRegion(circle, minX + circle.center.x, minY + circle.center.y) &&
        !pointInsideRegion(circle, minX + circle.center.x, maxY + circle.center.y) &&
        !pointInsideRegion(circle, maxX + circle.center.x, minY + circle.center.y) &&
        !pointInsideRegion(circle, maxX + circle.center.x, maxY + circle.center.y)) {
        return Real{0.0};
    }
    const Real middle = Real{0.5} * (left + right);
    const Real fLeft = circleVerticalOverlap(left, radius, minY, maxY);
    const Real fMiddle = circleVerticalOverlap(middle, radius, minY, maxY);
    const Real fRight = circleVerticalOverlap(right, radius, minY, maxY);
    const Real whole = (right - left) *
        (fLeft + Real{4.0} * fMiddle + fRight) / Real{6.0};
    const Real pixelArea = (maxX - minX) * (maxY - minY);
    return integrateCirclePixel(
        radius,
        minY,
        maxY,
        left,
        right,
        fLeft,
        fMiddle,
        fRight,
        whole,
        Real{1e-12} * pixelArea,
        18);
}

Real regionPixelIntersectionArea(const PatternRegion& region,
                                 const std::vector<std::array<Vec2, 3>>* triangles,
                                 Real minX,
                                 Real maxX,
                                 Real minY,
                                 Real maxY) {
    if (region.shape == PatternRegionShape::Circle) {
        return circlePixelIntersectionArea(region, minX, maxX, minY, maxY);
    }
    if (region.shape == PatternRegionShape::Rectangle) {
        return polygonPixelIntersectionArea(
            regionBoundaryVertices(region), minX, maxX, minY, maxY);
    }
    Real area{};
    if (triangles == nullptr) {
        throw std::invalid_argument("pattern polygon triangulation is unavailable");
    }
    for (const auto& triangle : *triangles) {
        area += polygonPixelIntersectionArea(
            {triangle[0], triangle[1], triangle[2]}, minX, maxX, minY, maxY);
    }
    return area;
}

Real regionArea(const PatternRegion& region) {
    if (region.shape == PatternRegionShape::Circle) {
        return pi * region.radiusUm * region.radiusUm;
    }
    if (region.shape == PatternRegionShape::Rectangle) {
        return Real{4.0} * region.halfwidthXUm * region.halfwidthYUm;
    }
    return std::abs(polygonSignedArea(region.vertices));
}

struct RegionCoveragePlan {
    std::vector<int> parent;
    std::vector<std::vector<std::array<Vec2, 3>>> polygonTriangles;
    bool exactIntersections{true};
};

RegionCoveragePlan makeRegionCoveragePlan(const PatternedLayer2D& layer) {
    RegionCoveragePlan plan;
    plan.parent.assign(layer.regions.size(), -1);
    plan.polygonTriangles.resize(layer.regions.size());
    for (std::size_t index = 0; index < layer.regions.size(); ++index) {
        if (layer.regions[index].shape == PatternRegionShape::Polygon) {
            plan.polygonTriangles[index] =
                triangulateSimplePolygon(layer.regions[index].vertices);
        }
    }
    for (std::size_t inner = 0; inner < layer.regions.size(); ++inner) {
        for (std::size_t later = inner + 1; later < layer.regions.size(); ++later) {
            if (regionContainsRegion(layer.regions[inner], layer.regions[later])) {
                plan.parent[inner] = -2;
                break;
            }
        }
    }
    for (std::size_t inner = 0; inner < layer.regions.size(); ++inner) {
        if (plan.parent[inner] == -2) {
            continue;
        }
        Real parentArea = std::numeric_limits<Real>::infinity();
        for (std::size_t outer = 0; outer < layer.regions.size(); ++outer) {
            if (inner == outer || plan.parent[outer] == -2 ||
                !regionContainsRegion(layer.regions[inner], layer.regions[outer])) {
                continue;
            }
            const Real area = regionArea(layer.regions[outer]);
            if (area < parentArea) {
                parentArea = area;
                plan.parent[inner] = static_cast<int>(outer);
            }
        }
    }
    for (std::size_t first = 0; first < layer.regions.size(); ++first) {
        if (plan.parent[first] == -2) {
            continue;
        }
        for (std::size_t second = first + 1; second < layer.regions.size(); ++second) {
            if (plan.parent[second] == -2) {
                continue;
            }
            const bool nested =
                regionContainsRegion(layer.regions[first], layer.regions[second]) ||
                regionContainsRegion(layer.regions[second], layer.regions[first]);
            if (!nested && regionsHaveInteriorOverlap(
                    layer.regions[first], layer.regions[second])) {
                plan.exactIntersections = false;
            }
        }
    }
    return plan;
}

std::vector<Real> pixelMaterialFractions(const PatternedLayer2D& layer,
                                         const RegionCoveragePlan& plan,
                                         int pixelX,
                                         int pixelY,
                                         int countX,
                                         int countY) {
    std::vector<Real> fractions(layer.regions.size() + 1, Real{0.0});
    if (!plan.exactIntersections) {
        const int samples = std::clamp(layer.fourierOptions.resolution, 2, 64);
        for (int sy = 0; sy < samples; ++sy) {
            const Real y = ((pixelY + (sy + Real{0.5}) / samples) / countY -
                Real{0.5}) * layer.periodYUm;
            for (int sx = 0; sx < samples; ++sx) {
                const Real x = ((pixelX + (sx + Real{0.5}) / samples) / countX -
                    Real{0.5}) * layer.periodXUm;
                fractions[patternedMaterialSlotAt(layer, x, y)] +=
                    Real{1.0} / (samples * samples);
            }
        }
        return fractions;
    }

    const Real minX = (static_cast<Real>(pixelX) / countX - Real{0.5}) *
        layer.periodXUm;
    const Real maxX = (static_cast<Real>(pixelX + 1) / countX - Real{0.5}) *
        layer.periodXUm;
    const Real minY = (static_cast<Real>(pixelY) / countY - Real{0.5}) *
        layer.periodYUm;
    const Real maxY = (static_cast<Real>(pixelY + 1) / countY - Real{0.5}) *
        layer.periodYUm;
    const Real inversePixelArea = Real{1.0} / ((maxX - minX) * (maxY - minY));
    fractions[0] = Real{1.0};
    // Independently follows S4 7fd00a2 Pattern_DiscretizeCell and
    // shape_get_intersection_area_quad: use shape/pixel intersection areas,
    // then apply the Kottke-Farjadpour-Johnson tau-tensor average (PRE 77,
    // 036611, 2008) only to genuinely mixed pixels.
    for (std::size_t regionIndex = 0; regionIndex < layer.regions.size(); ++regionIndex) {
        if (plan.parent[regionIndex] == -2) {
            continue;
        }
        Real area{};
        for (int shiftY = -1; shiftY <= 1; ++shiftY) {
            for (int shiftX = -1; shiftX <= 1; ++shiftX) {
                area += regionPixelIntersectionArea(
                    layer.regions[regionIndex],
                    layer.regions[regionIndex].shape == PatternRegionShape::Polygon
                        ? &plan.polygonTriangles[regionIndex]
                        : nullptr,
                    minX + shiftX * layer.periodXUm,
                    maxX + shiftX * layer.periodXUm,
                    minY + shiftY * layer.periodYUm,
                    maxY + shiftY * layer.periodYUm);
            }
        }
        const Real fraction = std::clamp(area * inversePixelArea, Real{0.0}, Real{1.0});
        fractions[regionIndex + 1] += fraction;
        const int parent = plan.parent[regionIndex];
        fractions[parent < 0 ? 0 : static_cast<std::size_t>(parent + 1)] -= fraction;
    }
    for (Real& fraction : fractions) {
        if (fraction < Real{0.0} && fraction > Real{-1e-10}) {
            fraction = Real{0.0};
        }
    }
    return fractions;
}

TensorFourierMatrices kottkePatternedTensors(const PatternedLayer2D& layer,
                                             const HarmonicBasis& basis,
                                             Real wavelengthUm) {
    const FormulationGrid grid = formulationGrid(layer, basis);
    const std::size_t total = checkedGridSize(
        grid.x, grid.y, "patterned formulation grid");
    const auto [epsBySlot, muBySlot] = patternedSlotTensors(layer, wavelengthUm);
    std::vector<Tensor3> epsSamples(total);
    std::vector<Tensor3> muSamples(total);
    const RegionCoveragePlan coveragePlan = makeRegionCoveragePlan(layer);
    for (int y = 0; y < grid.y; ++y) {
        for (int x = 0; x < grid.x; ++x) {
            const std::vector<Real> fractions = pixelMaterialFractions(
                layer, coveragePlan, x, y, grid.x, grid.y);
            std::vector<std::size_t> occupied;
            for (std::size_t slot = 0; slot < fractions.size(); ++slot) {
                if (fractions[slot] > Real{2.0} * std::numeric_limits<Real>::epsilon()) {
                    occupied.push_back(slot);
                }
            }
            const std::size_t index = wrappedGridIndex(x, y, grid.x, grid.y);
            if (occupied.size() == 1) {
                const std::size_t slot = occupied.front();
                epsSamples[index] = epsBySlot[slot];
                muSamples[index] = muBySlot[slot];
                continue;
            }

            if (occupied.size() == 2) {
                const std::size_t first = occupied[0];
                const std::size_t second = occupied[1];
                const Real fraction = fractions[first] / (fractions[first] + fractions[second]);
                const std::size_t boundarySlot = std::max(first, second);
                const Real px = ((x + Real{0.5}) / grid.x - Real{0.5}) *
                    layer.periodXUm;
                const Real py = ((y + Real{0.5}) / grid.y - Real{0.5}) *
                    layer.periodYUm;
                const Vec2 normal = boundarySlot == 0
                    ? Vec2{1.0, 0.0}
                    : boundaryNormal(layer.regions[boundarySlot - 1], px, py);
                epsSamples[index] = kottkeAverage(
                    epsBySlot[first],
                    epsBySlot[second],
                    fraction,
                    normal.x,
                    normal.y);
                muSamples[index] = kottkeAverage(
                    muBySlot[first],
                    muBySlot[second],
                    fraction,
                    normal.x,
                    normal.y);
                continue;
            }

            Tensor3 eps{};
            Tensor3 mu{};
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t col = 0; col < 3; ++col) {
                    eps(row, col) = {};
                    mu(row, col) = {};
                }
            }
            for (const std::size_t slot : occupied) {
                const Real fraction = fractions[slot];
                for (std::size_t row = 0; row < 3; ++row) {
                    for (std::size_t col = 0; col < 3; ++col) {
                        eps(row, col) += fraction * epsBySlot[slot](row, col);
                        mu(row, col) += fraction * muBySlot[slot](row, col);
                    }
                }
            }
            epsSamples[index] = eps;
            muSamples[index] = mu;
        }
    }
    TensorFourierMatrices out = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        grid.x,
        grid.y,
        basis);
    // Kottke averaging has already embedded the interface-normal inverse rule
    // into each mixed-cell tensor.  Use those effective tensor convolutions
    // directly; applying a second directional Li transform would double-count
    // the subpixel correction.
    out.liEpsQ = out.eps;
    out.liMuQ = out.mu;
    out.liEpsQZzInverse = inverse(out.liEpsQ[2][2]);
    out.liMuQZzInverse = inverse(out.liMuQ[2][2]);
    out.metadata.sequentialLiFactorization2d = false;
    out.metadata.factorizationUsesSampledGeometry = true;
    out.metadata.requiresJointConvergence = true;
    out.metadata.formulation = FourierFormulation::Kottke;
    out.metadata.lanczosSmoothingApplied = false;
    out.metadata.formulationResolution = layer.fourierOptions.resolution;
    return out;
}

} // namespace

TensorFourierMatrices tensorFourierMatricesPatternedLayer2d(
    const PatternedLayer2D& layer,
    const HarmonicBasis& basis,
    Real wavelengthUm) {
    validateFourierConvergenceOptions(layer.fourierOptions);
    const auto annotateApplicability = [&](TensorFourierMatrices& tensors) {
        tensors.metadata.tensorFactorizationApplicability =
            patternedTensorFactorizationApplicability(layer, wavelengthUm);
        if (tensors.metadata.tensorFactorizationApplicability !=
                TensorFactorizationApplicability::NotApplicable &&
            tensors.metadata.factorizationUsesSampledGeometry) {
            tensors.metadata.requiresJointConvergence = true;
        }
    };
    const FourierFormulation formulation =
        effectiveFourierFormulation(layer.fourierOptions);
    if (formulation == FourierFormulation::Kottke && !layer.regions.empty()) {
        TensorFourierMatrices out = kottkePatternedTensors(
            layer, basis, wavelengthUm);
        annotateApplicability(out);
        return out;
    }

    TensorFourierMatrices out =
        tensorFourierMatricesPatternedLayer2dDefault(layer, basis, wavelengthUm);
    annotateApplicability(out);
    out.metadata.formulation = formulation;
    out.metadata.formulationResolution = layer.fourierOptions.resolution;
    if (!layer.fourierOptions.lanczosSmoothing &&
        !layer.fourierOptions.polarizationDecomposition) {
        return out;
    }

    PatternedLayer2D reciprocalPattern;
    if (!makeReciprocalScalarPattern(layer, wavelengthUm, &reciprocalPattern)) {
        if (layer.fourierOptions.polarizationDecomposition) {
            throw std::invalid_argument(
                "PolBasisVL/NV/Jones require non-singular scalar epsilon and mu; "
                "tensor patterned media are not supported by this S4-compatible route");
        }
        smoothBlocks(out.eps, basis, layer);
        smoothBlocks(out.mu, basis, layer);
        smoothBlocks(out.liEpsQ, basis, layer);
        smoothBlocks(out.liMuQ, basis, layer);
        out.liEpsQZzInverse = inverse(out.liEpsQ[2][2]);
        out.liMuQZzInverse = inverse(out.liMuQ[2][2]);
        out.metadata.lanczosSmoothingApplied = true;
        return out;
    }

    TensorFourierMatrices reciprocal =
        tensorFourierMatricesPatternedLayer2dDefault(
            reciprocalPattern, basis, wavelengthUm);
    smoothBlocks(out.eps, basis, layer);
    smoothBlocks(out.mu, basis, layer);
    smoothBlocks(reciprocal.eps, basis, layer);
    smoothBlocks(reciprocal.mu, basis, layer);
    const Matrix& epsDirect = out.eps[0][0];
    const Matrix& epsReciprocal = reciprocal.eps[0][0];
    const Matrix& muDirect = out.mu[0][0];
    const Matrix& muReciprocal = reciprocal.mu[0][0];

    if (layer.fourierOptions.polarizationDecomposition) {
        const auto [epsDependsX, epsDependsY] =
            scalarConvolutionDependencies(epsDirect, basis);
        const auto [muDependsX, muDependsY] =
            scalarConvolutionDependencies(muDirect, basis);
        const bool dependsX = epsDependsX || muDependsX;
        const bool dependsY = epsDependsY || muDependsY;
        const TangentField field = dependsX && !dependsY
            ? constantTangentField(layer, basis, Real{0.0}, Real{1.0})
            : (dependsY && !dependsX
                ? constantTangentField(layer, basis, Real{1.0}, Real{0.0})
                : generateTangentField(layer, basis));
        setScalarPolarizationBlocks(
            out.liEpsQ,
            out.liEpsQZzInverse,
            epsDirect,
            epsReciprocal,
            field,
            basis,
            layer.fourierOptions.polarizationBasis);
        setScalarPolarizationBlocks(
            out.liMuQ,
            out.liMuQZzInverse,
            muDirect,
            muReciprocal,
            field,
            basis,
            layer.fourierOptions.polarizationBasis);
        out.metadata.sequentialLiFactorization2d = false;
    } else {
        const auto [epsDependsX, epsDependsY] =
            scalarConvolutionDependencies(epsDirect, basis);
        const auto [muDependsX, muDependsY] =
            scalarConvolutionDependencies(muDirect, basis);
        fillScalarFactorization(
            out.liEpsQ,
            out.liEpsQZzInverse,
            epsDirect,
            epsReciprocal,
            epsDependsX && !epsDependsY,
            epsDependsY && !epsDependsX);
        fillScalarFactorization(
            out.liMuQ,
            out.liMuQZzInverse,
            muDirect,
            muReciprocal,
            muDependsX && !muDependsY,
            muDependsY && !muDependsX);
        out.metadata.sequentialLiFactorization2d = false;
    }
    out.metadata.lanczosSmoothingApplied = layer.fourierOptions.lanczosSmoothing;
    return out;
}

PrecomputedPeriodicLayer2D precomputePatternedLayer2d(const PatternedLayer2D& layer,
                                                         const HarmonicBasis& basis,
                                                         Real wavelengthUm) {
    PrecomputedPeriodicLayer2D out;
    out.tensors = tensorFourierMatricesPatternedLayer2d(layer, basis, wavelengthUm);
    out.sourceMaterials.reserve(layer.regions.size() + 1);
    out.sourceMaterials.push_back(layer.background);
    for (const PatternRegion& region : layer.regions) {
        out.sourceMaterials.push_back(region.material);
    }
    out.periodXUm = layer.periodXUm;
    out.periodYUm = layer.periodYUm;
    out.thicknessUm = layer.thicknessUm;
    return out;
}

} // namespace rcwa
