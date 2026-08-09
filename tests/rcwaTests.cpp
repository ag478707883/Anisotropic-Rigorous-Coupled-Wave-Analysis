#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rcwa/layer.hpp"
#include "rcwa/physics.hpp"
#include "rcwa/solver.hpp"

namespace {

TEST_CASE("indexed scheduler caps concurrency and executes every job once") {
    using namespace rcwa;

    const int hardware = hardwareWorkerCount();
    const std::size_t jobs = static_cast<std::size_t>(hardware) + 7;
    REQUIRE(boundedWorkerCount(0, jobs) == hardware);
    REQUIRE(boundedWorkerCount(hardware + 7, jobs) == hardware);
    REQUIRE(boundedWorkerCount(-1, jobs) == 1);
    REQUIRE(boundedWorkerCount(hardware, 1) == 1);

    std::vector<std::atomic<int>> visits(jobs);
    for (auto& visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }
    runIndexedJobs(jobs, 0, [&](std::size_t index) {
        visits[index].fetch_add(1, std::memory_order_relaxed);
    });
    for (const auto& visit : visits) {
        REQUIRE(visit.load(std::memory_order_relaxed) == 1);
    }
}

TEST_CASE("canonical isotropic wave formulas use the outgoing branch") {
    using namespace rcwa;
    const Real tolerance = sizeof(Real) == sizeof(float) ? Real{2e-5f} : Real{1e-12};

    REQUIRE(outgoingSqrt(Complex{4.0, 0.0}) == Complex{2.0, 0.0});
    REQUIRE(std::abs(outgoingSqrt(Complex{4.0, -1e-15}) -
                     Complex{2.0, 0.0}) < tolerance);
    REQUIRE(std::imag(outgoingSqrt(Complex{-4.0, 0.0})) ==
            Catch::Approx(2.0).margin(tolerance));

    const auto properties = isotropicWaveProperties(
        {2.25, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0});
    REQUIRE(std::abs(properties.q - Complex{1.5, 0.0}) < tolerance);
    REQUIRE(std::abs(properties.teAdmittance - Complex{1.5, 0.0}) < tolerance);
    REQUIRE(std::abs(properties.tmAdmittance - Complex{1.5, 0.0}) < tolerance);

    const auto te = isotropicInterface(
        {1.0, 0.0}, {1.0, 0.0}, {2.25, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, Polarization::TE);
    REQUIRE(te.reflectance == Catch::Approx(0.04));
    REQUIRE(te.transmittance == Catch::Approx(0.96));

    const auto tm = isotropicInterface(
        {1.0, 0.0}, {1.0, 0.0}, {2.25, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, Polarization::TM);
    REQUIRE(tm.reflectance == Catch::Approx(te.reflectance));
    REQUIRE(tm.transmittance == Catch::Approx(te.transmittance));

    const auto film = isotropicFilmStack(
        {1.0, 0.0}, {1.0, 0.0}, {2.3104, 0.0}, {1.0, 0.0},
        {{{4.41, 0.0}, {1.0, 0.0}, 0.11}},
        {0.0, 0.0}, {0.0, 0.0}, 0.633, Polarization::TE);
    const Complex phase = std::exp(
        Complex{0.0, 4.0 * pi * 2.1 * 0.11 / 0.633});
    const Complex r01 = (Complex{1.0, 0.0} - Complex{2.1, 0.0}) /
                        (Complex{1.0, 0.0} + Complex{2.1, 0.0});
    const Complex r12 = (Complex{2.1, 0.0} - Complex{1.52, 0.0}) /
                        (Complex{2.1, 0.0} + Complex{1.52, 0.0});
    const Complex expectedReflection =
        (r01 + r12 * phase) / (Complex{1.0, 0.0} + r01 * r12 * phase);
    REQUIRE(film.reflectance == Catch::Approx(std::norm(expectedReflection)).margin(tolerance));

    const auto lossyFilm = isotropicFilmStack(
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
        {{{4.0, 0.6}, {1.0, 0.0}, 0.25}},
        {0.0, 0.0}, {0.0, 0.0}, 1.0, Polarization::TE);
    REQUIRE(lossyFilm.reflectance >= Real{0});
    REQUIRE(lossyFilm.transmittance >= Real{0});
    REQUIRE(lossyFilm.reflectance + lossyFilm.transmittance <=
            Real{1} + tolerance);

    const auto opaqueFilm = isotropicFilmStack(
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
        {{{-100.0, 20.0}, {1.0, 0.0}, 100.0}},
        {0.0, 0.0}, {0.0, 0.0}, 1.0, Polarization::TE);
    REQUIRE(std::isfinite(opaqueFilm.reflectance));
    REQUIRE(std::isfinite(opaqueFilm.transmittance));
    REQUIRE(opaqueFilm.transmittance < tolerance);
}

double airyReflectance(double n0,
                        double n1,
                        double n2,
                        double thicknessUm,
                        double wavelengthUm) {
    using C = std::complex<double>;
    const C r01 = (n0 - n1) / (n0 + n1);
    const C r12 = (n1 - n2) / (n1 + n2);
    const C phase = std::exp(C{0.0, 4.0 * rcwa::pi * n1 * thicknessUm / wavelengthUm});
    const C r = (r01 + r12 * phase) / (C{1.0, 0.0} + r01 * r12 * phase);
    return std::norm(r);
}

double airyReflectanceOblique(double n0,
                                double n1,
                                double n2,
                                double thicknessUm,
                                double wavelengthUm,
                                double thetaDeg,
                                rcwa::Polarization pol) {
    using C = std::complex<double>;
    const double theta = thetaDeg * rcwa::pi / 180.0;
    const C sin0 = std::sin(theta);
    const auto cosIn = [&](double n) {
        const C ratio = C{n0 / n, 0.0} * sin0;
        return std::sqrt(C{1.0, 0.0} - ratio * ratio);
    };
    const C cos0 = std::cos(theta);
    const C cos1 = cosIn(n1);
    const C cos2 = cosIn(n2);
    const auto admittance = [&](double n, C c) {
        return pol == rcwa::Polarization::TE ? C{n, 0.0} * c : C{n, 0.0} / c;
    };
    const C y0 = admittance(n0, cos0);
    const C y1 = admittance(n1, cos1);
    const C y2 = admittance(n2, cos2);
    const C r01 = (y0 - y1) / (y0 + y1);
    const C r12 = (y1 - y2) / (y1 + y2);
    const C phase = std::exp(C{0.0, 4.0 * rcwa::pi * n1 *
                                      thicknessUm / wavelengthUm} * cos1);
    const C r = (r01 + r12 * phase) / (C{1.0, 0.0} + r01 * r12 * phase);
    return std::norm(r);
}

double fresnelReflectance(double n0,
                           double n1,
                           double thetaDeg,
                           rcwa::Polarization pol) {
    return airyReflectanceOblique(n0, n1, n1, 0.0, 1.0, thetaDeg, pol);
}

template <typename Fn>
bool throwsContaining(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

void requireMatrixClose(const rcwa::Matrix& a,
                          const rcwa::Matrix& b,
                          double margin) {
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a.data()[i] - b.data()[i]) <= margin);
    }
}

void requireTensorClose(const rcwa::Tensor3& actual,
                        const rcwa::Tensor3& expected,
                        double margin) {
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            REQUIRE(std::abs(actual(row, column) - expected(row, column)) <= margin);
        }
    }
}

void requireScalarIdentityMatrix(const rcwa::Matrix& m,
                                    rcwa::Complex value,
                                    double margin) {
    REQUIRE(m.rows() == m.cols());
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            const rcwa::Complex expected = r == c ? value : rcwa::Complex{};
            REQUIRE(std::abs(m(r, c) - expected) <= margin);
        }
    }
}

void requireComplexMultisetClose(std::vector<rcwa::Complex> a,
                                    std::vector<rcwa::Complex> b,
                                    double margin) {
    REQUIRE(a.size() == b.size());
    auto keyLess = [](const rcwa::Complex& lhs, const rcwa::Complex& rhs) {
        if (std::real(lhs) != std::real(rhs)) {
            return std::real(lhs) < std::real(rhs);
        }
        return std::imag(lhs) < std::imag(rhs);
    };
    std::sort(a.begin(), a.end(), keyLess);
    std::sort(b.begin(), b.end(), keyLess);
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a[i] - b[i]) <= margin);
    }
}

std::pair<double, double> circularZeroOrderTotals(const rcwa::SParameterResult& s,
                                                      bool leftHanded) {
    using C = std::complex<double>;
    REQUIRE(s.N == 1);
    const C te = C{1.0 / std::sqrt(2.0), 0.0};
    const C tm = (leftHanded ? C{0.0, 1.0} : C{0.0, -1.0}) * te;
    const C rTe = s.S11(0, 0) * te + s.S11(0, 1) * tm;
    const C rTm = s.S11(1, 0) * te + s.S11(1, 1) * tm;
    const C tTe = s.S21(0, 0) * te + s.S21(0, 1) * tm;
    const C tTm = s.S21(1, 0) * te + s.S21(1, 1) * tm;
    return {
        std::norm(rTe) + std::norm(rTm),
        std::norm(tTe) + std::norm(tTm),
    };
}

struct DirectBerremanLayer {
    rcwa::Tensor3 epsilon;
    rcwa::Tensor3 mu;
    double thicknessUm{};
};

struct DirectBerremanPower {
    double reflectance{};
    double transmittance{};
};

rcwa::Matrix isotropicPortState(double epsilon,
                                rcwa::Complex kx,
                                bool forward,
                                rcwa::Polarization polarization) {
    using namespace rcwa;
    Complex gamma = std::sqrt(Complex{epsilon, 0.0} - kx * kx);
    if (std::real(gamma) < 0.0 ||
        (std::real(gamma) == 0.0 && std::imag(gamma) < 0.0)) {
        gamma = -gamma;
    }
    if (!forward) {
        gamma = -gamma;
    }

    Matrix state(4, 1);
    if (polarization == Polarization::TE) {
        state(1, 0) = {1.0, 0.0};
        state(2, 0) = -gamma;
    } else if (polarization == Polarization::TM) {
        state(0, 0) = {1.0, 0.0};
        state(3, 0) = epsilon / gamma;
    } else {
        throw std::invalid_argument("direct Berreman port state requires TE or TM");
    }
    return state;
}

double tangentialStateFlux(const rcwa::Matrix& state) {
    if (state.rows() != 4 || state.cols() != 1) {
        throw std::invalid_argument("Berreman flux state must be 4x1");
    }
    return std::real(
        state(0, 0) * std::conj(state(3, 0)) -
        state(1, 0) * std::conj(state(2, 0)));
}

rcwa::Matrix scaledState(const rcwa::Matrix& state, rcwa::Complex amplitude) {
    return amplitude * state;
}

// Test-only global transfer reference. Production multilayers use the stable
// local-interface S-matrix cascade. This deliberately different algebraic path
// propagates psi=[Ex,Ey,Hx,Hy]^T through exp(i*k0*Delta*d), then solves the four
// exterior boundary unknowns at once. It is suitable for these bounded test
// stacks, but must not replace the production cascade for optically thick media.
DirectBerremanPower directBerremanStackPower(
    const std::vector<DirectBerremanLayer>& layers,
    double wavelengthUm,
    double thetaDeg,
    double entryIndex,
    double substrateIndex,
    rcwa::Polarization incidentPolarization) {
    using namespace rcwa;
    const double theta = thetaDeg * pi / 180.0;
    const Complex kx{entryIndex * std::sin(theta), 0.0};
    const Complex ky{};
    const Complex phaseScale{0.0, 2.0 * pi / wavelengthUm};

    Matrix transfer = Matrix::identity(4);
    for (const auto& layer : layers) {
        const Eigensystem modes = eig(
            berremanStateMatrix(layer.epsilon, layer.mu, kx, ky));
        std::vector<Complex> phase(4);
        for (std::size_t mode = 0; mode < phase.size(); ++mode) {
            phase[mode] = std::exp(
                phaseScale * layer.thicknessUm * modes.values[mode]);
        }
        const Matrix propagation =
            rightScaleColumns(modes.vectors, phase) * inverse(modes.vectors);
        transfer = propagation * transfer;
    }

    const Matrix incident = isotropicPortState(
        entryIndex * entryIndex, kx, true, incidentPolarization);
    const Matrix reflectedTe = isotropicPortState(
        entryIndex * entryIndex, kx, false, Polarization::TE);
    const Matrix reflectedTm = isotropicPortState(
        entryIndex * entryIndex, kx, false, Polarization::TM);
    const Matrix transmittedTe = isotropicPortState(
        substrateIndex * substrateIndex, kx, true, Polarization::TE);
    const Matrix transmittedTm = isotropicPortState(
        substrateIndex * substrateIndex, kx, true, Polarization::TM);

    const Matrix propagatedIncident = transfer * incident;
    const Matrix propagatedReflectedTe = transfer * reflectedTe;
    const Matrix propagatedReflectedTm = transfer * reflectedTm;
    Matrix boundary(4, 4);
    Matrix rhs(4, 1);
    for (std::size_t row = 0; row < 4; ++row) {
        boundary(row, 0) = propagatedReflectedTe(row, 0);
        boundary(row, 1) = propagatedReflectedTm(row, 0);
        boundary(row, 2) = -transmittedTe(row, 0);
        boundary(row, 3) = -transmittedTm(row, 0);
        rhs(row, 0) = -propagatedIncident(row, 0);
    }
    const Matrix amplitudes = solveLinear(boundary, rhs);

    const Matrix reflected =
        scaledState(reflectedTe, amplitudes(0, 0)) +
        scaledState(reflectedTm, amplitudes(1, 0));
    const Matrix transmitted =
        scaledState(transmittedTe, amplitudes(2, 0)) +
        scaledState(transmittedTm, amplitudes(3, 0));
    const double incidentFlux = tangentialStateFlux(incident);
    return {
        -tangentialStateFlux(reflected) / incidentFlux,
        tangentialStateFlux(transmitted) / incidentFlux,
    };
}

} // namespace

TEST_CASE("diagonal operators use structured storage and explicit materialization") {
    using namespace rcwa;

    const std::vector<Complex> values{
        {1.25, -0.5},
        {-2.0, 0.25},
        {0.75, 1.5},
    };
    const DiagonalOperator diagonal(values);

    REQUIRE(diagonal.size() == values.size());
    REQUIRE(diagonal.values().size() == values.size());
    requireMatrixClose(diagonal.toDense(), Matrix::diagonal(values), 0.0);

    Matrix matrix(3, 2);
    matrix(0, 0) = {1.0, 2.0};
    matrix(0, 1) = {-0.5, 0.25};
    matrix(1, 0) = {3.0, -1.0};
    matrix(1, 1) = {0.75, 0.5};
    matrix(2, 0) = {-2.0, 0.0};
    matrix(2, 1) = {1.5, -0.75};

    requireMatrixClose(
        multiply(diagonal, matrix),
        diagonal.toDense() * matrix,
        1e-14);

    Matrix transposeShape(2, 3);
    transposeShape(0, 0) = {1.0, 2.0};
    transposeShape(0, 1) = {-0.5, 0.25};
    transposeShape(0, 2) = {0.75, 0.5};
    transposeShape(1, 0) = {3.0, -1.0};
    transposeShape(1, 1) = {-2.0, 0.0};
    transposeShape(1, 2) = {1.5, -0.75};
    requireMatrixClose(
        multiply(transposeShape, diagonal),
        transposeShape * diagonal.toDense(),
        1e-14);

}

TEST_CASE("prepared stacks retain transverse wavevectors in linear storage") {
    using namespace rcwa;

    static_assert(std::is_same_v<decltype(PreparedStack{}.Kx), DiagonalOperator>);
    static_assert(std::is_same_v<decltype(PreparedStack{}.Ky), DiagonalOperator>);

    const HarmonicBasis basis = makeHarmonicBasisOrders(
        2,
        1,
        LatticeTruncation::Parallelogramic);
    const auto [kx, ky] = makeTransverseWavevectorOperators(
        0.13, -0.08, 0.7, 0.9, basis);
    REQUIRE(kx.size() == basis.size());
    REQUIRE(ky.size() == basis.size());
    REQUIRE(kx.values().capacity() >= kx.size());
    REQUIRE(ky.values().capacity() >= ky.size());
}

TEST_CASE("third-party numerical backends are active") {
    using namespace rcwa;

    REQUIRE(fourierUsesFftw());
    REQUIRE(std::string(linearAlgebraBackendDetail()).find("LAPACKE") != std::string::npos);
    REQUIRE(std::string(matrixMultiplyBackendDetail()).find("CBLAS") != std::string::npos);

    Matrix A(2, 2);
    A(0, 0) = {2.0, 0.0};
    A(0, 1) = {1.0, 1.0};
    A(1, 0) = {0.0, -1.0};
    A(1, 1) = {3.0, 0.0};
    Matrix B(2, 1);
    B(0, 0) = {1.0, 0.0};
    B(1, 0) = {2.0, -1.0};
    const Matrix X = solveLinear(A, B);
    const Matrix residual = A * X - B;
    REQUIRE(maxAbs(residual) < 1e-11);

    Matrix D(2, 2);
    D(0, 0) = {1.0, 0.0};
    D(1, 1) = {2.0, 0.0};
    const auto eigResult = eig(D);
    REQUIRE(eigResult.values.size() == 2);
    REQUIRE(eigResult.vectors.rows() == 2);
    REQUIRE(eigResult.vectors.cols() == 2);

    // A real nonsymmetric matrix can have a complex-conjugate eigenpair.
    // This exercises reconstruction from LAPACK's real geev storage format.
    Matrix rotation(2, 2);
    rotation(0, 1) = {-1.0, 0.0};
    rotation(1, 0) = {1.0, 0.0};
    const auto rotationEig = eig(rotation);
    REQUIRE(rotationEig.values.size() == 2);
    for (std::size_t column = 0; column < 2; ++column) {
        Matrix vector(2, 1);
        vector(0, 0) = rotationEig.vectors(0, column);
        vector(1, 0) = rotationEig.vectors(1, column);
        const Matrix eigenResidual =
            rotation * vector - rotationEig.values[column] * vector;
        REQUIRE(maxAbs(eigenResidual) < 1e-12);
        REQUIRE(std::abs(std::real(rotationEig.values[column])) < 1e-12);
        REQUIRE(std::abs(std::abs(std::imag(rotationEig.values[column])) - 1.0) < 1e-12);
    }

    Matrix C(2, 3);
    C(0, 0) = {1.0, 0.0};
    C(0, 1) = {2.0, -1.0};
    C(0, 2) = {0.5, 0.25};
    C(1, 0) = {-1.0, 0.5};
    C(1, 1) = {0.0, 2.0};
    C(1, 2) = {3.0, -0.5};
    Matrix Dm(3, 2);
    Dm(0, 0) = {0.25, 0.5};
    Dm(0, 1) = {2.0, 0.0};
    Dm(1, 0) = {-1.0, 0.0};
    Dm(1, 1) = {0.0, 1.0};
    Dm(2, 0) = {0.5, -0.25};
    Dm(2, 1) = {-0.5, 0.5};
    const Matrix CD = C * Dm;
    REQUIRE(CD.rows() == 2);
    REQUIRE(CD.cols() == 2);
    REQUIRE(std::abs(CD(0, 0) - Complex{-1.4375, 1.5}) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(CD(0, 1) - Complex{2.625, 2.125}) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(CD(1, 0) - Complex{0.875, -3.375}) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(CD(1, 1) - Complex{-5.25, 2.75}) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("built-in material models are available from the core library") {
    using namespace rcwa;

    const Material air = builtinMaterialModel("incident", "Air");
    REQUIRE(air.name() == "incident");
    REQUIRE(std::abs(air.epsilon(1.0) - Complex{1.0, 0.0}) ==
            Catch::Approx(0.0).margin(1e-12));

    const Material silicon = builtinMaterialModel("Si", "silicon");
    REQUIRE(std::real(silicon.refractiveIndex(1.0)) == Catch::Approx(3.48).margin(1e-12));
    REQUIRE(std::imag(silicon.refractiveIndex(1.0)) == Catch::Approx(0.0).margin(1e-12));

    DrudeModelParameters drude;
    drude.plasmaFrequencyRadS = 1.37e16;
    drude.dampingRateRadS = 8.5e13;
    const Material metal = builtinMaterialModel("metal", "Drude", drude);
    REQUIRE(metal.isDispersive());
    REQUIRE(std::isfinite(std::real(metal.epsilon(1.55))));
    REQUIRE(std::isfinite(std::imag(metal.epsilon(1.55))));

    drude.dampingRateRadS = 0.0;
    const Complex losslessDrude =
        builtinMaterialModel("lossless metal", "Drude", drude).epsilon(1.55);
    REQUIRE(std::isfinite(std::real(losslessDrude)));
    REQUIRE(std::imag(losslessDrude) == Catch::Approx(0.0).margin(1e-15));

    drude.dampingRateRadS = -1.0;
    REQUIRE_THROWS_AS(
        builtinMaterialModel("active invalid metal", "Drude", drude),
        std::invalid_argument);

    PhononPolaritonModelParameters moo3;
    moo3.epsilonInf = {4.0, 5.2, 2.4};
    moo3.transverseWavenumberCmInv = {818.0, 545.0, 962.0};
    moo3.longitudinalWavenumberCmInv = {974.0, 851.0, 1010.0};
    moo3.dampingWavenumberCmInv = {4.0, 4.0, 2.0};
    const Material phononPolariton =
        builtinMaterialModel("alpha-MoO3", "PhononPolariton", moo3);
    REQUIRE(phononPolariton.isDispersive());
    const auto epsMoo3 = phononPolariton.epsilonTensor(16.0);
    REQUIRE(std::real(epsMoo3(0, 0)) > 0.0);
    REQUIRE(std::real(epsMoo3(1, 1)) < 0.0);
    REQUIRE(std::real(epsMoo3(2, 2)) > 0.0);
    REQUIRE(std::imag(epsMoo3(0, 0)) > 0.0);
    REQUIRE(std::imag(epsMoo3(1, 1)) > 0.0);
    REQUIRE(std::imag(epsMoo3(2, 2)) > 0.0);

    YigModelParameters yig;
    yig.diagonalEpsilon = Complex{4.0, 0.02};
    yig.offDiagonalGyration = Complex{0.1, 0.003};
    yig.magnetizationDirection = {0.0, 2.0, 0.0};
    const Material magnetizedYig = builtinMaterialModel("Ce:YIG +y", "YIG", yig);
    REQUIRE_FALSE(magnetizedYig.isDispersive());
    const auto epsYig = magnetizedYig.epsilonTensor(1.7);
    REQUIRE(std::abs(epsYig(0, 0) - yig.diagonalEpsilon) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYig(1, 1) - yig.diagonalEpsilon) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYig(2, 2) - yig.diagonalEpsilon) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYig(0, 2) + Complex{0.0, 1.0} *
                         yig.offDiagonalGyration) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYig(2, 0) - Complex{0.0, 1.0} *
                         yig.offDiagonalGyration) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYig(0, 1)) == Catch::Approx(0.0).margin(1e-12));

    yig.magnetizationDirection = {0.0, -1.0, 0.0};
    const auto epsYigReversed =
        builtinMaterialModel("Ce:YIG -y", "Ce:YIG", yig).epsilonTensor(1.7);
    REQUIRE(std::abs(epsYigReversed(0, 2) + epsYig(0, 2)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYigReversed(2, 0) + epsYig(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));

    yig.magnetizationDirection = {0.0, 0.0, 3.0};
    const auto epsYigAlongZ =
        builtinMaterialModel("Ce:YIG +z", "MagnetoOpticalYIG", yig)
            .epsilonTensor(1.7);
    REQUIRE(std::abs(epsYigAlongZ(0, 1) - Complex{0.0, 1.0} *
                         yig.offDiagonalGyration) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYigAlongZ(1, 0) + Complex{0.0, 1.0} *
                         yig.offDiagonalGyration) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsYigAlongZ(0, 2)) == Catch::Approx(0.0).margin(1e-12));

    yig.magnetizationDirection = {0.0, 0.0, 0.0};
    REQUIRE_THROWS_AS(
        builtinMaterialModel("invalid YIG direction", "YIG", yig),
        std::invalid_argument);

    InAsMagnetoModelParameters inas;
    inas.magneticFieldT = 1.5;
    const Material magnetizedInAs =
        builtinMaterialModel("InAs +y", "InAsMagneto", inas);
    const auto epsInAs = magnetizedInAs.epsilonTensor(15.7);
    REQUIRE(std::imag(epsInAs(0, 0)) > 0.0);
    REQUIRE(std::imag(epsInAs(1, 1)) > 0.0);
    REQUIRE(std::abs(epsInAs(0, 2) + epsInAs(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::imag(epsInAs(0, 2)) < 0.0);

    inas.magneticFieldT = -inas.magneticFieldT;
    const auto epsInAsReversed =
        builtinMaterialModel("InAs -y", "InAsMagneto", inas)
            .epsilonTensor(15.7);
    REQUIRE(std::abs(epsInAsReversed(0, 2) + epsInAs(0, 2)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsInAsReversed(2, 0) + epsInAs(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));

    InSbLorentzDrudeModelParameters insb;
    insb.magneticFieldT = 3.0;
    const Material magnetizedInsb = builtinMaterialModel("InSb", "InSbLorentzDrude", insb);
    REQUIRE(magnetizedInsb.isDispersive());
    const auto epsInsb = magnetizedInsb.epsilonTensor(5.153);
    REQUIRE(std::abs(epsInsb(0, 2)) > 0.0);
    REQUIRE(std::abs(epsInsb(0, 2) + epsInsb(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::imag(epsInsb(0, 2)) < 0.0);

    insb.magneticFieldT = -insb.magneticFieldT;
    const auto epsInsbReversed =
        builtinMaterialModel("InSb -y", "InSbLorentzDrude", insb)
            .epsilonTensor(5.153);
    REQUIRE(std::abs(epsInsbReversed(0, 2) + epsInsb(0, 2)) ==
            Catch::Approx(0.0).margin(1e-12));

    WsmModelParameters wsm;
    const Material wsmAlongY = builtinMaterialModel("WSM +y", "WSM", wsm);
    const auto epsWsmY = wsmAlongY.epsilonTensor(16.9);
    REQUIRE(std::abs(epsWsmY(0, 2)) > 0.0);
    REQUIRE(std::abs(epsWsmY(0, 2) + epsWsmY(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsWsmY(0, 1)) == Catch::Approx(0.0).margin(1e-12));

    wsm.nodeSeparationDirection = {0.0, 0.0, -2.0};
    const Material wsmAlongMinusZ =
        builtinMaterialModel("WSM -z", "WSM", wsm);
    const auto epsWsmMinusZ = wsmAlongMinusZ.epsilonTensor(16.9);
    REQUIRE(std::abs(epsWsmMinusZ(0, 1) - epsWsmY(0, 2)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsWsmMinusZ(1, 0) - epsWsmY(2, 0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsWsmMinusZ(0, 2)) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsWsmMinusZ(2, 0)) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsWsmMinusZ(0, 0) - epsWsmY(0, 0)) ==
            Catch::Approx(0.0).margin(1e-12));

    wsm.nodeSeparationDirection = {0.0, 0.0, 0.0};
    REQUIRE_THROWS_AS(
        builtinMaterialModel("invalid WSM direction", "WSM", wsm),
        std::invalid_argument);

    BlackPhosphorusModelParameters bp;
    bp.carrierDensityCm2 = 5.0e13;
    bp.scatteringRateEv = 0.010;
    bp.thicknessM = 1.0e-9;
    const Material blackPhosphorus =
        builtinMaterialModel("BP", "BlackPhosphorusDrude", bp);
    REQUIRE(blackPhosphorus.isDispersive());
    const auto epsBp = blackPhosphorus.epsilonTensor(10.0);
    REQUIRE(std::isfinite(std::real(epsBp(0, 0))));
    REQUIRE(std::isfinite(std::imag(epsBp(0, 0))));
    REQUIRE(std::isfinite(std::real(epsBp(1, 1))));
    REQUIRE(std::isfinite(std::imag(epsBp(1, 1))));
    REQUIRE(std::abs(epsBp(0, 0) - epsBp(1, 1)) > 0.0);
    REQUIRE(std::abs(epsBp(2, 2) - Complex{1.0, 0.0}) ==
            Catch::Approx(0.0).margin(1e-12));

    bp.twistDeg = 45.0;
    const Material twistedBlackPhosphorus =
        builtinMaterialModel("BP twisted", "BP", bp);
    const auto epsTwistedBp = twistedBlackPhosphorus.epsilonTensor(10.0);
    REQUIRE(std::abs(epsTwistedBp(0, 1)) > 0.0);
    REQUIRE(std::abs(epsTwistedBp(0, 1) - epsTwistedBp(1, 0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsTwistedBp(0, 0) - epsTwistedBp(1, 1)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(epsTwistedBp(2, 2) - Complex{1.0, 0.0}) ==
            Catch::Approx(0.0).margin(1e-12));

    const Material bpAlias = builtinMaterialModel("BP alias", "BP");
    REQUIRE(bpAlias.isDispersive());
}

TEST_CASE("tabulated nk materials are parsed and interpolated in the core library") {
    using namespace rcwa;

    const Material tabulated = Material::tabulatedIndex(
        "coating",
        {1.0, 2.0},
        {{1.5, 0.1}, {2.5, 0.3}});
    const Complex expectedIndex{2.0, 0.2};
    REQUIRE(std::abs(tabulated.refractiveIndex(1.5) - expectedIndex) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(tabulated.epsilon(1.5) - expectedIndex * expectedIndex) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE_THROWS_AS(tabulated.epsilon(0.9), std::invalid_argument);

    const Material clamped = Material::tabulatedIndex(
        "clamped coating",
        {1.0, 2.0},
        {{1.5, 0.1}, {2.5, 0.3}},
        MaterialExtrapolation::Clamp);
    REQUIRE(std::abs(clamped.refractiveIndex(0.9) - Complex{1.5, 0.1}) ==
            Catch::Approx(0.0).margin(1e-12));

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "asyrcwa_nk_material_test.txt";
    struct RemoveFile {
        std::filesystem::path path;
        ~RemoveFile() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } removeFile{path};

    {
        std::ofstream output(path);
        REQUIRE(output.good());
        output << "1000 1.5 0.1\n2000 2.5 0.3\n";
    }
    const Material fromFile = Material::fromNkFile(
        "file coating", path, Real{1e-3});
    REQUIRE(std::abs(fromFile.refractiveIndex(1.5) - expectedIndex) ==
            Catch::Approx(0.0).margin(1e-12));

    {
        std::ofstream output(path);
        REQUIRE(output.good());
        output << "wavelength_um n k\n1.0 1.5 0.1\n";
    }
    REQUIRE_THROWS_AS(
        Material::fromNkFile("header is invalid", path),
        std::invalid_argument);
}

TEST_CASE("material models match independently evaluated SI reference values") {
    using namespace rcwa;

    const auto scalarTensor = [](Complex value) {
        return Tensor3::isotropic(value);
    };

    GrapheneModelParameters graphene;
    graphene.fermiEnergyEv = 0.5;
    graphene.temperatureK = 300.0;
    graphene.relaxationTimeS = 500.0e-15;
    graphene.thicknessM = 0.34e-9;
    requireTensorClose(
        builtinMaterialModel("Graphene", "GrapheneKubo", graphene).epsilonTensor(10.0),
        scalarTensor({-541.447318234931, 5.9412697573027113}),
        2e-11);
    graphene.fermiEnergyEv = 100.0;
    const Tensor3 electronDopedGraphene =
        builtinMaterialModel("electron-doped Graphene", "GrapheneKubo", graphene)
            .epsilonTensor(10.0);
    graphene.fermiEnergyEv = -100.0;
    const Tensor3 holeDopedGraphene =
        builtinMaterialModel("hole-doped Graphene", "GrapheneKubo", graphene)
            .epsilonTensor(10.0);
    requireTensorClose(electronDopedGraphene, holeDopedGraphene, 1e-11);

    DrudeModelParameters drude;
    drude.epsInf = 3.4;
    drude.plasmaFrequencyRadS = 1.39e16;
    drude.dampingRateRadS = 2.7e13;
    requireTensorClose(
        builtinMaterialModel("Ag", "Drude", drude).epsilonTensor(1.55),
        scalarTensor({-127.36078893273591, 2.9051758360243096}),
        2e-12);

    PhononPolaritonModelParameters moo3;
    moo3.epsilonInf = {4.0, 5.2, 2.4};
    moo3.transverseWavenumberCmInv = {818.0, 545.0, 962.0};
    moo3.longitudinalWavenumberCmInv = {974.0, 851.0, 1010.0};
    moo3.dampingWavenumberCmInv = {4.0, 4.0, 2.0};
    moo3.twistDeg = 23.0;
    Tensor3 expectedMoo3;
    expectedMoo3(0, 0) = {3.9644616747388057, 0.12724151371050402};
    expectedMoo3(0, 1) = {9.5420001048737895, -0.21485817246271546};
    expectedMoo3(1, 0) = expectedMoo3(0, 1);
    expectedMoo3(1, 1) = {-14.46474310623028, 0.54221376435611757};
    expectedMoo3(2, 2) = {2.8247664331745583, 0.00099278081270148962};
    requireTensorClose(
        builtinMaterialModel("alpha-MoO3", "PhononPolariton", moo3)
            .epsilonTensor(16.0),
        expectedMoo3,
        2e-12);

    const Tensor3 actualWsm =
        builtinMaterialModel("WSM", "WSM", WsmModelParameters{})
            .epsilonTensor(16.9);
    Tensor3 expectedWsm =
        Tensor3::isotropic({-38.586651404080605, 0.62345821674069313});
    expectedWsm(0, 2) = {0.0, 24.990922327697444};
    expectedWsm(2, 0) = {0.0, -24.990922327697444};
    requireTensorClose(actualWsm, expectedWsm, 2e-11);

    BlackPhosphorusModelParameters bp;
    bp.carrierDensityCm2 = 5.0e13;
    bp.scatteringRateEv = 0.010;
    bp.thicknessM = 1.0e-9;
    bp.bandgapEv = 2.0;
    bp.latticeConstantM = 0.223e-9;
    bp.twistDeg = 23.0;
    Tensor3 expectedBp;
    expectedBp(0, 0) = {-86.41979882438271, 7.0508822884430851};
    expectedBp(0, 1) = {-10.08722916914976, 0.81358990119738106};
    expectedBp(1, 0) = expectedBp(0, 1);
    expectedBp(1, 1) = {-66.937550869273878, 5.4795330186776701};
    expectedBp(2, 2) = {1.0, 0.0};
    requireTensorClose(
        builtinMaterialModel("BP", "BP", bp).epsilonTensor(10.0),
        expectedBp,
        2e-11);

    InAsMagnetoModelParameters inas;
    inas.magneticFieldT = 1.5;
    Tensor3 expectedInas = Tensor3::isotropic(
        {7.1207922273044399, 0.0068419706767821695});
    expectedInas(1, 1) = {7.1440991922454824, 0.006751370495680125};
    expectedInas(0, 2) = {-0.00090778755743952972, -0.34977607940070787};
    expectedInas(2, 0) = {0.00090778755743952972, 0.34977607940070787};
    requireTensorClose(
        builtinMaterialModel("InAs", "InAsMagneto", inas).epsilonTensor(15.7),
        expectedInas,
        2e-12);

    CdTeLorentzDrudeModelParameters cdte;
    cdte.magneticFieldT = 2.0;
    Tensor3 expectedCdte =
        Tensor3::isotropic({4.933367738722187, 0.014563946286959782});
    expectedCdte(1, 1) = {4.9343192203610098, 0.014544894438219902};
    expectedCdte(0, 2) = {-8.4241778250072997e-05, -0.0063313618048943749};
    expectedCdte(2, 0) = {8.4241778250072997e-05, 0.0063313618048943749};
    requireTensorClose(
        builtinMaterialModel("CdTe", "CdTeLorentzDrude", cdte).epsilonTensor(10.0),
        expectedCdte,
        2e-12);

    InSbLorentzDrudeModelParameters insb;
    insb.magneticFieldT = 3.0;
    Tensor3 expectedInsb =
        Tensor3::isotropic({-0.093781373005735705, 0.089679902329608124});
    expectedInsb(1, 1) = {0.075316063547777076, 0.086844166856798383};
    expectedInsb(0, 2) = {-0.0011619614077712163, -0.10371393221133007};
    expectedInsb(2, 0) = {0.0011619614077712163, 0.10371393221133007};
    requireTensorClose(
        builtinMaterialModel("InSb", "InSbLorentzDrude", insb).epsilonTensor(5.153),
        expectedInsb,
        2e-12);
}

TEST_CASE("pattern polygon helpers are available from the core library") {
    using namespace rcwa;

    const auto ellipse = regularEllipseVertices(0.2, 0.1, 8);
    REQUIRE(ellipse.size() == 8);
    REQUIRE(ellipse.front().x == Catch::Approx(0.2).margin(1e-12));
    REQUIRE(ellipse.front().y == Catch::Approx(0.0).margin(1e-12));

    const auto square = regularPolygonVertices(4, 1.0);
    REQUIRE(square.size() == 4);
    REQUIRE(square.front().x == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(square.front().y == Catch::Approx(0.0).margin(1e-12));

    const auto sharpRectangle = roundedRectangleVertices(0.5, 0.25, 0.0, 4);
    REQUIRE(sharpRectangle.size() == 4);
    const auto roundedRectangle = roundedRectangleVertices(0.5, 0.25, 0.1, 4);
    REQUIRE(roundedRectangle.size() == 16);

    const auto capsule = capsuleVertices(1.0, 0.2, 8);
    REQUIRE(capsule.size() == 16);

    REQUIRE_THROWS_AS(regularEllipseVertices(0.0, 0.1, 8), std::invalid_argument);
    REQUIRE_THROWS_AS(regularPolygonVertices(2, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(roundedRectangleVertices(0.5, 0.25, 0.3, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(capsuleVertices(0.3, 0.2, 8), std::invalid_argument);
}

TEST_CASE("matrix shape errors throw in release builds") {
    using namespace rcwa;

    Matrix A(2, 3);
    Matrix B(2, 2);
    REQUIRE_THROWS_AS(A + B, std::invalid_argument);
    REQUIRE_THROWS_AS(A * B, std::invalid_argument);
    REQUIRE_THROWS_AS(A.block(1, 1, 2, 1), std::out_of_range);
    REQUIRE_THROWS_AS(A.setBlock(1, 2, Matrix(2, 2)), std::out_of_range);
    REQUIRE_THROWS_AS(Matrix(2, 2, std::vector<Complex>(3)), std::invalid_argument);

    Matrix singularInput(2, 2);
    singularInput(0, 0) = {1.0, 0.0};
    singularInput(1, 1) = {std::numeric_limits<double>::quiet_NaN(), 0.0};
    Matrix rhs(2, 1);
    rhs(0, 0) = {1.0, 0.0};
    rhs(1, 0) = {1.0, 0.0};
    REQUIRE(throwsContaining(
        [&] { (void)solveLinear(singularInput, rhs); },
        "non-finite"));
}

TEST_CASE("matrix workspaces and in-place GEMM preserve dense arithmetic") {
    using namespace rcwa;

    Matrix A(3, 2);
    Matrix B(2, 3);
    Matrix base(3, 3, Complex{0.2, -0.1});
    for (std::size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = {0.15 * static_cast<Real>(i + 1), -0.03 * i};
    }
    for (std::size_t i = 0; i < B.size(); ++i) {
        B.data()[i] = {-0.08 * static_cast<Real>(i + 1), 0.02 * i};
    }
    Matrix accumulated = base;
    multiplyAccumulate(accumulated, A, B, {0.7, -0.2});
    requireMatrixClose(
        accumulated,
        base + Complex{0.7, -0.2} * (A * B),
        2e-13);

    Matrix workspace(1, 9, Complex{3.0, 0.0});
    const Complex* allocation = workspace.data().data();
    workspace.resizeForOverwrite(3, 3);
    REQUIRE(workspace.data().data() == allocation);
    workspace.reset(3, 3, {0.4, 0.1});
    REQUIRE(std::all_of(
        workspace.data().begin(),
        workspace.data().end(),
        [](Complex value) { return value == Complex{0.4, 0.1}; }));

    REQUIRE_THROWS_AS(
        multiplyAccumulate(workspace, B, A),
        std::invalid_argument);
}

TEST_CASE("conditioned linear solve rejects unstable Schur complements") {
    using namespace rcwa;

    Matrix wellConditioned = Matrix::identity(3);
    wellConditioned(0, 1) = {0.2, -0.1};
    wellConditioned(1, 2) = {-0.15, 0.05};
    Matrix rhs(3, 2);
    rhs(0, 0) = {1.0, 0.2};
    rhs(1, 0) = {-0.3, 0.1};
    rhs(2, 0) = {0.7, -0.4};
    rhs(0, 1) = {-0.2, 0.0};
    rhs(1, 1) = {0.5, 0.3};
    rhs(2, 1) = {0.1, -0.6};

    const auto checked = trySolveLinearWellConditioned(
        wellConditioned,
        rhs,
        1e-12);
    REQUIRE(checked.has_value());
    requireMatrixClose(*checked, solveLinear(wellConditioned, rhs), 1e-12);

    Matrix illConditioned = Matrix::identity(3);
    illConditioned(2, 2) = {1e-16, 0.0};
    REQUIRE_FALSE(trySolveLinearWellConditioned(
        illConditioned,
        rhs,
        1e-12).has_value());
}

TEST_CASE("sampled Fourier coefficients use FFTW convention") {
    using namespace rcwa;

    std::vector<Complex> samples;
    samples.reserve(16);
    for (int s = 0; s < 16; ++s) {
        const double x = static_cast<double>(s) / 16.0;
        samples.push_back(Complex{2.0, 0.0} +
                          std::exp(Complex{0.0, twoPi * x}) * Complex{0.25, -0.1});
    }
    const auto coeffs = fourierCoeffsFromSamples(samples, 3);
    REQUIRE(std::real(coeffs[1]) == Catch::Approx(2.0).margin(1e-12));
    REQUIRE(std::imag(coeffs[1]) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::real(coeffs[2]) == Catch::Approx(0.25).margin(1e-12));
    REQUIRE(std::imag(coeffs[2]) == Catch::Approx(-0.1).margin(1e-12));

    std::vector<Complex> samples2d;
    constexpr int sxCount = 8;
    constexpr int syCount = 10;
    const Complex harmonicAmplitude{0.15, -0.07};
    samples2d.reserve(static_cast<std::size_t>(sxCount * syCount));
    for (int iy = 0; iy < syCount; ++iy) {
        const double y = (static_cast<double>(iy) + 0.5) / syCount - 0.5;
        for (int ix = 0; ix < sxCount; ++ix) {
            const double x = (static_cast<double>(ix) + 0.5) / sxCount - 0.5;
            samples2d.push_back(
                Complex{2.0, 0.0} +
                harmonicAmplitude * std::exp(Complex{0.0, twoPi * (x + y)}));
        }
    }
    const auto coeffs2d = fourierCoeffsFromSamples2d(
        samples2d,
        sxCount,
        syCount,
        3,
        3);
    REQUIRE(std::abs(coeffs2d[4] - Complex{2.0, 0.0}) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(coeffs2d[8] - harmonicAmplitude) ==
            Catch::Approx(0.0).margin(1e-12));

    const auto circular100 = makeHarmonicBasisCount(
        100,
        LatticeTruncation::Circular);
    REQUIRE(circular100.size() == 97);
    const auto parallelogramic100 = makeHarmonicBasisCount(
        100,
        LatticeTruncation::Parallelogramic);
    REQUIRE(parallelogramic100.size() == 81);

    // S4-compatible circular truncation is performed in the physical
    // reciprocal-lattice metric.  A 2:1 real-space rectangular lattice has
    // reciprocal weights 1/4 and 1, so a requested count of 13 stops before
    // the fourfold shell m^2/4+n^2=2 and retains 11 complete-shell orders.
    const auto rectangularCircular13 = makeHarmonicBasisCount(
        13,
        LatticeTruncation::Circular,
        2.0,
        1.0);
    REQUIRE(rectangularCircular13.size() == 11);
    REQUIRE(std::find_if(
        rectangularCircular13.orders.begin(),
        rectangularCircular13.orders.end(),
        [](const HarmonicIndex& h) { return h.m == 1 && h.n == 1; }) !=
        rectangularCircular13.orders.end());
    REQUIRE(std::find_if(
        rectangularCircular13.orders.begin(),
        rectangularCircular13.orders.end(),
        [](const HarmonicIndex& h) { return h.m == 0 && h.n == 2; }) ==
        rectangularCircular13.orders.end());

    // Explicit circular orders are maximum index bounds.  The retained disk
    // is inscribed in those bounds using |m*b1+n*b2|, not an index-space
    // ellipse that ignores the lattice periods.
    const auto rectangularCircularOrders = makeHarmonicBasisOrders(
        2,
        2,
        LatticeTruncation::Circular,
        2.0,
        1.0);
    REQUIRE(rectangularCircularOrders.size() == 7);
    REQUIRE(std::find_if(
        rectangularCircularOrders.orders.begin(),
        rectangularCircularOrders.orders.end(),
        [](const HarmonicIndex& h) { return h.m == 2 && h.n == 0; }) !=
        rectangularCircularOrders.orders.end());
    REQUIRE(std::find_if(
        rectangularCircularOrders.orders.begin(),
        rectangularCircularOrders.orders.end(),
        [](const HarmonicIndex& h) { return h.m == 1 && h.n == 1; }) ==
        rectangularCircularOrders.orders.end());

    REQUIRE_THROWS_AS(
        fourierCoeffsBinaryGrating(
            std::numeric_limits<double>::quiet_NaN(),
            {2.0, 0.0},
            {1.0, 0.0},
            3),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        fourierCoeffsFromSamples(
            {{1.0, 0.0}, {std::numeric_limits<double>::infinity(), 0.0}},
            3),
        std::invalid_argument);
}

TEST_CASE("vacuum and Fresnel stacks conserve power") {
    using namespace rcwa;

    RcwaSolver vacuum;
    vacuum.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    vacuum.setSuperstrate(Material::vacuum());
    vacuum.setSubstrate(Material::vacuum());
    vacuum.setPolarization(Polarization::TE);
    const auto rv = vacuum.solve();
    REQUIRE(rv.rTotal == Catch::Approx(0.0).margin(1e-8));
    REQUIRE(rv.tTotal == Catch::Approx(1.0).margin(1e-8));

    RcwaSolver vacuum2d;
    vacuum2d.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    vacuum2d.setSuperstrate(Material::vacuum());
    vacuum2d.setSubstrate(Material::vacuum());
    vacuum2d.setPolarization(Polarization::TE);
    const auto r2d = vacuum2d.solve();
    REQUIRE(r2d.N == 9);
    REQUIRE(r2d.centerIdx == 4);
    REQUIRE(r2d.rTotal == Catch::Approx(0.0).margin(1e-8));
    REQUIRE(r2d.tTotal == Catch::Approx(1.0).margin(1e-8));

    RcwaSolver circularVacuum;
    circularVacuum.setHarmonicCount(9);
    circularVacuum.setSuperstrate(Material::vacuum());
    circularVacuum.setSubstrate(Material::vacuum());
    circularVacuum.setPolarization(Polarization::TE);
    const auto rcirc = circularVacuum.solve();
    REQUIRE(rcirc.N == 9);
    REQUIRE(rcirc.rTotal == Catch::Approx(0.0).margin(1e-8));
    REQUIRE(rcirc.tTotal == Catch::Approx(1.0).margin(1e-8));

    // The +/-1 exterior orders are exactly at Rayleigh cutoff here.  Their
    // forward/backward columns coalesce, but an identical vacuum interface is
    // still the transparent limiting problem and must not become singular.
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        RcwaSolver exactCutoff;
        exactCutoff.setWavelength(1.0);
        exactCutoff.setHarmonicOrders(1, 0, LatticeTruncation::Parallelogramic);
        exactCutoff.setSuperstrate(Material::vacuum());
        exactCutoff.setSubstrate(Material::vacuum());
        exactCutoff.setPolarization(pol);
        const auto cutoffResult = exactCutoff.solve();
        REQUIRE(cutoffResult.rTotal == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(cutoffResult.tTotal == Catch::Approx(1.0).margin(1e-12));
    }

    RcwaSolver cutoffField;
    cutoffField.setWavelength(1.0);
    cutoffField.setHarmonicOrders(1, 0, LatticeTruncation::Parallelogramic);
    cutoffField.setSuperstrate(Material::vacuum());
    cutoffField.setSubstrate(Material::vacuum());
    FieldPlaneRequest cutoffFieldRequest;
    cutoffFieldRequest.plane = FieldPlane::XZ;
    cutoffFieldRequest.uPoints = 1;
    cutoffFieldRequest.vMinUm = -0.1;
    cutoffFieldRequest.vMaxUm = -0.1;
    cutoffFieldRequest.vPoints = 1;
    cutoffFieldRequest.components = {FieldComponent::E};
    REQUIRE_THROWS_AS(
        cutoffField.solveFieldPlane(cutoffFieldRequest),
        std::invalid_argument);

    GratingLayer grazingGrating;
    grazingGrating.materialRidge = Material::constant("ridge", {2.25, 0.0});
    grazingGrating.materialGroove = Material::vacuum();
    grazingGrating.fillFactor = 0.5;
    grazingGrating.periodUm = 1.0;
    grazingGrating.periodYUm = 1.0;
    grazingGrating.thicknessUm = 0.1;

    RcwaSolver grazing;
    grazing.setWavelength(1.0);
    grazing.setHarmonicOrders(1, 0, LatticeTruncation::Parallelogramic);
    grazing.setSuperstrate(Material::vacuum());
    grazing.setSubstrate(Material::vacuum());
    grazing.setPolarization(Polarization::TE);
    grazing.addGratingLayer(grazingGrating);
    const auto rgrazing = grazing.solve();
    REQUIRE(rgrazing.rOrders.size() == 3);
    REQUIRE(rgrazing.tOrders.size() == 3);
    for (std::size_t i = 0; i < rgrazing.rOrders.size(); ++i) {
        REQUIRE(std::isfinite(rgrazing.rOrders[i]));
        REQUIRE(std::isfinite(rgrazing.tOrders[i]));
    }
    REQUIRE(rgrazing.conservation == Catch::Approx(1.0).margin(1e-8));

    RcwaSolver interfaceSolver;
    interfaceSolver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    interfaceSolver.setSuperstrate(Material::vacuum());
    interfaceSolver.setSubstrate(Material::fromIndex("n1.5", {1.5, 0.0}));
    interfaceSolver.setPolarization(Polarization::TE);
    const auto ri = interfaceSolver.solve();
    const double fresnelR = std::pow((1.0 - 1.5) / (1.0 + 1.5), 2.0);
    REQUIRE(ri.rTotal == Catch::Approx(fresnelR).margin(1e-8));
    REQUIRE(ri.conservation == Catch::Approx(1.0).margin(1e-8));

    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        RcwaSolver obliqueInterface;
        obliqueInterface.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
        obliqueInterface.setIncidence(37.0, 0.0);
        obliqueInterface.setSuperstrate(Material::vacuum());
        obliqueInterface.setSubstrate(Material::fromIndex("n1.7", {1.7, 0.0}));
        obliqueInterface.setPolarization(pol);
        const auto ro = obliqueInterface.solve();
        REQUIRE(ro.rTotal ==
                Catch::Approx(fresnelReflectance(1.0, 1.7, 37.0, pol)).margin(1e-8));
        REQUIRE(ro.conservation == Catch::Approx(1.0).margin(1e-8));
    }
}

TEST_CASE("uniform film agrees with Airy result") {
    using namespace rcwa;

    RcwaSolver solver;
    solver.setWavelength(0.633);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    solver.setPolarization(Polarization::TE);
    solver.addUniformLayer(Material::fromIndex("film", {2.1, 0.0}), 0.11);
    const auto r = solver.solve();
    REQUIRE(r.rTotal == Catch::Approx(airyReflectance(1.0, 2.1, 1.52, 0.11, 0.633)).margin(1e-8));
    REQUIRE(r.conservation == Catch::Approx(1.0).margin(1e-8));

    RcwaSolver circularSolver;
    circularSolver.setWavelength(0.633);
    circularSolver.setHarmonicCount(9);
    circularSolver.setSuperstrate(Material::vacuum());
    circularSolver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    circularSolver.setPolarization(Polarization::TE);
    circularSolver.addUniformLayer(Material::fromIndex("film", {2.1, 0.0}), 0.11);
    const auto circularR = circularSolver.solve();
    REQUIRE(circularR.rTotal ==
            Catch::Approx(airyReflectance(1.0, 2.1, 1.52, 0.11, 0.633)).margin(1e-8));
    REQUIRE(circularR.conservation == Catch::Approx(1.0).margin(1e-8));

    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        RcwaSolver oblique;
        oblique.setWavelength(0.633);
        oblique.setIncidence(31.0, 0.0);
        oblique.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
        oblique.setSuperstrate(Material::vacuum());
        oblique.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
        oblique.setPolarization(pol);
        oblique.addUniformLayer(Material::fromIndex("film", {2.1, 0.0}), 0.11);
        const auto ro = oblique.solve();
        REQUIRE(ro.rTotal == Catch::Approx(airyReflectanceOblique(
            1.0, 2.1, 1.52, 0.11, 0.633, 31.0, pol)).margin(1e-8));
        REQUIRE(ro.conservation == Catch::Approx(1.0).margin(1e-8));
    }
}

TEST_CASE("compiled absorptivity peak refinement returns an interior maximum") {
    using namespace rcwa;

    RcwaSolver solver;
    solver.setWavelength(1.0);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::fromIndex("lossy film", {2.0, 0.15}), 0.25);
    const SpectrumResult peak = solver.findAbsorptivityPeak(
        0.8, 1.2, Polarization::TE, 9, 3, 1);
    REQUIRE(peak.wavelengthUm > 0.8);
    REQUIRE(peak.wavelengthUm < 1.2);
    const Real peakAbsorptivity = Real{1.0} - peak.rTotal - peak.tTotal;

    RcwaSolver edge = solver;
    edge.setWavelength(0.8);
    const SpectrumResult left = edge.solveSpectrumOnly();
    edge.setWavelength(1.2);
    const SpectrumResult right = edge.solveSpectrumOnly();
    REQUIRE(peakAbsorptivity >= Real{1.0} - left.rTotal - left.tTotal);
    REQUIRE(peakAbsorptivity >= Real{1.0} - right.rTotal - right.tTotal);
}

TEST_CASE("RcwaSolver direct state preserves copy and move behavior") {
    using namespace rcwa;

    RcwaSolver solver;
    solver.setWavelength(0.633);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    solver.addUniformLayer(Material::fromIndex("film", {2.1, 0.0}), 0.11);

    const SpectrumResult original = solver.solve();
    RcwaSolver copied = solver;
    copied.setWavelength(0.7);
    const SpectrumResult copiedResult = copied.solve();
    REQUIRE(solver.solve().rTotal == Catch::Approx(original.rTotal).margin(1e-12));
    REQUIRE(copiedResult.wavelengthUm == Catch::Approx(0.7));

    RcwaSolver moved = std::move(copied);
    const SpectrumResult movedResult = moved.solve();
    REQUIRE(movedResult.rTotal == Catch::Approx(copiedResult.rTotal).margin(1e-12));
    REQUIRE(movedResult.tTotal == Catch::Approx(copiedResult.tTotal).margin(1e-12));
}

TEST_CASE("spectrum and field calculations are separate S4-style entry points") {
    using namespace rcwa;

    RcwaSolver solver;
    solver.setWavelength(1.0);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.setPolarization(Polarization::TE);
    solver.addUniformLayer(Material::vacuum(), 0.5);

    const auto legacy = solver.solve();
    const auto spectrum = solver.solveSpectrumOnly();
    REQUIRE(legacy.rTotal == Catch::Approx(spectrum.rTotal).margin(1e-12));
    REQUIRE(legacy.tTotal == Catch::Approx(spectrum.tTotal).margin(1e-12));
    REQUIRE(spectrum.conservation == Catch::Approx(1.0).margin(1e-12));

    const auto bothLight = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(bothLight.size() == 2);
    REQUIRE(bothLight[0].rTotal == Catch::Approx(spectrum.rTotal).margin(1e-12));
    REQUIRE(bothLight[0].tTotal == Catch::Approx(spectrum.tTotal).margin(1e-12));

    solver.setPolarization(Polarization::TM);
    const auto tm = solver.solveSpectrumOnly();
    REQUIRE(bothLight[1].rTotal == Catch::Approx(tm.rTotal).margin(1e-12));
    REQUIRE(bothLight[1].tTotal == Catch::Approx(tm.tTotal).margin(1e-12));
    solver.setPolarization(Polarization::TE);

    const auto sParameters = solver.solveSParameters();
    REQUIRE(sParameters.N == 1);
    REQUIRE(sParameters.centerIdx == 0);
    REQUIRE(sParameters.orders.size() == 1);
    REQUIRE(sParameters.S11.rows() == 2);
    REQUIRE(sParameters.S11.cols() == 2);
    REQUIRE(sParameters.S12.rows() == 2);
    REQUIRE(sParameters.S21.rows() == 2);
    REQUIRE(sParameters.S22.rows() == 2);
    REQUIRE(maxAbs(sParameters.S11) < 1e-12);
    REQUIRE(maxAbs(sParameters.S22) < 1e-12);
    REQUIRE(std::abs(sParameters.S12(0, 0)) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(std::abs(sParameters.S12(1, 1)) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(std::abs(sParameters.S21(0, 0)) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(std::abs(sParameters.S21(1, 1)) == Catch::Approx(1.0).margin(1e-12));
    const auto spectrumAfterSParameters = solver.solveSpectrumOnly();
    REQUIRE(spectrumAfterSParameters.rTotal ==
            Catch::Approx(spectrum.rTotal).margin(1e-12));
    REQUIRE(spectrumAfterSParameters.tTotal ==
            Catch::Approx(spectrum.tTotal).margin(1e-12));

    RcwaSolver fresnelSolver;
    fresnelSolver.setWavelength(1.0);
    fresnelSolver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    fresnelSolver.setSuperstrate(Material::vacuum());
    fresnelSolver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    fresnelSolver.setPolarization(Polarization::TE);
    const auto fresnelSpectrum = fresnelSolver.solveSpectrumOnly();
    const auto fresnelS = fresnelSolver.solveSParameters();
    REQUIRE(std::norm(fresnelS.S11(0, 0)) ==
            Catch::Approx(fresnelSpectrum.rTotal).margin(1e-12));

    const auto zeroOrder = solver.solveZeroOrderAmplitudes();
    REQUIRE(zeroOrder.N == sParameters.N);
    REQUIRE(zeroOrder.centerIdx == sParameters.centerIdx);
    REQUIRE(zeroOrder.R.rows() == 2);
    REQUIRE(zeroOrder.R.cols() == 2);
    REQUIRE(zeroOrder.T.rows() == 2);
    REQUIRE(zeroOrder.T.cols() == 2);
    REQUIRE(zeroOrder.R(0, 0) == sParameters.S11(0, 0));
    REQUIRE(zeroOrder.R(0, 1) == sParameters.S11(0, 1));
    REQUIRE(zeroOrder.R(1, 0) == sParameters.S11(1, 0));
    REQUIRE(zeroOrder.R(1, 1) == sParameters.S11(1, 1));
    REQUIRE(zeroOrder.T(0, 0) == sParameters.S21(0, 0));
    REQUIRE(zeroOrder.T(0, 1) == sParameters.S21(0, 1));
    REQUIRE(zeroOrder.T(1, 0) == sParameters.S21(1, 0));
    REQUIRE(zeroOrder.T(1, 1) == sParameters.S21(1, 1));

    const auto zeroOrderBatch = solver.solveZeroOrderAmplitudesBatchOnly(
        std::vector<Real>{0.9, 1.0},
        2);
    REQUIRE(zeroOrderBatch.size() == 2);
    RcwaSolver zeroOrderPoint = solver;
    zeroOrderPoint.setWavelength(0.9);
    const auto zeroOrderAt09 = zeroOrderPoint.solveZeroOrderAmplitudes();
    requireMatrixClose(zeroOrderBatch[0].R, zeroOrderAt09.R, 1e-12);
    requireMatrixClose(zeroOrderBatch[0].T, zeroOrderAt09.T, 1e-12);
    requireMatrixClose(zeroOrderBatch[1].R, zeroOrder.R, 1e-12);
    requireMatrixClose(zeroOrderBatch[1].T, zeroOrder.T, 1e-12);

    const auto batch = solver.solveSpectrumBatchOnly(
        std::vector<Real>{0.9, 1.0},
        {Polarization::TE, Polarization::TM},
        2);
    const auto batchTotals = solver.solveSpectrumBatchTotalsOnly(
        std::vector<Real>{0.9, 1.0},
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(batch.size() == 4);
    REQUIRE(batchTotals.size() == batch.size());
    for (std::size_t iw = 0; iw < 2; ++iw) {
        RcwaSolver pointSolver = solver;
        pointSolver.setWavelength(iw == 0 ? 0.9 : 1.0);
        const auto pointResults = pointSolver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
        REQUIRE(batch[2 * iw].rTotal == Catch::Approx(pointResults[0].rTotal).margin(1e-12));
        REQUIRE(batch[2 * iw].tTotal == Catch::Approx(pointResults[0].tTotal).margin(1e-12));
        REQUIRE(batch[2 * iw + 1].rTotal ==
                Catch::Approx(pointResults[1].rTotal).margin(1e-12));
        REQUIRE(batch[2 * iw + 1].tTotal ==
                Catch::Approx(pointResults[1].tTotal).margin(1e-12));
        REQUIRE(batchTotals[2 * iw].rTotal ==
                Catch::Approx(batch[2 * iw].rTotal).margin(1e-12));
        REQUIRE(batchTotals[2 * iw].tTotal ==
                Catch::Approx(batch[2 * iw].tTotal).margin(1e-12));
        REQUIRE(batchTotals[2 * iw + 1].rTotal ==
                Catch::Approx(batch[2 * iw + 1].rTotal).margin(1e-12));
        REQUIRE(batchTotals[2 * iw + 1].tTotal ==
                Catch::Approx(batch[2 * iw + 1].tTotal).margin(1e-12));
    }

    const auto angleBatch = solver.solveSpectrumBatchForAngles(
        std::vector<Real>{15.0, -15.0},
        0.0,
        std::vector<Real>{0.9, 1.0},
        {Polarization::TE, Polarization::TM},
        2);
    const auto angleTotals = solver.solveSpectrumBatchTotalsForAngles(
        std::vector<Real>{15.0, -15.0},
        0.0,
        std::vector<Real>{0.9, 1.0},
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(angleBatch.size() == 8);
    REQUIRE(angleTotals.size() == angleBatch.size());
    for (std::size_t angleIndex = 0; angleIndex < 2; ++angleIndex) {
        for (std::size_t iw = 0; iw < 2; ++iw) {
            RcwaSolver pointSolver = solver;
            pointSolver.setIncidence(angleIndex == 0 ? 15.0 : -15.0, 0.0);
            pointSolver.setWavelength(iw == 0 ? 0.9 : 1.0);
            const auto pointResults = pointSolver.solveSpectrumOnlyForPolarizations(
                {Polarization::TE, Polarization::TM});
            const std::size_t offset = (angleIndex * 2 + iw) * 2;
            REQUIRE(angleBatch[offset].rTotal ==
                    Catch::Approx(pointResults[0].rTotal).margin(1e-12));
            REQUIRE(angleBatch[offset].tTotal ==
                    Catch::Approx(pointResults[0].tTotal).margin(1e-12));
            REQUIRE(angleBatch[offset + 1].rTotal ==
                    Catch::Approx(pointResults[1].rTotal).margin(1e-12));
            REQUIRE(angleBatch[offset + 1].tTotal ==
                    Catch::Approx(pointResults[1].tTotal).margin(1e-12));
            REQUIRE(angleTotals[offset].rTotal ==
                    Catch::Approx(angleBatch[offset].rTotal).margin(1e-12));
            REQUIRE(angleTotals[offset].tTotal ==
                    Catch::Approx(angleBatch[offset].tTotal).margin(1e-12));
            REQUIRE(angleTotals[offset + 1].rTotal ==
                    Catch::Approx(angleBatch[offset + 1].rTotal).margin(1e-12));
            REQUIRE(angleTotals[offset + 1].tTotal ==
                    Catch::Approx(angleBatch[offset + 1].tTotal).margin(1e-12));
        }
    }

    FieldPlaneRequest fieldRequest;
    fieldRequest.plane = FieldPlane::XZ;
    fieldRequest.uMinUm = 0.0;
    fieldRequest.uMaxUm = 0.0;
    fieldRequest.uPoints = 1;
    fieldRequest.vMinUm = 0.0;
    fieldRequest.vMaxUm = 0.25;
    fieldRequest.vPoints = 2;
    fieldRequest.fixedUm = 0.0;
    fieldRequest.components = {
        FieldComponent::Ex,
        FieldComponent::Ey,
        FieldComponent::Ez,
        FieldComponent::Hx,
        FieldComponent::Hy,
        FieldComponent::Hz,
        FieldComponent::E,
    };
    const auto fields = solver.solveFieldPlane(fieldRequest);
    REQUIRE(fields.samples.size() == 14);
    REQUIRE(std::abs(fields.samples[0].value) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(fields.samples[1].value) == Catch::Approx(1.0).margin(1e-10));
    REQUIRE(std::abs(fields.samples[2].value) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(fields.samples[3].value) == Catch::Approx(1.0).margin(1e-10));
    REQUIRE(std::abs(fields.samples[4].value) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(fields.samples[5].value) == Catch::Approx(0.0).margin(1e-10));
    REQUIRE(fields.samples[6].magnitude == Catch::Approx(1.0).margin(1e-10));
    REQUIRE(fields.samples[7].iv == 1);
    REQUIRE(fields.samples[7].zUm == Catch::Approx(0.25).margin(1e-12));
    REQUIRE(std::abs(fields.samples[8].value) == Catch::Approx(1.0).margin(1e-10));

    FieldPlaneRequest outsideRequest;
    outsideRequest.plane = FieldPlane::XZ;
    outsideRequest.uMinUm = 0.0;
    outsideRequest.uMaxUm = 0.0;
    outsideRequest.uPoints = 1;
    outsideRequest.vMinUm = -0.2;
    outsideRequest.vMaxUm = 0.7;
    outsideRequest.vPoints = 3;
    outsideRequest.fixedUm = 0.0;
    outsideRequest.components = {FieldComponent::E};
    const auto outside = solver.solveFieldPlane(outsideRequest);
    REQUIRE(outside.samples.size() == 3);
    REQUIRE(outside.samples[0].layerIndex == -1);
    REQUIRE(outside.samples[2].layerIndex == 1);
    for (const auto& sample : outside.samples) {
        REQUIRE(sample.magnitude == Catch::Approx(1.0).margin(1e-10));
    }

    FieldPlaneRequest substrateRequest;
    substrateRequest.plane = FieldPlane::XY;
    substrateRequest.uMinUm = 0.0;
    substrateRequest.uMaxUm = 0.0;
    substrateRequest.uPoints = 1;
    substrateRequest.vMinUm = 0.0;
    substrateRequest.vMaxUm = 0.0;
    substrateRequest.vPoints = 1;
    substrateRequest.fixedUm = 0.75;
    substrateRequest.components = {FieldComponent::E};
    const auto substratePlane = solver.solveFieldPlane(substrateRequest);
    REQUIRE(substratePlane.samples.size() == 1);
    REQUIRE(substratePlane.samples[0].layerIndex == 1);
    REQUIRE(substratePlane.samples[0].magnitude == Catch::Approx(1.0).margin(1e-10));

    RcwaSolver physicalIncident;
    physicalIncident.setWavelength(1.0);
    physicalIncident.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    physicalIncident.setSuperstrate(Material::fromIndex("incident", {1.45, 0.0}));
    physicalIncident.setSubstrate(Material::vacuum());
    physicalIncident.setPolarization(Polarization::TM);
    physicalIncident.addUniformLayer(Material::vacuum(), 0.3);
    FieldPlaneRequest transmittedFieldRequest;
    transmittedFieldRequest.plane = FieldPlane::XZ;
    transmittedFieldRequest.uMinUm = 0.0;
    transmittedFieldRequest.uMaxUm = 0.0;
    transmittedFieldRequest.uPoints = 1;
    transmittedFieldRequest.vMinUm = 0.1;
    transmittedFieldRequest.vMaxUm = 0.4;
    transmittedFieldRequest.vPoints = 2;
    transmittedFieldRequest.fixedUm = 0.0;
    transmittedFieldRequest.components = {FieldComponent::E, FieldComponent::H};
    const auto transmittedField = physicalIncident.solveFieldPlane(transmittedFieldRequest);
    const double expectedT = 2.0 * 1.45 / (1.45 + 1.0);
    REQUIRE(transmittedField.samples.size() == 4);
    REQUIRE(transmittedField.samples[0].magnitude == Catch::Approx(expectedT).margin(1e-10));
    REQUIRE(transmittedField.samples[1].magnitude == Catch::Approx(expectedT).margin(1e-10));
    REQUIRE(transmittedField.samples[2].magnitude == Catch::Approx(expectedT).margin(1e-10));
    REQUIRE(transmittedField.samples[3].magnitude == Catch::Approx(expectedT).margin(1e-10));

    FieldPlaneRequest xzPlaneRequest;
    xzPlaneRequest.plane = FieldPlane::XZ;
    xzPlaneRequest.uMinUm = 0.0;
    xzPlaneRequest.uMaxUm = 0.2;
    xzPlaneRequest.uPoints = 3;
    xzPlaneRequest.vMinUm = 0.17;
    xzPlaneRequest.vMaxUm = 0.17;
    xzPlaneRequest.vPoints = 1;
    xzPlaneRequest.fixedUm = 0.0;
    xzPlaneRequest.components = {FieldComponent::Ey};
    xzPlaneRequest.workers = 0;

    RcwaSolver phased;
    phased.setWavelength(1.0);
    phased.setIncidence(23.0, 0.0);
    phased.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    phased.setSuperstrate(Material::vacuum());
    phased.setSubstrate(Material::vacuum());
    phased.setPolarization(Polarization::TE);
    phased.addUniformLayer(Material::vacuum(), 0.5);
    const auto xzPlane = phased.solveFieldPlane(xzPlaneRequest);
    const auto xzGrid = phased.solveFieldGrid(xzPlaneRequest);
    REQUIRE(xzPlane.samples.size() == 3);
    REQUIRE(xzGrid.values.size() == 3);
    REQUIRE(xzGrid.components == xzPlaneRequest.components);
    REQUIRE(xzGrid.uUm.size() == 3);
    REQUIRE(xzGrid.uUm[0] == Catch::Approx(0.0));
    REQUIRE(xzGrid.uUm[1] == Catch::Approx(0.1));
    REQUIRE(xzGrid.uUm[2] == Catch::Approx(0.2));
    REQUIRE(xzGrid.vUm.size() == 1);
    REQUIRE(xzGrid.vUm[0] == Catch::Approx(0.17));
    REQUIRE(xzGrid.layerIndexByV == std::vector<int>{0});
    REQUIRE(xzGrid.zLocalUmByV.size() == 1);
    REQUIRE(xzGrid.zLocalUmByV[0] == Catch::Approx(0.17));
    REQUIRE(std::string(fieldPlaneName(xzPlane.plane)) == "xz");
    const Complex expectedXPhase =
        std::exp(Complex{0.0, 2.0 * pi * std::sin(23.0 * pi / 180.0) * 0.1});
    for (std::size_t i = 0; i < xzPlane.samples.size(); ++i) {
        REQUIRE(xzPlane.samples[i].iu == static_cast<int>(i));
        REQUIRE(xzPlane.samples[i].iv == 0);
        REQUIRE(xzPlane.samples[i].yUm == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(xzPlane.samples[i].zUm == Catch::Approx(0.17).margin(1e-12));
    }
    REQUIRE(std::abs(xzPlane.samples[1].value / xzPlane.samples[0].value -
                     expectedXPhase) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(xzPlane.samples[2].value / xzPlane.samples[1].value -
                     expectedXPhase) == Catch::Approx(0.0).margin(1e-12));

    FieldPlaneRequest xyPlaneRequest;
    xyPlaneRequest.plane = FieldPlane::XY;
    xyPlaneRequest.uMinUm = 0.0;
    xyPlaneRequest.uMaxUm = 0.2;
    xyPlaneRequest.uPoints = 3;
    xyPlaneRequest.vMinUm = 0.0;
    xyPlaneRequest.vMaxUm = 0.0;
    xyPlaneRequest.vPoints = 1;
    xyPlaneRequest.fixedUm = 0.17;
    xyPlaneRequest.components = {FieldComponent::Ey};
    xyPlaneRequest.workers = 2;
    const auto xyPlane = phased.solveFieldPlane(xyPlaneRequest);
    REQUIRE(xyPlane.samples.size() == 3);
    REQUIRE(std::string(fieldPlaneName(xyPlane.plane)) == "xy");
    for (std::size_t i = 0; i < xyPlane.samples.size(); ++i) {
        REQUIRE(xyPlane.samples[i].xUm == Catch::Approx(xzPlane.samples[i].xUm).margin(1e-12));
        REQUIRE(xyPlane.samples[i].yUm == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(xyPlane.samples[i].zUm == Catch::Approx(0.17).margin(1e-12));
        REQUIRE(std::abs(xyPlane.samples[i].value - xzPlane.samples[i].value) ==
                Catch::Approx(0.0).margin(1e-12));
    }

    FieldPlaneRequest yzPlaneRequest;
    yzPlaneRequest.plane = FieldPlane::YZ;
    yzPlaneRequest.uMinUm = 0.0;
    yzPlaneRequest.uMaxUm = 0.2;
    yzPlaneRequest.uPoints = 3;
    yzPlaneRequest.vMinUm = 0.17;
    yzPlaneRequest.vMaxUm = 0.17;
    yzPlaneRequest.vPoints = 1;
    yzPlaneRequest.fixedUm = 0.0;
    yzPlaneRequest.components = {FieldComponent::Ey};
    yzPlaneRequest.workers = 2;
    const auto yzPlane = phased.solveFieldPlane(yzPlaneRequest);
    REQUIRE(yzPlane.samples.size() == 3);
    REQUIRE(std::string(fieldPlaneName(yzPlane.plane)) == "yz");
    for (std::size_t i = 0; i < yzPlane.samples.size(); ++i) {
        REQUIRE(yzPlane.samples[i].xUm == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(yzPlane.samples[i].zUm == Catch::Approx(0.17).margin(1e-12));
        REQUIRE(std::abs(yzPlane.samples[i].value - yzPlane.samples[0].value) ==
                Catch::Approx(0.0).margin(1e-12));
    }

    const auto circularFieldComponents = [](Polarization pol) {
        RcwaSolver circular;
        circular.setWavelength(1.0);
        circular.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
        circular.setSuperstrate(Material::vacuum());
        circular.setSubstrate(Material::vacuum());
        circular.setPolarization(pol);
        circular.addUniformLayer(Material::vacuum(), 0.5);

        FieldPlaneRequest request;
        request.plane = FieldPlane::XZ;
        request.uMinUm = 0.0;
        request.uMaxUm = 0.0;
        request.uPoints = 1;
        request.vMinUm = 0.25;
        request.vMaxUm = 0.25;
        request.vPoints = 1;
        request.fixedUm = 0.0;
        request.components = {
            FieldComponent::Ex,
            FieldComponent::Ey,
            FieldComponent::Hx,
            FieldComponent::Hy,
        };
        const auto result = circular.solveFieldPlane(request);
        REQUIRE(result.samples.size() == 4);
        return std::vector<Complex>{
            result.samples[0].value,
            result.samples[1].value,
            result.samples[2].value,
            result.samples[3].value,
        };
    };
    const auto lcpField = circularFieldComponents(Polarization::LCP);
    const auto rcpField = circularFieldComponents(Polarization::RCP);
    REQUIRE(std::abs(lcpField[0] / lcpField[1] - Complex{0.0, 1.0}) ==
            Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(rcpField[0] / rcpField[1] - Complex{0.0, -1.0}) ==
            Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(lcpField[3] / lcpField[2] - Complex{0.0, -1.0}) ==
            Catch::Approx(0.0).margin(1e-10));
    REQUIRE(std::abs(rcpField[3] / rcpField[2] - Complex{0.0, 1.0}) ==
            Catch::Approx(0.0).margin(1e-10));

    FieldPlaneRequest bad = xzPlaneRequest;
    bad.components.clear();
    REQUIRE_THROWS_AS(solver.solveFieldPlane(bad), std::invalid_argument);

    RcwaSolver reflective;
    reflective.setWavelength(1.0);
    reflective.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    reflective.setSuperstrate(Material::vacuum());
    reflective.setSubstrate(Material::fromIndex("glass", {1.8, 0.0}));
    reflective.setPolarization(Polarization::TE);
    reflective.addUniformLayer(Material::vacuum(), 0.5);
    FieldPlaneRequest reflectiveRequest;
    reflectiveRequest.plane = FieldPlane::XZ;
    reflectiveRequest.uMinUm = 0.0;
    reflectiveRequest.uMaxUm = 0.0;
    reflectiveRequest.uPoints = 1;
    reflectiveRequest.vMinUm = 0.0;
    reflectiveRequest.vMaxUm = 0.5;
    reflectiveRequest.vPoints = 2;
    reflectiveRequest.fixedUm = 0.0;
    reflectiveRequest.components = {FieldComponent::Ey};
    const auto reflectiveFields = reflective.solveFieldPlane(reflectiveRequest);
    REQUIRE(reflectiveFields.samples.size() == 2);
    REQUIRE(std::abs(reflectiveFields.samples[0].value - reflectiveFields.samples[1].value) >
            1e-3);

    RcwaSolver continuity;
    continuity.setWavelength(1.0);
    continuity.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    continuity.setSuperstrate(Material::vacuum());
    continuity.setSubstrate(Material::fromIndex("glass", {1.8, 0.0}));
    continuity.setPolarization(Polarization::TM);
    continuity.addUniformLayer(Material::fromIndex("film", {1.45, 0.0}), 0.37);
    continuity.addUniformLayer(Material::fromIndex("film", {1.45, 0.0}), 0.22);
    FieldPlaneRequest continuityRequest;
    continuityRequest.plane = FieldPlane::XZ;
    continuityRequest.uMinUm = 0.0;
    continuityRequest.uMaxUm = 0.0;
    continuityRequest.uPoints = 1;
    continuityRequest.vMinUm = 0.37 - 1e-9;
    continuityRequest.vMaxUm = 0.37 + 1e-9;
    continuityRequest.vPoints = 2;
    continuityRequest.fixedUm = 0.0;
    continuityRequest.components = {
        FieldComponent::Ex,
        FieldComponent::Ey,
        FieldComponent::Ez,
        FieldComponent::Hx,
        FieldComponent::Hy,
        FieldComponent::Hz,
    };
    const auto continuityFields = continuity.solveFieldPlane(continuityRequest);
    const auto continuityGrid = continuity.solveFieldGrid(continuityRequest);
    REQUIRE(continuityFields.samples.size() == 12);
    REQUIRE(continuityGrid.values.size() == 12);
    REQUIRE(continuityGrid.layerIndexByV == std::vector<int>{0, 1});
    REQUIRE(continuityFields.samples[0].layerIndex == 0);
    REQUIRE(continuityFields.samples[6].layerIndex == 1);
    for (std::size_t component = 0; component < 6; ++component) {
        REQUIRE(std::abs(continuityGrid.values[2 * component] -
                         continuityFields.samples[component].value) < 1e-12);
        REQUIRE(std::abs(continuityGrid.values[2 * component + 1] -
                         continuityFields.samples[6 + component].value) < 1e-12);
        REQUIRE(std::abs(continuityFields.samples[component].value -
                         continuityFields.samples[6 + component].value) ==
                Catch::Approx(0.0).margin(1e-7));
    }

}

TEST_CASE("direct Bloch S parameters match the equivalent real incidence") {
    using namespace rcwa;

    constexpr Real wavelengthUm = 0.8;
    constexpr Real thetaDeg = 20.0;
    constexpr Real phiDeg = 30.0;
    RcwaSolver solver;
    solver.setWavelength(wavelengthUm);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::constant("slab", {4.0, 0.0}), 0.25);
    solver.setIncidence(thetaDeg, phiDeg);

    const SParameterResult angle = solver.solveSParameters();
    const Real theta = degToRad(thetaDeg);
    const Real phi = degToRad(phiDeg);
    const Real blochKxReduced =
        std::sin(theta) * std::cos(phi) / wavelengthUm;
    const Real blochKyReduced =
        std::sin(theta) * std::sin(phi) / wavelengthUm;
    const SParameterResult bloch = solver.solveSParametersAnalyticContinuation(
        blochKxReduced,
        blochKyReduced);
    requireMatrixClose(bloch.S11, angle.S11, 2e-11);
    requireMatrixClose(bloch.S12, angle.S12, 2e-11);
    requireMatrixClose(bloch.S21, angle.S21, 2e-11);
    requireMatrixClose(bloch.S22, angle.S22, 2e-11);

    const SParameterResult gamma =
        solver.solveSParametersAnalyticContinuationAtGamma();
    const SParameterResult directGamma =
        solver.solveSParametersAnalyticContinuation(0.0, 0.0);
    requireMatrixClose(directGamma.S11, gamma.S11, 0.0);
    requireMatrixClose(directGamma.S12, gamma.S12, 0.0);
    requireMatrixClose(directGamma.S21, gamma.S21, 0.0);
    requireMatrixClose(directGamma.S22, gamma.S22, 0.0);
}

TEST_CASE("reconstructed fields satisfy Maxwell curls and material interface conditions") {
    using namespace rcwa;

    const std::vector<FieldComponent> components{
        FieldComponent::Ex,
        FieldComponent::Ey,
        FieldComponent::Ez,
        FieldComponent::Hx,
        FieldComponent::Hy,
        FieldComponent::Hz,
    };

    RcwaSolver planeWave;
    planeWave.setWavelength(1.0);
    planeWave.setIncidence(23.0, 31.0);
    planeWave.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    planeWave.setSuperstrate(Material::vacuum());
    planeWave.setSubstrate(Material::vacuum());
    planeWave.setPolarization(Polarization::LCP);
    planeWave.addUniformLayer(Material::vacuum(), 0.5);

    const Real h = 1.0e-5;
    const auto sampleLine = [&](FieldPlane plane,
                                Real uMin,
                                Real uMax,
                                Real vMin,
                                Real vMax,
                                Real fixed) {
        FieldPlaneRequest request;
        request.plane = plane;
        request.uMinUm = uMin;
        request.uMaxUm = uMax;
        request.uPoints = uMin == uMax ? 1 : 3;
        request.vMinUm = vMin;
        request.vMaxUm = vMax;
        request.vPoints = vMin == vMax ? 1 : 3;
        request.fixedUm = fixed;
        request.components = components;
        return planeWave.solveFieldPlane(request);
    };

    const FieldPlaneResult xLine =
        sampleLine(FieldPlane::XZ, -h, h, 0.2, 0.2, 0.0);
    const FieldPlaneResult yLine =
        sampleLine(FieldPlane::YZ, -h, h, 0.2, 0.2, 0.0);
    const FieldPlaneResult zLine =
        sampleLine(FieldPlane::XZ, 0.0, 0.0, 0.2 - h, 0.2 + h, 0.0);
    REQUIRE(xLine.samples.size() == 18);
    REQUIRE(yLine.samples.size() == 18);
    REQUIRE(zLine.samples.size() == 18);

    const auto derivative = [&](const FieldPlaneResult& line, std::size_t component) {
        return (line.samples[12 + component].value -
                line.samples[component].value) /
            (Real{2.0} * h);
    };
    std::array<Complex, 6> center{};
    for (std::size_t component = 0; component < center.size(); ++component) {
        center[component] = zLine.samples[6 + component].value;
    }
    const Complex dExDx = derivative(xLine, 0);
    const Complex dEyDx = derivative(xLine, 1);
    const Complex dEzDx = derivative(xLine, 2);
    const Complex dHxDx = derivative(xLine, 3);
    const Complex dHyDx = derivative(xLine, 4);
    const Complex dHzDx = derivative(xLine, 5);
    const Complex dExDy = derivative(yLine, 0);
    const Complex dEyDy = derivative(yLine, 1);
    const Complex dEzDy = derivative(yLine, 2);
    const Complex dHxDy = derivative(yLine, 3);
    const Complex dHyDy = derivative(yLine, 4);
    const Complex dHzDy = derivative(yLine, 5);
    const Complex dExDz = derivative(zLine, 0);
    const Complex dEyDz = derivative(zLine, 1);
    const Complex dEzDz = derivative(zLine, 2);
    const Complex dHxDz = derivative(zLine, 3);
    const Complex dHyDz = derivative(zLine, 4);
    const Complex dHzDz = derivative(zLine, 5);
    (void)dExDx;
    (void)dEyDy;
    (void)dEzDz;
    (void)dHxDx;
    (void)dHyDy;
    (void)dHzDz;

    const std::array<Complex, 3> curlE{
        dEzDy - dEyDz,
        dExDz - dEzDx,
        dEyDx - dExDy,
    };
    const std::array<Complex, 3> curlH{
        dHzDy - dHyDz,
        dHxDz - dHzDx,
        dHyDx - dHxDy,
    };
    const Real k0 = twoPi;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        REQUIRE(std::abs(curlE[axis] - iu * k0 * center[3 + axis]) < 2e-7);
        REQUIRE(std::abs(curlH[axis] + iu * k0 * center[axis]) < 2e-7);
    }

    RcwaSolver interface;
    interface.setWavelength(1.0);
    interface.setIncidence(27.0, 19.0);
    interface.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    interface.setSuperstrate(Material::vacuum());
    interface.setSubstrate(Material::fromIndex("substrate", {1.3, 0.0}));
    interface.setPolarization(Polarization::LCP);
    interface.addUniformLayer(Material::fromIndex("layer 1", {1.4, 0.0}), 0.3);
    interface.addUniformLayer(Material::fromIndex("layer 2", {1.9, 0.0}), 0.4);

    FieldPlaneRequest interfaceRequest;
    interfaceRequest.plane = FieldPlane::XZ;
    interfaceRequest.uMinUm = 0.0;
    interfaceRequest.uMaxUm = 0.0;
    interfaceRequest.uPoints = 1;
    interfaceRequest.vMinUm = 0.3 - 1e-9;
    interfaceRequest.vMaxUm = 0.3 + 1e-9;
    interfaceRequest.vPoints = 2;
    interfaceRequest.fixedUm = 0.0;
    interfaceRequest.components = components;
    const FieldPlaneResult interfaceFields = interface.solveFieldPlane(interfaceRequest);
    REQUIRE(interfaceFields.samples.size() == 12);
    REQUIRE(interfaceFields.samples[0].layerIndex == 0);
    REQUIRE(interfaceFields.samples[6].layerIndex == 1);
    for (const std::size_t tangential : {std::size_t{0}, std::size_t{1},
                                         std::size_t{3}, std::size_t{4}}) {
        REQUIRE(std::abs(interfaceFields.samples[tangential].value -
                         interfaceFields.samples[6 + tangential].value) < 2e-7);
    }
    const Complex dzAbove = Complex{1.4 * 1.4, 0.0} *
        interfaceFields.samples[2].value;
    const Complex dzBelow = Complex{1.9 * 1.9, 0.0} *
        interfaceFields.samples[8].value;
    REQUIRE(std::abs(dzAbove - dzBelow) < 2e-7);
    REQUIRE(std::abs(interfaceFields.samples[5].value -
                     interfaceFields.samples[11].value) < 2e-7);
}

TEST_CASE("anisotropic uniform layers remain finite and conservative") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    eps(0, 2) = {0.12, 0.0};
    eps(2, 0) = {0.12, 0.0};

    RcwaSolver solver;
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::anisotropic("biaxial", eps), 0.3);
    const auto r = solver.solve();
    REQUIRE(std::isfinite(r.rTotal));
    REQUIRE(std::isfinite(r.tTotal));
    REQUIRE(r.conservation == Catch::Approx(1.0).margin(1e-8));

    eps(0, 2) = {0.08, 0.0};
    eps(2, 0) = {0.08, 0.0};
    RcwaSolver tilted;
    tilted.setWavelength(0.633);
    tilted.setIncidence(15.0, 0.0);
    tilted.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    tilted.setSuperstrate(Material::vacuum());
    tilted.setSubstrate(Material::vacuum());
    tilted.setPolarization(Polarization::TE);
    tilted.addUniformLayer(Material::anisotropic("tilted_biaxial", eps), 0.25);
    REQUIRE(tilted.solve().conservation == Catch::Approx(1.0).margin(1e-8));

    Tensor3 mu = Tensor3::diagonal({1.05, 0.0}, {1.12, 0.0}, {0.97, 0.0});
    mu(0, 2) = {0.03, 0.0};
    mu(2, 0) = {0.03, 0.0};
    RcwaSolver magnetic;
    magnetic.setWavelength(0.71);
    magnetic.setIncidence(11.0, 23.0);
    magnetic.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    magnetic.setSuperstrate(Material::vacuum());
    magnetic.setSubstrate(Material::vacuum());
    magnetic.addUniformLayer(Material::anisotropic("tensor_mu_slab", eps, mu), 0.21);
    const auto rm = magnetic.solve();
    REQUIRE(std::isfinite(rm.rTotal));
    REQUIRE(std::isfinite(rm.tTotal));
    REQUIRE(rm.conservation == Catch::Approx(1.0).margin(1e-7));
}

TEST_CASE("uniform TMM uses the canonical Berreman 4x4 state convention") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal(
        {2.31, 0.04}, {2.67, 0.03}, {3.08, 0.02});
    eps(0, 1) = {0.07, -0.02};
    eps(1, 0) = {-0.03, 0.01};
    eps(0, 2) = {0.11, 0.03};
    eps(2, 0) = {0.09, -0.02};
    eps(1, 2) = {-0.05, 0.01};
    eps(2, 1) = {0.04, 0.02};
    const Tensor3 mu = Tensor3::isotropic({1.0, 0.0});
    const Complex kx{0.27, 0.0};

    // Transform the project's [Ex,Ey,Hx,Hy]^T matrix into the canonical
    // psi=[Ex,Hy,Ey,-Hx]^T state and compare every element with the explicit
    // Maxwell-elimination formula below.
    const Matrix state = berremanStateMatrix(eps, mu, kx, {});
    constexpr std::array<std::size_t, 4> component{0, 3, 1, 2};
    constexpr std::array<Real, 4> sign{1.0, 1.0, 1.0, -1.0};
    Matrix transformed(4, 4);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            transformed(row, column) = sign[row] *
                state(component[row], component[column]) * sign[column];
        }
    }

    const Complex invEzz = Complex{1.0, 0.0} / eps(2, 2);
    Matrix reference(4, 4);
    reference(0, 0) = -kx * eps(2, 0) * invEzz;
    reference(0, 1) = Complex{1.0, 0.0} - kx * kx * invEzz;
    reference(0, 2) = -kx * eps(2, 1) * invEzz;
    reference(1, 0) = eps(0, 0) - eps(0, 2) * eps(2, 0) * invEzz;
    reference(1, 1) = -kx * eps(0, 2) * invEzz;
    reference(1, 2) = eps(0, 1) - eps(0, 2) * eps(2, 1) * invEzz;
    reference(2, 3) = {1.0, 0.0};
    reference(3, 0) = eps(1, 0) - eps(1, 2) * eps(2, 0) * invEzz;
    reference(3, 1) = -kx * eps(1, 2) * invEzz;
    reference(3, 2) = eps(1, 1) - kx * kx -
        eps(1, 2) * eps(2, 1) * invEzz;
    requireMatrixClose(transformed, reference, 2e-13);

    // Also validate the generalized epsilon/mu, nonzero-ky matrix directly
    // against all six frequency-domain Maxwell equations for every mode.
    Tensor3 tensorMu = Tensor3::diagonal(
        {1.13, 0.01}, {0.94, 0.02}, {1.07, 0.01});
    tensorMu(0, 1) = {0.02, -0.01};
    tensorMu(1, 0) = {-0.01, 0.015};
    tensorMu(0, 2) = {0.03, 0.01};
    tensorMu(2, 0) = {0.025, -0.005};
    tensorMu(1, 2) = {-0.02, 0.004};
    tensorMu(2, 1) = {0.015, 0.006};
    const Complex ky{-0.14, 0.0};
    const Matrix generalized = berremanStateMatrix(eps, tensorMu, kx, ky);
    const Eigensystem modes = eig(generalized);
    for (std::size_t mode = 0; mode < 4; ++mode) {
        const Complex gamma = modes.values[mode];
        const Complex Ex = modes.vectors(0, mode);
        const Complex Ey = modes.vectors(1, mode);
        const Complex Hx = modes.vectors(2, mode);
        const Complex Hy = modes.vectors(3, mode);
        const Complex Ez = (
            ky * Hx - kx * Hy - eps(2, 0) * Ex - eps(2, 1) * Ey) /
            eps(2, 2);
        const Complex Hz = (
            kx * Ey - ky * Ex - tensorMu(2, 0) * Hx - tensorMu(2, 1) * Hy) /
            tensorMu(2, 2);
        const std::array<Complex, 3> electric{Ex, Ey, Ez};
        const std::array<Complex, 3> magnetic{Hx, Hy, Hz};
        const std::array<Complex, 3> kCrossE{
            ky * Ez - gamma * Ey,
            gamma * Ex - kx * Ez,
            kx * Ey - ky * Ex,
        };
        const std::array<Complex, 3> kCrossH{
            ky * Hz - gamma * Hy,
            gamma * Hx - kx * Hz,
            kx * Hy - ky * Hx,
        };
        for (std::size_t row = 0; row < 3; ++row) {
            Complex muH{};
            Complex epsE{};
            for (std::size_t column = 0; column < 3; ++column) {
                muH += tensorMu(row, column) * magnetic[column];
                epsE += eps(row, column) * electric[column];
            }
            const Real scale = std::max({
                Real{1.0}, std::abs(muH), std::abs(epsE),
                std::abs(kCrossE[row]), std::abs(kCrossH[row])});
            REQUIRE(std::abs(kCrossE[row] - muH) <= 2e-10 * scale);
            REQUIRE(std::abs(kCrossH[row] + epsE) <= 2e-10 * scale);
        }
    }
}

TEST_CASE("uniform 4x4 TMM matches the local direct Berreman stack reference") {
    using namespace rcwa;

    Tensor3 eps;
    eps(0, 0) = {2.50, 0.04};
    eps(0, 1) = {0.12, 0.005};
    eps(0, 2) = {0.18, 0.010};
    eps(1, 0) = {0.12, 0.005};
    eps(1, 1) = {2.90, 0.03};
    eps(1, 2) = {-0.07, 0.003};
    eps(2, 0) = {0.18, 0.010};
    eps(2, 1) = {-0.07, 0.003};
    eps(2, 2) = {3.30, 0.02};

    RcwaSolver solver;
    solver.setWavelength(0.6328);
    solver.setIncidence(23.0, 0.0);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("n=1.52", {1.52, 0.0}));
    solver.addUniformLayer(Material::anisotropic("Berreman slab", eps), 0.173);

    const auto directTm = directBerremanStackPower(
        {{eps, Tensor3::isotropic({1.0, 0.0}), 0.173}},
        0.6328,
        23.0,
        1.0,
        1.52,
        Polarization::TM);
    const auto directTe = directBerremanStackPower(
        {{eps, Tensor3::isotropic({1.0, 0.0}), 0.173}},
        0.6328,
        23.0,
        1.0,
        1.52,
        Polarization::TE);

    solver.setPolarization(Polarization::TM);
    const auto p = solver.solveSpectrumOnly();
    REQUIRE(p.rTotal == Catch::Approx(directTm.reflectance).margin(3e-12));
    REQUIRE(p.tTotal == Catch::Approx(directTm.transmittance).margin(3e-12));
    REQUIRE(p.conservation == Catch::Approx(0.9637203488932448).margin(3e-12));
    REQUIRE(directTm.reflectance ==
            Catch::Approx(0.034470574954951826).margin(3e-12));
    REQUIRE(directTm.transmittance ==
            Catch::Approx(0.929249773938293).margin(3e-12));

    solver.setPolarization(Polarization::TE);
    const auto s = solver.solveSpectrumOnly();
    REQUIRE(s.rTotal == Catch::Approx(directTe.reflectance).margin(3e-12));
    REQUIRE(s.tTotal == Catch::Approx(directTe.transmittance).margin(3e-12));
    REQUIRE(s.conservation == Catch::Approx(0.9711959108407970).margin(3e-12));
    REQUIRE(directTe.reflectance ==
            Catch::Approx(0.057979359696230000).margin(3e-12));
    REQUIRE(directTe.transmittance ==
            Catch::Approx(0.913216551144567).margin(3e-12));
}

TEST_CASE("all physically uniform layer specifications use the same TMM backend") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal(
        {2.20, 0.01}, {2.55, 0.02}, {2.85, 0.015});
    eps(0, 2) = {0.08, -0.004};
    eps(2, 0) = {0.08, -0.004};
    const Material target = Material::anisotropic("uniform tensor", eps);
    const Material other = Material::constant("other", {4.1, 0.03});

    const auto prepareOne = [&](LayerSpec layer) {
        std::vector<LayerSpec> layers;
        layers.push_back(std::move(layer));
        const StackBuilder builder(
            1, 1, 1, false, LatticeTruncation::Parallelogramic, layers);
        return builder.prepare(
            0.81, 13.0, 7.0, Material::vacuum(), Material::vacuum());
    };

    const PreparedStack direct = prepareOne(UniformLayer{target, 0.23});
    REQUIRE(direct.media[1].kind == LayerComputationKind::UniformTmm);

    GratingLayer emptyRidge;
    emptyRidge.materialRidge = other;
    emptyRidge.materialGroove = target;
    emptyRidge.fillFactor = 0.0;
    emptyRidge.periodUm = 1.0;
    emptyRidge.periodYUm = 1.0;
    emptyRidge.thicknessUm = 0.23;

    GratingLayer fullRidge = emptyRidge;
    fullRidge.materialRidge = target;
    fullRidge.materialGroove = other;
    fullRidge.fillFactor = 1.0;

    GratingLayer equalMaterials = emptyRidge;
    equalMaterials.materialRidge = target;
    equalMaterials.materialGroove = target;
    equalMaterials.fillFactor = 0.37;

    PeriodicLayer2D sampledUniform;
    sampledUniform.cells = {target};
    sampledUniform.sampleCountX = 1;
    sampledUniform.sampleCountY = 1;
    sampledUniform.periodXUm = 1.0;
    sampledUniform.periodYUm = 1.0;
    sampledUniform.thicknessUm = 0.23;

    PatternedLayer2D emptyPattern;
    emptyPattern.background = target;
    emptyPattern.periodXUm = 1.0;
    emptyPattern.periodYUm = 1.0;
    emptyPattern.thicknessUm = 0.23;

    PatternedLayer2D opticallyUniformPattern = emptyPattern;
    opticallyUniformPattern.regions.push_back(PatternRegion::rectangle(
        target, {0.0, 0.0}, 0.0, {0.2, 0.2}));

    const HarmonicBasis precomputedBasis = makeHarmonicBasisOrders(
        1, 1, LatticeTruncation::Parallelogramic);
    auto precomputedStorage = std::make_shared<PrecomputedPeriodicLayer2D>();
    precomputedStorage->tensors = tensorFourierMatricesUniform(
        eps,
        Tensor3::isotropic({1.0, 0.0}),
        precomputedBasis);
    precomputedStorage->periodXUm = 1.0;
    precomputedStorage->periodYUm = 1.0;
    precomputedStorage->thicknessUm = 0.23;
    const PrecomputedPeriodicLayer2DPtr precomputedUniform = precomputedStorage;

    const std::vector<LayerSpec> uniformRepresentations{
        emptyRidge,
        fullRidge,
        equalMaterials,
        sampledUniform,
        emptyPattern,
        opticallyUniformPattern,
        precomputedUniform,
    };
    for (const auto& layer : uniformRepresentations) {
        const PreparedStack stack = prepareOne(layer);
        REQUIRE(stack.media[1].kind == LayerComputationKind::UniformTmm);
        requireComplexMultisetClose(stack.media[1].gamma, direct.media[1].gamma, 2e-12);
    }

    GratingLayer trueGrating = emptyRidge;
    trueGrating.fillFactor = 0.43;
    const PreparedStack grating = prepareOne(trueGrating);
    REQUIRE(grating.media[1].kind == LayerComputationKind::PatternedRcwa);

    PatternedLayer2D truePattern = emptyPattern;
    truePattern.regions.push_back(PatternRegion::rectangle(
        other, {0.0, 0.0}, 0.0, {0.2, 0.2}));
    const PreparedStack patterned = prepareOne(truePattern);
    REQUIRE(patterned.media[1].kind == LayerComputationKind::PatternedRcwa);
}

TEST_CASE("enhanced T-matrix is stable for the deep Moharam multilevel grating") {
    using namespace rcwa;

    RcwaSolver solver;
    solver.setWavelength(1.0);
    solver.setIncidence(10.0, 0.0);
    solver.setHarmonicOrders(3, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("n=2.04", {2.04, 0.0}));

    // JOSA A 12, 1077-1086 (1995), Figs. 3-4: a 16-level asymmetric
    // staircase represented by 15 binary layers at a total depth of 49 lambda.
    for (int level = 1; level <= 15; ++level) {
        GratingLayer layer;
        layer.materialRidge = Material::fromIndex("ridge n=2.04", {2.04, 0.0});
        layer.materialGroove = Material::vacuum();
        layer.fillFactor = static_cast<Real>(level) / Real{16.0};
        layer.periodUm = 1.0;
        layer.periodYUm = 1.0;
        layer.ridgeOffsetUm = -Real{0.5} * (Real{1.0} - layer.fillFactor);
        layer.thicknessUm = Real{49.0} / Real{15.0};
        solver.addGratingLayer(std::move(layer));
    }

    REQUIRE(solver.stackingAlgorithm() == StackingAlgorithm::ScatteringMatrix);
    const auto scatteringSpectra = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    const SParameterResult scattering = solver.solveSParameters();
    const ZeroOrderAmplitudeResult scatteringZero = solver.solveZeroOrderAmplitudes();

    solver.setStackingAlgorithm(StackingAlgorithm::EnhancedTransmittanceMatrix);
    REQUIRE(solver.stackingAlgorithm() ==
            StackingAlgorithm::EnhancedTransmittanceMatrix);
    const auto enhancedSpectra = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    const SParameterResult enhanced = solver.solveSParameters();
    const ZeroOrderAmplitudeResult enhancedZero = solver.solveZeroOrderAmplitudes();

    REQUIRE(enhancedSpectra.size() == scatteringSpectra.size());
    for (std::size_t i = 0; i < enhancedSpectra.size(); ++i) {
        REQUIRE(std::isfinite(enhancedSpectra[i].rTotal));
        REQUIRE(std::isfinite(enhancedSpectra[i].tTotal));
        REQUIRE(enhancedSpectra[i].conservation == Catch::Approx(1.0).margin(2e-10));
        REQUIRE(enhancedSpectra[i].rTotal ==
                Catch::Approx(scatteringSpectra[i].rTotal).margin(2e-10));
        REQUIRE(enhancedSpectra[i].tTotal ==
                Catch::Approx(scatteringSpectra[i].tTotal).margin(2e-10));
    }
    requireMatrixClose(enhanced.S11, scattering.S11, 3e-10);
    requireMatrixClose(enhanced.S12, scattering.S12, 3e-10);
    requireMatrixClose(enhanced.S21, scattering.S21, 3e-10);
    requireMatrixClose(enhanced.S22, scattering.S22, 3e-10);
    requireMatrixClose(enhancedZero.R, scatteringZero.R, 3e-10);
    requireMatrixClose(enhancedZero.T, scatteringZero.T, 3e-10);
}

TEST_CASE("enhanced T-matrix constructs both ports of a nonreciprocal stack") {
    using namespace rcwa;

    Tensor3 gyrotropic = Tensor3::diagonal(
        {3.2, 0.02}, {3.4, 0.03}, {3.1, 0.01});
    gyrotropic(0, 2) = {0.0, -0.18};
    gyrotropic(2, 0) = {0.0, 0.18};

    RcwaSolver solver;
    solver.setWavelength(1.03);
    solver.setIncidence(21.0, 13.0);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::constant("lossy substrate", {2.1, 0.04}));
    solver.addUniformLayer(Material::anisotropic("gyrotropic layer", gyrotropic), 0.37);
    solver.addUniformLayer(Material::constant("lossy cap", {2.1, 0.04}), 0.19);

    const SParameterResult scattering = solver.solveSParameters();
    const DirectionalThermalChannelResult scatteringThermal =
        solver.solveDirectionalThermalChannels();
    solver.setStackingAlgorithm(StackingAlgorithm::EnhancedTransmittanceMatrix);
    const SParameterResult enhanced = solver.solveSParameters();
    const DirectionalThermalChannelResult enhancedThermal =
        solver.solveDirectionalThermalChannels();

    requireMatrixClose(enhanced.S11, scattering.S11, 3e-12);
    requireMatrixClose(enhanced.S12, scattering.S12, 3e-12);
    requireMatrixClose(enhanced.S21, scattering.S21, 3e-12);
    requireMatrixClose(enhanced.S22, scattering.S22, 3e-12);
    REQUIRE(enhancedThermal.teAbsorptivity ==
            Catch::Approx(scatteringThermal.teAbsorptivity).margin(3e-12));
    REQUIRE(enhancedThermal.tmAbsorptivity ==
            Catch::Approx(scatteringThermal.tmAbsorptivity).margin(3e-12));
    REQUIRE(enhancedThermal.teEmissivity ==
            Catch::Approx(scatteringThermal.teEmissivity).margin(3e-12));
    REQUIRE(enhancedThermal.tmEmissivity ==
            Catch::Approx(scatteringThermal.tmEmissivity).margin(3e-12));
}

TEST_CASE("uniform TMM selects forward modes by decay or energy flow") {
    using namespace rcwa;

    const auto basis = makeHarmonicBasisOrders(
        0, 0, LatticeTruncation::Parallelogramic);
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.0, 0.0, 1.0, 1.0, basis);
    const LayerModes negativeIndexModes = computeUniformModes(
        Tensor3::isotropic({-1.0, 0.0}),
        Tensor3::isotropic({-1.0, 0.0}),
        Kx,
        Ky,
        0.37,
        1.0);
    REQUIRE(negativeIndexModes.kind == LayerComputationKind::UniformTmm);
    REQUIRE(negativeIndexModes.gamma.size() == 4);
    REQUIRE(std::real(negativeIndexModes.gamma[0]) < 0.0);
    REQUIRE(std::real(negativeIndexModes.gamma[1]) < 0.0);
    REQUIRE(propagatingFluxWeight(negativeIndexModes, 0) > 0.0);
    REQUIRE(propagatingFluxWeight(negativeIndexModes, 1) > 0.0);

    Tensor3 weaklyAnisotropic = Tensor3::isotropic({2.25, 0.0});
    weaklyAnisotropic(0, 1) = {1e-15, 0.0};
    const LayerModes weakTensorModes = computeUniformModes(
        weaklyAnisotropic,
        Tensor3::isotropic({1.0, 0.0}),
        Kx,
        Ky,
        0.1,
        1.0);
    REQUIRE(weakTensorModes.kind == LayerComputationKind::UniformTmm);
    REQUIRE_FALSE(weakTensorModes.usesAnalyticScalarBasis);

    // epsilon=mu=-1 has the same wave impedance as vacuum.  A finite slab is
    // therefore exactly reflectionless even though its forward phase wave has
    // negative kz.  This fails if sqrt(kz^2)'s positive real root is blindly
    // labelled as the forward TMM mode.
    RcwaSolver solver;
    solver.setWavelength(1.0);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(
        Material::constant(
            "impedance-matched negative index",
            {-1.0, 0.0},
            {-1.0, 0.0}),
        0.37);
    for (const Polarization polarization : {Polarization::TE, Polarization::TM}) {
        solver.setPolarization(polarization);
        const auto result = solver.solveSpectrumOnly();
        REQUIRE(result.rTotal == Catch::Approx(0.0).margin(2e-12));
        REQUIRE(result.tTotal == Catch::Approx(1.0).margin(2e-12));
        REQUIRE(result.conservation == Catch::Approx(1.0).margin(2e-12));
    }
}

TEST_CASE("uniform anisotropic modes match periodic tensor route") {
    using namespace rcwa;

    const Real wavelengthUm = 0.74;
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.18, -0.07, wavelengthUm, wavelengthUm, basis);

    Tensor3 eps = Tensor3::diagonal({2.25, 0.01}, {2.56, 0.02}, {2.89, 0.015});
    eps(0, 1) = {0.04, 0.003};
    eps(1, 0) = {0.04, 0.003};
    eps(0, 2) = {0.09, -0.002};
    eps(2, 0) = {0.09, -0.002};
    eps(1, 2) = {-0.03, 0.001};
    eps(2, 1) = {-0.03, 0.001};

    Tensor3 mu = Tensor3::diagonal({1.04, 0.0}, {1.08, 0.0}, {0.97, 0.0});
    mu(0, 2) = {0.02, 0.0};
    mu(2, 0) = {0.02, 0.0};

    const LayerModes uniformModes =
        computeUniformModes(eps, mu, Kx, Ky, 0.19, wavelengthUm);
    const TensorFourierMatrices tensors = tensorFourierMatricesUniform(eps, mu, basis);
    const LayerModes periodicModes =
        computePeriodicModes(tensors, Kx, Ky, 0.19, wavelengthUm);

    requireComplexMultisetClose(uniformModes.gamma, periodicModes.gamma, 2e-9);

    const Tensor3 vacuum = Tensor3::isotropic({1.0, 0.0});
    const Tensor3 substrateEps = Tensor3::isotropic({2.13, 0.015});
    const Tensor3 exteriorMu = Tensor3::isotropic({1.0, 0.0});
    const LayerModes superstrate = computeUniformModes(
        vacuum, exteriorMu, Kx, Ky, 0.0, wavelengthUm);
    const LayerModes substrate = computeUniformModes(
        substrateEps, exteriorMu, Kx, Ky, 0.0, wavelengthUm);
    const std::vector<LayerModes> uniformMedia{
        superstrate, uniformModes, substrate};
    const std::vector<LayerModes> periodicMedia{
        superstrate, periodicModes, substrate};

    const SMatrix uniformS = computeStackSmatrix(uniformMedia);
    const SMatrix periodicS = computeStackSmatrix(periodicMedia);
    requireMatrixClose(uniformS.S11, periodicS.S11, 3e-8);
    requireMatrixClose(uniformS.S12, periodicS.S12, 3e-8);
    requireMatrixClose(uniformS.S21, periodicS.S21, 3e-8);
    requireMatrixClose(uniformS.S22, periodicS.S22, 3e-8);

    const std::vector<Polarization> polarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP,
    };
    const auto uniformTotals = computeStackTotalsForPolarizations(
        uniformMedia,
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    const auto periodicTotals = computeStackTotalsForPolarizations(
        periodicMedia,
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    REQUIRE(uniformTotals.size() == periodicTotals.size());
    for (std::size_t index = 0; index < uniformTotals.size(); ++index) {
        REQUIRE(uniformTotals[index].R0 ==
                Catch::Approx(periodicTotals[index].R0).margin(3e-8));
        REQUIRE(uniformTotals[index].T0 ==
                Catch::Approx(periodicTotals[index].T0).margin(3e-8));
        REQUIRE(uniformTotals[index].rTotal ==
                Catch::Approx(periodicTotals[index].rTotal).margin(3e-8));
        REQUIRE(uniformTotals[index].tTotal ==
                Catch::Approx(periodicTotals[index].tTotal).margin(3e-8));
    }

    const auto makePrepared = [&](const LayerModes& finiteModes) {
        PreparedStack stack;
        stack.Kx = Kx;
        stack.Ky = Ky;
        stack.media = {superstrate, finiteModes, substrate};
        stack.basis = basis;
        stack.totalHarmonics = static_cast<int>(basis.size());
        stack.centerIdx = basis.centerIndex();
        stack.lattice = {1.0, 1.0, true};
        return stack;
    };
    const PreparedStack uniformStack = makePrepared(uniformModes);
    const PreparedStack periodicStack = makePrepared(periodicModes);
    const auto physicalStateAtDepth = [&](const LayerModes& modes,
                                          const StackAmplitudes& amplitudes,
                                          Real depthUm) {
        const std::size_t harmonics = basis.size();
        const std::size_t directionalModes = 2 * harmonics;
        Matrix modal(4 * harmonics, 1);
        const Real downDistance = twoPi * depthUm / wavelengthUm;
        const Real upDistance = twoPi * (Real{0.19} - depthUm) / wavelengthUm;
        for (std::size_t mode = 0; mode < directionalModes; ++mode) {
            modal(mode, 0) = amplitudes.layerDownTop.front()(mode, 0) *
                std::exp(iu * modes.gamma[mode] * downDistance);
            const std::size_t upMode = directionalModes + mode;
            modal(upMode, 0) = amplitudes.layerUpBottom.front()(mode, 0) *
                std::exp(iu * (-modes.gamma[upMode]) * upDistance);
        }
        return materializeModeMatrix(modes) * modal;
    };
    for (const Polarization polarization : {Polarization::TE, Polarization::TM}) {
        const Matrix incident = makeIncidentAmplitudes(
            basis.size(), basis.centerIndex(), polarization);
        const StackAmplitudes uniformAmplitudes =
            solveStackAmplitudes(uniformStack, incident);
        const StackAmplitudes periodicAmplitudes =
            solveStackAmplitudes(periodicStack, incident);
        for (const Real depthUm : {Real{0.0}, Real{0.071}, Real{0.19}}) {
            requireMatrixClose(
                physicalStateAtDepth(uniformModes, uniformAmplitudes, depthUm),
                physicalStateAtDepth(periodicModes, periodicAmplitudes, depthUm),
                6e-8);
        }
    }
}

TEST_CASE("reduced uniform Berreman modes preserve the full 4x4 eigensystem") {
    using namespace rcwa;

    const Real wavelengthUm = 0.83;
    const auto basis = makeHarmonicBasisOrders(
        2, 1, LatticeTruncation::Parallelogramic);
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.16,
        -0.09,
        wavelengthUm / 1.17,
        wavelengthUm / 0.91,
        basis);
    Tensor3 eps = Tensor3::diagonal(
        {2.41, 0.03}, {2.78, 0.02}, {3.16, 0.015});
    eps(0, 1) = {0.06, 0.08};
    eps(1, 0) = {-0.04, -0.07};
    const Tensor3 mu = Tensor3::isotropic({1.07, 0.01});

    const LayerModes reduced =
        computeUniformModes(eps, mu, Kx, Ky, 0.27, wavelengthUm);
    REQUIRE(reduced.kind == LayerComputationKind::UniformTmm);
    REQUIRE_FALSE(reduced.usesAnalyticScalarBasis);
    const std::size_t N = basis.size();
    REQUIRE(reduced.gamma.size() == 4 * N);

    for (std::size_t harmonic = 0; harmonic < N; ++harmonic) {
        const Matrix state = berremanStateMatrix(
            eps,
            mu,
            Kx(harmonic, harmonic),
            Ky(harmonic, harmonic));
        const Eigensystem full = eig(state);
        std::vector<Complex> localGamma(4);
        for (std::size_t slot = 0; slot < 4; ++slot) {
            const std::size_t column = slot * N + harmonic;
            localGamma[slot] = reduced.gamma[column];
            for (std::size_t row = 0; row < 4; ++row) {
                Complex residual{};
                for (std::size_t inner = 0; inner < 4; ++inner) {
                    residual += state(row, inner) * modeStateValue(
                        reduced,
                        inner * N + harmonic,
                        column);
                }
                residual -= localGamma[slot] * modeStateValue(
                    reduced,
                    row * N + harmonic,
                    column);
                REQUIRE(std::abs(residual) <= 3e-10);
            }
        }
        requireComplexMultisetClose(localGamma, full.values, 3e-10);
    }
}

TEST_CASE("y-mirror periodic eigenproblems split generically and fall back strictly") {
    using namespace rcwa;

    const Real wavelengthUm = 0.78;
    const Real periodXUm = 1.20;
    const Real periodYUm = 0.90;
    const auto basis =
        makeHarmonicBasisOrders(2, 2, LatticeTruncation::Circular);
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.11,
        0.0,
        wavelengthUm / periodXUm,
        wavelengthUm / periodYUm,
        basis);

    // The x offset deliberately prevents relying on x symmetry. Only the
    // centered y coordinate and unrotated rectangle provide y-mirror symmetry.
    PatternedLayer2D symmetricLayer;
    symmetricLayer.background =
        Material::constant("generic background", {4.4, 0.03});
    symmetricLayer.periodXUm = periodXUm;
    symmetricLayer.periodYUm = periodYUm;
    symmetricLayer.thicknessUm = 0.23;
    symmetricLayer.regions.push_back(PatternRegion::rectangle(
        Material::constant("generic inclusion", {1.7, 0.01}),
        {0.13, 0.0},
        0.0,
        {0.18, 0.12}));

    const TensorFourierMatrices tensors =
        tensorFourierMatricesPatternedLayer2d(
            symmetricLayer,
            basis,
            wavelengthUm);
    const LayerModes split = computePeriodicModes(
        tensors,
        Kx,
        Ky,
        symmetricLayer.thicknessUm,
        wavelengthUm);
    const LayerModes full = computePeriodicModes(
        tensors,
        Kx,
        Ky,
        symmetricLayer.thicknessUm,
        wavelengthUm,
        false);
    REQUIRE(split.usedYMirrorSymmetrySplit);
    REQUIRE_FALSE(full.usedYMirrorSymmetrySplit);
    requireComplexMultisetClose(split.gamma, full.gamma, 2e-9);

    const Tensor3 vacuum = Tensor3::isotropic({1.0, 0.0});
    const Tensor3 glass = Tensor3::isotropic({2.25, 0.0});
    const Tensor3 mu = Tensor3::isotropic({1.0, 0.0});
    const LayerModes superstrate =
        computeUniformModes(vacuum, mu, Kx, Ky, 0.0, wavelengthUm);
    const LayerModes substrate =
        computeUniformModes(glass, mu, Kx, Ky, 0.0, wavelengthUm);
    const std::vector<Polarization> polarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP,
    };
    const auto splitTotals = computeStackTotalsForPolarizations(
        {superstrate, split, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    const auto fullTotals = computeStackTotalsForPolarizations(
        {superstrate, full, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    REQUIRE(splitTotals.size() == fullTotals.size());
    for (std::size_t i = 0; i < splitTotals.size(); ++i) {
        REQUIRE(splitTotals[i].R0 ==
                Catch::Approx(fullTotals[i].R0).margin(2e-9));
        REQUIRE(splitTotals[i].T0 ==
                Catch::Approx(fullTotals[i].T0).margin(2e-9));
        REQUIRE(splitTotals[i].rTotal ==
                Catch::Approx(fullTotals[i].rTotal).margin(2e-9));
        REQUIRE(splitTotals[i].tTotal ==
                Catch::Approx(fullTotals[i].tTotal).margin(2e-9));
    }

    const auto [offPlaneKx, offPlaneKy] = makeTransverseWavevectorOperators(
        0.11,
        0.025,
        wavelengthUm / periodXUm,
        wavelengthUm / periodYUm,
        basis);
    const LayerModes offPlane = computePeriodicModes(
        tensors,
        offPlaneKx,
        offPlaneKy,
        symmetricLayer.thicknessUm,
        wavelengthUm);
    REQUIRE_FALSE(offPlane.usedYMirrorSymmetrySplit);

    TensorFourierMatrices perturbed = tensors;
    std::size_t positiveY = basis.size();
    for (std::size_t i = 0; i < basis.orders.size(); ++i) {
        if (basis.orders[i].n > 0) {
            positiveY = i;
            break;
        }
    }
    REQUIRE(positiveY < basis.size());
    perturbed.liEpsQ[0][0](positiveY, positiveY) +=
        Complex{1e-5, 2e-6};
    const LayerModes nonMirrorFourier = computePeriodicModes(
        perturbed,
        Kx,
        Ky,
        symmetricLayer.thicknessUm,
        wavelengthUm);
    REQUIRE_FALSE(nonMirrorFourier.usedYMirrorSymmetrySplit);
}

TEST_CASE("spatially transformed layer modes reproduce opposite-angle solves") {
    using namespace rcwa;

    const Real wavelengthUm = 0.79;
    const Real periodXUm = 1.13;
    const Real periodYUm = 0.94;
    const auto basis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Circular);
    const auto [sourceKx, sourceKy] = makeTransverseWavevectorOperators(
        0.14,
        0.0,
        wavelengthUm / periodXUm,
        wavelengthUm / periodYUm,
        basis);
    const auto [targetKx, targetKy] = makeTransverseWavevectorOperators(
        -0.14,
        0.0,
        wavelengthUm / periodXUm,
        wavelengthUm / periodYUm,
        basis);

    PatternedLayer2D centered;
    centered.background = Material::constant("mirror background", {4.3, 0.02});
    centered.periodXUm = periodXUm;
    centered.periodYUm = periodYUm;
    centered.thicknessUm = 0.31;
    centered.regions.push_back(PatternRegion::rectangle(
        Material::constant("mirror inclusion", {1.6, 0.01}),
        {0.0, 0.0},
        0.0,
        {0.19, 0.14}));
    const TensorFourierMatrices tensors =
        tensorFourierMatricesPatternedLayer2d(centered, basis, wavelengthUm);
    const LayerModes source = computePeriodicModes(
        tensors,
        sourceKx,
        sourceKy,
        centered.thicknessUm,
        wavelengthUm);
    const LayerModes directTarget = computePeriodicModes(
        tensors,
        targetKx,
        targetKy,
        centered.thicknessUm,
        wavelengthUm);
    const auto mirroredTarget = tryMakeXMirroredPeriodicModes(
        source, tensors, sourceKx, sourceKy, targetKx, targetKy);
    REQUIRE(mirroredTarget.has_value());
    requireComplexMultisetClose(
        mirroredTarget->gamma, directTarget.gamma, 3e-9);

    const Tensor3 vacuum = Tensor3::isotropic({1.0, 0.0});
    const Tensor3 substrateEps = Tensor3::isotropic({2.31, 0.01});
    const Tensor3 mu = Tensor3::isotropic({1.0, 0.0});
    const LayerModes superstrate = computeUniformModes(
        vacuum, mu, targetKx, targetKy, 0.0, wavelengthUm);
    const LayerModes substrate = computeUniformModes(
        substrateEps, mu, targetKx, targetKy, 0.0, wavelengthUm);
    const std::vector<Polarization> polarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP,
    };
    const TransverseSymmetryTransform xMirrorTransform{
        Real{-1},
        Real{1},
        std::array<Real, 4>{Real{-1}, Real{1}, Real{1}, Real{-1}},
        std::array<Real, 3>{Real{-1}, Real{1}, Real{1}},
    };
    Tensor3 uniformEps =
        Tensor3::diagonal({2.24, 0.03}, {2.61, 0.02}, {2.89, 0.01});
    uniformEps(1, 2) = {0.035, -0.008};
    uniformEps(2, 1) = {-0.025, 0.006};
    const Tensor3 uniformMu = Tensor3::isotropic({1.0, 0.0});
    const LayerModes uniformSource = computeUniformModes(
        uniformEps,
        uniformMu,
        sourceKx,
        sourceKy,
        0.23,
        wavelengthUm);
    const LayerModes directUniformTarget = computeUniformModes(
        uniformEps,
        uniformMu,
        targetKx,
        targetKy,
        0.23,
        wavelengthUm);
    const auto mirroredUniformTarget = tryMakeTransverseSymmetryUniformModes(
        uniformSource,
        uniformEps,
        uniformMu,
        sourceKx,
        sourceKy,
        targetKx,
        targetKy,
        xMirrorTransform);
    REQUIRE(mirroredUniformTarget.has_value());
    requireComplexMultisetClose(
        mirroredUniformTarget->gamma, directUniformTarget.gamma, 3e-9);

    const auto mirroredUniformTotals = computeStackTotalsForPolarizations(
        {superstrate, *mirroredUniformTarget, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    const auto directUniformTotals = computeStackTotalsForPolarizations(
        {superstrate, directUniformTarget, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    REQUIRE(mirroredUniformTotals.size() == directUniformTotals.size());
    for (std::size_t i = 0; i < directUniformTotals.size(); ++i) {
        REQUIRE(mirroredUniformTotals[i].R0 ==
                Catch::Approx(directUniformTotals[i].R0).margin(3e-9));
        REQUIRE(mirroredUniformTotals[i].T0 ==
                Catch::Approx(directUniformTotals[i].T0).margin(3e-9));
        REQUIRE(mirroredUniformTotals[i].rTotal ==
                Catch::Approx(directUniformTotals[i].rTotal).margin(3e-9));
        REQUIRE(mirroredUniformTotals[i].tTotal ==
                Catch::Approx(directUniformTotals[i].tTotal).margin(3e-9));
    }
    Tensor3 xOddEps = uniformEps;
    xOddEps(0, 1) = {0.04, 0.0};
    REQUIRE_FALSE(tryMakeTransverseSymmetryUniformModes(
        uniformSource,
        xOddEps,
        uniformMu,
        sourceKx,
        sourceKy,
        targetKx,
        targetKy,
        xMirrorTransform));

    const auto mirroredTotals = computeStackTotalsForPolarizations(
        {superstrate, *mirroredTarget, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    const auto directTotals = computeStackTotalsForPolarizations(
        {superstrate, directTarget, substrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    REQUIRE(mirroredTotals.size() == directTotals.size());
    for (std::size_t i = 0; i < directTotals.size(); ++i) {
        REQUIRE(mirroredTotals[i].R0 ==
                Catch::Approx(directTotals[i].R0).margin(3e-9));
        REQUIRE(mirroredTotals[i].T0 ==
                Catch::Approx(directTotals[i].T0).margin(3e-9));
        REQUIRE(mirroredTotals[i].rTotal ==
                Catch::Approx(directTotals[i].rTotal).margin(3e-9));
        REQUIRE(mirroredTotals[i].tTotal ==
                Catch::Approx(directTotals[i].tTotal).margin(3e-9));
    }

    PatternedLayer2D shifted = centered;
    shifted.regions.clear();
    shifted.regions.push_back(PatternRegion::rectangle(
        Material::constant("mirror inclusion", {1.6, 0.01}),
        {0.11, 0.0},
        0.0,
        {0.19, 0.14}));
    const TensorFourierMatrices shiftedTensors =
        tensorFourierMatricesPatternedLayer2d(shifted, basis, wavelengthUm);
    REQUIRE_FALSE(tryMakeXMirroredPeriodicModes(
        source, shiftedTensors, sourceKx, sourceKy, targetKx, targetKy));

    const auto [conicalSourceKx, conicalSourceKy] =
        makeTransverseWavevectorOperators(
            0.14,
            0.09,
            wavelengthUm / periodXUm,
            wavelengthUm / periodYUm,
            basis);
    const auto [invertedTargetKx, invertedTargetKy] =
        makeTransverseWavevectorOperators(
            -0.14,
            -0.09,
            wavelengthUm / periodXUm,
            wavelengthUm / periodYUm,
            basis);
    const LayerModes conicalSource = computePeriodicModes(
        tensors,
        conicalSourceKx,
        conicalSourceKy,
        centered.thicknessUm,
        wavelengthUm);
    const LayerModes directInvertedTarget = computePeriodicModes(
        tensors,
        invertedTargetKx,
        invertedTargetKy,
        centered.thicknessUm,
        wavelengthUm);
    const auto invertedTarget = tryMakeInPlaneInvertedPeriodicModes(
        conicalSource,
        tensors,
        conicalSourceKx,
        conicalSourceKy,
        invertedTargetKx,
        invertedTargetKy);
    REQUIRE(invertedTarget.has_value());
    requireComplexMultisetClose(
        invertedTarget->gamma, directInvertedTarget.gamma, 3e-9);

    const LayerModes invertedSuperstrate = computeUniformModes(
        vacuum, mu, invertedTargetKx, invertedTargetKy, 0.0, wavelengthUm);
    const LayerModes invertedSubstrate = computeUniformModes(
        substrateEps,
        mu,
        invertedTargetKx,
        invertedTargetKy,
        0.0,
        wavelengthUm);
    const auto invertedTotals = computeStackTotalsForPolarizations(
        {invertedSuperstrate, *invertedTarget, invertedSubstrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    const auto directInvertedTotals = computeStackTotalsForPolarizations(
        {invertedSuperstrate, directInvertedTarget, invertedSubstrate},
        static_cast<int>(basis.size()),
        basis.centerIndex(),
        polarizations);
    REQUIRE(invertedTotals.size() == directInvertedTotals.size());
    for (std::size_t i = 0; i < directInvertedTotals.size(); ++i) {
        REQUIRE(invertedTotals[i].R0 ==
                Catch::Approx(directInvertedTotals[i].R0).margin(3e-9));
        REQUIRE(invertedTotals[i].T0 ==
                Catch::Approx(directInvertedTotals[i].T0).margin(3e-9));
        REQUIRE(invertedTotals[i].rTotal ==
                Catch::Approx(directInvertedTotals[i].rTotal).margin(3e-9));
        REQUIRE(invertedTotals[i].tTotal ==
                Catch::Approx(directInvertedTotals[i].tTotal).margin(3e-9));
    }
    REQUIRE_FALSE(tryMakeInPlaneInvertedPeriodicModes(
        conicalSource,
        shiftedTensors,
        conicalSourceKx,
        conicalSourceKy,
        invertedTargetKx,
        invertedTargetKy));
}

TEST_CASE("compact uniform modes preserve dense stack scattering results") {
    using namespace rcwa;

    Tensor3 leftEps =
        Tensor3::diagonal({2.25, 0.03}, {2.65, 0.02}, {2.95, 0.01});
    leftEps(0, 1) = {0.07, 0.02};
    leftEps(1, 0) = {-0.04, -0.01};
    leftEps(0, 2) = {0.08, -0.01};
    leftEps(2, 0) = {0.08, -0.01};

    Tensor3 rightEps =
        Tensor3::diagonal({3.10, 0.04}, {2.80, 0.03}, {3.35, 0.02});
    rightEps(0, 1) = {-0.05, 0.015};
    rightEps(1, 0) = {0.03, -0.01};

    PatternedLayer2D patterned;
    patterned.background = Material::constant("pattern background", {5.2, 0.04});
    patterned.periodXUm = 1.1;
    patterned.periodYUm = 0.9;
    patterned.thicknessUm = 0.24;
    patterned.regions.push_back(PatternRegion::rectangle(
        Material::vacuum(),
        {0.03, -0.04},
        13.0,
        {0.21, 0.16}));

    std::vector<LayerSpec> layers;
    layers.push_back(UniformLayer{
        Material::anisotropic("left tensor", leftEps),
        0.17});
    layers.push_back(patterned);
    layers.push_back(UniformLayer{
        Material::anisotropic("right tensor", rightEps),
        0.13});

    const StackBuilder builder(
        1,
        2,
        2,
        false,
        LatticeTruncation::Circular,
        layers);
    PreparedStack compact = builder.prepare(
        0.83,
        19.0,
        11.0,
        Material::vacuum(),
        Material::fromIndex("glass", {1.47, 0.01}));

    REQUIRE(hasCompactUniformModes(compact.media.front()));
    REQUIRE(compact.media.front().W.empty());
    REQUIRE(compact.media.front().uniformBlocks.size() ==
            static_cast<std::size_t>(compact.totalHarmonics));
    const Matrix materialized = materializeModeMatrix(compact.media.front());
    REQUIRE(materialized.rows() ==
            4 * static_cast<std::size_t>(compact.totalHarmonics));
    REQUIRE(materialized.cols() == materialized.rows());

    PreparedStack dense = compact;
    for (LayerModes& modes : dense.media) {
        if (!hasCompactUniformModes(modes)) {
            continue;
        }
        modes.W = materializeModeMatrix(modes);
        modes.uniformBlocks.clear();
    }

    const std::vector<Polarization> polarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP,
    };
    const auto compactTotals = computeStackTotalsForPolarizations(
        compact.media,
        compact.totalHarmonics,
        compact.centerIdx,
        polarizations);
    const auto denseTotals = computeStackTotalsForPolarizations(
        dense.media,
        dense.totalHarmonics,
        dense.centerIdx,
        polarizations);
    REQUIRE(compactTotals.size() == denseTotals.size());
    for (std::size_t i = 0; i < compactTotals.size(); ++i) {
        REQUIRE(compactTotals[i].R0 ==
                Catch::Approx(denseTotals[i].R0).margin(2e-10));
        REQUIRE(compactTotals[i].T0 ==
                Catch::Approx(denseTotals[i].T0).margin(2e-10));
        REQUIRE(compactTotals[i].rTotal ==
                Catch::Approx(denseTotals[i].rTotal).margin(2e-10));
        REQUIRE(compactTotals[i].tTotal ==
                Catch::Approx(denseTotals[i].tTotal).margin(2e-10));
    }

    const SMatrix compactS = computeStackSmatrix(compact.media);
    const SMatrix denseS = computeStackSmatrix(dense.media);
    requireMatrixClose(compactS.S11, denseS.S11, 5e-9);
    requireMatrixClose(compactS.S12, denseS.S12, 5e-9);
    requireMatrixClose(compactS.S21, denseS.S21, 5e-9);
    requireMatrixClose(compactS.S22, denseS.S22, 5e-9);
}

TEST_CASE("compact Schur route matches dense route for the WSM square-hole stack") {
    using namespace rcwa;

    BlackPhosphorusModelParameters bp;
    bp.carrierDensityCm2 = 3.0e13;
    bp.scatteringRateEv = 0.010;
    bp.thicknessM = 0.5e-9;
    bp.twistDeg = 80.0;

    PhononPolaritonModelParameters moo3;
    moo3.epsilonInf = {4.0, 5.2, 2.4};
    moo3.transverseWavenumberCmInv = {818.0, 545.0, 962.0};
    moo3.longitudinalWavenumberCmInv = {974.0, 851.0, 1010.0};
    moo3.dampingWavenumberCmInv = {4.0, 4.0, 2.0};

    WsmModelParameters wsm;
    wsm.nodeSeparationMInv = 2.0e9;
    wsm.fermiVelocityMS = 0.83e5;
    wsm.cutoffXi = 3.0;
    wsm.backgroundEpsilon = 6.2;
    wsm.temperatureK = 300.0;
    wsm.degeneracy = 2.0;
    wsm.relaxationTimeS = 1000.0e-15;
    wsm.fermiEnergyEv = 0.15;

    DrudeModelParameters silver;
    silver.epsInf = 3.4;
    silver.plasmaFrequencyRadS = 1.39e16;
    silver.dampingRateRadS = 2.7e13;

    PatternedLayer2D squareHole;
    squareHole.background = Material::constant("Ge", {16.0, 0.0});
    squareHole.periodXUm = 5.54;
    squareHole.periodYUm = 5.54;
    squareHole.thicknessUm = 2.8;
    squareHole.regions.push_back(PatternRegion::rectangle(
        Material::air(),
        {0.0, 0.0},
        0.0,
        {0.77, 0.77}));

    std::vector<LayerSpec> layers;
    layers.push_back(UniformLayer{
        builtinMaterialModel("BP", "BP", bp),
        0.0005});
    layers.push_back(squareHole);
    layers.push_back(UniformLayer{
        builtinMaterialModel("MoO3", "PhononPolariton", moo3),
        0.3});
    layers.push_back(UniformLayer{
        builtinMaterialModel("WSM", "WSM", wsm),
        1.6});
    layers.push_back(UniformLayer{
        builtinMaterialModel("Ag", "Drude", silver),
        0.5});

    const StackBuilder builder(
        1,
        5,
        5,
        false,
        LatticeTruncation::Circular,
        layers);
    for (const Real thetaDeg : {Real{1.0}, Real{-1.0}}) {
        PreparedStack compact = builder.prepare(
            12.345,
            thetaDeg,
            0.0,
            Material::air(),
            Material::air());
        PreparedStack dense = compact;
        for (LayerModes& modes : dense.media) {
            if (!hasCompactUniformModes(modes)) {
                continue;
            }
            modes.W = materializeModeMatrix(modes);
            modes.uniformBlocks.clear();
        }

        const std::vector<Polarization> polarizations{
            Polarization::TE,
            Polarization::TM,
        };
        const auto compactTotals = computeStackTotalsForPolarizations(
            compact.media,
            compact.totalHarmonics,
            compact.centerIdx,
            polarizations);
        const auto denseTotals = computeStackTotalsForPolarizations(
            dense.media,
            dense.totalHarmonics,
            dense.centerIdx,
            polarizations);
        REQUIRE(compactTotals.size() == denseTotals.size());
        for (std::size_t i = 0; i < compactTotals.size(); ++i) {
            REQUIRE(compactTotals[i].R0 ==
                    Catch::Approx(denseTotals[i].R0).margin(5e-11));
            REQUIRE(compactTotals[i].T0 ==
                    Catch::Approx(denseTotals[i].T0).margin(5e-11));
            REQUIRE(compactTotals[i].rTotal ==
                    Catch::Approx(denseTotals[i].rTotal).margin(5e-11));
            REQUIRE(compactTotals[i].tTotal ==
                    Catch::Approx(denseTotals[i].tTotal).margin(5e-11));
        }
    }
}

TEST_CASE("anisotropic spectra use the spectrum-only path consistently") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.10, 0.02}, {2.45, 0.01}, {2.80, 0.03});
    eps(0, 1) = {0.05, 0.01};
    eps(1, 0) = {-0.05, -0.01};
    eps(0, 2) = {0.08, 0.00};
    eps(2, 0) = {0.08, 0.00};

    RcwaSolver solver;
    solver.setWavelength(1.15);
    solver.setIncidence(17.0, 9.0);
    solver.setHarmonicOrders(2, 2, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("glass", {1.45, 0.0}));
    solver.setPolarization(Polarization::TE);
    solver.addUniformLayer(Material::anisotropic("gyrotropic slab", eps), 0.35);
    solver.addUniformLayer(Material::constant("lossy cap", {2.25, 0.08}), 0.12);

    const auto viaSolve = solver.solve();
    const auto te = solver.solveSpectrumOnly();
    REQUIRE(te.N == viaSolve.N);
    REQUIRE(te.rTotal == Catch::Approx(viaSolve.rTotal).margin(1e-12));
    REQUIRE(te.tTotal == Catch::Approx(viaSolve.tTotal).margin(1e-12));
    REQUIRE(std::isfinite(te.conservation));

    const auto both = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(both.size() == 2);
    REQUIRE(both[0].rTotal == Catch::Approx(te.rTotal).margin(1e-12));
    REQUIRE(both[0].tTotal == Catch::Approx(te.tTotal).margin(1e-12));

    solver.setPolarization(Polarization::TM);
    const auto tm = solver.solveSpectrumOnly();
    REQUIRE(both[1].rTotal == Catch::Approx(tm.rTotal).margin(1e-12));
    REQUIRE(both[1].tTotal == Catch::Approx(tm.tTotal).margin(1e-12));

    const auto batch = solver.solveSpectrumBatchOnly(
        std::vector<Real>{1.10, 1.15},
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(batch.size() == 4);
    REQUIRE(batch[2].rTotal == Catch::Approx(both[0].rTotal).margin(1e-12));
    REQUIRE(batch[2].tTotal == Catch::Approx(both[0].tTotal).margin(1e-12));
    REQUIRE(batch[3].rTotal == Catch::Approx(both[1].rTotal).margin(1e-12));
    REQUIRE(batch[3].tTotal == Catch::Approx(both[1].tTotal).margin(1e-12));
}

TEST_CASE("uniform WSM slab matches the closed-form phi-zero transfer matrix") {
    using namespace rcwa;

    WsmModelParameters parameters;
    parameters.nodeSeparationMInv = -2.0e9;
    parameters.fermiVelocityMS = 0.83e5;
    parameters.cutoffXi = 3.0;
    parameters.backgroundEpsilon = 6.2;
    parameters.temperatureK = 300.0;
    parameters.degeneracy = 2.0;
    parameters.relaxationTimeS = 1000.0e-15;
    parameters.fermiEnergyEv = 0.15;

    RcwaSolver solver;
    solver.setWavelength(16.9);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(
        builtinMaterialModel("WSM", "WSM", parameters),
        2.0);

    const auto solveAt = [&](Real thetaDeg, Polarization polarization) {
        solver.setIncidence(thetaDeg, 0.0);
        solver.setPolarization(polarization);
        return solver.solveSpectrumOnly();
    };
    const auto teForward = solveAt(5.0, Polarization::TE);
    const auto teReverse = solveAt(-5.0, Polarization::TE);
    const auto tmForward = solveAt(5.0, Polarization::TM);
    const auto tmReverse = solveAt(-5.0, Polarization::TM);

    // Independent 2x2 matrix exponentials for the decoupled TE and TM
    // Maxwell systems at phi=0 give these power coefficients.
    REQUIRE(teForward.rTotal == Catch::Approx(0.994915594489).margin(2e-10));
    REQUIRE(teForward.tTotal == Catch::Approx(0.000037807022).margin(2e-10));
    REQUIRE(teReverse.rTotal == Catch::Approx(teForward.rTotal).margin(2e-12));
    REQUIRE(teReverse.tTotal == Catch::Approx(teForward.tTotal).margin(2e-12));
    REQUIRE(tmForward.rTotal == Catch::Approx(0.982837822128).margin(2e-10));
    REQUIRE(tmForward.tTotal == Catch::Approx(0.000567466733).margin(2e-10));
    REQUIRE(tmReverse.rTotal == Catch::Approx(0.983863243559).margin(2e-10));
    REQUIRE(tmReverse.tTotal == Catch::Approx(0.000567466733).margin(2e-10));

    WsmModelParameters oppositeParameters = parameters;
    oppositeParameters.nodeSeparationMInv = -parameters.nodeSeparationMInv;
    RcwaSolver oppositeSolver;
    oppositeSolver.setWavelength(16.9);
    oppositeSolver.setHarmonicOrders(0, 0, LatticeTruncation::Circular);
    oppositeSolver.setSuperstrate(Material::vacuum());
    oppositeSolver.setSubstrate(Material::vacuum());
    oppositeSolver.addUniformLayer(
        builtinMaterialModel("oppositely magnetized WSM", "WSM", oppositeParameters),
        2.0);
    const auto solveOppositeAt = [&](Real thetaDeg, Polarization polarization) {
        oppositeSolver.setIncidence(thetaDeg, 0.0);
        oppositeSolver.setPolarization(polarization);
        return oppositeSolver.solveSpectrumOnly();
    };
    const auto oppositeTeForward = solveOppositeAt(5.0, Polarization::TE);
    const auto oppositeTeReverse = solveOppositeAt(-5.0, Polarization::TE);
    const auto oppositeTmForward = solveOppositeAt(5.0, Polarization::TM);
    const auto oppositeTmReverse = solveOppositeAt(-5.0, Polarization::TM);

    // Onsager-Casimir symmetry requires S(theta, b) = S(-theta, -b)^T.
    // For this mirror-symmetric two-port slab it reduces to equality of the
    // exterior power coefficients under simultaneous angle and b reversal.
    REQUIRE(teForward.rTotal == Catch::Approx(oppositeTeReverse.rTotal).margin(2e-12));
    REQUIRE(teForward.tTotal == Catch::Approx(oppositeTeReverse.tTotal).margin(2e-12));
    REQUIRE(teReverse.rTotal == Catch::Approx(oppositeTeForward.rTotal).margin(2e-12));
    REQUIRE(teReverse.tTotal == Catch::Approx(oppositeTeForward.tTotal).margin(2e-12));
    REQUIRE(tmForward.rTotal == Catch::Approx(oppositeTmReverse.rTotal).margin(2e-12));
    REQUIRE(tmForward.tTotal == Catch::Approx(oppositeTmReverse.tTotal).margin(2e-12));
    REQUIRE(tmReverse.rTotal == Catch::Approx(oppositeTmForward.rTotal).margin(2e-12));
    REQUIRE(tmReverse.tTotal == Catch::Approx(oppositeTmForward.tTotal).margin(2e-12));

    WsmModelParameters reciprocalParameters = parameters;
    reciprocalParameters.nodeSeparationMInv = 0.0;
    RcwaSolver reciprocalSolver;
    reciprocalSolver.setWavelength(16.9);
    reciprocalSolver.setHarmonicOrders(0, 0, LatticeTruncation::Circular);
    reciprocalSolver.setSuperstrate(Material::vacuum());
    reciprocalSolver.setSubstrate(Material::vacuum());
    reciprocalSolver.addUniformLayer(
        builtinMaterialModel("zero-b WSM", "WSM", reciprocalParameters),
        2.0);
    reciprocalSolver.setPolarization(Polarization::TM);
    reciprocalSolver.setIncidence(5.0, 0.0);
    const auto reciprocalForward = reciprocalSolver.solveSpectrumOnly();
    reciprocalSolver.setIncidence(-5.0, 0.0);
    const auto reciprocalReverse = reciprocalSolver.solveSpectrumOnly();
    REQUIRE(reciprocalForward.rTotal ==
            Catch::Approx(reciprocalReverse.rTotal).margin(2e-12));
    REQUIRE(reciprocalForward.tTotal ==
            Catch::Approx(reciprocalReverse.tTotal).margin(2e-12));
}

TEST_CASE("uniform Si Ge WSM Ag stack matches the closed-form multilayer transfer matrix") {
    using namespace rcwa;

    WsmModelParameters wsm;
    wsm.nodeSeparationMInv = -2.0e9;
    wsm.fermiVelocityMS = 0.83e5;
    wsm.cutoffXi = 3.0;
    wsm.backgroundEpsilon = 6.2;
    wsm.temperatureK = 300.0;
    wsm.degeneracy = 2.0;
    wsm.relaxationTimeS = 1000.0e-15;
    wsm.fermiEnergyEv = 0.15;

    DrudeModelParameters silver;
    silver.epsInf = 3.4;
    silver.plasmaFrequencyRadS = 1.39e16;
    silver.dampingRateRadS = 2.7e13;

    RcwaSolver solver;
    solver.setWavelength(16.9);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::constant("Si", {3.48 * 3.48, 0.0}), 2.3);
    solver.addUniformLayer(Material::constant("Ge", {16.0, 0.0}), 2.3);
    solver.addUniformLayer(builtinMaterialModel("WSM", "WSM", wsm), 2.0);
    solver.addUniformLayer(builtinMaterialModel("Ag", "Drude", silver), 0.5);

    const auto solveAt = [&](Real thetaDeg, Polarization polarization) {
        solver.setIncidence(thetaDeg, 0.0);
        solver.setPolarization(polarization);
        return solver.solveSpectrumOnly();
    };
    const auto teForward = solveAt(5.0, Polarization::TE);
    const auto teReverse = solveAt(-5.0, Polarization::TE);
    const auto tmForward = solveAt(5.0, Polarization::TM);
    const auto tmReverse = solveAt(-5.0, Polarization::TM);

    // Independent cascaded 2x2 Maxwell matrix exponentials for the four
    // homogeneous layers give these exterior power coefficients.  The Ag
    // layer is over ten skin depths thick, so its physical transmission is
    // below the reliable dynamic range of a direct transfer-matrix product.
    REQUIRE(teForward.rTotal == Catch::Approx(0.993924603943).margin(2e-10));
    REQUIRE(teReverse.rTotal == Catch::Approx(teForward.rTotal).margin(2e-12));
    REQUIRE(tmForward.rTotal == Catch::Approx(0.979422852075).margin(2e-10));
    REQUIRE(tmReverse.rTotal == Catch::Approx(0.980800977772).margin(2e-10));
    REQUIRE(teForward.tTotal < 1e-20);
    REQUIRE(teReverse.tTotal < 1e-20);
    REQUIRE(tmForward.tTotal < 1e-20);
    REQUIRE(tmReverse.tTotal < 1e-20);
}

TEST_CASE("Wu WSM SiO2 multilayer matches the local direct Berreman stack") {
    using namespace rcwa;

    WsmModelParameters wsm;
    wsm.nodeSeparationMInv = 2.0e9;
    wsm.nodeSeparationDirection = {0.0, 1.0, 0.0};
    wsm.fermiVelocityMS = 0.83e5;
    wsm.cutoffXi = 3.0;
    wsm.backgroundEpsilon = 6.2;
    wsm.temperatureK = 300.0;
    wsm.degeneracy = 2.0;
    wsm.relaxationTimeS = 1000.0e-15;
    wsm.fermiEnergyEv = 0.15;

    DrudeModelParameters silver;
    silver.epsInf = 3.4;
    silver.plasmaFrequencyRadS = 1.39e16;
    silver.dampingRateRadS = 2.7e13;

    RcwaSolver solver;
    solver.setWavelength(9.882);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("substrate", {1.45, 0.0}));
    const Material wsmMaterial = builtinMaterialModel("WSM", "WSM", wsm);
    const Material silica = Material::fromIndex("SiO2", {1.45, 0.0});
    std::vector<DirectBerremanLayer> directLayers;
    for (std::size_t pair = 0; pair < 5; ++pair) {
        solver.addUniformLayer(wsmMaterial, 0.050);
        solver.addUniformLayer(silica, 3.652);
        const auto [wsmEpsilon, wsmMu] = wsmMaterial.tensors(9.882);
        const auto [silicaEpsilon, silicaMu] = silica.tensors(9.882);
        directLayers.push_back({wsmEpsilon, wsmMu, 0.050});
        directLayers.push_back({silicaEpsilon, silicaMu, 3.652});
    }
    const Material silverMaterial = builtinMaterialModel("Ag", "Drude", silver);
    solver.addUniformLayer(silverMaterial, 0.300);
    const auto [silverEpsilon, silverMu] = silverMaterial.tensors(9.882);
    directLayers.push_back({silverEpsilon, silverMu, 0.300});
    solver.setPolarization(Polarization::TM);

    solver.setIncidence(30.0, 0.0);
    const auto forward = solver.solveSpectrumOnly();
    const auto directForward = directBerremanStackPower(
        directLayers, 9.882, 30.0, 1.0, 1.45, Polarization::TM);
    REQUIRE(forward.rTotal == Catch::Approx(directForward.reflectance).margin(3e-11));
    REQUIRE(forward.tTotal == Catch::Approx(directForward.transmittance).margin(3e-16));
    REQUIRE(directForward.reflectance ==
            Catch::Approx(0.839782340557609).margin(3e-11));
    REQUIRE(directForward.transmittance ==
            Catch::Approx(2.17407305201711e-14).margin(3e-16));

    solver.setIncidence(-30.0, 0.0);
    const auto reverse = solver.solveSpectrumOnly();
    const auto directReverse = directBerremanStackPower(
        directLayers, 9.882, -30.0, 1.0, 1.45, Polarization::TM);
    REQUIRE(reverse.rTotal == Catch::Approx(directReverse.reflectance).margin(3e-11));
    REQUIRE(reverse.tTotal == Catch::Approx(directReverse.transmittance).margin(3e-16));
    REQUIRE(directReverse.reflectance ==
            Catch::Approx(0.196321891152458).margin(3e-11));
    REQUIRE(directReverse.transmittance ==
            Catch::Approx(5.93467228115177e-14).margin(3e-16));
    REQUIRE(std::abs(
        (Real{1.0} - forward.rTotal - forward.tTotal) -
        0.160217659442369) < 3e-11);
    REQUIRE(std::abs(
        (Real{1.0} - reverse.rTotal - reverse.tTotal) -
        0.803678108847482) < 3e-11);
}

TEST_CASE("circular polarization spectra use coherent TE/TM amplitudes") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.15, 0.04}, {2.30, 0.06}, {2.65, 0.02});
    eps(0, 1) = {0.08, 0.18};
    eps(1, 0) = {-0.02, -0.12};

    RcwaSolver solver;
    solver.setWavelength(1.05);
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::anisotropic("coherent tensor slab", eps), 0.42);

    const auto rows = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM, Polarization::Both,
         Polarization::LCP, Polarization::RCP});
    REQUIRE(rows.size() == 5);
    for (const auto& row : rows) {
        REQUIRE(std::isfinite(row.rTotal));
        REQUIRE(std::isfinite(row.tTotal));
        REQUIRE(std::isfinite(row.conservation));
    }

    const auto s = solver.solveSParameters();
    const auto expectedLcp = circularZeroOrderTotals(s, true);
    const auto expectedRcp = circularZeroOrderTotals(s, false);
    REQUIRE(rows[3].rTotal == Catch::Approx(expectedLcp.first).margin(1e-11));
    REQUIRE(rows[3].tTotal == Catch::Approx(expectedLcp.second).margin(1e-11));
    REQUIRE(rows[4].rTotal == Catch::Approx(expectedRcp.first).margin(1e-11));
    REQUIRE(rows[4].tTotal == Catch::Approx(expectedRcp.second).margin(1e-11));
    REQUIRE(std::abs(rows[3].tTotal - rows[4].tTotal) > 1e-7);

    solver.setPolarization(Polarization::LCP);
    const auto singleLcp = solver.solveSpectrumOnly();
    REQUIRE(singleLcp.rTotal == Catch::Approx(rows[3].rTotal).margin(1e-12));
    REQUIRE(singleLcp.tTotal == Catch::Approx(rows[3].tTotal).margin(1e-12));

    solver.setPolarization(Polarization::RCP);
    const auto singleRcp = solver.solveSpectrumOnly();
    REQUIRE(singleRcp.rTotal == Catch::Approx(rows[4].rTotal).margin(1e-12));
    REQUIRE(singleRcp.tTotal == Catch::Approx(rows[4].tTotal).margin(1e-12));
}

TEST_CASE("angle-polarization spectrum batches preserve requested pair rows") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.10, 0.03}, {2.40, 0.04}, {2.70, 0.02});
    eps(0, 1) = {0.05, 0.11};
    eps(1, 0) = {-0.04, -0.09};

    RcwaSolver solver;
    solver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addUniformLayer(Material::anisotropic("pair batch tensor slab", eps), 0.37);

    const std::vector<Real> wavelengths{1.02, 1.08, 1.14};
    const std::vector<SpectrumAnglePolarization> requests{
        {-4.0, Polarization::LCP},
        {0.0, Polarization::LCP},
        {0.0, Polarization::RCP},
        {4.0, Polarization::RCP},
        {4.0, Polarization::RCP},
    };
    const double phiDeg = 17.0;

    const auto rows = solver.solveSpectrumBatchTotalsForAnglePolarizations(
        requests,
        phiDeg,
        wavelengths,
        3);
    REQUIRE(rows.size() == requests.size() * wavelengths.size());

    for (std::size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
        for (std::size_t wavelengthIndex = 0; wavelengthIndex < wavelengths.size();
             ++wavelengthIndex) {
            const auto baseline = solver.solveSpectrumBatchTotalsForAngles(
                {requests[requestIndex].thetaDeg},
                phiDeg,
                {wavelengths[wavelengthIndex]},
                {requests[requestIndex].polarization},
                1);
            REQUIRE(baseline.size() == 1);

            const auto& row = rows[requestIndex * wavelengths.size() + wavelengthIndex];
            REQUIRE(row.wavelengthUm ==
                    Catch::Approx(wavelengths[wavelengthIndex]).margin(1e-12));
            REQUIRE(row.thetaDeg ==
                    Catch::Approx(requests[requestIndex].thetaDeg).margin(1e-12));
            REQUIRE(row.phiDeg == Catch::Approx(phiDeg).margin(1e-12));
            REQUIRE(row.rTotal == Catch::Approx(baseline[0].rTotal).margin(1e-12));
            REQUIRE(row.tTotal == Catch::Approx(baseline[0].tTotal).margin(1e-12));
            REQUIRE(row.conservation == Catch::Approx(baseline[0].conservation).margin(1e-12));
        }
    }

    const auto fullRows = solver.solveSpectrumBatchForAnglePolarizations(
        requests,
        phiDeg,
        wavelengths,
        2);
    REQUIRE(fullRows.size() == rows.size());
    REQUIRE(fullRows[0].rOrders.size() == 1);
    REQUIRE(fullRows[0].rTotal == Catch::Approx(rows[0].rTotal).margin(1e-12));
    REQUIRE(fullRows[0].tTotal == Catch::Approx(rows[0].tTotal).margin(1e-12));
}

TEST_CASE("uniform layer runs are handled by the generic local transfer path") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.15, 0.01}, {2.45, 0.02}, {2.75, 0.01});
    eps(0, 1) = {0.03, 0.0};
    eps(1, 0) = {-0.03, 0.0};

    RcwaSolver solver;
    solver.setWavelength(1.35);
    solver.setIncidence(21.0, 13.0);
    solver.setHarmonicOrders(2, 2, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    solver.setPolarization(Polarization::TE);
    solver.addUniformLayer(Material::constant("iso film", {2.25, 0.01}), 0.12);
    solver.addUniformLayer(Material::anisotropic("tensor film", eps), 0.21);
    solver.addUniformLayer(Material::constant("metal cap", {-12.0, 0.8}), 0.06);

    const auto te = solver.solveSpectrumOnly();
    REQUIRE(te.N == 13);
    REQUIRE(std::isfinite(te.rTotal));
    REQUIRE(std::isfinite(te.tTotal));
    REQUIRE(std::isfinite(te.conservation));

    const auto both = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(both.size() == 2);
    REQUIRE(both[0].rTotal == Catch::Approx(te.rTotal).margin(1e-12));
    REQUIRE(both[0].tTotal == Catch::Approx(te.tTotal).margin(1e-12));
    REQUIRE(std::isfinite(both[1].rTotal));
    REQUIRE(std::isfinite(both[1].tTotal));
}

TEST_CASE("full S-parameters share uniform run blocks with spectrum and zero-order solves") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.10, 0.01}, {2.42, 0.02}, {2.80, 0.01});
    eps(0, 1) = {0.04, 0.01};
    eps(1, 0) = {-0.02, -0.01};
    eps(0, 2) = {0.06, 0.0};
    eps(2, 0) = {0.06, 0.0};

    RcwaSolver solver;
    solver.setWavelength(0.91);
    solver.setIncidence(13.0, 7.0);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    solver.setPolarization(Polarization::TE);
    solver.addUniformLayer(Material::constant("front film", {2.25, 0.0}), 0.17);
    solver.addUniformLayer(Material::anisotropic("coupling film", eps), 0.11);
    solver.addUniformLayer(Material::constant("absorbing cap", {2.05, 0.03}), 0.08);

    const auto s = solver.solveSParameters();
    REQUIRE(s.N == 9);
    REQUIRE(s.orders.size() == static_cast<std::size_t>(s.N));
    const std::size_t N = static_cast<std::size_t>(s.N);
    const std::size_t channelCount = 2 * N;
    const std::size_t te0 = static_cast<std::size_t>(s.centerIdx);
    const std::size_t tm0 = N + te0;

    for (const Matrix* block : std::array<const Matrix*, 4>{&s.S11, &s.S12, &s.S21, &s.S22}) {
        REQUIRE(block->rows() == channelCount);
        REQUIRE(block->cols() == channelCount);
        for (const Complex value : block->data()) {
            REQUIRE(std::isfinite(std::real(value)));
            REQUIRE(std::isfinite(std::imag(value)));
        }
    }

    const auto zeroOrder = solver.solveZeroOrderAmplitudes();
    REQUIRE(zeroOrder.N == s.N);
    REQUIRE(zeroOrder.centerIdx == s.centerIdx);
    REQUIRE(std::abs(zeroOrder.R(0, 0) - s.S11(te0, te0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.R(0, 1) - s.S11(te0, tm0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.R(1, 0) - s.S11(tm0, te0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.R(1, 1) - s.S11(tm0, tm0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.T(0, 0) - s.S21(te0, te0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.T(0, 1) - s.S21(te0, tm0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.T(1, 0) - s.S21(tm0, te0)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(zeroOrder.T(1, 1) - s.S21(tm0, tm0)) ==
            Catch::Approx(0.0).margin(1e-12));

    for (std::size_t row = 0; row < channelCount; ++row) {
        if (row == te0 || row == tm0) {
            continue;
        }
        REQUIRE(std::abs(s.S11(row, te0)) == Catch::Approx(0.0).margin(1e-10));
        REQUIRE(std::abs(s.S11(row, tm0)) == Catch::Approx(0.0).margin(1e-10));
        REQUIRE(std::abs(s.S21(row, te0)) == Catch::Approx(0.0).margin(1e-10));
        REQUIRE(std::abs(s.S21(row, tm0)) == Catch::Approx(0.0).margin(1e-10));
    }

    const auto te = solver.solveSpectrumOnly();
    const double teR = std::norm(s.S11(te0, te0)) + std::norm(s.S11(tm0, te0));
    const double teT = std::norm(s.S21(te0, te0)) + std::norm(s.S21(tm0, te0));
    REQUIRE(te.rTotal == Catch::Approx(teR).margin(1e-10));
    REQUIRE(te.tTotal == Catch::Approx(teT).margin(1e-10));

    solver.setPolarization(Polarization::TM);
    const auto tm = solver.solveSpectrumOnly();
    const double tmR = std::norm(s.S11(te0, tm0)) + std::norm(s.S11(tm0, tm0));
    const double tmT = std::norm(s.S21(te0, tm0)) + std::norm(s.S21(tm0, tm0));
    REQUIRE(tm.rTotal == Catch::Approx(tmR).margin(1e-10));
    REQUIRE(tm.tTotal == Catch::Approx(tmT).margin(1e-10));
}

TEST_CASE("directional thermal channels are power-normalized scattering deficits") {
    using namespace rcwa;

    RcwaSolver lossless;
    lossless.setWavelength(1.05);
    lossless.setIncidence(23.0, 17.0);
    lossless.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    lossless.setSuperstrate(Material::vacuum());
    lossless.setSubstrate(Material::vacuum());
    lossless.addUniformLayer(Material::constant("lossless film", {2.25, 0.0}), 0.37);
    const auto losslessEmission = lossless.solveDirectionalThermalChannels();
    REQUIRE(losslessEmission.teAbsorptivity == Catch::Approx(0.0).margin(2e-12));
    REQUIRE(losslessEmission.tmAbsorptivity == Catch::Approx(0.0).margin(2e-12));
    REQUIRE(losslessEmission.teEmissivity == Catch::Approx(0.0).margin(2e-12));
    REQUIRE(losslessEmission.tmEmissivity == Catch::Approx(0.0).margin(2e-12));
    REQUIRE(losslessEmission.teIncidentScattering == Catch::Approx(1.0).margin(2e-12));
    REQUIRE(losslessEmission.tmIncidentScattering == Catch::Approx(1.0).margin(2e-12));
    REQUIRE(losslessEmission.teOutgoingScattering == Catch::Approx(1.0).margin(2e-12));
    REQUIRE(losslessEmission.tmOutgoingScattering == Catch::Approx(1.0).margin(2e-12));

    Tensor3 eps;
    eps(0, 0) = {2.30, 0.25};
    eps(0, 1) = {0.00, 0.35};
    eps(0, 2) = {0.18, 0.12};
    eps(1, 0) = {0.00, -0.12};
    eps(1, 1) = {2.00, 0.18};
    eps(1, 2) = {0.08, 0.06};
    eps(2, 0) = {-0.08, -0.07};
    eps(2, 1) = {0.03, 0.02};
    eps(2, 2) = {2.60, 0.20};

    RcwaSolver mixing;
    mixing.setWavelength(1.05);
    mixing.setIncidence(31.0, 23.0);
    mixing.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    mixing.setSuperstrate(Material::vacuum());
    mixing.setSubstrate(Material::vacuum());
    mixing.addUniformLayer(Material::anisotropic("absorbing channel mixer", eps), 0.37);

    const auto emission = mixing.solveDirectionalThermalChannels();
    const auto s = mixing.solveSParameters();
    REQUIRE(s.N == 1);
    const double teRowScattering =
        std::norm(s.S11(0, 0)) + std::norm(s.S11(0, 1)) +
        std::norm(s.S12(0, 0)) + std::norm(s.S12(0, 1));
    const double tmRowScattering =
        std::norm(s.S11(1, 0)) + std::norm(s.S11(1, 1)) +
        std::norm(s.S12(1, 0)) + std::norm(s.S12(1, 1));
    REQUIRE(emission.teOutgoingScattering ==
            Catch::Approx(teRowScattering).margin(2e-12));
    REQUIRE(emission.tmOutgoingScattering ==
            Catch::Approx(tmRowScattering).margin(2e-12));
    REQUIRE(emission.teEmissivity ==
            Catch::Approx(1.0 - teRowScattering).margin(2e-12));
    REQUIRE(emission.tmEmissivity ==
            Catch::Approx(1.0 - tmRowScattering).margin(2e-12));

    mixing.setPolarization(Polarization::TE);
    const auto teAbsorption = mixing.solveSpectrumOnly();
    mixing.setPolarization(Polarization::TM);
    const auto tmAbsorption = mixing.solveSpectrumOnly();
    REQUIRE(emission.teAbsorptivity ==
            Catch::Approx(1.0 - teAbsorption.conservation).margin(2e-12));
    REQUIRE(emission.tmAbsorptivity ==
            Catch::Approx(1.0 - tmAbsorption.conservation).margin(2e-12));
    REQUIRE(std::abs(emission.teEmissivity - emission.teAbsorptivity) > 1e-2);
    REQUIRE(emission.lcpAbsorptivity + emission.rcpAbsorptivity ==
            Catch::Approx(emission.teAbsorptivity + emission.tmAbsorptivity)
                .margin(2e-12));
    REQUIRE(emission.lcpEmissivity + emission.rcpEmissivity ==
            Catch::Approx(emission.teEmissivity + emission.tmEmissivity)
                .margin(2e-12));

    mixing.setPolarization(Polarization::LCP);
    const auto lcpSpectrum = mixing.solveSpectrumOnly();
    mixing.setPolarization(Polarization::RCP);
    const auto rcpSpectrum = mixing.solveSpectrumOnly();
    REQUIRE(emission.lcpAbsorptivity ==
            Catch::Approx(1.0 - lcpSpectrum.rTotal - lcpSpectrum.tTotal)
                .margin(2e-12));
    REQUIRE(emission.rcpAbsorptivity ==
            Catch::Approx(1.0 - rcpSpectrum.rTotal - rcpSpectrum.tTotal)
                .margin(2e-12));
    REQUIRE(emission.lcpIncidentScattering ==
            Catch::Approx(lcpSpectrum.rTotal + lcpSpectrum.tTotal).margin(2e-12));
    REQUIRE(emission.rcpIncidentScattering ==
            Catch::Approx(rcpSpectrum.rTotal + rcpSpectrum.tTotal).margin(2e-12));
    REQUIRE(std::abs(emission.tmEmissivity - emission.tmAbsorptivity) > 1e-2);

    const auto batch = mixing.solveDirectionalThermalChannelsBatchOnly({1.05, 1.08}, 2);
    REQUIRE(batch.size() == 2);
    REQUIRE(batch[0].teAbsorptivity ==
            Catch::Approx(emission.teAbsorptivity).margin(2e-12));
    REQUIRE(batch[0].tmEmissivity ==
            Catch::Approx(emission.tmEmissivity).margin(2e-12));

    const auto combined = mixing.solveSpectrumThermalBatchForAngles(
        {31.0, -31.0},
        23.0,
        {1.05, 1.08},
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(combined.spectra.size() == 8);
    REQUIRE(combined.thermalChannels.size() == 4);
    REQUIRE(combined.spectra[0].rTotal ==
            Catch::Approx(teAbsorption.rTotal).margin(2e-12));
    REQUIRE(combined.spectra[0].tTotal ==
            Catch::Approx(teAbsorption.tTotal).margin(2e-12));
    REQUIRE(combined.spectra[1].rTotal ==
            Catch::Approx(tmAbsorption.rTotal).margin(2e-12));
    REQUIRE(combined.spectra[1].tTotal ==
            Catch::Approx(tmAbsorption.tTotal).margin(2e-12));
    REQUIRE(combined.thermalChannels[0].teAbsorptivity ==
            Catch::Approx(emission.teAbsorptivity).margin(2e-12));
    REQUIRE(combined.thermalChannels[0].tmEmissivity ==
            Catch::Approx(emission.tmEmissivity).margin(2e-12));

    RcwaSolver absorbingHalfSpace;
    absorbingHalfSpace.setWavelength(1.05);
    absorbingHalfSpace.setIncidence(20.0, 11.0);
    absorbingHalfSpace.setHarmonicOrders(
        0, 0, LatticeTruncation::Parallelogramic);
    absorbingHalfSpace.setSuperstrate(Material::vacuum());
    absorbingHalfSpace.setSubstrate(
        Material::constant("absorbing half-space", {2.25, 0.45}));
    const auto halfSpaceS = absorbingHalfSpace.solveSParameters();
    const auto halfSpaceThermal =
        absorbingHalfSpace.solveDirectionalThermalChannels();
    const double expectedTe = 1.0 - std::norm(halfSpaceS.S11(0, 0)) -
        std::norm(halfSpaceS.S11(1, 0));
    const double expectedTm = 1.0 - std::norm(halfSpaceS.S11(0, 1)) -
        std::norm(halfSpaceS.S11(1, 1));
    REQUIRE(halfSpaceThermal.teAbsorptivity ==
            Catch::Approx(expectedTe).margin(2e-12));
    REQUIRE(halfSpaceThermal.tmAbsorptivity ==
            Catch::Approx(expectedTm).margin(2e-12));
    REQUIRE(halfSpaceThermal.teEmissivity ==
            Catch::Approx(expectedTe).margin(2e-12));
    REQUIRE(halfSpaceThermal.tmEmissivity ==
            Catch::Approx(expectedTm).margin(2e-12));
    REQUIRE(expectedTe > 1e-2);
    REQUIRE(expectedTm > 1e-2);
    const auto projectedHalfSpace =
        absorbingHalfSpace.solveSpectrumThermalBatchForAngles(
            {20.0},
            11.0,
            {1.05},
            {Polarization::TE, Polarization::TM},
            1);
    REQUIRE(projectedHalfSpace.thermalChannels.size() == 1);
    REQUIRE(projectedHalfSpace.thermalChannels[0].teAbsorptivity ==
            Catch::Approx(halfSpaceThermal.teAbsorptivity).margin(2e-12));
    REQUIRE(projectedHalfSpace.thermalChannels[0].tmAbsorptivity ==
            Catch::Approx(halfSpaceThermal.tmAbsorptivity).margin(2e-12));
    REQUIRE(projectedHalfSpace.thermalChannels[0].teEmissivity ==
            Catch::Approx(halfSpaceThermal.teEmissivity).margin(2e-12));
    REQUIRE(projectedHalfSpace.thermalChannels[0].tmEmissivity ==
            Catch::Approx(halfSpaceThermal.tmEmissivity).margin(2e-12));

    Tensor3 reciprocalEps = Tensor3::diagonal(
        {2.20, 0.16}, {2.45, 0.12}, {2.70, 0.10});
    reciprocalEps(0, 1) = reciprocalEps(1, 0) = {0.07, 0.015};
    reciprocalEps(0, 2) = reciprocalEps(2, 0) = {0.11, 0.010};
    reciprocalEps(1, 2) = reciprocalEps(2, 1) = {0.05, 0.008};
    RcwaSolver reciprocal;
    reciprocal.setWavelength(1.08);
    reciprocal.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    reciprocal.setSuperstrate(Material::vacuum());
    reciprocal.setSubstrate(Material::fromIndex("glass", {1.52, 0.0}));
    reciprocal.addUniformLayer(
        Material::anisotropic("reciprocal lossy mixer", reciprocalEps), 0.43);
    reciprocal.setIncidence(28.0, 19.0);
    const auto reciprocalPlus = reciprocal.solveDirectionalThermalChannels();
    reciprocal.setIncidence(-28.0, 19.0);
    const auto reciprocalMinus = reciprocal.solveDirectionalThermalChannels();
    REQUIRE(reciprocalPlus.teAbsorptivity ==
            Catch::Approx(reciprocalMinus.teEmissivity).margin(3e-11));
    REQUIRE(reciprocalPlus.tmAbsorptivity ==
            Catch::Approx(reciprocalMinus.tmEmissivity).margin(3e-11));
}

TEST_CASE("thermal channels require positive-semidefinite material loss operators") {
    using namespace rcwa;

    Tensor3 passiveGyrotropic = Tensor3::diagonal(
        {3.2, 0.02}, {3.4, 0.03}, {3.1, 0.01});
    passiveGyrotropic(0, 2) = {0.0, -0.18};
    passiveGyrotropic(2, 0) = {0.0, 0.18};
    const MaterialPassivityResult passive = materialPassivity(
        passiveGyrotropic,
        Tensor3::isotropic({1.0, 0.0}));
    REQUIRE(passive.passive);
    REQUIRE(passive.epsilonMinimumLossEigenvalue ==
            Catch::Approx(0.01).margin(2e-14));

    // Positive imaginary diagonal entries alone are insufficient: this
    // Hermitian loss operator has eigenvalues 0.06, 0.01, and -0.04.
    Tensor3 indefiniteLoss = Tensor3::diagonal(
        {2.0, 0.01}, {2.1, 0.01}, {2.2, 0.01});
    indefiniteLoss(0, 1) = {0.0, 0.05};
    indefiniteLoss(1, 0) = {0.0, 0.05};
    const Material nonPassiveTensor = Material::anisotropic(
        "indefinite tensor loss",
        indefiniteLoss);
    const MaterialPassivityResult indefinite = nonPassiveTensor.passivity(1.0);
    REQUIRE_FALSE(indefinite.passive);
    REQUIRE(indefinite.epsilonMinimumLossEigenvalue ==
            Catch::Approx(-0.04).margin(2e-14));

    RcwaSolver tensorSolver;
    tensorSolver.setWavelength(1.0);
    tensorSolver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    tensorSolver.setSuperstrate(Material::vacuum());
    tensorSolver.setSubstrate(Material::vacuum());
    tensorSolver.addUniformLayer(nonPassiveTensor, 0.1);
    REQUIRE_NOTHROW(tensorSolver.solveSParameters());
    REQUIRE(throwsContaining(
        [&] { (void)tensorSolver.solveDirectionalThermalChannels(); },
        "positive"));

    RcwaSolver gainSolver;
    gainSolver.setWavelength(1.0);
    gainSolver.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    gainSolver.setSuperstrate(Material::vacuum());
    gainSolver.setSubstrate(Material::vacuum());
    gainSolver.addUniformLayer(
        Material::constant("gain film", {2.0, -0.002}),
        0.05);
    REQUIRE_NOTHROW(gainSolver.solveSpectrumOnly());
    REQUIRE(throwsContaining(
        [&] { (void)gainSolver.solveDirectionalThermalChannels(); },
        "gain film"));
    REQUIRE(throwsContaining(
        [&] {
            (void)gainSolver.solveSpectrumThermalBatchForAngles(
                {0.0},
                0.0,
                {0.9, 1.0},
                {Polarization::TE},
                2);
        },
        "not thermodynamic emissivities"));
}

TEST_CASE("circular thermal deficits include coherent diffracted TE TM channels") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material absorber = Material::constant("absorber", {2.7, 0.35});
    const Material tailFilm = Material::constant("tail film", {2.1, 0.08});
    PatternedLayer2D pattern;
    pattern.background = air;
    pattern.periodXUm = 1.0;
    pattern.periodYUm = 0.9;
    pattern.thicknessUm = 0.31;
    pattern.regions.push_back(PatternRegion::rectangle(
        absorber,
        {0.08, -0.06},
        17.0,
        {0.27, 0.19}));

    const std::vector<LayerSpec> layers{
        pattern,
        UniformLayer{tailFilm, 0.19}};
    const PreparedStack prepared = prepareStack(
        0.72,
        24.0,
        31.0,
        1,
        1,
        1,
        false,
        LatticeTruncation::Circular,
        air,
        air,
        layers);
    const std::vector<Polarization> allPolarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP};
    const SMatrix fullS = computeStackSmatrix(prepared.media);
    const auto fullThermal = computeTopDirectionalThermalChannels(
        prepared.media,
        fullS);
    const auto exactProjection =
        computeStackTotalsAndTopDirectionalThermalChannels(
            prepared.media,
            prepared.totalHarmonics,
            prepared.centerIdx,
            allPolarizations);
    const std::size_t center = static_cast<std::size_t>(prepared.centerIdx);
    const std::size_t tmCenter =
        static_cast<std::size_t>(prepared.totalHarmonics) + center;
    for (const std::size_t channel : {center, tmCenter}) {
        REQUIRE(exactProjection.thermal.absorptivity[channel] ==
                Catch::Approx(fullThermal.absorptivity[channel]).margin(3e-11));
        REQUIRE(exactProjection.thermal.emissivity[channel] ==
                Catch::Approx(fullThermal.emissivity[channel]).margin(3e-11));
        REQUIRE(exactProjection.thermal.incidentScattering[channel] ==
                Catch::Approx(fullThermal.incidentScattering[channel]).margin(3e-11));
        REQUIRE(exactProjection.thermal.outgoingScattering[channel] ==
                Catch::Approx(fullThermal.outgoingScattering[channel]).margin(3e-11));
    }
    REQUIRE(std::abs(
                exactProjection.thermal.teTmAbsorptivityCoherence[center] -
                fullThermal.teTmAbsorptivityCoherence[center]) < 3e-11);
    REQUIRE(std::abs(
                exactProjection.thermal.teTmEmissivityCoherence[center] -
                fullThermal.teTmEmissivityCoherence[center]) < 3e-11);
    const auto fullTotals = computeStackTotalsForPolarizations(
        prepared.media,
        prepared.totalHarmonics,
        prepared.centerIdx,
        allPolarizations,
        fullS);
    REQUIRE(exactProjection.spectra.size() == fullTotals.size());
    for (std::size_t i = 0; i < fullTotals.size(); ++i) {
        REQUIRE(exactProjection.spectra[i].rTotal ==
                Catch::Approx(fullTotals[i].rTotal).margin(3e-11));
        REQUIRE(exactProjection.spectra[i].tTotal ==
                Catch::Approx(fullTotals[i].tTotal).margin(3e-11));
    }

    RcwaSolver solver;
    solver.setWavelength(0.72);
    solver.setIncidence(24.0, 31.0);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Circular);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.addPatternedLayer2d(pattern);
    solver.addUniformLayer(tailFilm, 0.19);

    const auto thermal = solver.solveDirectionalThermalChannels();
    const auto projected = solver.solveSpectrumThermalBatchForAngles(
        {24.0},
        31.0,
        {0.72},
        allPolarizations,
        1);
    REQUIRE(projected.spectra.size() == 4);
    REQUIRE(projected.thermalChannels.size() == 1);
    const auto& projectedThermal = projected.thermalChannels.front();
    REQUIRE(projectedThermal.teAbsorptivity ==
            Catch::Approx(thermal.teAbsorptivity).margin(3e-11));
    REQUIRE(projectedThermal.tmAbsorptivity ==
            Catch::Approx(thermal.tmAbsorptivity).margin(3e-11));
    REQUIRE(projectedThermal.lcpAbsorptivity ==
            Catch::Approx(thermal.lcpAbsorptivity).margin(3e-11));
    REQUIRE(projectedThermal.rcpAbsorptivity ==
            Catch::Approx(thermal.rcpAbsorptivity).margin(3e-11));
    REQUIRE(projectedThermal.teEmissivity ==
            Catch::Approx(thermal.teEmissivity).margin(3e-11));
    REQUIRE(projectedThermal.tmEmissivity ==
            Catch::Approx(thermal.tmEmissivity).margin(3e-11));
    REQUIRE(projectedThermal.lcpEmissivity ==
            Catch::Approx(thermal.lcpEmissivity).margin(3e-11));
    REQUIRE(projectedThermal.rcpEmissivity ==
            Catch::Approx(thermal.rcpEmissivity).margin(3e-11));
    solver.setPolarization(Polarization::LCP);
    const auto lcp = solver.solveSpectrumOnly();
    solver.setPolarization(Polarization::RCP);
    const auto rcp = solver.solveSpectrumOnly();
    REQUIRE(thermal.lcpAbsorptivity ==
            Catch::Approx(1.0 - lcp.rTotal - lcp.tTotal).margin(3e-11));
    REQUIRE(thermal.rcpAbsorptivity ==
            Catch::Approx(1.0 - rcp.rTotal - rcp.tTotal).margin(3e-11));
    REQUIRE(projected.spectra[2].rTotal == Catch::Approx(lcp.rTotal).margin(3e-11));
    REQUIRE(projected.spectra[2].tTotal == Catch::Approx(lcp.tTotal).margin(3e-11));
    REQUIRE(projected.spectra[3].rTotal == Catch::Approx(rcp.rTotal).margin(3e-11));
    REQUIRE(projected.spectra[3].tTotal == Catch::Approx(rcp.tTotal).margin(3e-11));
    // Linear and circular channels are projected from the same coherent
    // scattering operator.
    REQUIRE(std::isfinite(thermal.lcpAbsorptivity));
    REQUIRE(std::isfinite(thermal.rcpAbsorptivity));
    REQUIRE(std::abs(thermal.lcpAbsorptivity - thermal.rcpAbsorptivity) > 1e-7);
}

TEST_CASE("batch spectra with periodic layers reuse wavelength caches consistently") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("dielectric", {2.25, 0.0});

    PatternedLayer2D layer;
    layer.background = air;
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.25;
    layer.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.18, 0.28}));

    RcwaSolver solver;
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Circular);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TE);
    solver.addPatternedLayer2d(layer);

    const auto batch = solver.solveSpectrumBatchForAngles(
        std::vector<Real>{12.0, -12.0},
        0.0,
        std::vector<Real>{0.9, 1.0},
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(batch.size() == 8);

    for (std::size_t angleIndex = 0; angleIndex < 2; ++angleIndex) {
        for (std::size_t iw = 0; iw < 2; ++iw) {
            RcwaSolver pointSolver = solver;
            pointSolver.setIncidence(angleIndex == 0 ? 12.0 : -12.0, 0.0);
            pointSolver.setWavelength(iw == 0 ? 0.9 : 1.0);
            const auto pointResults = pointSolver.solveSpectrumOnlyForPolarizations(
                {Polarization::TE, Polarization::TM});
            const std::size_t offset = (angleIndex * 2 + iw) * 2;
            REQUIRE(batch[offset].rTotal ==
                    Catch::Approx(pointResults[0].rTotal).margin(1e-12));
            REQUIRE(batch[offset].tTotal ==
                    Catch::Approx(pointResults[0].tTotal).margin(1e-12));
            REQUIRE(batch[offset + 1].rTotal ==
                    Catch::Approx(pointResults[1].rTotal).margin(1e-12));
            REQUIRE(batch[offset + 1].tTotal ==
                    Catch::Approx(pointResults[1].tTotal).margin(1e-12));
        }
    }
}

TEST_CASE("stack builder classifies static and dynamic periodic tensor caches once") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("static dielectric", {2.25, 0.0});
    const Material dispersive = Material::dispersive(
        "dispersive dielectric",
        [](Real wavelengthUm) {
            return std::pair<Complex, Complex>{
                Complex{2.0 + 0.1 / wavelengthUm, 0.01},
                Complex{1.0, 0.0}};
        });

    PatternedLayer2D staticPattern;
    staticPattern.background = air;
    staticPattern.periodXUm = 1.0;
    staticPattern.periodYUm = 1.0;
    staticPattern.thicknessUm = 0.2;
    staticPattern.regions.push_back(PatternRegion::rectangle(
        dielectric, {0.0, 0.0}, 0.0, {0.2, 0.25}));

    PatternedLayer2D dynamicPattern = staticPattern;
    dynamicPattern.background = dispersive;

    const std::vector<LayerSpec> layers{
        UniformLayer{dispersive, 0.1},
        staticPattern,
        dynamicPattern,
    };
    const StackBuilder builder(
        1, 1, 1, false, LatticeTruncation::Circular, layers);
    REQUIRE(builder.hasPeriodicLayers());
    REQUIRE(builder.hasDynamicPeriodicLayers());
    REQUIRE(builder.hasDispersiveMaterials());

    const PeriodicTensorCache staticCache = builder.makeStaticPeriodicCache(1.0);
    REQUIRE(staticCache.size() == layers.size());
    REQUIRE_FALSE(staticCache[0]);
    REQUIRE(staticCache[1]);
    REQUIRE_FALSE(staticCache[2]);

    const PeriodicTensorCache wavelengthCache =
        builder.makePeriodicTensorCache(1.1, &staticCache);
    REQUIRE(wavelengthCache.size() == layers.size());
    REQUIRE_FALSE(wavelengthCache[0]);
    REQUIRE(wavelengthCache[1] == staticCache[1]);
    REQUIRE(wavelengthCache[2]);
}

TEST_CASE("stack builder reuses only exactly equal thickness-independent modal bases") {
    using namespace rcwa;

    constexpr Real wavelengthUm = 0.94;
    const Real modalTolerance = sizeof(Real) == sizeof(float)
        ? Real{2e-4f}
        : Real{2e-12};
    const Real powerTolerance = sizeof(Real) == sizeof(float)
        ? Real{3e-4f}
        : Real{3e-11};
    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("modal cache dielectric", {3.1, 0.04});
    PatternedLayer2D first;
    first.background = air;
    first.periodXUm = 1.0;
    first.periodYUm = 0.9;
    first.thicknessUm = 0.17;
    first.regions.push_back(PatternRegion::rectangle(
        dielectric, {0.03, -0.04}, 11.0, {0.21, 0.16}));

    PatternedLayer2D repeated = first;
    repeated.thicknessUm = 0.31;
    PatternedLayer2D distinct = first;
    distinct.thicknessUm = 0.13;
    distinct.regions.front().halfwidthXUm += 0.01;

    const std::vector<LayerSpec> layers{first, repeated, distinct};
    const StackBuilder builder(
        1, 1, 1, false, LatticeTruncation::Circular, layers);
    const PeriodicTensorCache tensors = builder.makePeriodicTensorCache(wavelengthUm);
    REQUIRE(tensors.size() == layers.size());
    REQUIRE(tensors[0]);
    REQUIRE(tensors[1]);
    REQUIRE(tensors[2]);

    const PreparedStack cached = builder.prepare(
        wavelengthUm, 13.0, 7.0, air, air, &tensors);
    REQUIRE(cached.finiteLayerModalBasisBuilds == 2);
    REQUIRE(cached.finiteLayerModalBasisReuses == 1);
    requireMatrixClose(cached.media[1].W, cached.media[2].W, 0.0);
    REQUIRE(cached.media[1].gamma == cached.media[2].gamma);
    REQUIRE(cached.media[1].phase != cached.media[2].phase);

    LayerModes directRepeated = computeLayerModes(
        layers[1],
        cached.Kx,
        cached.Ky,
        cached.basis,
        wavelengthUm,
        tensors[1].get());
    requireMatrixClose(cached.media[2].W, directRepeated.W, modalTolerance);
    REQUIRE(cached.media[2].gamma.size() == directRepeated.gamma.size());
    REQUIRE(cached.media[2].phase.size() == directRepeated.phase.size());
    for (std::size_t i = 0; i < directRepeated.gamma.size(); ++i) {
        REQUIRE(std::abs(cached.media[2].gamma[i] - directRepeated.gamma[i]) <
                modalTolerance);
        REQUIRE(std::abs(cached.media[2].phase[i] - directRepeated.phase[i]) <
                modalTolerance);
    }

    PreparedStack independent = cached;
    independent.media[2] = std::move(directRepeated);
    const std::vector<Polarization> polarizations{
        Polarization::TE,
        Polarization::TM,
        Polarization::LCP,
        Polarization::RCP,
    };
    const auto cachedTotals = computeStackTotalsForPolarizations(
        cached.media, cached.totalHarmonics, cached.centerIdx, polarizations);
    const auto independentTotals = computeStackTotalsForPolarizations(
        independent.media,
        independent.totalHarmonics,
        independent.centerIdx,
        polarizations);
    REQUIRE(cachedTotals.size() == independentTotals.size());
    for (std::size_t i = 0; i < cachedTotals.size(); ++i) {
        REQUIRE(cachedTotals[i].rTotal ==
                Catch::Approx(independentTotals[i].rTotal).margin(powerTolerance));
        REQUIRE(cachedTotals[i].tTotal ==
                Catch::Approx(independentTotals[i].tTotal).margin(powerTolerance));
    }
}

TEST_CASE("single-angle periodic spectrum batches reuse wavelength caches consistently") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("dielectric", {2.25, 0.0});

    PatternedLayer2D layer;
    layer.background = air;
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.25;
    layer.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.18, 0.28}));

    RcwaSolver solver;
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Circular);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TE);
    solver.addPatternedLayer2d(layer);

    const std::vector<Real> wavelengths{0.9, 1.0};
    const auto batch = solver.solveSpectrumBatchOnly(
        wavelengths,
        {Polarization::TE, Polarization::TM},
        2);
    REQUIRE(batch.size() == 4);

    for (std::size_t iw = 0; iw < wavelengths.size(); ++iw) {
        RcwaSolver pointSolver = solver;
        pointSolver.setWavelength(wavelengths[iw]);
        const auto pointResults = pointSolver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
        REQUIRE(batch[2 * iw].rTotal ==
                Catch::Approx(pointResults[0].rTotal).margin(1e-12));
        REQUIRE(batch[2 * iw].tTotal ==
                Catch::Approx(pointResults[0].tTotal).margin(1e-12));
        REQUIRE(batch[2 * iw + 1].rTotal ==
                Catch::Approx(pointResults[1].rTotal).margin(1e-12));
        REQUIRE(batch[2 * iw + 1].tTotal ==
                Catch::Approx(pointResults[1].tTotal).margin(1e-12));
    }

    const std::vector<SpectrumAnglePolarization> requests{
        {12.0, Polarization::TE},
        {12.0, Polarization::TM},
    };
    const auto angleBatch = solver.solveSpectrumBatchForAnglePolarizations(
        requests,
        0.0,
        wavelengths,
        2);
    REQUIRE(angleBatch.size() == requests.size() * wavelengths.size());

    for (std::size_t iw = 0; iw < wavelengths.size(); ++iw) {
        RcwaSolver pointSolver = solver;
        pointSolver.setIncidence(12.0, 0.0);
        pointSolver.setWavelength(wavelengths[iw]);
        const auto pointResults = pointSolver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
        for (std::size_t requestIndex = 0; requestIndex < requests.size(); ++requestIndex) {
            const auto& row = angleBatch[requestIndex * wavelengths.size() + iw];
            REQUIRE(row.rTotal ==
                    Catch::Approx(pointResults[requestIndex].rTotal).margin(1e-12));
            REQUIRE(row.tTotal ==
                    Catch::Approx(pointResults[requestIndex].tTotal).margin(1e-12));
        }
    }
}

TEST_CASE("uniform limit of patterned grating matches uniform tensor layer") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    eps(0, 1) = {0.04, 0.0};
    eps(1, 0) = {0.04, 0.0};
    eps(0, 2) = {0.12, 0.0};
    eps(2, 0) = {0.12, 0.0};

    RcwaSolver uniform;
    uniform.setWavelength(1.0);
    uniform.setIncidence(20.0, 0.0);
    uniform.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    uniform.setSuperstrate(Material::vacuum());
    uniform.setSubstrate(Material::vacuum());
    uniform.setPolarization(Polarization::TE);
    uniform.addUniformLayer(Material::anisotropic("uniform tensor", eps), 0.3);
    const auto ru = uniform.solve();

    GratingLayer g;
    g.materialRidge = Material::anisotropic("same tensor ridge", eps);
    g.materialGroove = Material::anisotropic("same tensor groove", eps);
    g.fillFactor = 0.5;
    g.periodUm = 1.0;
    g.thicknessUm = 0.3;

    RcwaSolver grating;
    grating.setWavelength(1.0);
    grating.setIncidence(20.0, 0.0);
    grating.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    grating.setSuperstrate(Material::vacuum());
    grating.setSubstrate(Material::vacuum());
    grating.setPolarization(Polarization::TE);
    grating.addGratingLayer(g);
    const auto rg = grating.solve();
    REQUIRE(rg.conservation == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(rg.rTotal == Catch::Approx(ru.rTotal).margin(1e-6));
    REQUIRE(rg.tTotal == Catch::Approx(ru.tTotal).margin(1e-6));
}

TEST_CASE("adaptive tapered grating expands to midpoint lamellar slices") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material silicon = Material::constant("Si", {12.25, 0.0});

    REQUIRE(adaptiveTaperedGratingSliceCount(0.2, 0.6, 0.11, 2, 16) == 4);
    REQUIRE(adaptiveTaperedGratingMidpointFill(0.2, 0.6, 0, 4) ==
            Catch::Approx(0.25).margin(1e-12));
    REQUIRE(adaptiveTaperedGratingMidpointFill(0.2, 0.6, 3, 4) ==
            Catch::Approx(0.55).margin(1e-12));
    REQUIRE(throwsContaining(
        [] {
            (void)adaptiveTaperedGratingSliceCount(0.2, 0.6, 0.0, 1, 16);
        },
        "fill_step_tolerance"));

    AdaptiveTaperedGratingLayer tapered;
    tapered.materialRidge = silicon;
    tapered.materialGroove = air;
    tapered.topFillFactor = 0.2;
    tapered.bottomFillFactor = 0.6;
    tapered.periodUm = 0.7;
    tapered.periodYUm = 0.7;
    tapered.ridgeOffsetUm = 0.08;
    tapered.thicknessUm = 0.4;
    tapered.fillStepTolerance = 0.11;
    tapered.minSlices = 2;
    tapered.maxSlices = 16;

    const auto slices = expandAdaptiveTaperedGratingLayer(tapered);
    REQUIRE(slices.size() == 4);
    REQUIRE(slices.front().thicknessUm == Catch::Approx(0.1).margin(1e-12));
    REQUIRE(slices.front().fillFactor == Catch::Approx(0.25).margin(1e-12));
    REQUIRE(slices.front().ridgeOffsetUm == Catch::Approx(0.08).margin(1e-12));
    REQUIRE(slices.back().fillFactor == Catch::Approx(0.55).margin(1e-12));
    REQUIRE(slices.back().ridgeOffsetUm == Catch::Approx(0.08).margin(1e-12));

    RcwaSolver adaptive;
    adaptive.setWavelength(1.0);
    adaptive.setIncidence(11.0, 0.0);
    adaptive.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    adaptive.setSuperstrate(air);
    adaptive.setSubstrate(air);
    adaptive.setPolarization(Polarization::TE);
    adaptive.addAdaptiveTaperedGratingLayer(tapered);
    const auto ra = adaptive.solve();

    RcwaSolver manual;
    manual.setWavelength(1.0);
    manual.setIncidence(11.0, 0.0);
    manual.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    manual.setSuperstrate(air);
    manual.setSubstrate(air);
    manual.setPolarization(Polarization::TE);
    for (const auto& slice : slices) {
        manual.addGratingLayer(slice);
    }
    const auto rm = manual.solve();

    REQUIRE(ra.N == rm.N);
    REQUIRE(ra.rTotal == Catch::Approx(rm.rTotal).margin(1e-12));
    REQUIRE(ra.tTotal == Catch::Approx(rm.tTotal).margin(1e-12));
    REQUIRE(ra.conservation == Catch::Approx(1.0).margin(1e-7));
}

TEST_CASE("adaptive profiled grating allocates slices over nonlinear fill profiles") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material silicon = Material::constant("Si", {12.25, 0.0});
    const std::vector<FillFactorProfilePoint> profile{
        {0.0, 0.2},
        {0.25, 0.5},
        {1.0, 0.35},
    };

    REQUIRE(adaptiveProfiledGratingSliceCount(profile, 0.11, 2, 16) == 5);
    REQUIRE(adaptiveProfiledGratingFillAt(profile, 0.125) ==
            Catch::Approx(0.35).margin(1e-12));
    REQUIRE(adaptiveProfiledGratingFillAt(profile, 0.625) ==
            Catch::Approx(0.425).margin(1e-12));
    REQUIRE(throwsContaining(
        [&] {
            (void)adaptiveProfiledGratingSliceCount(
                {{0.0, 0.2}, {0.5, 0.4}},
                0.1,
                1,
                8);
        },
        "end at 1"));

    AdaptiveProfiledGratingLayer profiled;
    profiled.materialRidge = silicon;
    profiled.materialGroove = air;
    profiled.fillProfile = profile;
    profiled.periodUm = 0.7;
    profiled.periodYUm = 0.7;
    profiled.ridgeOffsetUm = -0.06;
    profiled.thicknessUm = 0.4;
    profiled.fillStepTolerance = 0.11;
    profiled.minSlices = 2;
    profiled.maxSlices = 16;

    const auto slices = expandAdaptiveProfiledGratingLayer(profiled);
    REQUIRE(slices.size() == 5);
    REQUIRE(slices[0].thicknessUm == Catch::Approx(0.4 * 0.25 / 3.0).margin(1e-12));
    REQUIRE(slices[0].fillFactor == Catch::Approx(0.25).margin(1e-12));
    REQUIRE(slices[0].ridgeOffsetUm == Catch::Approx(-0.06).margin(1e-12));
    REQUIRE(slices[2].fillFactor == Catch::Approx(0.45).margin(1e-12));
    REQUIRE(slices[3].thicknessUm == Catch::Approx(0.4 * 0.75 / 2.0).margin(1e-12));
    REQUIRE(slices[4].fillFactor == Catch::Approx(0.3875).margin(1e-12));
    REQUIRE(slices[4].ridgeOffsetUm == Catch::Approx(-0.06).margin(1e-12));

    RcwaSolver adaptive;
    adaptive.setWavelength(1.0);
    adaptive.setIncidence(9.0, 0.0);
    adaptive.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    adaptive.setSuperstrate(air);
    adaptive.setSubstrate(air);
    adaptive.setPolarization(Polarization::TM);
    adaptive.addAdaptiveProfiledGratingLayer(profiled);
    const auto ra = adaptive.solve();

    RcwaSolver manual;
    manual.setWavelength(1.0);
    manual.setIncidence(9.0, 0.0);
    manual.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    manual.setSuperstrate(air);
    manual.setSubstrate(air);
    manual.setPolarization(Polarization::TM);
    for (const auto& slice : slices) {
        manual.addGratingLayer(slice);
    }
    const auto rm = manual.solve();

    REQUIRE(ra.N == rm.N);
    REQUIRE(ra.rTotal == Catch::Approx(rm.rTotal).margin(1e-12));
    REQUIRE(ra.tTotal == Catch::Approx(rm.tTotal).margin(1e-12));
    REQUIRE(ra.conservation == Catch::Approx(1.0).margin(1e-7));
}

TEST_CASE("anisotropic grating and lossy metal cases are stable") {
    using namespace rcwa;

    Tensor3 epsRidge = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    epsRidge(0, 1) = {0.04, 0.0};
    epsRidge(1, 0) = {0.04, 0.0};
    epsRidge(0, 2) = {0.36, 0.0};
    epsRidge(2, 0) = {0.36, 0.0};
    epsRidge(1, 2) = {0.16, 0.0};
    epsRidge(2, 1) = {0.16, 0.0};

    GratingLayer g;
    g.materialRidge = Material::anisotropic("biaxial ridge", epsRidge);
    g.materialGroove = Material::anisotropic("air groove tensor", Tensor3::isotropic({1.0, 0.0}));
    g.fillFactor = 0.5;
    g.periodUm = 1.0;
    g.thicknessUm = 0.4;

    RcwaSolver grating;
    grating.setWavelength(1.0);
    grating.setIncidence(30.0, 0.0);
    grating.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    grating.setSuperstrate(Material::vacuum());
    grating.setSubstrate(Material::vacuum());
    grating.setPolarization(Polarization::TE);
    grating.addGratingLayer(g);
    const auto rg = grating.solve();
    REQUIRE(std::isfinite(rg.rTotal));
    REQUIRE(std::isfinite(rg.tTotal));
    REQUIRE(rg.conservation == Catch::Approx(1.0).margin(1e-6));

    RcwaSolver metal;
    metal.setWavelength(15.0);
    metal.setIncidence(26.0, 0.0);
    metal.setHarmonicOrders(0, 0, LatticeTruncation::Parallelogramic);
    metal.setSuperstrate(Material::vacuum());
    metal.setSubstrate(Material::vacuum());
    metal.setPolarization(Polarization::TM);
    metal.addUniformLayer(Material::constant("lossy metal", {-100.0, 10.0}), 1.0);
    const auto rm = metal.solve();
    REQUIRE(std::isfinite(rm.rTotal));
    REQUIRE(std::isfinite(rm.tTotal));
}

TEST_CASE("lamellar grating uses the same Li factorization in 1D and 2D harmonic grids") {
    using namespace rcwa;

    Tensor3 epsRidge = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    epsRidge(0, 1) = {0.04, 0.0};
    epsRidge(1, 0) = {0.04, 0.0};
    epsRidge(0, 2) = {0.36, 0.0};
    epsRidge(2, 0) = {0.36, 0.0};
    epsRidge(1, 2) = {0.16, 0.0};
    epsRidge(2, 1) = {0.16, 0.0};

    GratingLayer g;
    g.materialRidge = Material::anisotropic("biaxial ridge", epsRidge);
    g.materialGroove = Material::vacuum();
    g.fillFactor = 0.5;
    g.periodUm = 1.0;
    g.periodYUm = 1.0;
    g.thicknessUm = 0.4;

    RcwaSolver oneDimensional;
    oneDimensional.setWavelength(1.0);
    oneDimensional.setIncidence(25.0, 17.0);
    oneDimensional.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    oneDimensional.setSuperstrate(Material::vacuum());
    oneDimensional.setSubstrate(Material::vacuum());
    oneDimensional.setPolarization(Polarization::TE);
    oneDimensional.addGratingLayer(g);
    const auto r1 = oneDimensional.solve();

    RcwaSolver liftedTo2d;
    liftedTo2d.setWavelength(1.0);
    liftedTo2d.setIncidence(25.0, 17.0);
    liftedTo2d.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    liftedTo2d.setSuperstrate(Material::vacuum());
    liftedTo2d.setSubstrate(Material::vacuum());
    liftedTo2d.setPolarization(Polarization::TE);
    liftedTo2d.addGratingLayer(g);
    const auto r2 = liftedTo2d.solve();

    REQUIRE(r2.N == 9);
    REQUIRE(r2.rTotal == Catch::Approx(r1.rTotal).margin(1e-7));
    REQUIRE(r2.tTotal == Catch::Approx(r1.tTotal).margin(1e-7));
}

TEST_CASE("zero-order lamellar Li modes separate TM harmonic and TE arithmetic means") {
    using namespace rcwa;

    constexpr Real fill = Real{0.25};
    const Complex epsRidge{9.0, 0.0};
    const Complex epsGroove{1.0, 0.0};
    const Complex arithmetic = fill * epsRidge + (Real{1} - fill) * epsGroove;
    const Complex harmonic = Complex{1.0, 0.0} /
        (fill / epsRidge + (Real{1} - fill) / epsGroove);
    const Tensor3 ridge = Tensor3::isotropic(epsRidge);
    const Tensor3 groove = Tensor3::isotropic(epsGroove);
    const Tensor3 vacuum = Tensor3::isotropic({1.0, 0.0});
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        0, 0, LatticeTruncation::Parallelogramic);

    const TensorFourierMatrices xNormal = tensorFourierMatricesBinaryGrating(
        fill, ridge, groove, vacuum, vacuum, basis);
    REQUIRE(std::abs(xNormal.eps[0][0](0, 0) - arithmetic) <= 1e-13);
    REQUIRE(std::abs(xNormal.liEpsQ[0][0](0, 0) - harmonic) <= 1e-13);
    REQUIRE(std::abs(xNormal.liEpsQ[1][1](0, 0) - arithmetic) <= 1e-13);
    REQUIRE(std::abs(xNormal.liEpsQ[2][2](0, 0) - arithmetic) <= 1e-13);

    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.0, 0.0, 1.0, 1.0, basis);
    const LayerModes modes = computePeriodicModes(
        xNormal, Kx, Ky, 0.2, 1.0, false);
    REQUIRE(modes.gamma.size() == 4);

    int tmCount = 0;
    int teCount = 0;
    for (std::size_t mode = 0; mode < 2; ++mode) {
        const bool isTm = std::abs(modes.W(0, mode)) > std::abs(modes.W(1, mode));
        if (isTm) {
            ++tmCount;
            REQUIRE(std::abs(modes.gamma[mode] - std::sqrt(harmonic)) <= 1e-12);
        } else {
            ++teCount;
            REQUIRE(std::abs(modes.gamma[mode] - std::sqrt(arithmetic)) <= 1e-12);
        }
    }
    REQUIRE(tmCount == 1);
    REQUIRE(teCount == 1);

    const TensorFourierMatrices yNormal = tensorFourierMatricesBinaryGratingY(
        fill, ridge, groove, vacuum, vacuum, basis);
    REQUIRE(std::abs(yNormal.liEpsQ[0][0](0, 0) - arithmetic) <= 1e-13);
    REQUIRE(std::abs(yNormal.liEpsQ[1][1](0, 0) - harmonic) <= 1e-13);
    REQUIRE(std::abs(yNormal.liEpsQ[2][2](0, 0) - arithmetic) <= 1e-13);
}

TEST_CASE("y-only sampled tensor layers use y-normal Li factorization") {
    using namespace rcwa;

    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);
    Tensor3 epsRidge = Tensor3::diagonal({2.25, 0.0}, {3.24, 0.0}, {2.89, 0.0});

    std::vector<Tensor3> epsSamples(25, Tensor3::isotropic({1.0, 0.0}));
    std::vector<Tensor3> muSamples(25, Tensor3::isotropic({1.0, 0.0}));
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            if (y == 1 || y == 2) {
                epsSamples[static_cast<std::size_t>(y * 5 + x)] = epsRidge;
            }
        }
    }

    const TensorFourierMatrices tensors = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        5,
        5,
        basis);

    std::vector<Tensor3> invEyySamples;
    invEyySamples.reserve(epsSamples.size());
    for (const Tensor3& eps : epsSamples) {
        invEyySamples.push_back(Tensor3::isotropic(Complex{1.0, 0.0} / eps(1, 1)));
    }
    const TensorFourierMatrices invEyyTensors = tensorFourierMatricesFromSamples2d(
        invEyySamples,
        muSamples,
        5,
        5,
        basis);

    requireMatrixClose(tensors.liEpsQ[0][0], tensors.eps[0][0], 1e-12);
    requireMatrixClose(
        tensors.liEpsQ[1][1],
        inverse(invEyyTensors.eps[0][0]),
        1e-12);
    requireMatrixClose(tensors.liEpsQ[2][2], tensors.eps[2][2], 1e-12);
    REQUIRE(maxAbs(tensors.liEpsQ[1][1] - tensors.eps[1][1]) > 1e-3);

    PeriodicLayer2D sampled;
    sampled.sampleCountX = 5;
    sampled.sampleCountY = 5;
    sampled.periodXUm = 1.0;
    sampled.periodYUm = 1.0;
    sampled.thicknessUm = 0.2;
    sampled.cells.reserve(25);
    for (const Tensor3& eps : epsSamples) {
        sampled.cells.push_back(Material::anisotropic("sampled tensor", eps));
    }

    RcwaSolver solver;
    solver.setWavelength(0.9);
    solver.setIncidence(13.0, 9.0);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPeriodicLayer2d(sampled);
    const auto result = solver.solve();
    REQUIRE(std::isfinite(result.rTotal));
    REQUIRE(std::isfinite(result.tTotal));
    REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("binary 1D grating matches the equivalent centered rectangle layer") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("ridge", {3.24, 0.0});
    const Real periodX = 1.3;
    const Real periodY = 0.9;
    const Real fill = 0.37;
    const Real thickness = 0.28;
    const Real wavelength = 0.82;
    const auto basis = makeHarmonicBasisOrders(
        2,
        1,
        LatticeTruncation::Parallelogramic);

    const auto [epsRidge, muRidge] = dielectric.tensors(wavelength);
    const auto [epsGroove, muGroove] = air.tensors(wavelength);
    const TensorFourierMatrices gratingTensors =
        tensorFourierMatricesBinaryGrating(
            fill,
            epsRidge,
            epsGroove,
            muRidge,
            muGroove,
            basis);

    PatternedLayer2D rectangle;
    rectangle.background = air;
    rectangle.periodXUm = periodX;
    rectangle.periodYUm = periodY;
    rectangle.thicknessUm = thickness;
    rectangle.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.5 * fill * periodX, 0.5 * periodY}));
    const TensorFourierMatrices rectangleTensors =
        tensorFourierMatricesPatternedLayer2d(rectangle, basis, wavelength);

    requireMatrixClose(gratingTensors.eps[0][0], rectangleTensors.eps[0][0], 1e-12);
    requireMatrixClose(gratingTensors.eps[1][1], rectangleTensors.eps[1][1], 1e-12);
    requireMatrixClose(gratingTensors.eps[2][2], rectangleTensors.eps[2][2], 1e-12);
    requireMatrixClose(gratingTensors.mu[0][0], rectangleTensors.mu[0][0], 1e-12);
    requireMatrixClose(gratingTensors.liEpsQ[1][1], rectangleTensors.eps[1][1], 1e-12);
    requireMatrixClose(gratingTensors.liEpsQ[2][2], rectangleTensors.eps[2][2], 1e-12);
    requireMatrixClose(gratingTensors.liMuQ[1][1], rectangleTensors.mu[1][1], 1e-12);
    requireMatrixClose(gratingTensors.liMuQ[2][2], rectangleTensors.mu[2][2], 1e-12);

    GratingLayer grating;
    grating.materialRidge = dielectric;
    grating.materialGroove = air;
    grating.fillFactor = fill;
    grating.periodUm = periodX;
    grating.periodYUm = periodY;
    grating.thicknessUm = thickness;

    RcwaSolver native;
    native.setWavelength(wavelength);
    native.setIncidence(17.0, 0.0);
    native.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    native.setSuperstrate(air);
    native.setSubstrate(air);
    native.setPolarization(Polarization::TE);
    native.addGratingLayer(grating);
    const auto rn = native.solve();

    RcwaSolver patterned;
    patterned.setWavelength(wavelength);
    patterned.setIncidence(17.0, 0.0);
    patterned.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    patterned.setSuperstrate(air);
    patterned.setSubstrate(air);
    patterned.setPolarization(Polarization::TE);
    patterned.addPatternedLayer2d(rectangle);
    const auto rp = patterned.solve();

    REQUIRE(rn.N == rp.N);
    REQUIRE(rn.rTotal == Catch::Approx(rp.rTotal).margin(1e-10));
    REQUIRE(rn.tTotal == Catch::Approx(rp.tTotal).margin(1e-10));
    REQUIRE(rn.conservation == Catch::Approx(1.0).margin(1e-8));
}

TEST_CASE("binary 1D grating offset matches the equivalent shifted rectangle layer") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("shifted ridge", {3.24, 0.0});
    const Real periodX = 1.3;
    const Real periodY = 0.9;
    const Real fill = 0.37;
    const Real ridgeOffset = 0.17;
    const Real thickness = 0.28;
    const Real wavelength = 0.82;
    const auto basis = makeHarmonicBasisOrders(
        2,
        1,
        LatticeTruncation::Parallelogramic);

    const auto [epsRidge, muRidge] = dielectric.tensors(wavelength);
    const auto [epsGroove, muGroove] = air.tensors(wavelength);
    const TensorFourierMatrices gratingTensors =
        tensorFourierMatricesBinaryGrating(
            fill,
            epsRidge,
            epsGroove,
            muRidge,
            muGroove,
            basis,
            ridgeOffset / periodX);

    PatternedLayer2D rectangle;
    rectangle.background = air;
    rectangle.periodXUm = periodX;
    rectangle.periodYUm = periodY;
    rectangle.thicknessUm = thickness;
    rectangle.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {ridgeOffset, 0.0},
        0.0,
        {0.5 * fill * periodX, 0.5 * periodY}));
    const TensorFourierMatrices rectangleTensors =
        tensorFourierMatricesPatternedLayer2d(rectangle, basis, wavelength);

    requireMatrixClose(gratingTensors.eps[0][0], rectangleTensors.eps[0][0], 1e-12);
    requireMatrixClose(gratingTensors.eps[1][1], rectangleTensors.eps[1][1], 1e-12);
    requireMatrixClose(gratingTensors.eps[2][2], rectangleTensors.eps[2][2], 1e-12);
    requireMatrixClose(gratingTensors.mu[0][0], rectangleTensors.mu[0][0], 1e-12);
    requireMatrixClose(gratingTensors.liEpsQ[0][0], rectangleTensors.liEpsQ[0][0], 1e-10);
    requireMatrixClose(gratingTensors.liEpsQ[1][1], rectangleTensors.liEpsQ[1][1], 1e-12);
    requireMatrixClose(gratingTensors.liEpsQ[2][2], rectangleTensors.liEpsQ[2][2], 1e-12);
    requireMatrixClose(gratingTensors.liMuQ[0][0], rectangleTensors.liMuQ[0][0], 1e-10);
    requireMatrixClose(gratingTensors.liMuQ[1][1], rectangleTensors.liMuQ[1][1], 1e-12);
    requireMatrixClose(gratingTensors.liMuQ[2][2], rectangleTensors.liMuQ[2][2], 1e-12);

    GratingLayer grating;
    grating.materialRidge = dielectric;
    grating.materialGroove = air;
    grating.fillFactor = fill;
    grating.periodUm = periodX;
    grating.periodYUm = periodY;
    grating.ridgeOffsetUm = ridgeOffset;
    grating.thicknessUm = thickness;

    RcwaSolver native;
    native.setWavelength(wavelength);
    native.setIncidence(17.0, 0.0);
    native.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    native.setSuperstrate(air);
    native.setSubstrate(air);
    native.setPolarization(Polarization::TE);
    native.addGratingLayer(grating);
    const auto rn = native.solve();

    RcwaSolver patterned;
    patterned.setWavelength(wavelength);
    patterned.setIncidence(17.0, 0.0);
    patterned.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    patterned.setSuperstrate(air);
    patterned.setSubstrate(air);
    patterned.setPolarization(Polarization::TE);
    patterned.addPatternedLayer2d(rectangle);
    const auto rp = patterned.solve();

    REQUIRE(rn.N == rp.N);
    REQUIRE(rn.rTotal == Catch::Approx(rp.rTotal).margin(1e-10));
    REQUIRE(rn.tTotal == Catch::Approx(rp.tTotal).margin(1e-10));
    REQUIRE(rn.conservation == Catch::Approx(1.0).margin(1e-8));
}

TEST_CASE("multi-strip full-height grating uses analytic lamellar Fourier blocks") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material metal = Material::constant("metal", {-80.0, 7.0});
    const Real periodX = 8.8;
    const Real periodY = 8.8;
    const Real w1 = 1.76;
    const Real gap = 3.168;
    const Real w2 = 2.2;
    const auto basis = makeHarmonicBasisOrders(2, 0, LatticeTruncation::Parallelogramic);

    PatternedLayer2D single;
    single.background = air;
    single.periodXUm = periodX;
    single.periodYUm = periodY;
    single.thicknessUm = 0.55;
    single.regions.push_back(PatternRegion::rectangle(
        metal,
        {0.0, 0.0},
        0.0,
        {0.5 * w1, 0.5 * periodY}));
    const auto singlePattern = tensorFourierMatricesPatternedLayer2d(single, basis, 13.25);
    const auto [epsMetal, muMetal] = metal.tensors(13.25);
    const auto [epsAir, muAir] = air.tensors(13.25);
    const auto singleGrating = tensorFourierMatricesBinaryGrating(
        w1 / periodX,
        epsMetal,
        epsAir,
        muMetal,
        muAir,
        basis);
    requireMatrixClose(singlePattern.eps[0][0], singleGrating.eps[0][0], 1e-11);
    requireMatrixClose(singlePattern.liEpsQ[1][1], singleGrating.liEpsQ[1][1], 1e-11);

    PatternedLayer2D fourPart;
    fourPart.background = air;
    fourPart.periodXUm = periodX;
    fourPart.periodYUm = periodY;
    fourPart.thicknessUm = 0.55;
    const Real left = -0.5 * periodX;
    fourPart.regions.push_back(PatternRegion::rectangle(
        metal,
        {left + 0.5 * w1, 0.0},
        0.0,
        {0.5 * w1, 0.5 * periodY}));
    fourPart.regions.push_back(PatternRegion::rectangle(
        metal,
        {left + w1 + gap + 0.5 * w2, 0.0},
        0.0,
        {0.5 * w2, 0.5 * periodY}));
    const auto fourPartTensors =
        tensorFourierMatricesPatternedLayer2d(fourPart, basis, 13.25);
    REQUIRE(fourPartTensors.eps[0][0].rows() == basis.size());
    REQUIRE(fourPartTensors.liEpsQ[0][0].rows() == basis.size());

    RcwaSolver solver;
    solver.setWavelength(13.25);
    solver.setIncidence(25.0, 0.0);
    solver.setHarmonicOrders(2, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TM);
    solver.addPatternedLayer2d(fourPart);
    const auto result = solver.solve();
    REQUIRE(std::isfinite(result.rTotal));
    REQUIRE(std::isfinite(result.tTotal));
}

TEST_CASE("tensor permeability patterned layers solve instead of using scalar-mu shortcuts") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    Tensor3 mu = Tensor3::diagonal({1.0, 0.0}, {1.1, 0.0}, {0.95, 0.0});
    mu(0, 2) = {0.02, 0.0};
    mu(2, 0) = {0.02, 0.0};

    GratingLayer magneticGrating;
    magneticGrating.materialRidge = Material::anisotropic("tensor mu ridge", eps, mu);
    magneticGrating.materialGroove = Material::anisotropic("scalar mu groove", eps);
    magneticGrating.fillFactor = 0.5;
    magneticGrating.periodUm = 1.0;
    magneticGrating.thicknessUm = 0.2;

    RcwaSolver gratingSolver;
    gratingSolver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    gratingSolver.setSuperstrate(Material::vacuum());
    gratingSolver.setSubstrate(Material::vacuum());
    gratingSolver.addGratingLayer(magneticGrating);
    const auto gratingResult = gratingSolver.solve();
    REQUIRE(std::isfinite(gratingResult.rTotal));
    REQUIRE(std::isfinite(gratingResult.tTotal));
    REQUIRE(gratingResult.conservation == Catch::Approx(1.0).margin(1e-6));

    PatternedLayer2D patterned;
    patterned.background = Material::vacuum();
    patterned.periodXUm = 1.0;
    patterned.periodYUm = 1.0;
    patterned.thicknessUm = 0.16;
    patterned.regions.push_back(PatternRegion::rectangle(
        Material::anisotropic("tensor mu rectangle", eps, mu),
        {0.0, 0.0},
        0.0,
        {0.22, 0.18}));
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);
    const auto defaultSampled = samplePatternedLayer2d(patterned, basis);
    REQUIRE(defaultSampled.sampleCountX == recommendedPatternSampleCount(basis));
    REQUIRE(defaultSampled.sampleCountY == recommendedPatternSampleCount(basis));
    patterned.sampleCountX = 33;
    patterned.sampleCountY = 35;
    const auto userSampled = samplePatternedLayer2d(patterned, basis);
    REQUIRE(userSampled.sampleCountX == 33);
    REQUIRE(userSampled.sampleCountY == 35);
    patterned.sampleCountX = 0;
    patterned.sampleCountY = 0;

    RcwaSolver patternedSolver;
    patternedSolver.setWavelength(0.92);
    patternedSolver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    patternedSolver.setSuperstrate(Material::vacuum());
    patternedSolver.setSubstrate(Material::vacuum());
    patternedSolver.addPatternedLayer2d(patterned);
    const auto patternedResult = patternedSolver.solve();
    REQUIRE(std::isfinite(patternedResult.rTotal));
    REQUIRE(std::isfinite(patternedResult.tTotal));
    REQUIRE(patternedResult.conservation == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("tensor permeability grating matches fixed numerical benchmark") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    eps(0, 2) = {0.05, 0.0};
    eps(2, 0) = {0.05, 0.0};
    Tensor3 mu = Tensor3::diagonal({1.05, 0.0}, {1.12, 0.0}, {0.97, 0.0});
    mu(0, 2) = {0.02, 0.0};
    mu(2, 0) = {0.02, 0.0};

    GratingLayer grating;
    grating.materialRidge = Material::anisotropic("magnetic benchmark ridge", eps, mu);
    grating.materialGroove = Material::vacuum();
    grating.fillFactor = 0.45;
    grating.periodUm = 1.0;
    grating.periodYUm = 1.0;
    grating.thicknessUm = 0.16;

    RcwaSolver solver;
    solver.setWavelength(1.0);
    solver.setIncidence(7.0, 0.0);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.setPolarization(Polarization::TE);
    solver.addGratingLayer(grating);
    const auto result = solver.solve();
    REQUIRE(result.rTotal == Catch::Approx(0.15265524353243431).margin(1e-11));
    REQUIRE(result.tTotal == Catch::Approx(0.8473447564675656).margin(1e-11));
    REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("invalid lattice inputs are rejected") {
    using namespace rcwa;

    PeriodicLayer2D underSampled;
    underSampled.sampleCountX = 3;
    underSampled.sampleCountY = 5;
    underSampled.periodXUm = 1.0;
    underSampled.periodYUm = 1.0;
    underSampled.thicknessUm = 0.1;
    underSampled.cells.assign(15, Material::vacuum());
    underSampled.cells.front() = Material::constant("sample inclusion", {2.25, 0.0});
    RcwaSolver sampledSolver;
    sampledSolver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    sampledSolver.addPeriodicLayer2d(underSampled);
    REQUIRE(throwsContaining([&] { (void)sampledSolver.solve(); }, "sampling"));

    GratingLayer g;
    g.materialRidge = Material::constant("hi", {2.25, 0.0});
    g.materialGroove = Material::vacuum();
    g.periodUm = 1.0;
    g.periodYUm = 1.0;
    PeriodicLayer2D p;
    p.sampleCountX = 5;
    p.sampleCountY = 5;
    p.periodXUm = 1.1;
    p.periodYUm = 1.0;
    p.thicknessUm = 0.1;
    p.cells.assign(25, Material::vacuum());
    RcwaSolver mixed;
    mixed.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    mixed.addGratingLayer(g);
    mixed.addPeriodicLayer2d(p);
    REQUIRE(throwsContaining([&] { (void)mixed.solve(); }, "lattice"));
}

TEST_CASE("invalid global solve parameters are rejected before computation") {
    using namespace rcwa;

    RcwaSolver badWavelength;
    badWavelength.setWavelength(0.0);
    REQUIRE(throwsContaining([&] { (void)badWavelength.solve(); }, "wavelength"));

    RcwaSolver badAngle;
    badAngle.setIncidence(std::numeric_limits<double>::quiet_NaN(), 0.0);
    REQUIRE(throwsContaining([&] { (void)badAngle.solve(); }, "theta_deg"));

    RcwaSolver grazingIncident;
    grazingIncident.setIncidence(90.0, 0.0);
    REQUIRE(throwsContaining([&] { (void)grazingIncident.solve(); }, "theta_deg"));

    RcwaSolver lossyIncident;
    lossyIncident.setSuperstrate(Material::fromIndex("lossy incident", {1.4, 0.01}));
    REQUIRE(throwsContaining([&] { (void)lossyIncident.solve(); }, "lossless"));

    REQUIRE_THROWS_AS(
        Material::constant(
            "invalid epsilon",
            {std::numeric_limits<double>::quiet_NaN(), 0.0}),
        std::invalid_argument);

    const Complex growing = boundedExp({0.25, 0.5});
    REQUIRE(std::abs(growing) == Catch::Approx(std::exp(0.25)).margin(1e-14));

    RcwaSolver badHarmonics;
    badHarmonics.setHarmonicCount(0, LatticeTruncation::Parallelogramic);
    REQUIRE(throwsContaining([&] { (void)badHarmonics.solve(); }, "harmonic count"));

    REQUIRE(throwsContaining(
        [&] {
            (void)makeHarmonicBasisOrders(
                std::numeric_limits<int>::max(),
                1,
                LatticeTruncation::Parallelogramic);
        },
        "integer index range"));
}

TEST_CASE("2D sampled layer cell count is validated without integer overflow") {
    using namespace rcwa;

    PeriodicLayer2D huge;
    huge.sampleCountX = 50000;
    huge.sampleCountY = 50000;
    huge.periodXUm = 1.0;
    huge.periodYUm = 1.0;
    huge.thicknessUm = 0.1;
    huge.cells.assign(1, Material::vacuum());

    RcwaSolver solver;
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.addPeriodicLayer2d(huge);
    REQUIRE(throwsContaining([&] { (void)solver.solve(); }, "cells size"));
}

TEST_CASE("sampled 2D periodic layers are finite and match uniform limit") {
    using namespace rcwa;

    PeriodicLayer2D layer;
    layer.sampleCountX = 5;
    layer.sampleCountY = 5;
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.2;
    layer.thicknessUm = 0.2;
    layer.cells.reserve(25);
    for (int iy = 0; iy < layer.sampleCountY; ++iy) {
        for (int ix = 0; ix < layer.sampleCountX; ++ix) {
            layer.cells.push_back((ix == iy || ix + iy == 4)
                ? Material::constant("hi", {2.25, 0.0})
                : Material::vacuum());
        }
    }

    RcwaSolver solver;
    solver.setWavelength(0.8);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.setPolarization(Polarization::TE);
    solver.addPeriodicLayer2d(layer);
    const auto r = solver.solve();
    REQUIRE(r.N == 9);
    REQUIRE(r.centerIdx == 4);
    REQUIRE(std::isfinite(r.rTotal));
    REQUIRE(std::isfinite(r.tTotal));
    REQUIRE(std::isfinite(r.conservation));

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    eps(0, 1) = {0.03, 0.0};
    eps(1, 0) = {0.03, 0.0};
    eps(0, 2) = {0.07, 0.0};
    eps(2, 0) = {0.07, 0.0};
    RcwaSolver uniform;
    uniform.setWavelength(0.9);
    uniform.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    uniform.setSuperstrate(Material::vacuum());
    uniform.setSubstrate(Material::vacuum());
    uniform.addUniformLayer(Material::anisotropic("uniform 3D tensor", eps), 0.18);
    const auto ru = uniform.solve();

    PeriodicLayer2D sampled;
    sampled.sampleCountX = 5;
    sampled.sampleCountY = 5;
    sampled.periodXUm = 1.0;
    sampled.periodYUm = 1.1;
    sampled.thicknessUm = 0.18;
    sampled.cells.assign(25, Material::anisotropic("sampled 3D tensor", eps));
    RcwaSolver periodic;
    periodic.setWavelength(0.9);
    periodic.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    periodic.setSuperstrate(Material::vacuum());
    periodic.setSubstrate(Material::vacuum());
    periodic.addPeriodicLayer2d(sampled);
    const auto rp = periodic.solve();
    REQUIRE(rp.rTotal == Catch::Approx(ru.rTotal).margin(1e-7));
    REQUIRE(rp.tTotal == Catch::Approx(ru.tTotal).margin(1e-7));

    std::vector<Tensor3> epsSamples(sampled.cells.size());
    std::vector<Tensor3> muSamples(sampled.cells.size());
    for (std::size_t i = 0; i < sampled.cells.size(); ++i) {
        epsSamples[i] = sampled.cells[i].epsilonTensor(0.9);
        muSamples[i] = sampled.cells[i].muTensor(0.9);
    }
    PrecomputedPeriodicLayer2D precomputed;
    precomputed.tensors = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampled.sampleCountX,
        sampled.sampleCountY,
        3,
        3);
    precomputed.periodXUm = sampled.periodXUm;
    precomputed.periodYUm = sampled.periodYUm;
    precomputed.thicknessUm = sampled.thicknessUm;

    RcwaSolver precomputedSolver;
    precomputedSolver.setWavelength(0.9);
    precomputedSolver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    precomputedSolver.setSuperstrate(Material::vacuum());
    precomputedSolver.setSubstrate(Material::vacuum());
    precomputedSolver.addPrecomputedPeriodicLayer2d(precomputed);
    const auto rpre = precomputedSolver.solve();
    REQUIRE(rpre.rTotal == Catch::Approx(rp.rTotal).margin(1e-10));
    REQUIRE(rpre.tTotal == Catch::Approx(rp.tTotal).margin(1e-10));
}

TEST_CASE("sampled 2D layers use ordered Li L2L1 blocks and keep uniform mu blocks") {
    using namespace rcwa;

    const auto basis = makeHarmonicBasisOrders(2, 2, LatticeTruncation::Parallelogramic);
    PeriodicLayer2D sampled;
    sampled.sampleCountX = 9;
    sampled.sampleCountY = 9;
    sampled.periodXUm = 1.0;
    sampled.periodYUm = 1.0;
    sampled.thicknessUm = 0.2;
    sampled.cells.reserve(81);
    for (int iy = 0; iy < sampled.sampleCountY; ++iy) {
        for (int ix = 0; ix < sampled.sampleCountX; ++ix) {
            sampled.cells.push_back((ix == iy || ix + iy == 8)
                ? Material::constant("hi", {3.61, 0.0})
                : Material::vacuum());
        }
    }

    std::vector<Tensor3> epsSamples(sampled.cells.size());
    std::vector<Tensor3> muSamples(sampled.cells.size());
    for (std::size_t i = 0; i < sampled.cells.size(); ++i) {
        const auto [eps, mu] = sampled.cells[i].tensors(0.9);
        epsSamples[i] = eps;
        muSamples[i] = mu;
    }
    const TensorFourierMatrices tensors = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampled.sampleCountX,
        sampled.sampleCountY,
        basis);
    REQUIRE(tensors.metadata.sequentialLiFactorization2d);
    REQUIRE(maxAbs(tensors.eps[0][0] - tensors.mu[0][0]) > 1e-3);
    REQUIRE(maxAbs(tensors.liEpsQ[0][0] - tensors.eps[0][0]) > 1e-3);
    REQUIRE(maxAbs(tensors.liEpsQ[1][1] - tensors.eps[1][1]) > 1e-3);
    requireMatrixClose(
        tensors.liEpsQZzInverse,
        inverse(tensors.liEpsQ[2][2]),
        1e-12);
    requireScalarIdentityMatrix(tensors.mu[0][0], {1.0, 0.0}, 1e-12);
    requireScalarIdentityMatrix(tensors.liMuQ[0][0], {1.0, 0.0}, 1e-12);
    requireScalarIdentityMatrix(tensors.liMuQZzInverse, {1.0, 0.0}, 1e-12);
}

TEST_CASE("ordered Li L2L1 is built before circular harmonic projection") {
    using namespace rcwa;

    constexpr int sampleCountX = 11;
    constexpr int sampleCountY = 9;
    Tensor3 background = Tensor3::diagonal({1.2, 0.0}, {1.4, 0.0}, {1.1, 0.0});
    Tensor3 inclusion = Tensor3::diagonal({4.0, 0.0}, {3.2, 0.0}, {2.6, 0.0});
    inclusion(0, 1) = {0.0, 0.35};
    inclusion(1, 0) = {0.0, -0.35};
    inclusion(0, 2) = {0.12, 0.03};
    inclusion(2, 0) = {0.12, -0.03};

    std::vector<Tensor3> epsSamples(
        static_cast<std::size_t>(sampleCountX * sampleCountY),
        background);
    std::vector<Tensor3> muSamples(
        epsSamples.size(),
        Tensor3::isotropic({1.0, 0.0}));
    for (int y = 2; y <= 6; ++y) {
        for (int x = 1; x <= 7; ++x) {
            if (!(x >= 5 && y >= 4)) {
                epsSamples[static_cast<std::size_t>(y * sampleCountX + x)] = inclusion;
            }
        }
    }

    const HarmonicBasis rectangularBasis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Parallelogramic);
    const HarmonicBasis circularBasis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Circular);
    const TensorFourierMatrices rectangular = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampleCountX,
        sampleCountY,
        rectangularBasis);
    const TensorFourierMatrices circular = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampleCountX,
        sampleCountY,
        circularBasis);

    std::vector<std::size_t> rectangularIndices;
    for (const HarmonicIndex& retained : circularBasis.orders) {
        const auto found = std::find_if(
            rectangularBasis.orders.begin(),
            rectangularBasis.orders.end(),
            [&](const HarmonicIndex& candidate) {
                return candidate.m == retained.m && candidate.n == retained.n;
            });
        REQUIRE(found != rectangularBasis.orders.end());
        rectangularIndices.push_back(static_cast<std::size_t>(
            std::distance(rectangularBasis.orders.begin(), found)));
    }

    for (std::size_t tensorRow = 0; tensorRow < 3; ++tensorRow) {
        for (std::size_t tensorColumn = 0; tensorColumn < 3; ++tensorColumn) {
            Matrix projected(circularBasis.size(), circularBasis.size());
            for (std::size_t row = 0; row < circularBasis.size(); ++row) {
                for (std::size_t column = 0; column < circularBasis.size(); ++column) {
                    projected(row, column) = rectangular.liEpsQ[tensorRow][tensorColumn](
                        rectangularIndices[row],
                        rectangularIndices[column]);
                }
            }
            requireMatrixClose(
                circular.liEpsQ[tensorRow][tensorColumn],
                projected,
                2e-12);
        }
    }
    requireMatrixClose(
        circular.liEpsQZzInverse,
        inverse(circular.liEpsQ[2][2]),
        2e-12);
}

TEST_CASE("finite ordered Li L2L1 differs from reversed L1L2 for a 2D tensor pattern") {
    using namespace rcwa;

    constexpr int sampleCount = 9;
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        1, 1, LatticeTruncation::Parallelogramic);
    Tensor3 low = Tensor3::diagonal({1.4, 0.0}, {1.7, 0.0}, {1.2, 0.0});
    low(0, 1) = {0.08, 0.04};
    low(1, 0) = {0.03, -0.02};
    Tensor3 high = Tensor3::diagonal({5.0, 0.0}, {3.4, 0.0}, {2.8, 0.0});
    high(0, 1) = {0.0, 0.5};
    high(1, 0) = {0.0, -0.5};
    high(0, 2) = {0.16, 0.02};
    high(2, 0) = {0.09, -0.01};

    std::vector<Tensor3> originalSamples(
        static_cast<std::size_t>(sampleCount * sampleCount),
        low);
    for (int y = 1; y <= 6; ++y) {
        for (int x = 2; x <= 7; ++x) {
            if (x + 2 * y < 14 || (x >= 6 && y <= 3)) {
                originalSamples[static_cast<std::size_t>(y * sampleCount + x)] = high;
            }
        }
    }
    const std::vector<Tensor3> muSamples(
        originalSamples.size(),
        Tensor3::isotropic({1.0, 0.0}));
    const TensorFourierMatrices ordered = tensorFourierMatricesFromSamples2d(
        originalSamples,
        muSamples,
        sampleCount,
        sampleCount,
        basis);

    const std::array<std::size_t, 3> swapAxis{1, 0, 2};
    const auto swapTensorXY = [&](const Tensor3& tensor) {
        Tensor3 swapped;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                swapped(row, column) = tensor(
                    swapAxis[row],
                    swapAxis[column]);
            }
        }
        return swapped;
    };
    std::vector<Tensor3> swappedSamples(originalSamples.size());
    for (int newY = 0; newY < sampleCount; ++newY) {
        for (int newX = 0; newX < sampleCount; ++newX) {
            swappedSamples[static_cast<std::size_t>(newY * sampleCount + newX)] =
                swapTensorXY(originalSamples[
                    static_cast<std::size_t>(newX * sampleCount + newY)]);
        }
    }
    const TensorFourierMatrices swappedOrdered = tensorFourierMatricesFromSamples2d(
        swappedSamples,
        muSamples,
        sampleCount,
        sampleCount,
        basis);

    const auto swappedHarmonicIndex = [&](const HarmonicIndex& originalOrder) {
        const auto found = std::find_if(
            basis.orders.begin(),
            basis.orders.end(),
            [&](const HarmonicIndex& candidate) {
                return candidate.m == originalOrder.n &&
                    candidate.n == originalOrder.m;
            });
        REQUIRE(found != basis.orders.end());
        return static_cast<std::size_t>(std::distance(basis.orders.begin(), found));
    };

    Real largestOrderDifference{};
    for (std::size_t tensorRow = 0; tensorRow < 3; ++tensorRow) {
        for (std::size_t tensorColumn = 0; tensorColumn < 3; ++tensorColumn) {
            Matrix reversed(basis.size(), basis.size());
            for (std::size_t row = 0; row < basis.size(); ++row) {
                const std::size_t swappedRow = swappedHarmonicIndex(basis.orders[row]);
                for (std::size_t column = 0; column < basis.size(); ++column) {
                    const std::size_t swappedColumn =
                        swappedHarmonicIndex(basis.orders[column]);
                    reversed(row, column) = swappedOrdered.liEpsQ
                        [swapAxis[tensorRow]][swapAxis[tensorColumn]](
                            swappedRow,
                            swappedColumn);
                }
            }
            largestOrderDifference = std::max(
                largestOrderDifference,
                maxAbs(ordered.liEpsQ[tensorRow][tensorColumn] - reversed));
        }
    }
    REQUIRE(ordered.metadata.sequentialLiFactorization2d);
    REQUIRE(swappedOrdered.metadata.sequentialLiFactorization2d);
    REQUIRE(largestOrderDifference > Real{1e-4});
    requireMatrixClose(
        ordered.liEpsQ[2][2] * ordered.liEpsQZzInverse,
        Matrix::identity(basis.size()),
        2e-11);
}

TEST_CASE("square inclusion solves with RETICOLO ordered Li blocks for both polarizations") {
    using namespace rcwa;

    PatternedLayer2D square;
    square.background = Material::vacuum();
    square.periodXUm = 1.0;
    square.periodYUm = 1.0;
    square.thicknessUm = 0.5;
    square.regions.push_back(PatternRegion::rectangle(
        Material::constant("epsilon 12 square", {12.0, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.20, 0.20}));

    RcwaSolver solver;
    solver.setWavelength(1.25);
    solver.setIncidence(20.0, 0.0);
    solver.setHarmonicCount(401, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(square);

    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(results.size() == 2);

    REQUIRE(std::isfinite(results[0].rTotal));
    REQUIRE(std::isfinite(results[0].tTotal));
    REQUIRE(std::isfinite(results[1].rTotal));
    REQUIRE(std::isfinite(results[1].tTotal));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(5e-4));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(5e-4));
}

TEST_CASE("concentric square ring solves with RETICOLO ordered Li blocks for both polarizations") {
    using namespace rcwa;

    PatternedLayer2D ring;
    ring.background = Material::vacuum();
    ring.periodXUm = 1.0;
    ring.periodYUm = 1.0;
    ring.thicknessUm = 0.5;
    ring.regions.push_back(PatternRegion::rectangle(
        Material::constant("epsilon 12 square ring", {12.0, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.30, 0.30}));
    ring.regions.push_back(PatternRegion::rectangle(
        Material::vacuum(),
        {0.0, 0.0},
        0.0,
        {0.10, 0.10}));

    RcwaSolver solver;
    solver.setWavelength(1.25);
    solver.setIncidence(20.0, 0.0);
    solver.setHarmonicCount(401, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(ring);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(results.size() == 2);

    REQUIRE(std::isfinite(results[0].rTotal));
    REQUIRE(std::isfinite(results[0].tTotal));
    REQUIRE(std::isfinite(results[1].rTotal));
    REQUIRE(std::isfinite(results[1].tTotal));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(3e-3));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(2e-3));
}

TEST_CASE("concentric square groove solves with RETICOLO ordered Li blocks for both polarizations") {
    using namespace rcwa;

    // This is the complementary material topology used by the Lao 2026
    // square-groove example: a through-cut air ring separates the central and
    // exterior parts of one silicon layer.
    PatternedLayer2D groove;
    groove.background = Material::constant("Si", {3.48 * 3.48, 0.0});
    groove.periodXUm = 5.04;
    groove.periodYUm = 5.04;
    groove.thicknessUm = 4.8;
    groove.regions.push_back(PatternRegion::rectangle(
        Material::vacuum(),
        {0.0, 0.0},
        0.0,
        {1.96, 1.96}));
    groove.regions.push_back(PatternRegion::rectangle(
        groove.background,
        {0.0, 0.0},
        0.0,
        {1.47, 1.47}));

    RcwaSolver solver;
    solver.setWavelength(13.452);
    solver.setIncidence(36.0, 0.0);
    solver.setHarmonicOrders(7, 7, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(groove);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(results.size() == 2);

    REQUIRE(std::isfinite(results[0].rTotal));
    REQUIRE(std::isfinite(results[0].tTotal));
    REQUIRE(std::isfinite(results[1].rTotal));
    REQUIRE(std::isfinite(results[1].tTotal));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(1e-10));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("Huang square-hole stack matches RETICOLO V9 li=1 regression points") {
    using namespace rcwa;

    PatternedLayer2D siliconHole;
    siliconHole.background = Material::constant("Si", {3.48 * 3.48, 0.0});
    siliconHole.periodXUm = 8.3;
    siliconHole.periodYUm = 8.3;
    siliconHole.thicknessUm = 2.3;
    siliconHole.regions.push_back(PatternRegion::rectangle(
        Material::vacuum(),
        {0.0, 0.0},
        0.0,
        {1.2, 1.2}));

    WsmModelParameters wsm;
    wsm.nodeSeparationMInv = 2.0e9;
    wsm.nodeSeparationDirection = {0.0, -1.0, 0.0};
    wsm.fermiVelocityMS = 0.83e5;
    wsm.cutoffXi = 3.0;
    wsm.backgroundEpsilon = 6.2;
    wsm.temperatureK = 300.0;
    wsm.degeneracy = 2.0;
    wsm.relaxationTimeS = 1000.0e-15;
    wsm.fermiEnergyEv = 0.15;

    DrudeModelParameters silver;
    silver.epsInf = 3.4;
    silver.plasmaFrequencyRadS = 1.39e16;
    silver.dampingRateRadS = 2.7e13;

    RcwaSolver solver;
    solver.setHarmonicOrders(5, 5, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(siliconHole);
    solver.addUniformLayer(Material::constant("Ge", {16.0, 0.0}), 2.3);
    solver.addUniformLayer(builtinMaterialModel("WSM", "WSM", wsm), 2.0);
    solver.addUniformLayer(builtinMaterialModel("Ag", "Drude", silver), 0.5);

    struct ReticoloPoint {
        Real wavelengthUm;
        Polarization polarization;
        Real expectedForwardAbsorptance;
        Real expectedReverseAbsorptance;
        Real expectedEta;
    };
    const std::array<ReticoloPoint, 4> points{{
        {16.592, Polarization::TE,
         0.0210180464794983, 0.992348282548286, 0.971330236068787},
        {16.898, Polarization::TM,
         0.0394706829690527, 0.995944471367708, 0.956473788398656},
        {17.069, Polarization::TE,
         0.0164632116174235, 0.928202045962887, 0.911738834345463},
        {17.450, Polarization::TM,
         0.979526271971859, 0.169727191896630, 0.809799080075229},
    }};
    for (const ReticoloPoint& point : points) {
        solver.setWavelength(point.wavelengthUm);
        solver.setPolarization(point.polarization);
        solver.setIncidence(5.0, 0.0);
        const SpectrumResult forward = solver.solveSpectrumOnly();
        solver.setIncidence(-5.0, 0.0);
        const SpectrumResult reverse = solver.solveSpectrumOnly();
        const Real forwardAbsorptance =
            Real{1.0} - forward.rTotal - forward.tTotal;
        const Real reverseAbsorptance =
            Real{1.0} - reverse.rTotal - reverse.tTotal;
        const Real eta = std::abs(forwardAbsorptance - reverseAbsorptance);
        INFO("wavelength_um=" << point.wavelengthUm);
        REQUIRE(forward.N == 121);
        REQUIRE(reverse.N == 121);
        REQUIRE(forwardAbsorptance == Catch::Approx(
            point.expectedForwardAbsorptance).margin(2e-7));
        REQUIRE(reverseAbsorptance == Catch::Approx(
            point.expectedReverseAbsorptance).margin(2e-7));
        REQUIRE(eta == Catch::Approx(point.expectedEta).margin(2e-7));
    }
}

TEST_CASE("Fang square-ring stack uses RETICOLO ordered Li blocks for both polarizations") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material silicon = Material::constant("Si", {12.1104, 0.0});
    PatternedLayer2D ring;
    ring.background = air;
    ring.periodXUm = 1.3;
    ring.periodYUm = 1.3;
    ring.thicknessUm = 0.6;
    ring.regions.push_back(PatternRegion::rectangle(
        silicon,
        {0.0, 0.0},
        0.0,
        {0.455, 0.455}));
    ring.regions.push_back(PatternRegion::rectangle(
        air,
        {0.0, 0.0},
        0.0,
        {0.195, 0.195}));

    RcwaSolver solver;
    solver.setWavelength(1.66924);
    solver.setIncidence(0.8, 0.0);
    solver.setHarmonicOrders(7, 7, LatticeTruncation::Circular);
    solver.setSuperstrate(air);
    solver.setSubstrate(silicon);
    solver.addPatternedLayer2d(ring);
    solver.addUniformLayer(Material::constant("reciprocal Ce:YIG", {4.0, 0.0}), 0.8);
    solver.addUniformLayer(Material::constant(
        "Ag",
        {-148.24128935091596, 3.628268718157638}), 0.2);

    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].N == 149);
    REQUIRE(results[1].N == 149);

    REQUIRE(std::isfinite(results[0].rTotal));
    REQUIRE(std::isfinite(results[1].rTotal));
    REQUIRE(results[0].rTotal >= 0.0);
    REQUIRE(results[0].rTotal <= 1.0);
    REQUIRE(results[1].rTotal >= 0.0);
    REQUIRE(results[1].rTotal <= 1.0);
    REQUIRE(results[0].tTotal < 1e-6);
    REQUIRE(results[1].tTotal < 1e-6);
}

TEST_CASE("circular inclusion solves with RETICOLO ordered Li blocks for both polarizations") {
    using namespace rcwa;

    PatternedLayer2D circle;
    circle.background = Material::vacuum();
    circle.periodXUm = 1.0;
    circle.periodYUm = 1.0;
    circle.thicknessUm = 0.5;
    circle.regions.push_back(PatternRegion::circle(
        Material::constant("epsilon 12 circle", {12.0, 0.0}),
        {0.0, 0.0},
        0.20));

    RcwaSolver solver;
    solver.setWavelength(1.25);
    solver.setIncidence(20.0, 0.0);
    solver.setHarmonicCount(401, LatticeTruncation::Circular);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(circle);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});
    REQUIRE(results.size() == 2);

    REQUIRE(std::isfinite(results[0].rTotal));
    REQUIRE(std::isfinite(results[0].tTotal));
    REQUIRE(std::isfinite(results[1].rTotal));
    REQUIRE(std::isfinite(results[1].tTotal));
    // Pinned ordered y-to-x Li regression using RETICOLO's ten-rectangle,
    // area-normalized circle geometry on the complete mode/interface path.
    REQUIRE(results[0].rTotal ==
            Catch::Approx(0.04559546765498528).margin(3e-11));
    REQUIRE(results[0].tTotal ==
            Catch::Approx(0.9544045323450995).margin(3e-11));
    REQUIRE(results[1].rTotal ==
            Catch::Approx(0.05004494969415518).margin(3e-11));
    REQUIRE(results[1].tTotal ==
            Catch::Approx(0.9499550503057733).margin(3e-11));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(1e-3));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(1e-3));
}

TEST_CASE("circular inclusion matches independent RETICOLO V9 res0 defaults") {
    using namespace rcwa;

    PatternedLayer2D circle;
    circle.background = Material::vacuum();
    circle.periodXUm = 1.0;
    circle.periodYUm = 1.0;
    circle.thicknessUm = 0.5;
    circle.regions.push_back(PatternRegion::circle(
        Material::constant("epsilon 12 circle", {12.0, 0.0}),
        {0.0, 0.0},
        0.20,
        10));

    RcwaSolver solver;
    solver.setWavelength(1.25);
    solver.setIncidence(20.0, 0.0);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(circle);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});

    // Exported independently from MATLAB R2024a + RETICOLO V9 using res0,
    // res1.li=1, res1.angles=1 and texture selector k=10.
    REQUIRE(results[0].rTotal ==
            Catch::Approx(0.41355764220855507).margin(3e-12));
    REQUIRE(results[0].tTotal ==
            Catch::Approx(0.58644235779144549).margin(3e-12));
    REQUIRE(results[1].rTotal ==
            Catch::Approx(0.31970352001138425).margin(3e-12));
    REQUIRE(results[1].tTotal ==
            Catch::Approx(0.68029647998861398).margin(3e-12));
}

TEST_CASE("RETICOLO circular-ring Li blocks are covariant under periodic translation") {
    using namespace rcwa;

    constexpr int sampleCount = 129;
    const Real shiftX = Real{13.0} / sampleCount;
    const Real shiftY = -Real{17.0} / sampleCount;
    const Material dielectric = Material::constant(
        "epsilon 12 translated ring",
        {12.0, 0.0});
    const auto makeRing = [&](Real dx, Real dy) {
        PatternedLayer2D ring;
        ring.background = Material::vacuum();
        ring.periodXUm = 1.0;
        ring.periodYUm = 1.0;
        ring.thicknessUm = 0.5;
        ring.sampleCountX = sampleCount;
        ring.sampleCountY = sampleCount;
        ring.regions.push_back(PatternRegion::circle(
            dielectric,
            {0.02 + dx, -0.03 + dy},
            0.30));
        ring.regions.push_back(PatternRegion::circle(
            Material::vacuum(),
            {0.08 + dx, -0.01 + dy},
            0.11));
        return ring;
    };
    const auto solve = [&](const PatternedLayer2D& ring) {
        RcwaSolver solver;
        solver.setWavelength(1.25);
        solver.setIncidence(23.0, 37.0);
        solver.setHarmonicOrders(4, 4, LatticeTruncation::Parallelogramic);
        solver.setSuperstrate(Material::vacuum());
        solver.setSubstrate(Material::vacuum());
        solver.addPatternedLayer2d(ring);
        return solver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
    };

    const auto reference = solve(makeRing(0.0, 0.0));
    const auto translated = solve(makeRing(shiftX, shiftY));
    REQUIRE(reference.size() == 2);
    REQUIRE(translated.size() == 2);
    for (std::size_t polarization = 0; polarization < 2; ++polarization) {
        REQUIRE(translated[polarization].rTotal ==
                Catch::Approx(reference[polarization].rTotal).margin(2e-9));
        REQUIRE(translated[polarization].tTotal ==
                Catch::Approx(reference[polarization].tTotal).margin(2e-9));
    }
}

TEST_CASE("analytic lamellar grating matches the independent S4 proper FFF benchmark") {
    using namespace rcwa;

    PatternedLayer2D lamellar;
    lamellar.background = Material::vacuum();
    lamellar.periodXUm = 1.0;
    lamellar.periodYUm = 1.0;
    lamellar.thicknessUm = 0.5;
    lamellar.regions.push_back(PatternRegion::rectangle(
        Material::constant("epsilon 12 ridge", {12.0, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.20, 0.50}));

    RcwaSolver solver;
    solver.setWavelength(1.25);
    solver.setIncidence(20.0, 0.0);
    solver.setHarmonicOrders(10, 0, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPatternedLayer2d(lamellar);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});

    // Independent S4 7fd00a2, 21 retained 1-D Fourier harmonics.
    REQUIRE(results[0].rTotal == Catch::Approx(0.622323055267167).margin(2e-12));
    REQUIRE(results[0].tTotal == Catch::Approx(0.377676944732833).margin(2e-12));
    REQUIRE(results[1].rTotal == Catch::Approx(0.226742839955480).margin(2e-12));
    REQUIRE(results[1].tTotal == Catch::Approx(0.773257160044518).margin(2e-12));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(2e-12));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(2e-12));
}

TEST_CASE("genuinely 2D square pillar uses the RETICOLO li=1 regression") {
    using namespace rcwa;

    // CELERIS aac766c selftest [8b] uses this TiO2/fused-silica cell.  This is
    // a genuinely 2D interface, so the pinned spectrum distinguishes
    // RETICOLO li=1 from the former ordinary-convolution and independent-axis
    // rules.
    PatternedLayer2D pillar;
    pillar.background = Material::vacuum();
    pillar.periodXUm = 0.35;
    pillar.periodYUm = 0.35;
    pillar.thicknessUm = 0.60;
    pillar.regions.push_back(PatternRegion::rectangle(
        Material::constant("TiO2 n=2.45", {2.45 * 2.45, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.0875, 0.0875}));

    RcwaSolver solver;
    solver.setWavelength(0.532);
    solver.setHarmonicOrders(6, 6, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::constant(
        "fused silica at 532 nm",
        {2.1336630260081355, 0.0}));
    solver.addPatternedLayer2d(pillar);
    const auto results = solver.solveSpectrumOnlyForPolarizations(
        {Polarization::TE, Polarization::TM});

    REQUIRE(results.size() == 2);
    REQUIRE(results[0].N == 169);
    REQUIRE(results[0].rTotal ==
            Catch::Approx(0.08182271223251396).margin(4e-11));
    REQUIRE(results[0].tTotal ==
            Catch::Approx(0.9181772877675739).margin(4e-11));
    REQUIRE(results[1].rTotal ==
            Catch::Approx(0.08207425904925131).margin(4e-11));
    REQUIRE(results[1].tTotal ==
            Catch::Approx(0.9179257409507274).margin(4e-11));
    REQUIRE(results[0].conservation == Catch::Approx(1.0).margin(3e-11));
    REQUIRE(results[1].conservation == Catch::Approx(1.0).margin(3e-11));
}

TEST_CASE("S4 normal-vector formulation improves the square-pillar convergence sequence") {
    using namespace rcwa;

    PatternedLayer2D pillar;
    pillar.background = Material::vacuum();
    pillar.periodXUm = 0.35;
    pillar.periodYUm = 0.35;
    pillar.thicknessUm = 0.60;
    pillar.regions.push_back(PatternRegion::rectangle(
        Material::constant("TiO2 n=2.45", {2.45 * 2.45, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.0875, 0.0875}));
    pillar.fourierOptions.polarizationDecomposition = true;
    pillar.fourierOptions.polarizationBasis = PolarizationBasis::Normal;
    pillar.fourierOptions.resolution = 8;

    // Independent S4 7fd00a2, FMMGetEpsilon_PolBasisNV with the same complete
    // square G grids.  The tolerances cover local LAPACK/FFTW conventions but
    // reject the former fixed-iteration neighborhood diffusion field.
    const std::array<int, 3> orders{2, 4, 6};
    const std::array<Real, 3> expectedR{
        0.1032605495289372,
        0.02557032452195428,
        0.03364727441167452,
    };
    const std::array<Real, 3> expectedT{
        0.8986172058386869,
        0.9740217884228237,
        0.9668080683411946,
    };
    const std::array<Real, 3> margins{2.1e-3, 3.0e-4, 2.0e-4};
    for (std::size_t index = 0; index < orders.size(); ++index) {
        RcwaSolver solver;
        solver.setWavelength(0.532);
        solver.setHarmonicOrders(
            orders[index], orders[index], LatticeTruncation::Parallelogramic);
        solver.setSuperstrate(Material::vacuum());
        solver.setSubstrate(Material::constant(
            "fused silica at 532 nm",
            {2.1336630260081355, 0.0}));
        solver.setFourierConvergenceOptions(pillar.fourierOptions);
        solver.addPatternedLayer2d(pillar);
        const SpectrumResult result = solver.solve();
        REQUIRE(result.rTotal == Catch::Approx(expectedR[index]).margin(margins[index]));
        REQUIRE(result.tTotal == Catch::Approx(expectedT[index]).margin(margins[index]));
    }
}

TEST_CASE("y-invariant 2D grid reduces to the S4 one-dimensional limit") {
    using namespace rcwa;

    PatternedLayer2D lamellar;
    lamellar.background = Material::vacuum();
    lamellar.periodXUm = 0.30;
    lamellar.periodYUm = 0.30;
    lamellar.thicknessUm = 0.50;
    lamellar.regions.push_back(PatternRegion::rectangle(
        Material::constant("n=1.5 ridge", {2.25, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.075, 0.150}));

    const auto solve = [&](int orderY) {
        RcwaSolver solver;
        solver.setWavelength(0.50);
        solver.setHarmonicOrders(4, orderY, LatticeTruncation::Parallelogramic);
        solver.setSuperstrate(Material::vacuum());
        solver.setSubstrate(Material::vacuum());
        solver.addPatternedLayer2d(lamellar);
        return solver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
    };

    const auto oneDimensional = solve(0);
    const auto twoDimensionalGrid = solve(2);
    REQUIRE(oneDimensional.size() == 2);
    REQUIRE(twoDimensionalGrid.size() == 2);
    REQUIRE(oneDimensional[0].tTotal ==
            Catch::Approx(0.93334495211482837).margin(3e-12));
    REQUIRE(oneDimensional[1].tTotal ==
            Catch::Approx(0.96326685470441886).margin(3e-12));
    for (std::size_t polarization = 0; polarization < 2; ++polarization) {
        REQUIRE(twoDimensionalGrid[polarization].rTotal ==
                Catch::Approx(oneDimensional[polarization].rTotal).margin(3e-12));
        REQUIRE(twoDimensionalGrid[polarization].tTotal ==
                Catch::Approx(oneDimensional[polarization].tTotal).margin(3e-12));
    }
}

TEST_CASE("rectangular scalar Li matrices match RETICOLO V9 ret2li li=1") {
    using namespace rcwa;

    // Independent values exported from RETICOLO V9 retcouche.m/ret2li with
    // res0.res1.li=1, nx=ny=3.  The eccentric non-square hole distinguishes
    // y->x ordering from both direct convolution and automatic-axis ordering.
    PatternedLayer2D layer;
    layer.background = Material::constant("epsilon 12", {12.0, 0.0});
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.5;
    layer.regions.push_back(PatternRegion::rectangle(
        Material::vacuum(),
        {0.07, -0.11},
        0.0,
        {0.20, 0.30}));
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        1, 1, LatticeTruncation::Parallelogramic);
    const PrecomputedPeriodicLayer2D precomputed =
        precomputePatternedLayer2d(layer, basis, 1.25);
    const TensorFourierMatrices& tensors = precomputed.tensors;
    const std::size_t center = static_cast<std::size_t>(basis.centerIndex());
    constexpr std::size_t referenceColumn = 1;
    const auto frobeniusNorm = [](const Matrix& matrix) {
        Real sum{};
        for (const Complex value : matrix.data()) {
            sum += std::norm(value);
        }
        return std::sqrt(sum);
    };

    REQUIRE(tensors.metadata.sequentialLiFactorization2d);
    REQUIRE_FALSE(tensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE(std::abs(
        tensors.liEpsQ[0][0](center, center) -
        Complex{8.6426077870962281, 0.0}) <= 2e-13);
    REQUIRE(std::abs(
        tensors.liEpsQ[0][0](center, referenceColumn) -
        Complex{-1.3724925446302587, -1.135424578228988}) <= 2e-13);
    REQUIRE(frobeniusNorm(tensors.liEpsQ[0][0]) ==
        Catch::Approx(27.096772446513562).margin(2e-13));
    REQUIRE(std::abs(
        tensors.liEpsQ[1][1](center, center) -
        Complex{8.3361480756572597, 0.0}) <= 2e-13);
    REQUIRE(std::abs(
        tensors.liEpsQ[1][1](center, referenceColumn) -
        Complex{-0.44364464210578936, -0.36701476639511887}) <= 2e-13);
    REQUIRE(frobeniusNorm(tensors.liEpsQ[1][1]) ==
        Catch::Approx(26.799436746167416).margin(2e-13));
    REQUIRE(std::abs(
        tensors.liEpsQZzInverse(center, center) -
        Complex{0.16365360707341964, 0.0}) <= 2e-13);
    REQUIRE(std::abs(
        tensors.liEpsQZzInverse(center, referenceColumn) -
        Complex{0.037215494311020926, 0.030787334399005887}) <= 2e-13);
    REQUIRE(frobeniusNorm(tensors.liEpsQZzInverse) ==
        Catch::Approx(0.50621556258171696).margin(2e-13));
    requireMatrixClose(
        tensors.liEpsQ[2][2], tensors.eps[2][2], 2e-13);
    requireMatrixClose(
        tensors.liEpsQ[2][2] * tensors.liEpsQZzInverse,
        Matrix::identity(basis.size()),
        2e-13);
    REQUIRE(maxAbs(tensors.liEpsQ[0][0] - tensors.eps[0][0]) > 1.0);
    REQUIRE(maxAbs(tensors.liEpsQ[1][1] - tensors.eps[1][1]) > 1.0);
}

TEST_CASE("library patterned 2D layers parse simple regions") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("dielectric", {2.25, 0.0});
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);

    PatternedLayer2D rectangle;
    rectangle.background = air;
    rectangle.periodXUm = 1.0;
    rectangle.periodYUm = 1.0;
    rectangle.thicknessUm = 0.2;
    rectangle.sampleCountX = 3;
    rectangle.sampleCountY = 3;
    rectangle.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.20, 0.20}));

    const PrecomputedPeriodicLayer2D precomputed =
        precomputePatternedLayer2d(rectangle, basis, 0.8);
    const int zeroOrder = basis.centerIndex();
    const double rectangleFill = 4.0 * 0.20 * 0.20;
    const double expectedRectangleAverageEps = 1.0 + (2.25 - 1.0) * rectangleFill;
    REQUIRE(std::real(precomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedRectangleAverageEps).margin(1e-12));
    requireScalarIdentityMatrix(precomputed.tensors.mu[0][0], {1.0, 0.0}, 1e-12);
    requireScalarIdentityMatrix(precomputed.tensors.liMuQZzInverse, {1.0, 0.0}, 1e-12);
    REQUIRE(precomputed.tensors.metadata.sequentialLiFactorization2d);
    REQUIRE(maxAbs(
        precomputed.tensors.liEpsQ[0][0] -
        precomputed.tensors.eps[0][0]) > 1e-3);
    REQUIRE(maxAbs(
        precomputed.tensors.liEpsQ[1][1] -
        precomputed.tensors.eps[1][1]) > 1e-3);
    requireMatrixClose(
        precomputed.tensors.liEpsQ[2][2],
        precomputed.tensors.eps[2][2],
        1e-12);
    requireMatrixClose(
        precomputed.tensors.liEpsQZzInverse,
        inverse(precomputed.tensors.eps[2][2]),
        1e-12);
    Tensor3 epsTensor = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    epsTensor(0, 1) = {0.07, 0.0};
    epsTensor(1, 0) = {0.07, 0.0};
    PatternedLayer2D anisotropicRectangle = rectangle;
    anisotropicRectangle.regions.clear();
    anisotropicRectangle.regions.push_back(PatternRegion::rectangle(
        Material::anisotropic("anisotropic rectangle", epsTensor),
        {0.0, 0.0},
        0.0,
        {0.20, 0.20}));
    const PrecomputedPeriodicLayer2D precomputedAnisotropic =
        precomputePatternedLayer2d(anisotropicRectangle, basis, 0.8);
    REQUIRE(std::real(precomputedAnisotropic.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(1.0 + (2.25 - 1.0) * rectangleFill).margin(1e-12));
    REQUIRE(std::real(precomputedAnisotropic.tensors.eps[0][1](zeroOrder, zeroOrder)) ==
            Catch::Approx(0.07 * rectangleFill).margin(1e-12));
    REQUIRE(precomputedAnisotropic.tensors.metadata.sequentialLiFactorization2d);
    REQUIRE_FALSE(
        precomputedAnisotropic.tensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(precomputedAnisotropic.tensors.metadata.requiresJointConvergence);
    requireMatrixClose(
        precomputedAnisotropic.tensors.eps[0][1],
        Complex{0.07 / 1.25, 0.0} *
            (precomputed.tensors.eps[0][0] - Matrix::identity(basis.size())),
        1e-12);
    REQUIRE(maxAbs(
        precomputedAnisotropic.tensors.liEpsQ[0][0] -
        precomputedAnisotropic.tensors.eps[0][0]) > 1e-3);
    requireMatrixClose(
        precomputedAnisotropic.tensors.liEpsQZzInverse,
        inverse(precomputedAnisotropic.tensors.liEpsQ[2][2]),
        1e-12);

    PatternedLayer2D xLamellar = rectangle;
    xLamellar.regions.clear();
    xLamellar.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.25, 0.50}));
    const PrecomputedPeriodicLayer2D precomputedXLamellar =
        precomputePatternedLayer2d(xLamellar, basis, 0.8);
    PatternedLayer2D reciprocalXLamellar = xLamellar;
    reciprocalXLamellar.background =
        Material::constant("air reciprocal epsilon", {1.0, 0.0});
    reciprocalXLamellar.regions.clear();
    reciprocalXLamellar.regions.push_back(PatternRegion::rectangle(
        Material::constant("dielectric reciprocal epsilon", {1.0 / 2.25, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.25, 0.50}));
    const TensorFourierMatrices reciprocalXTensors =
        tensorFourierMatricesPatternedLayer2d(reciprocalXLamellar, basis, 0.8);
    const Matrix expectedXLi = inverse(reciprocalXTensors.eps[0][0]);
    requireMatrixClose(precomputedXLamellar.tensors.liEpsQ[0][0], expectedXLi, 1e-11);
    requireMatrixClose(
        precomputedXLamellar.tensors.liEpsQ[1][1],
        precomputedXLamellar.tensors.eps[1][1],
        1e-12);

    PatternedLayer2D yLamellar = rectangle;
    yLamellar.regions.clear();
    yLamellar.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.50, 0.25}));
    const PrecomputedPeriodicLayer2D precomputedYLamellar =
        precomputePatternedLayer2d(yLamellar, basis, 0.8);
    PatternedLayer2D reciprocalYLamellar = yLamellar;
    reciprocalYLamellar.background =
        Material::constant("air reciprocal epsilon y", {1.0, 0.0});
    reciprocalYLamellar.regions.clear();
    reciprocalYLamellar.regions.push_back(PatternRegion::rectangle(
        Material::constant("dielectric reciprocal epsilon y", {1.0 / 2.25, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.50, 0.25}));
    const TensorFourierMatrices reciprocalYTensors =
        tensorFourierMatricesPatternedLayer2d(reciprocalYLamellar, basis, 0.8);
    const Matrix expectedYLi = inverse(reciprocalYTensors.eps[1][1]);
    requireMatrixClose(
        precomputedYLamellar.tensors.liEpsQ[0][0],
        precomputedYLamellar.tensors.eps[0][0],
        1e-12);
    requireMatrixClose(precomputedYLamellar.tensors.liEpsQ[1][1], expectedYLi, 1e-11);

    RcwaSolver solver;
    solver.setWavelength(0.8);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TE);
    solver.addPatternedLayer2d(rectangle);
    const auto r = solver.solve();
    REQUIRE(r.N == 9);
    REQUIRE(std::isfinite(r.rTotal));
    REQUIRE(std::isfinite(r.tTotal));

    PatternedLayer2D squareHole = rectangle;
    squareHole.regions.clear();
    squareHole.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.30, 0.30}));
    squareHole.regions.push_back(PatternRegion::rectangle(
        air,
        {0.0, 0.0},
        0.0,
        {0.10, 0.10}));
    const PrecomputedPeriodicLayer2D precomputedHole =
        precomputePatternedLayer2d(squareHole, basis, 0.8);
    const double outerFill = 4.0 * 0.30 * 0.30;
    const double innerFill = 4.0 * 0.10 * 0.10;
    const double expectedAverageEps = 1.0 + (2.25 - 1.0) * (outerFill - innerFill);
    REQUIRE(std::real(precomputedHole.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedAverageEps).margin(1e-12));
    REQUIRE(std::imag(precomputedHole.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(precomputedHole.tensors.metadata.sequentialLiFactorization2d);
    REQUIRE(maxAbs(
        precomputedHole.tensors.liEpsQ[0][0] -
        precomputedHole.tensors.eps[0][0]) > 1e-3);
    REQUIRE(maxAbs(
        precomputedHole.tensors.liEpsQ[1][1] -
        precomputedHole.tensors.eps[1][1]) > 1e-3);
    requireMatrixClose(
        precomputedHole.tensors.liEpsQ[2][2],
        precomputedHole.tensors.eps[2][2],
        1e-12);
    requireMatrixClose(
        precomputedHole.tensors.liEpsQZzInverse,
        inverse(precomputedHole.tensors.eps[2][2]),
        1e-12);
    requireScalarIdentityMatrix(
        precomputedHole.tensors.liMuQZzInverse,
        {1.0, 0.0},
        1e-12);

    PatternedLayer2D outerSquareOnly = squareHole;
    outerSquareOnly.regions.resize(1);
    PatternedLayer2D redundantNestedSquare = outerSquareOnly;
    redundantNestedSquare.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.10, 0.10}));
    const TensorFourierMatrices outerSquareOnlyTensors =
        tensorFourierMatricesPatternedLayer2d(outerSquareOnly, basis, 0.8);
    const TensorFourierMatrices redundantNestedSquareTensors =
        tensorFourierMatricesPatternedLayer2d(
            redundantNestedSquare,
            basis,
            0.8);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            requireMatrixClose(
                redundantNestedSquareTensors.eps[row][column],
                outerSquareOnlyTensors.eps[row][column],
                1e-12);
            requireMatrixClose(
                redundantNestedSquareTensors.liEpsQ[row][column],
                outerSquareOnlyTensors.liEpsQ[row][column],
                1e-12);
        }
    }
    requireMatrixClose(
        redundantNestedSquareTensors.liEpsQZzInverse,
        outerSquareOnlyTensors.liEpsQZzInverse,
        1e-12);

    RcwaSolver squareHoleSolver;
    squareHoleSolver.setWavelength(0.8);
    squareHoleSolver.setIncidence(17.0, 23.0);
    squareHoleSolver.setHarmonicOrders(
        2,
        2,
        LatticeTruncation::Parallelogramic);
    squareHoleSolver.setSuperstrate(air);
    squareHoleSolver.setSubstrate(air);
    squareHoleSolver.addPatternedLayer2d(squareHole);
    const auto squareHoleResults =
        squareHoleSolver.solveSpectrumOnlyForPolarizations(
            {Polarization::TE, Polarization::TM});
    REQUIRE(squareHoleResults.size() == 2);
    for (const auto& result : squareHoleResults) {
        REQUIRE(std::isfinite(result.rTotal));
        REQUIRE(std::isfinite(result.tTotal));
        REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-3));
    }

    PatternedLayer2D rectangleCross = rectangle;
    rectangleCross.sampleCountX = 3;
    rectangleCross.sampleCountY = 3;
    rectangleCross.regions.clear();
    rectangleCross.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.15, 0.50}));
    rectangleCross.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.50, 0.15}));
    const PrecomputedPeriodicLayer2D precomputedCross =
        precomputePatternedLayer2d(rectangleCross, basis, 0.8);
    const double crossFill = 0.30 + 0.30 - 0.09;
    const double expectedCrossAverageEps = 1.0 + (2.25 - 1.0) * crossFill;
    REQUIRE(std::real(precomputedCross.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedCrossAverageEps).margin(1e-12));

    PatternedLayer2D threeRectangles = rectangle;
    threeRectangles.sampleCountX = 3;
    threeRectangles.sampleCountY = 3;
    threeRectangles.regions.clear();
    threeRectangles.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.0, 0.0},
        0.0,
        {0.10, 0.50}));
    threeRectangles.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.20, -0.25},
        0.0,
        {0.10, 0.10}));
    threeRectangles.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {-0.20, 0.25},
        0.0,
        {0.10, 0.10}));
    const PrecomputedPeriodicLayer2D precomputedThreeRectangles =
        precomputePatternedLayer2d(threeRectangles, basis, 0.8);
    const double threeRectangleFill = 0.20 + 0.04 + 0.04;
    const double expectedThreeRectangleAverageEps =
        1.0 + (2.25 - 1.0) * threeRectangleFill;
    REQUIRE(std::real(
                precomputedThreeRectangles.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedThreeRectangleAverageEps).margin(1e-12));

    PatternedLayer2D eccentricRing = rectangle;
    eccentricRing.regions.clear();
    eccentricRing.regions.push_back(PatternRegion::circle(
        dielectric,
        {0.02, -0.03},
        0.30));
    eccentricRing.regions.push_back(PatternRegion::circle(
        air,
        {0.08, -0.01},
        0.11));
    const PrecomputedPeriodicLayer2D precomputedRing =
        precomputePatternedLayer2d(eccentricRing, basis, 0.8);
    const double outerCircleFill = rcwa::pi * 0.30 * 0.30;
    const double innerCircleFill = rcwa::pi * 0.11 * 0.11;
    const double expectedRingAverageEps =
        1.0 + (2.25 - 1.0) * (outerCircleFill - innerCircleFill);
    REQUIRE(std::real(precomputedRing.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedRingAverageEps).margin(1e-12));
    REQUIRE(std::imag(precomputedRing.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(0.0).margin(1e-12));
    REQUIRE(precomputedRing.tensors.metadata.sequentialLiFactorization2d);
    REQUIRE(maxAbs(
        precomputedRing.tensors.liEpsQ[0][0] -
        precomputedRing.tensors.eps[0][0]) > 1e-3);
    REQUIRE(maxAbs(
        precomputedRing.tensors.liEpsQ[1][1] -
        precomputedRing.tensors.eps[1][1]) > 1e-3);
    requireMatrixClose(
        precomputedRing.tensors.liEpsQ[2][2] *
            precomputedRing.tensors.liEpsQZzInverse,
        Matrix::identity(basis.size()),
        1e-12);

    PatternedLayer2D polygon = rectangle;
    polygon.regions.clear();
    polygon.sampleCountX = 9;
    polygon.sampleCountY = 9;
    polygon.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.0, 0.0},
        0.0,
        {{-0.2, -0.2}, {0.2, -0.2}, {0.0, 0.2}}));
    const auto sampled = samplePatternedLayer2d(polygon, basis);
    REQUIRE(sampled.sampleCountX == 9);
    REQUIRE(sampled.sampleCountY == 9);
    REQUIRE(sampled.cells.size() == 81);
}

TEST_CASE("arbitrary shapes use a common rectilinear geometry for direct and Li blocks") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("dielectric", {2.25, 0.0});
    const Material highIndex = Material::constant("high index", {3.61, 0.0});
    const Real wavelength = 0.83;
    const auto basis = makeHarmonicBasisOrders(2, 2, LatticeTruncation::Parallelogramic);

    PatternedLayer2D rectangle;
    rectangle.background = air;
    rectangle.periodXUm = 1.20;
    rectangle.periodYUm = 0.95;
    rectangle.thicknessUm = 0.18;
    rectangle.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.07, -0.04},
        17.0,
        {0.18, 0.13}));

    PatternedLayer2D polygonRectangle = rectangle;
    polygonRectangle.regions.clear();
    polygonRectangle.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.07, -0.04},
        17.0,
        {{-0.18, -0.13}, {0.18, -0.13}, {0.18, 0.13}, {-0.18, 0.13}}));

    const TensorFourierMatrices rectangleTensors =
        tensorFourierMatricesPatternedLayer2d(rectangle, basis, wavelength);
    const TensorFourierMatrices polygonRectangleTensors =
        tensorFourierMatricesPatternedLayer2d(polygonRectangle, basis, wavelength);
    requireMatrixClose(rectangleTensors.eps[0][0], polygonRectangleTensors.eps[0][0], 1e-12);
    requireMatrixClose(rectangleTensors.mu[1][1], polygonRectangleTensors.mu[1][1], 1e-12);
    requireMatrixClose(
        rectangleTensors.liEpsQZzInverse,
        polygonRectangleTensors.liEpsQZzInverse,
        1e-10);

    PatternedLayer2D triangle;
    triangle.background = air;
    triangle.periodXUm = 1.0;
    triangle.periodYUm = 1.0;
    triangle.thicknessUm = 0.20;
    const std::vector<Vec2> triangleVertices{
        {-0.22, -0.16},
        {0.18, -0.16},
        {-0.04, 0.19},
    };
    triangle.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.05, -0.03},
        0.0,
        triangleVertices));
    const PrecomputedPeriodicLayer2D trianglePrecomputed =
        precomputePatternedLayer2d(triangle, basis, wavelength);
    const int zeroOrder = basis.centerIndex();
    const double triangleArea = 0.5 * 0.40 * 0.35;
    const double expectedTriangleAverageEps = 1.0 + (2.25 - 1.0) * triangleArea;
    REQUIRE(std::real(trianglePrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedTriangleAverageEps).margin(2e-3));
    REQUIRE(std::imag(trianglePrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(0.0).margin(1e-12));

    PatternedLayer2D reversedTriangle = triangle;
    reversedTriangle.regions.clear();
    std::vector<Vec2> reversedVertices = triangleVertices;
    std::reverse(reversedVertices.begin(), reversedVertices.end());
    reversedTriangle.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.05, -0.03},
        0.0,
        reversedVertices));
    const PrecomputedPeriodicLayer2D reversedTrianglePrecomputed =
        precomputePatternedLayer2d(reversedTriangle, basis, wavelength);
    requireMatrixClose(
        trianglePrecomputed.tensors.eps[0][0],
        reversedTrianglePrecomputed.tensors.eps[0][0],
        1e-12);

    PatternedLayer2D disjointPolygons = triangle;
    disjointPolygons.regions.clear();
    disjointPolygons.regions.push_back(PatternRegion::polygon(
        dielectric,
        {-0.24, 0.12},
        0.0,
        {{-0.08, -0.07}, {0.08, -0.07}, {0.0, 0.07}}));
    disjointPolygons.regions.push_back(PatternRegion::polygon(
        highIndex,
        {0.24, -0.12},
        30.0,
        {{-0.07, -0.06}, {0.07, -0.06}, {0.0, 0.06}}));
    const PrecomputedPeriodicLayer2D disjointPrecomputed =
        precomputePatternedLayer2d(disjointPolygons, basis, wavelength);
    const double firstArea = 0.5 * 0.16 * 0.14;
    const double secondArea = 0.5 * 0.14 * 0.12;
    const double expectedDisjointAverageEps =
        1.0 + (2.25 - 1.0) * firstArea + (3.61 - 1.0) * secondArea;
    REQUIRE(std::real(disjointPrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedDisjointAverageEps).margin(3e-3));

    PatternedLayer2D mixedDisjoint = triangle;
    mixedDisjoint.regions.clear();
    mixedDisjoint.regions.push_back(PatternRegion::circle(
        dielectric,
        {-0.26, -0.20},
        0.075));
    mixedDisjoint.regions.push_back(PatternRegion::rectangle(
        highIndex,
        {0.23, 0.20},
        -24.0,
        {0.08, 0.055}));
    const PrecomputedPeriodicLayer2D mixedDisjointPrecomputed =
        precomputePatternedLayer2d(mixedDisjoint, basis, wavelength);
    const double circleArea = rcwa::pi * 0.075 * 0.075;
    const double rotatedRectangleArea = 4.0 * 0.08 * 0.055;
    const double expectedMixedAverageEps =
        1.0 + (2.25 - 1.0) * circleArea + (3.61 - 1.0) * rotatedRectangleArea;
    REQUIRE(std::real(
                mixedDisjointPrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedMixedAverageEps).margin(1e-2));

    PatternedLayer2D polygonHole = triangle;
    polygonHole.regions.clear();
    polygonHole.regions.push_back(PatternRegion::polygon(
        highIndex,
        {0.0, 0.0},
        0.0,
        {{-0.28, -0.24}, {0.28, -0.24}, {0.28, 0.24}, {-0.28, 0.24}}));
    polygonHole.regions.push_back(PatternRegion::polygon(
        air,
        {0.0, 0.0},
        15.0,
        {{-0.10, -0.08}, {0.10, -0.08}, {0.10, 0.08}, {-0.10, 0.08}}));
    polygonHole.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.0, 0.0},
        -20.0,
        {{-0.035, -0.030}, {0.035, -0.030}, {0.0, 0.035}}));
    const PrecomputedPeriodicLayer2D polygonHolePrecomputed =
        precomputePatternedLayer2d(polygonHole, basis, wavelength);
    const double outerArea = 0.56 * 0.48;
    const double holeArea = 0.20 * 0.16;
    const double islandArea = 0.5 * 0.07 * 0.065;
    const double expectedHoleAverageEps =
        1.0 + (3.61 - 1.0) * outerArea +
        (1.0 - 3.61) * holeArea +
        (2.25 - 1.0) * islandArea;
    REQUIRE(std::real(polygonHolePrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedHoleAverageEps).margin(1e-2));

    PatternedLayer2D laterOuterCoversInner = triangle;
    laterOuterCoversInner.regions.clear();
    laterOuterCoversInner.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.0, 0.0},
        0.0,
        {{-0.04, -0.04}, {0.04, -0.04}, {0.0, 0.05}}));
    laterOuterCoversInner.regions.push_back(PatternRegion::circle(
        highIndex,
        {0.0, 0.0},
        0.18));
    const PrecomputedPeriodicLayer2D laterOuterPrecomputed =
        precomputePatternedLayer2d(laterOuterCoversInner, basis, wavelength);
    const double expectedLaterOuterAverageEps =
        1.0 + (3.61 - 1.0) * rcwa::pi * 0.18 * 0.18;
    REQUIRE(std::real(laterOuterPrecomputed.tensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(expectedLaterOuterAverageEps).margin(1e-2));

    RcwaSolver solver;
    solver.setWavelength(wavelength);
    solver.setHarmonicOrders(2, 2, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TE);
    solver.addPatternedLayer2d(disjointPolygons);
    const auto result = solver.solve();
    REQUIRE(result.N == 25);
    REQUIRE(std::isfinite(result.rTotal));
    REQUIRE(std::isfinite(result.tTotal));
    REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-4));
}

TEST_CASE("rectilinear shape Fourier coefficients approach midpoint integration") {
    using namespace rcwa;

    constexpr Real periodX = 1.30;
    constexpr Real periodY = 1.10;
    constexpr int deltaM = 2;
    constexpr int deltaN = -1;
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Parallelogramic);
    const auto harmonicIndex = [&](int m, int n) {
        const auto it = std::find_if(
            basis.orders.begin(),
            basis.orders.end(),
            [&](const HarmonicIndex& order) {
                return order.m == m && order.n == n;
            });
        REQUIRE(it != basis.orders.end());
        return static_cast<std::size_t>(std::distance(basis.orders.begin(), it));
    };
    const auto numericalIndicatorCoefficient = [&](const PatternRegion& region) {
        constexpr int samplesX = 801;
        constexpr int samplesY = 797;
        const Real gx = twoPi * static_cast<Real>(deltaM) / periodX;
        const Real gy = twoPi * static_cast<Real>(deltaN) / periodY;
        Complex sum{};
        for (int iy = 0; iy < samplesY; ++iy) {
            const Real y =
                ((iy + Real{0.5}) / samplesY - Real{0.5}) * periodY;
            for (int ix = 0; ix < samplesX; ++ix) {
                const Real x =
                    ((ix + Real{0.5}) / samplesX - Real{0.5}) * periodX;
                if (pointInsideRegion(region, x, y)) {
                    sum += std::exp(Complex{0.0, -(gx * x + gy * y)});
                }
            }
        }
        return sum / static_cast<Real>(samplesX * samplesY);
    };
    const auto requireRectilinearCoefficient = [&](PatternRegion region, Real tolerance) {
        PatternedLayer2D layer;
        layer.background = Material::vacuum();
        layer.periodXUm = periodX;
        layer.periodYUm = periodY;
        layer.thicknessUm = 0.2;
        layer.sampleCountX = 3;
        layer.sampleCountY = 3;
        layer.regions.push_back(region);
        const TensorFourierMatrices tensors =
            tensorFourierMatricesPatternedLayer2d(layer, basis, 0.9);
        const Complex actual = tensors.eps[0][0](
            harmonicIndex(deltaM, deltaN),
            harmonicIndex(0, 0));
        const Complex expected = numericalIndicatorCoefficient(region);
        REQUIRE(std::abs(actual - expected) < tolerance);
    };

    const Material contrastOne = Material::constant("epsilon 2", {2.0, 0.0});
    requireRectilinearCoefficient(PatternRegion::circle(
        contrastOne, {0.11, -0.08}, 0.21), Real{2e-2});
    requireRectilinearCoefficient(PatternRegion::rectangle(
        contrastOne, {-0.09, 0.07}, 23.0, {0.22, 0.13}), Real{2e-2});
    requireRectilinearCoefficient(PatternRegion::polygon(
        contrastOne,
        {0.05, -0.04},
        -17.0,
        {{-0.24, -0.18}, {0.22, -0.18}, {0.22, -0.04},
         {0.02, -0.04}, {0.02, 0.20}, {-0.24, 0.20}}), Real{2e-2});
}

TEST_CASE("cross-cell RETICOLO rectangles wrap periodically") {
    using namespace rcwa;

    PatternedLayer2D preferred;
    preferred.background = Material::vacuum();
    preferred.periodXUm = 1.0;
    preferred.periodYUm = 1.0;
    preferred.thicknessUm = 0.2;
    preferred.sampleCountX = 81;
    preferred.sampleCountY = 79;
    preferred.regions.push_back(PatternRegion::circle(
        Material::constant("epsilon 3", {3.0, 0.0}),
        {0.46, -0.43},
        0.18));

    PatternedLayer2D sampled = preferred;
    sampled.preferAnalytic = false;
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Parallelogramic);
    const TensorFourierMatrices preferredTensors =
        tensorFourierMatricesPatternedLayer2d(preferred, basis, 0.9);
    const TensorFourierMatrices sampledTensors =
        tensorFourierMatricesPatternedLayer2d(sampled, basis, 0.9);
    REQUIRE_FALSE(preferredTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(preferredTensors.metadata.requiresJointConvergence);
    REQUIRE(sampledTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE(sampledTensors.metadata.requiresJointConvergence);
    const std::size_t zeroOrder = static_cast<std::size_t>(basis.centerIndex());
    REQUIRE(std::real(preferredTensors.eps[0][0](zeroOrder, zeroOrder)) ==
            Catch::Approx(1.0 + 2.0 * rcwa::pi * 0.18 * 0.18).margin(1e-12));
    REQUIRE(maxAbs(preferredTensors.eps[0][0] - sampledTensors.eps[0][0]) >
            Real{1e-6});

    PatternedLayer2D periodShifted = preferred;
    periodShifted.regions.front().center.x -= preferred.periodXUm;
    periodShifted.regions.front().center.y += preferred.periodYUm;
    const TensorFourierMatrices shiftedTensors =
        tensorFourierMatricesPatternedLayer2d(periodShifted, basis, 0.9);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            requireMatrixClose(
                preferredTensors.eps[row][column],
                shiftedTensors.eps[row][column],
                1e-12);
            requireMatrixClose(
                preferredTensors.liEpsQ[row][column],
                shiftedTensors.liEpsQ[row][column],
                1e-12);
        }
    }
}

TEST_CASE("sampled patterned library supports composed polygon-style shapes") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    const Material dielectric = Material::constant("dielectric", {2.25, 0.0});
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Circular);

    PatternedLayer2D shapes;
    shapes.background = air;
    shapes.periodXUm = 1.0;
    shapes.periodYUm = 1.0;
    shapes.thicknessUm = 0.2;
    shapes.sampleCountX = 17;
    shapes.sampleCountY = 17;
    shapes.regions.push_back(PatternRegion::polygon(
        dielectric,
        {-0.22, -0.18},
        15.0,
        {{0.12, 0.0}, {0.06, 0.10}, {-0.06, 0.10},
         {-0.12, 0.0}, {-0.06, -0.10}, {0.06, -0.10}}));
    shapes.regions.push_back(PatternRegion::polygon(
        dielectric,
        {0.20, -0.18},
        0.0,
        {{-0.04, -0.14}, {0.04, -0.14}, {0.04, -0.04}, {0.14, -0.04},
         {0.14, 0.04}, {0.04, 0.04}, {0.04, 0.14}, {-0.04, 0.14},
         {-0.04, 0.04}, {-0.14, 0.04}, {-0.14, -0.04}, {-0.04, -0.04}}));
    shapes.regions.push_back(PatternRegion::circle(
        dielectric,
        {0.18, 0.20},
        0.11));
    shapes.regions.push_back(PatternRegion::circle(
        air,
        {0.18, 0.20},
        0.05));

    const auto sampled = samplePatternedLayer2d(shapes, basis);
    REQUIRE(sampled.cells.size() == 289);

    RcwaSolver solver;
    solver.setWavelength(0.8);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Circular);
    solver.setSuperstrate(air);
    solver.setSubstrate(air);
    solver.setPolarization(Polarization::TE);
    solver.addPatternedLayer2d(shapes);
    const auto result = solver.solve();
    REQUIRE(result.N == 5);
    REQUIRE(std::isfinite(result.rTotal));
    REQUIRE(std::isfinite(result.tTotal));
    REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-4));
}

TEST_CASE("patterned 2D geometry cache matches sampled tensor construction") {
    using namespace rcwa;

    const Material air = Material::vacuum();
    Tensor3 epsTensor = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    epsTensor(0, 1) = {0.04, 0.01};
    epsTensor(1, 0) = {0.04, -0.01};
    const Material anisotropic = Material::anisotropic("anisotropic", epsTensor);
    const Material dielectric = Material::constant("dielectric", {3.1, 0.2});

    PatternedLayer2D direct;
    direct.background = air;
    direct.periodXUm = 1.3;
    direct.periodYUm = 1.1;
    direct.thicknessUm = 0.18;
    direct.sampleCountX = 17;
    direct.sampleCountY = 15;
    direct.preferAnalytic = false;
    direct.regions.push_back(PatternRegion::circle(anisotropic, {-0.12, 0.06}, 0.24));
    direct.regions.push_back(PatternRegion::rectangle(
        dielectric,
        {0.18, -0.08},
        27.0,
        {0.18, 0.10}));

    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Circular);
    const Real wavelengthUm = 0.91;
    const TensorFourierMatrices expected =
        tensorFourierMatricesPatternedLayer2d(direct, basis, wavelengthUm);

    PatternedLayer2D cached = direct;
    cached.geometryCache = std::make_shared<const PatternedLayer2DGeometryCache>(
        precomputePatternedLayer2dGeometry(direct, basis));
    const TensorFourierMatrices actual =
        tensorFourierMatricesPatternedLayer2d(cached, basis, wavelengthUm);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            requireMatrixClose(actual.eps[r][c], expected.eps[r][c], 1e-12);
            requireMatrixClose(actual.mu[r][c], expected.mu[r][c], 1e-12);
            requireMatrixClose(actual.liEpsQ[r][c], expected.liEpsQ[r][c], 1e-12);
            requireMatrixClose(actual.liMuQ[r][c], expected.liMuQ[r][c], 1e-12);
        }
    }
    requireMatrixClose(actual.liEpsQZzInverse, expected.liEpsQZzInverse, 1e-12);
    requireMatrixClose(actual.liMuQZzInverse, expected.liMuQZzInverse, 1e-12);
}

TEST_CASE("in-plane tensor patterned layers match full tensor mode route") {
    using namespace rcwa;

    auto epsTensor = [](Complex exz) {
        Tensor3 eps = Tensor3::diagonal({4.0, 0.04}, {4.3, 0.03}, {4.8, 0.02});
        eps(0, 1) = {0.18, 0.12};
        eps(1, 0) = {-0.18, -0.12};
        eps(0, 2) = exz;
        eps(2, 0) = exz;
        return eps;
    };

    auto makeLayer = [&](Complex exz) {
        PatternedLayer2D layer;
        layer.background = Material::anisotropic("background", epsTensor(exz));
        layer.periodXUm = 1.25;
        layer.periodYUm = 1.10;
        layer.thicknessUm = 0.22;
        layer.sampleCountX = 17;
        layer.sampleCountY = 17;
        layer.preferAnalytic = false;
        Tensor3 regionEps = epsTensor(exz);
        regionEps(0, 0) += Complex{1.1, 0.02};
        regionEps(1, 1) += Complex{0.8, 0.01};
        layer.regions.push_back(PatternRegion::circle(
            Material::anisotropic("region", regionEps),
            {-0.10, 0.05},
            0.24));
        return layer;
    };

    const auto basis = makeHarmonicBasisOrders(2, 2, LatticeTruncation::Circular);
    const Real wavelengthUm = 0.92;
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.12,
        0.05,
        wavelengthUm / 1.25,
        wavelengthUm / 1.10,
        basis);

    auto reducedTensors =
        tensorFourierMatricesPatternedLayer2d(makeLayer({0.0, 0.0}), basis, wavelengthUm);
    auto fullTensors =
        tensorFourierMatricesPatternedLayer2d(makeLayer({1e-6, 0.0}), basis, wavelengthUm);
    annotateTensorFourierMetadata(reducedTensors, basis.size());
    annotateTensorFourierMetadata(fullTensors, basis.size());
    const LayerModes reduced =
        computePeriodicModes(reducedTensors, Kx, Ky, 0.22, wavelengthUm);
    const LayerModes full =
        computePeriodicModes(fullTensors, Kx, Ky, 0.22, wavelengthUm);

    REQUIRE(reduced.W.rows() == full.W.rows());
    REQUIRE(reduced.gamma.size() == full.gamma.size());
    for (const Complex gamma : reduced.gamma) {
        Real nearest = std::numeric_limits<Real>::infinity();
        for (const Complex reference : full.gamma) {
            nearest = std::min(nearest, std::abs(gamma - reference));
        }
        REQUIRE(nearest < 1e-4);
    }
}

TEST_CASE("sampled 2D anisotropic patterned layers solve through full tensor route") {
    using namespace rcwa;

    Tensor3 eps = Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    eps(0, 1) = {0.05, 0.0};
    eps(1, 0) = {0.05, 0.0};

    PeriodicLayer2D sampled;
    sampled.sampleCountX = 5;
    sampled.sampleCountY = 5;
    sampled.periodXUm = 1.0;
    sampled.periodYUm = 1.0;
    sampled.thicknessUm = 0.18;
    sampled.cells.assign(25, Material::vacuum());
    sampled.cells[12] = Material::anisotropic("anisotropic cell", eps);

    std::vector<Tensor3> epsSamples(sampled.cells.size());
    std::vector<Tensor3> muSamples(sampled.cells.size());
    for (std::size_t i = 0; i < sampled.cells.size(); ++i) {
        epsSamples[i] = sampled.cells[i].epsilonTensor(0.9);
        muSamples[i] = sampled.cells[i].muTensor(0.9);
    }
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);
    const auto tensors = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampled.sampleCountX,
        sampled.sampleCountY,
        basis);
    REQUIRE(tensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::Unknown);
    REQUIRE(tensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE(tensors.metadata.requiresJointConvergence);
    REQUIRE(std::abs(tensors.eps[0][1](basis.centerIndex(), basis.centerIndex())) >
            0.0);
    const auto singleAxisBasis =
        makeHarmonicBasisOrders(1, 0, LatticeTruncation::Parallelogramic);
    const auto singleAxisTensors = tensorFourierMatricesFromSamples2d(
        epsSamples,
        muSamples,
        sampled.sampleCountX,
        sampled.sampleCountY,
        singleAxisBasis);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            requireMatrixClose(
                singleAxisTensors.liEpsQ[row][column],
                singleAxisTensors.eps[row][column],
                1e-12);
        }
    }

    RcwaSolver solver;
    solver.setWavelength(0.9);
    solver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    solver.setSuperstrate(Material::vacuum());
    solver.setSubstrate(Material::vacuum());
    solver.addPeriodicLayer2d(sampled);
    const auto result = solver.solve();
    REQUIRE(result.N == 9);
    REQUIRE(std::isfinite(result.rTotal));
    REQUIRE(std::isfinite(result.tTotal));
    REQUIRE(result.conservation == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("tensor pattern metadata distinguishes strict Li applicability from convergence routes") {
    using namespace rcwa;

    Tensor3 anisotropic = Tensor3::diagonal(
        {2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    anisotropic(0, 1) = {0.07, 0.02};
    anisotropic(1, 0) = {0.07, -0.02};
    const Material tensorMaterial = Material::anisotropic("tensor", anisotropic);
    const HarmonicBasis basis = makeHarmonicBasisOrders(
        1, 1, LatticeTruncation::Parallelogramic);

    PatternedLayer2D aligned;
    aligned.background = Material::vacuum();
    aligned.periodXUm = 1.0;
    aligned.periodYUm = 1.0;
    aligned.thicknessUm = 0.2;
    aligned.regions.push_back(PatternRegion::rectangle(
        tensorMaterial,
        {0.0, 0.0},
        0.0,
        {0.2, 0.3}));
    const TensorFourierMatrices alignedTensors =
        tensorFourierMatricesPatternedLayer2d(aligned, basis, 0.9);
    REQUIRE(alignedTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::CoordinateAligned);
    REQUIRE_FALSE(alignedTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(alignedTensors.metadata.requiresJointConvergence);

    PatternedLayer2D lamellar = aligned;
    lamellar.regions.front() = PatternRegion::rectangle(
        tensorMaterial,
        {0.0, 0.0},
        0.0,
        {0.2, 0.5});
    const TensorFourierMatrices lamellarTensors =
        tensorFourierMatricesPatternedLayer2d(lamellar, basis, 0.9);
    REQUIRE(lamellarTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::CoordinateAligned);
    REQUIRE_FALSE(lamellarTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(lamellarTensors.metadata.requiresJointConvergence);

    PatternedLayer2D lamellarY = aligned;
    lamellarY.regions.front() = PatternRegion::rectangle(
        tensorMaterial,
        {0.0, 0.0},
        0.0,
        {0.5, 0.2});
    const TensorFourierMatrices lamellarYTensors =
        tensorFourierMatricesPatternedLayer2d(lamellarY, basis, 0.9);
    REQUIRE(lamellarYTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::CoordinateAligned);
    REQUIRE_FALSE(lamellarYTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(lamellarYTensors.metadata.requiresJointConvergence);
    REQUIRE(maxAbs(
        lamellarYTensors.liEpsQ[1][1] -
        lamellarYTensors.eps[1][1]) > 1e-3);

    PatternedLayer2D rotated = aligned;
    rotated.regions.front().angleDeg = 17.0;
    const TensorFourierMatrices rotatedTensors =
        tensorFourierMatricesPatternedLayer2d(rotated, basis, 0.9);
    REQUIRE(rotatedTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::CurvedOrOblique);
    REQUIRE(rotatedTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE(rotatedTensors.metadata.requiresJointConvergence);

    PatternedLayer2D curved = aligned;
    curved.regions.front() = PatternRegion::circle(
        tensorMaterial,
        {0.0, 0.0},
        0.2);
    const TensorFourierMatrices curvedTensors =
        tensorFourierMatricesPatternedLayer2d(curved, basis, 0.9);
    REQUIRE(curvedTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::CurvedOrOblique);
    REQUIRE_FALSE(curvedTensors.metadata.factorizationUsesSampledGeometry);
    REQUIRE_FALSE(curvedTensors.metadata.requiresJointConvergence);

    PatternedLayer2D scalarCircle = curved;
    scalarCircle.regions.front().material = Material::constant(
        "scalar", {3.0, 0.0});
    const TensorFourierMatrices scalarTensors =
        tensorFourierMatricesPatternedLayer2d(scalarCircle, basis, 0.9);
    REQUIRE(scalarTensors.metadata.tensorFactorizationApplicability ==
            TensorFactorizationApplicability::NotApplicable);
}

TEST_CASE("periodic modes handle scalar and tensor epsilon blocks") {
    using namespace rcwa;

    const Real wavelengthUm = 0.9;
    const auto basis = makeHarmonicBasisOrders(1, 1, LatticeTruncation::Parallelogramic);
    const auto [Kx, Ky] = makeTransverseWavevectorOperators(
        0.0, 0.0, wavelengthUm, wavelengthUm, basis);

    std::vector<Tensor3> epsScalar(25, Tensor3::isotropic({1.0, 0.0}));
    std::vector<Tensor3> muSamples(25, Tensor3::isotropic({1.0, 0.0}));
    epsScalar[12] = Tensor3::isotropic({2.25, 0.0});
    const auto scalarTensors = tensorFourierMatricesFromSamples2d(
        epsScalar,
        muSamples,
        5,
        5,
        basis);
    const auto scalarAuto = computePeriodicModes(
        scalarTensors,
        Kx,
        Ky,
        0.18,
        wavelengthUm);
    REQUIRE(scalarAuto.W.rows() == 4 * basis.size());
    REQUIRE(scalarAuto.kind == LayerComputationKind::PatternedRcwa);

    TensorFourierMatrices staleMetadataTensors = scalarTensors;
    staleMetadataTensors.metadata.valid = true;
    staleMetadataTensors.metadata.harmonicCount = basis.size();
    staleMetadataTensors.metadata.factorizationComplete = false;
    staleMetadataTensors.metadata.scalarEpsilonBlocks = true;
    staleMetadataTensors.metadata.commonScalarMuIdentity = true;
    staleMetadataTensors.metadata.commonScalarMu = {2.0, 0.0};
    staleMetadataTensors.metadata.diagonalEpsilonBlocks = true;
    const auto staleMetadataAuto = computePeriodicModes(
        staleMetadataTensors,
        Kx,
        Ky,
        0.18,
        wavelengthUm);
    requireComplexMultisetClose(scalarAuto.gamma, staleMetadataAuto.gamma, 1e-10);

    std::vector<Tensor3> epsTensor = epsScalar;
    Tensor3 anisotropicEps =
        Tensor3::diagonal({2.25, 0.0}, {2.56, 0.0}, {2.89, 0.0});
    anisotropicEps(0, 1) = {0.05, 0.0};
    anisotropicEps(1, 0) = {0.05, 0.0};
    epsTensor[12] = anisotropicEps;
    const auto tensorTensors = tensorFourierMatricesFromSamples2d(
        epsTensor,
        muSamples,
        5,
        5,
        basis);
    REQUIRE(tensorTensors.metadata.sequentialLiFactorization2d);
    REQUIRE(maxAbs(tensorTensors.liEpsQ[0][0] - tensorTensors.eps[0][0]) > 1e-3);
    requireMatrixClose(
        tensorTensors.liEpsQZzInverse,
        inverse(tensorTensors.liEpsQ[2][2]),
        1e-12);
    const auto tensorAuto = computePeriodicModes(
        tensorTensors,
        Kx,
        Ky,
        0.18,
        wavelengthUm);
    REQUIRE(tensorAuto.W.rows() == 4 * basis.size());
    REQUIRE(tensorAuto.kind == LayerComputationKind::PatternedRcwa);

    std::vector<Tensor3> muTensorSamples = muSamples;
    Tensor3 tensorMu = Tensor3::diagonal({1.05, 0.0}, {1.12, 0.0}, {0.97, 0.0});
    tensorMu(0, 2) = {0.02, 0.0};
    tensorMu(2, 0) = {0.02, 0.0};
    muTensorSamples[12] = tensorMu;
    const auto tensorMuTensors = tensorFourierMatricesFromSamples2d(
        epsScalar,
        muTensorSamples,
        5,
        5,
        basis);
    REQUIRE(tensorMuTensors.liMuQ[2][2].rows() == basis.size());
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            requireMatrixClose(
                tensorMuTensors.liEpsQ[row][column],
                scalarTensors.liEpsQ[row][column],
                1e-12);
        }
    }
    const auto tensorMuAuto = computePeriodicModes(
        tensorMuTensors,
        Kx,
        Ky,
        0.18,
        wavelengthUm);
    REQUIRE(tensorMuAuto.W.rows() == 4 * basis.size());
    REQUIRE(tensorMuAuto.kind == LayerComputationKind::PatternedRcwa);

    TensorFourierMatrices legacyTensors = tensorMuTensors;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            legacyTensors.liMuQ[r][c] = Matrix{};
        }
    }
    legacyTensors.liMuQZzInverse = Matrix{};
    REQUIRE_THROWS_AS(
        computePeriodicModes(
            legacyTensors,
            Kx,
            Ky,
            0.18,
            wavelengthUm),
        std::invalid_argument);

    PrecomputedPeriodicLayer2D legacyPrecomputed;
    legacyPrecomputed.tensors = legacyTensors;
    legacyPrecomputed.periodXUm = 1.0;
    legacyPrecomputed.periodYUm = 1.0;
    legacyPrecomputed.thicknessUm = 0.18;
    RcwaSolver legacySolver;
    legacySolver.setWavelength(wavelengthUm);
    legacySolver.setHarmonicOrders(1, 1, LatticeTruncation::Parallelogramic);
    legacySolver.setSuperstrate(Material::vacuum());
    legacySolver.setSubstrate(Material::vacuum());
    legacySolver.addPrecomputedPeriodicLayer2d(legacyPrecomputed);
    REQUIRE_THROWS_AS(legacySolver.solve(), std::invalid_argument);
    FieldPlaneRequest request;
    request.plane = FieldPlane::XY;
    request.uPoints = 1;
    request.vPoints = 1;
    request.fixedUm = 0.09;
    request.components = {FieldComponent::Hz};
    REQUIRE_THROWS_AS(legacySolver.solveFieldPlane(request), std::invalid_argument);
}

TEST_CASE("Lanczos smoothing preserves DC and damps nonzero Fourier orders") {
    using namespace rcwa;

    const HarmonicBasis basis = makeHarmonicBasisOrders(
        3, 3, LatticeTruncation::Parallelogramic);
    FourierConvergenceOptions options;
    options.lanczosSmoothing = true;
    options.lanczosPower = 2;

    REQUIRE(lanczosSmoothingFactor(basis, 0, 0, 1.0, 1.0, options) ==
            Catch::Approx(1.0));
    const Real first = lanczosSmoothingFactor(basis, 1, 0, 1.0, 1.0, options);
    const Real farther = lanczosSmoothingFactor(basis, 4, 3, 1.0, 1.0, options);
    REQUIRE(first > Real{0.0});
    REQUIRE(first < Real{1.0});
    REQUIRE(std::abs(farther) < first);

    Matrix convolution(basis.size(), basis.size(), Complex{1.0, 0.0});
    applyLanczosSmoothing(convolution, basis, 1.0, 1.0, options);
    const Real smoothingTolerance = sizeof(Real) == sizeof(float)
        ? Real{2e-5f}
        : Real{1e-14};
    for (std::size_t row = 0; row < basis.size(); ++row) {
        for (std::size_t column = 0; column < basis.size(); ++column) {
            const int dm = basis.orders[row].m - basis.orders[column].m;
            const int dn = basis.orders[row].n - basis.orders[column].n;
            REQUIRE(std::real(convolution(row, column)) ==
                    Catch::Approx(lanczosSmoothingFactor(
                        basis, dm, dn, 1.0, 1.0, options)).margin(smoothingTolerance));
            REQUIRE(std::imag(convolution(row, column)) == Catch::Approx(0.0));
        }
    }
}

TEST_CASE("all PolBasis formulations reduce to directional Li factorization in 1D") {
    using namespace rcwa;

    const HarmonicBasis basis = makeHarmonicBasisOrders(
        3, 0, LatticeTruncation::Parallelogramic);
    PatternedLayer2D layer;
    layer.background = Material::constant("air", {1.0, 0.0});
    layer.regions.push_back(PatternRegion::rectangle(
        Material::constant("high", {12.0, 0.0}),
        {0.0, 0.0},
        0.0,
        {0.25, 0.5}));
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.2;
    layer.sampleCountX = 65;
    layer.sampleCountY = 9;

    const TensorFourierMatrices expected =
        tensorFourierMatricesPatternedLayer2d(layer, basis, 1.0);
    for (const PolarizationBasis polarizationBasis : {
             PolarizationBasis::VectorField,
             PolarizationBasis::Normal,
             PolarizationBasis::Jones}) {
        layer.fourierOptions.polarizationDecomposition = true;
        layer.fourierOptions.polarizationBasis = polarizationBasis;
        const TensorFourierMatrices actual =
            tensorFourierMatricesPatternedLayer2d(layer, basis, 1.0);
        requireMatrixClose(actual.liEpsQ[0][0], expected.liEpsQ[0][0], 1e-10);
        requireMatrixClose(actual.liEpsQ[1][1], expected.liEpsQ[1][1], 1e-10);
        requireMatrixClose(actual.liEpsQ[2][2], expected.liEpsQ[2][2], 1e-10);
        REQUIRE(actual.metadata.formulation != FourierFormulation::Default);
    }
}

TEST_CASE("Kottke subpixel smoothing builds interface-oriented effective tensors") {
    using namespace rcwa;

    const HarmonicBasis basis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Parallelogramic);
    PatternedLayer2D layer;
    layer.background = Material::constant("air", {1.0, 0.0});
    layer.regions.push_back(PatternRegion::circle(
        Material::constant("high", {12.0, 0.0}),
        {0.0, 0.0},
        0.25));
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.2;
    layer.sampleCountX = 65;
    layer.sampleCountY = 65;

    const TensorFourierMatrices ordinary =
        tensorFourierMatricesPatternedLayer2d(layer, basis, 1.0);
    layer.fourierOptions.subpixelSmoothing = true;
    layer.fourierOptions.resolution = 4;
    const TensorFourierMatrices smoothed =
        tensorFourierMatricesPatternedLayer2d(layer, basis, 1.0);

    REQUIRE(smoothed.metadata.formulation == FourierFormulation::Kottke);
    REQUIRE_FALSE(smoothed.metadata.sequentialLiFactorization2d);
    REQUIRE_FALSE(smoothed.metadata.lanczosSmoothingApplied);
    REQUIRE(maxAbs(smoothed.eps[0][1]) > Real{1e-6});
    REQUIRE(maxAbs(smoothed.eps[0][0] - ordinary.eps[0][0]) > Real{1e-6});
    const std::size_t center = static_cast<std::size_t>(basis.centerIndex());
    const Real exactAverageEpsilon = Real{1.0} +
        (Real{12.0} - Real{1.0}) * pi * Real{0.25} * Real{0.25};
    REQUIRE(std::real(smoothed.eps[2][2](center, center)) ==
            Catch::Approx(exactAverageEpsilon).margin(1e-10));
    REQUIRE(smoothed.liEpsQZzInverse.rows() == basis.size());
}

TEST_CASE("2D PolBasis variants produce complete finite Maxwell factorization blocks") {
    using namespace rcwa;

    const HarmonicBasis basis = makeHarmonicBasisOrders(
        2, 2, LatticeTruncation::Parallelogramic);
    PatternedLayer2D layer;
    layer.background = Material::constant("air", {1.0, 0.0});
    layer.regions.push_back(PatternRegion::circle(
        Material::constant("high", {10.0, 0.0}),
        {0.0, 0.0},
        0.24));
    layer.periodXUm = 1.0;
    layer.periodYUm = 1.0;
    layer.thicknessUm = 0.15;
    layer.sampleCountX = 65;
    layer.sampleCountY = 65;
    layer.fourierOptions.polarizationDecomposition = true;
    layer.fourierOptions.lanczosSmoothing = true;

    for (const PolarizationBasis polarizationBasis : {
             PolarizationBasis::VectorField,
             PolarizationBasis::Normal,
             PolarizationBasis::Jones}) {
        layer.fourierOptions.polarizationBasis = polarizationBasis;
        TensorFourierMatrices tensors =
            tensorFourierMatricesPatternedLayer2d(layer, basis, 1.0);
        annotateTensorFourierMetadata(tensors, basis.size());
        REQUIRE(tensors.metadata.factorizationComplete);
        REQUIRE_FALSE(tensors.metadata.sequentialLiFactorization2d);
        REQUIRE(tensors.metadata.lanczosSmoothingApplied);
        for (const auto& row : tensors.liEpsQ) {
            for (const Matrix& block : row) {
                REQUIRE(block.rows() == basis.size());
                for (const Complex value : block.data()) {
                    REQUIRE(std::isfinite(std::real(value)));
                    REQUIRE(std::isfinite(std::imag(value)));
                }
            }
        }
    }
}
