#pragma once

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rcwa {

#ifdef RCWA_SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif
using Complex = std::complex<Real>;

inline constexpr Complex iu{Real{0.0}, Real{1.0}};
inline constexpr Real pi = static_cast<Real>(3.141592653589793238462643383279502884);
inline constexpr Real twoPi = Real{2.0} * pi;

inline constexpr Real degToRad(Real deg) {
    return deg * pi / Real{180.0};
}

inline Complex boundedExp(Complex exponent) {
    const Real real = std::real(exponent);
    const Real imag = std::imag(exponent);
    if (!std::isfinite(real) || !std::isfinite(imag)) {
        throw std::runtime_error("complex exponential received a non-finite exponent");
    }
    if (real > std::log(std::numeric_limits<Real>::max())) {
        throw std::overflow_error("complex exponential magnitude exceeds floating-point range");
    }
    if (real < std::log(std::numeric_limits<Real>::min())) {
        return {};
    }
    return std::exp(exponent);
}

inline void requirePositive(Real value, const char* name) {
    if (!(value > Real{0.0}) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be a positive finite value");
    }
}

inline void requireFinite(Real value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

[[nodiscard]] inline bool isFinite(Complex value) {
    return std::isfinite(std::real(value)) && std::isfinite(std::imag(value));
}

inline void requireFinite(Complex value, const char* name) {
    if (!isFinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

inline std::size_t checkedProduct(std::size_t a, std::size_t b, const char* context) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string(context) + " count is too large");
    }
    return a * b;
}

inline std::size_t checkedGridSize(int x, int y, const char* context) {
    if (x <= 0 || y <= 0) {
        throw std::invalid_argument(std::string(context) + " dimensions must be positive");
    }
    return checkedProduct(
        static_cast<std::size_t>(x),
        static_cast<std::size_t>(y),
        context);
}

// Shared indexed work scheduler for spectra, fields, and binding utilities.
// Zero requests automatic hardware-limited concurrency.
inline int hardwareWorkerCount() noexcept {
    const unsigned int available = std::thread::hardware_concurrency();
    if (available == 0) {
        return 1;
    }
    return static_cast<int>(std::min<unsigned int>(
        available,
        static_cast<unsigned int>(std::numeric_limits<int>::max())));
}

inline int boundedWorkerCount(int workers, std::size_t jobs) {
    if (jobs == 0) {
        return 1;
    }
    const int hardwareLimit = hardwareWorkerCount();
    workers = workers == 0
        ? hardwareLimit
        : std::clamp(workers, 1, hardwareLimit);
    if (jobs < static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::min(workers, static_cast<int>(jobs));
    }
    return workers;
}

class WorkerErrorState {
public:
    void captureCurrentException() {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mFirstError) {
            mFirstError = std::current_exception();
            mHasError.store(true, std::memory_order_release);
        }
    }

    [[nodiscard]] bool hasError() const noexcept {
        return mHasError.load(std::memory_order_acquire);
    }

    void rethrowIfSet() const {
        if (mFirstError) {
            std::rethrow_exception(mFirstError);
        }
    }

private:
    std::exception_ptr mFirstError;
    std::atomic<bool> mHasError{false};
    mutable std::mutex mMutex;
};

template <typename JobFn>
void runIndexedJobs(std::size_t jobCount, int workers, JobFn&& jobFn) {
    if (jobCount == 0) {
        return;
    }

    const int workerCount = boundedWorkerCount(workers, jobCount);
    auto&& fn = jobFn;
    if (workerCount == 1) {
        for (std::size_t index = 0; index < jobCount; ++index) {
            fn(index);
        }
        return;
    }

    WorkerErrorState errors;
    std::atomic<std::size_t> next{0};
    const std::size_t chunkSize = std::max<std::size_t>(
        1,
        jobCount / (static_cast<std::size_t>(workerCount) * 8));
    const auto work = [&] {
        while (!errors.hasError()) {
            const std::size_t begin = next.fetch_add(chunkSize, std::memory_order_relaxed);
            if (begin >= jobCount) {
                break;
            }
            const std::size_t end = std::min(jobCount, begin + chunkSize);
            for (std::size_t index = begin; index < end; ++index) {
                fn(index);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workerCount - 1));
    try {
        for (int worker = 1; worker < workerCount; ++worker) {
            pool.emplace_back([&] {
                try {
                    work();
                } catch (...) {
                    errors.captureCurrentException();
                }
            });
        }
    } catch (...) {
        errors.captureCurrentException();
    }
    try {
        work();
    } catch (...) {
        errors.captureCurrentException();
    }
    for (auto& worker : pool) {
        worker.join();
    }
    errors.rethrowIfSet();
}

struct Vec2 {
    Real x{};
    Real y{};

    constexpr Vec2() = default;
    constexpr Vec2(Real xValue, Real yValue) : x(xValue), y(yValue) {}
};

struct Vec3 {
    Real x{};
    Real y{};
    Real z{};

    constexpr Vec3() = default;
    constexpr Vec3(Real xValue, Real yValue, Real zValue) : x(xValue), y(yValue), z(zValue) {}
};

struct HarmonicIndex {
    int m{};
    int n{};
};

enum class Polarization {
    TE,
    TM,
    Both,
    LCP,
    RCP,
};

enum class LatticeTruncation {
    Circular,
    Parallelogramic,
};

struct PolarizationAmplitudes {
    Complex te{};
    Complex tm{};
};

// Normalized Jones amplitudes in the solver's TE/TM modal basis.  Keeping this
// map in one place prevents the stack solver and the incident-field builder
// from drifting apart on the LCP/RCP phase convention.
[[nodiscard]] inline PolarizationAmplitudes polarizationAmplitudes(Polarization pol) {
    const Complex unit{1.0 / std::sqrt(Real{2.0}), 0.0};
    switch (pol) {
    case Polarization::TE:
        return {{1.0, 0.0}, {}};
    case Polarization::TM:
        return {{}, {1.0, 0.0}};
    case Polarization::Both:
        return {unit, unit};
    case Polarization::LCP:
        return {unit, iu * unit};
    case Polarization::RCP:
        return {unit, -iu * unit};
    }
    throw std::invalid_argument("unsupported polarization");
}

inline std::string toString(Polarization p) {
    switch (p) {
        case Polarization::TE: return "TE";
        case Polarization::TM: return "TM";
        case Polarization::Both: return "Both";
        case Polarization::LCP: return "LCP";
        case Polarization::RCP: return "RCP";
    }
    return "Unknown";
}

} // namespace rcwa
