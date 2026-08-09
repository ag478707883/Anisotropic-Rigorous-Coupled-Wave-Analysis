from __future__ import annotations

import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DEFAULT_BUILD_DIR = ROOT / "build"
ONEAPI_MKL_BIN = Path(os.environ.get(
    "RCWA_ONEAPI_MKL_BIN",
    r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
))
ONEAPI_COMPILER_BIN = Path(os.environ.get(
    "RCWA_ONEAPI_COMPILER_BIN",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
))
MINGW_BIN = Path(os.environ.get("RCWA_MINGW_BIN", r"C:\mingw64\bin"))

_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_EXTENSION_GLOBS = ("rcwa_cpp*.pyd", "rcwa_cpp*.so", "rcwa_cpp*.dylib")
_REQUIRED_SIMULATION_METHODS = (
    "AddLayer",
    "SetRegionRectangle",
    "SetMaterialNK",
    "GetSpectrum",
    "GetDiffractionOrders",
    "GetSpectrumForAngles",
    "GetSpectrumAndDirectionalThermalChannelsForAngles",
    "GetDiffractionOrdersForAngles",
    "GetSpectrumForAnglePolarizationPairs",
    "GetDiffractionOrdersForAnglePolarizationPairs",
    "GetSParameters",
    "GetSParametersAnalyticContinuation",
    "GetSParametersAnalyticContinuationAtGamma",
    "FindAbsorptivityPeak",
    "GetZeroOrderAmplitudes",
    "GetDirectionalThermalChannels",
    "GetFieldGrid",
)
_REQUIRED_MODULE_FUNCTIONS = (
    "MaterialModelTensors",
    "MaterialModelTensorsBatch",
    "New",
)
_API_SOURCE_ROOTS = (
    ROOT / "bindings",
    ROOT / "include" / "rcwa",
    ROOT / "src" / "rcwa",
)
_API_SOURCE_PATHS = (
    ROOT / "CMakeLists.txt",
    ROOT / "setup.py",
)


def load_rcwa_cpp():
    """Import the compiled extension and prepare native runtime search paths."""
    apply_runtime_environment()
    module_dir = module_directory()
    add_dll_directories([module_dir, Path(sys.executable).parent])
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("rcwa_cpp")
    except ImportError as exc:
        raise RuntimeError(
            "rcwa_cpp Python extension is not available. Build target rcwa_cpp first, "
            "run `python -m pip install -e .` from the project root, or set "
            "RCWA_BUILD_DIR to the CMake build directory containing python/rcwa_cpp."
        ) from exc
    require_simulation_api(module)
    return module


def module_directory() -> Path:
    configured_extension_dir = os.environ.get("RCWA_EXTENSION_DIR")
    if configured_extension_dir:
        return absolute_path(Path(configured_extension_dir))

    package_extension_dir = extension_directory_from(PACKAGE_DIR)
    if package_extension_dir is not None:
        return package_extension_dir

    return build_directory() / "python"


def build_directory() -> Path:
    configured = os.environ.get("RCWA_BUILD_DIR")
    if configured:
        return absolute_path(Path(configured))

    candidates = [
        candidate
        for candidate in candidate_build_directories()
        if has_extension(candidate / "python")
    ]
    source_mtime = api_source_mtime() if candidates else 0.0
    for candidate in candidates:
        if extension_is_current(candidate / "python", source_mtime):
            return candidate
    if candidates:
        return max(candidates, key=lambda path: newest_extension_mtime(path / "python") or 0.0)
    return DEFAULT_BUILD_DIR


def candidate_build_directories() -> list[Path]:
    return [
        DEFAULT_BUILD_DIR,
        ROOT / "build-release",
        ROOT / "build-debug",
    ]


def extension_directory_from(path: Path) -> Path | None:
    return path if has_extension(path) else None


def has_extension(module_dir: Path) -> bool:
    if not module_dir.exists():
        return False
    return any(any(module_dir.glob(pattern)) for pattern in _EXTENSION_GLOBS)


def extension_is_current(module_dir: Path, source_mtime: float | None = None) -> bool:
    extension_mtime = newest_extension_mtime(module_dir)
    if source_mtime is None:
        source_mtime = api_source_mtime()
    return extension_mtime is not None and extension_mtime >= source_mtime


def newest_extension_mtime(module_dir: Path) -> float | None:
    files = extension_files(module_dir)
    if not files:
        return None
    return max(path.stat().st_mtime for path in files)


def extension_files(module_dir: Path) -> list[Path]:
    if not module_dir.exists():
        return []
    files: list[Path] = []
    for pattern in _EXTENSION_GLOBS:
        files.extend(module_dir.glob(pattern))
    return files


def api_source_mtime() -> float:
    existing = [path for path in _API_SOURCE_PATHS if path.exists()]
    for root in _API_SOURCE_ROOTS:
        if root.exists():
            existing.extend(root.rglob("*.cpp"))
            existing.extend(root.rglob("*.hpp"))
    if not existing:
        return 0.0
    return max(path.stat().st_mtime for path in existing)


def require_simulation_api(module: object) -> None:
    simulation = getattr(module, "Simulation", None)
    missing = [
        method
        for method in _REQUIRED_SIMULATION_METHODS
        if simulation is None or not hasattr(simulation, method)
    ]
    missing_functions = [
        function
        for function in _REQUIRED_MODULE_FUNCTIONS
        if not hasattr(module, function)
    ]
    if not missing and not missing_functions:
        return
    module_file = getattr(module, "__file__", "<unknown>")
    missing_text = ", ".join(
        [f"Simulation.{method}" for method in missing] + missing_functions
    )
    raise RuntimeError(
        f"Loaded stale rcwa_cpp extension from {module_file}; missing {missing_text}. "
        "Rebuild target rcwa_cpp, or unset RCWA_BUILD_DIR so the package can select a newer build."
    )


def absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def apply_runtime_environment() -> None:
    env = command_environment()
    for key in ("PATH", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "MKL_THREADING_LAYER"):
        if key in env:
            os.environ[key] = env[key]
    add_dll_directories(runtime_library_dirs())


def command_environment() -> dict[str, str]:
    env = os.environ.copy()
    runtime_dirs = runtime_library_dirs()
    if runtime_dirs:
        existing_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join(
            [str(path) for path in runtime_dirs]
            + ([existing_path] if existing_path else [])
        )
    try:
        blas_threads = int(env.get("RCWA_BLAS_THREADS", "1"))
    except (TypeError, ValueError):
        # A malformed optional tuning variable should not prevent importing
        # the package.  Keep the deterministic single-thread default.
        blas_threads = 1
    if blas_threads > 0:
        env["MKL_NUM_THREADS"] = str(blas_threads)
        env["OMP_NUM_THREADS"] = str(blas_threads)
        env.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    return env


def runtime_library_dirs() -> list[Path]:
    candidates = [
        module_directory_without_environment(),
        ONEAPI_MKL_BIN,
        ONEAPI_COMPILER_BIN,
        MINGW_BIN,
        Path(sys.executable).parent,
    ]
    candidates.extend(_oneapi_bin_candidates())
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def module_directory_without_environment() -> Path:
    configured_extension_dir = os.environ.get("RCWA_EXTENSION_DIR")
    if configured_extension_dir:
        return absolute_path(Path(configured_extension_dir))
    package_extension_dir = extension_directory_from(PACKAGE_DIR)
    if package_extension_dir is not None:
        return package_extension_dir
    configured_build_dir = os.environ.get("RCWA_BUILD_DIR")
    if configured_build_dir:
        return absolute_path(Path(configured_build_dir)) / "python"
    return DEFAULT_BUILD_DIR / "python"


def _oneapi_bin_candidates() -> list[Path]:
    root = Path(r"C:\Program Files (x86)\Intel\oneAPI")
    if not root.exists():
        return []
    candidates: list[Path] = []
    for subdir in ("mkl", "compiler"):
        parent = root / subdir
        if parent.exists():
            for version in sorted(parent.iterdir(), reverse=True):
                candidates.append(version / "bin")
    for version in sorted(root.glob("20*"), reverse=True):
        candidates.append(version / "bin")
    return candidates


def add_dll_directories(paths: list[Path]) -> None:
    add_directory = getattr(os, "add_dll_directory", None)
    if add_directory is None:
        return
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key not in _DLL_DIRECTORY_PATHS:
            _DLL_DIRECTORY_HANDLES.append(add_directory(key))
            _DLL_DIRECTORY_PATHS.add(key)


@contextmanager
def timed_step(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[time] {label}: {time.perf_counter() - start:.3f}s")
