from __future__ import annotations

from .runtime import (
    apply_runtime_environment,
    build_directory,
    load_rcwa_cpp,
    module_directory,
    timed_step,
)

_backend = load_rcwa_cpp()

Simulation = _backend.Simulation
New = _backend.New
MaterialModelTensors = _backend.MaterialModelTensors
MaterialModelTensorsBatch = _backend.MaterialModelTensorsBatch
wavelength_grid = _backend.wavelength_grid

__all__ = [
    "MaterialModelTensors",
    "MaterialModelTensorsBatch",
    "New",
    "Simulation",
    "apply_runtime_environment",
    "build_directory",
    "load_rcwa_cpp",
    "module_directory",
    "timed_step",
    "wavelength_grid",
]


def __getattr__(name: str):
    return getattr(_backend, name)
