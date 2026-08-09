from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


TensorIndex = tuple[int, int]


def material_tensor_component_ratio(
    rcwa: object,
    model: str,
    wavelengths_um: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float],
    numerator: TensorIndex = (0, 2),
    denominator: TensorIndex = (0, 0),
) -> np.ndarray:
    """Return |epsilon[numerator]|/|epsilon[denominator]| from a built-in model."""
    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_um must be one-dimensional")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_um must contain finite positive values")

    # Build the dispersion model once and evaluate the complete wavelength
    # vector in native code.  The fallback keeps this helper usable with small
    # test doubles or older extension modules.
    batch_evaluator = getattr(rcwa, "MaterialModelTensorsBatch", None)
    if callable(batch_evaluator):
        tensors = batch_evaluator(model, wavelengths, dict(parameters))
        epsilon = np.asarray(tensors["epsilon"], dtype=np.complex128)
        expected_shape = (len(wavelengths), 3, 3)
        if epsilon.shape != expected_shape:
            raise ValueError(
                "MaterialModelTensorsBatch returned an unexpected epsilon shape"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(epsilon[:, numerator[0], numerator[1]]) / np.abs(
                epsilon[:, denominator[0], denominator[1]]
            )

    ratio = np.empty_like(wavelengths, dtype=float)
    for i, wavelength_um in enumerate(wavelengths):
        tensors = rcwa.MaterialModelTensors(
            model,
            float(wavelength_um),
            dict(parameters),
        )
        epsilon = np.asarray(tensors["epsilon"], dtype=np.complex128)
        ratio[i] = abs(epsilon[numerator]) / abs(epsilon[denominator])
    return ratio


def wsm_gamma(
    rcwa: object,
    wavelengths_um: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    return material_tensor_component_ratio(rcwa, "WSM", wavelengths_um, parameters)


def inas_magneto_gamma(
    rcwa: object,
    wavelengths_um: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    return material_tensor_component_ratio(rcwa, "InAsMagneto", wavelengths_um, parameters)


def bismuth_iron_garnet_epsilon_tensor(
    wavelength_um: float,
    *,
    refractive_index_a: float = 2.36,
    refractive_index_b_nm: float = 413.0,
    extinction_a_nm: float = 1660.0,
    extinction_b: float = 15.2,
    gyration_prefactor: float = 100.0,
    gyration_decay_per_nm: float = 0.011,
    gyration_floor: float = 0.0,
    magnetization_sign: float = 1.0,
) -> np.ndarray:
    """Return the empirical BIG tensor used by Yan et al. (2023).

    The tensor follows the paper ordering epsilon_xy=-i*g and epsilon_yx=+i*g.
    ``gyration_floor`` can retain a cited long-wave gyration value when the
    visible-range exponential fit is extended beyond its measured range.
    """
    if not np.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ValueError("wavelength_um must be positive and finite")
    wavelength_nm = 1000.0 * float(wavelength_um)
    refractive_index = refractive_index_a + (
        refractive_index_b_nm / wavelength_nm
    ) ** 2
    extinction = wavelength_nm / (4.0 * np.pi) * np.exp(
        (extinction_a_nm / wavelength_nm) ** 2 - extinction_b
    )
    diagonal = complex(refractive_index, extinction) ** 2
    gyration = max(
        gyration_prefactor * np.exp(-gyration_decay_per_nm * wavelength_nm),
        gyration_floor,
    )
    signed_gyration = float(magnetization_sign) * gyration
    return np.asarray(
        [
            [diagonal, -1j * signed_gyration, 0.0],
            [1j * signed_gyration, diagonal, 0.0],
            [0.0, 0.0, diagonal],
        ],
        dtype=np.complex128,
    )
