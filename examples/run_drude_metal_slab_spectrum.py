"""Compute the normal-incidence spectrum of a dispersive Drude metal slab.

The structure is air / 0.3-um Drude metal / air.  The RCWA backend uses its
multilayer scattering-matrix path, and the result is checked against the
closed-form Fresnel solution for a single slab.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


speed_of_light_m_s = 299_792_458.0
plasma_frequency_rad_s = 3.0e15
damping_rate_rad_s = 5.5e12
epsilon_infinity = 1.0
slab_thickness_um = 0.3

wavelength_min_um = 0.2
wavelength_max_um = 2.0
wavelength_point_count = 1000
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

polarization = "TE"
theta_deg = 0.0
phi_deg = 0.0
workers = 14

plot_path = project_root / "images" / "drude_metal_slab_spectrum.png"
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def drude_permittivity(wavelength_um: np.ndarray) -> np.ndarray:
    """Return epsilon_inf - omega_p^2 / (omega^2 + i*gamma*omega)."""
    wavelength_m = np.asarray(wavelength_um, dtype=float) * 1.0e-6
    omega_rad_s = 2.0 * np.pi * speed_of_light_m_s / wavelength_m
    return epsilon_infinity - plasma_frequency_rad_s**2 / (
        omega_rad_s**2 + 1j * damping_rate_rad_s * omega_rad_s
    )


def build_simulation() -> object:
    simulation = asyrcwa.New()
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialModel(
        "DrudeMetal",
        "Drude",
        {
            "eps_inf": epsilon_infinity,
            "plasma_frequency_rad_s": plasma_frequency_rad_s,
            "damping_rate_rad_s": damping_rate_rad_s,
        },
    )
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    simulation.AddLayer("metal_slab", slab_thickness_um, "DrudeMetal")
    simulation.SetExcitationPlanewave((theta_deg, phi_deg))
    return simulation


def fresnel_slab_spectrum(wavelength_um: np.ndarray) -> tuple[np.ndarray, ...]:
    """Closed-form normal-incidence power spectrum for an air/film/air stack."""
    n_air = 1.0
    n_metal = np.sqrt(drude_permittivity(wavelength_um))
    r01 = (n_air - n_metal) / (n_air + n_metal)
    r12 = (n_metal - n_air) / (n_metal + n_air)
    t01 = 2.0 * n_air / (n_air + n_metal)
    t12 = 2.0 * n_metal / (n_metal + n_air)
    phase = 2.0 * np.pi * n_metal * slab_thickness_um / wavelength_um
    round_trip = np.exp(2j * phase)
    denominator = 1.0 + r01 * r12 * round_trip
    reflection_amplitude = (r01 + r12 * round_trip) / denominator
    transmission_amplitude = (
        t01 * t12 * np.exp(1j * phase) / denominator
    )
    reflectance = np.abs(reflection_amplitude) ** 2
    transmittance = np.abs(transmission_amplitude) ** 2
    absorptance = 1.0 - reflectance - transmittance
    return reflectance, transmittance, absorptance


def solve_spectrum() -> dict[str, np.ndarray]:
    rows = build_simulation().GetSpectrum(
        wavelengths_um,
        (polarization,),
        Workers=workers,
    )
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete Drude slab spectrum")

    wavelength = np.asarray(rows["lambda_um"], dtype=float)
    reflectance = np.asarray(rows["R"], dtype=float)
    transmittance = np.asarray(rows["T"], dtype=float)
    absorptance = np.asarray(rows["A"], dtype=float)
    reference = fresnel_slab_spectrum(wavelength)

    maximum_error = max(
        float(np.max(np.abs(actual - expected)))
        for actual, expected in zip(
            (reflectance, transmittance, absorptance),
            reference,
        )
    )
    if maximum_error > 2.0e-10:
        raise RuntimeError(
            f"scattering/Fresnel spectrum mismatch: {maximum_error:.3e}"
        )
    if np.min(absorptance) < -2.0e-12:
        raise RuntimeError(
            f"passive slab has negative absorptance: {np.min(absorptance):.3e}"
        )

    return {
        "wavelength_um": wavelength,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
        "maximum_fresnel_error": np.array(maximum_error),
    }


def plot_spectrum(spectrum: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    wavelength = spectrum["wavelength_um"]
    axis.plot(wavelength, spectrum["reflectance"], linewidth=2.0, label="R")
    axis.plot(wavelength, spectrum["transmittance"], linewidth=2.0, label="T")
    axis.plot(wavelength, spectrum["absorptance"], linewidth=2.0, label="A")
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel("Power fraction")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectrum = solve_spectrum()
    print(
        "air / Drude metal / air, "
        f"thickness={slab_thickness_um:g} um, "
        f"max Fresnel error={float(spectrum['maximum_fresnel_error']):.3e}, "
        f"R+T+A error={np.max(np.abs(spectrum['reflectance'] + spectrum['transmittance'] + spectrum['absorptance'] - 1.0)):.3e}"
    )
    plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
