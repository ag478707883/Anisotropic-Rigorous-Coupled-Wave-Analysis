"""Compute the spectrum of the EMLab one-dimensional dielectric grating.

This is the scattering-matrix example with a 0.7-um period, a 0.46-um
thickness, and a 30-percent silicon ridge fill factor.  The analytic lamellar
Fourier representation replaces the raster FFT construction in the original
script.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


air_refractive_index = 1.0
ridge_refractive_index = 3.48
fill_factor = 0.30
period_um = 0.70
grating_thickness_um = 0.46

# The original num_ord=10 retains diffraction orders -10..+10.
orders = (10, 0)
workers = 0

wavelength_min_um = 0.50
wavelength_max_um = 2.30
wavelength_point_count = 300
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

theta_deg = 0.0
phi_deg = 0.0
polarization = "TE"

plot_path = project_root / "images" / "emlab_1d_grating_spectrum.png"
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation() -> object:
    simulation = asyrcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterial("Air", air_refractive_index**2 + 0.0j)
    simulation.SetMaterial("Silicon", ridge_refractive_index**2 + 0.0j)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    simulation.AddLayer("silicon_grating", grating_thickness_um, "Air")
    simulation.SetRegionRectangle(
        "silicon_grating",
        "Silicon",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * fill_factor * period_um, 0.5 * period_um),
    )
    simulation.SetExcitationPlanewave(
        (theta_deg, phi_deg),
        sAmplitude=1.0 + 0.0j,
        pAmplitude=0.0 + 0.0j,
    )
    return simulation


def solve_spectrum() -> dict[str, np.ndarray | float]:
    rows = build_simulation().GetSpectrum(
        wavelengths_um,
        (polarization,),
        Workers=workers,
    )
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete EMLab grating spectrum")

    wavelength = np.asarray(rows["lambda_um"], dtype=float)
    reflectance = np.asarray(rows["R"], dtype=float)
    transmittance = np.asarray(rows["T"], dtype=float)
    power_sum = reflectance + transmittance
    if not all(
        np.all(np.isfinite(values))
        for values in (wavelength, reflectance, transmittance, power_sum)
    ):
        raise RuntimeError("spectrum contains non-finite values")

    energy_error = float(np.max(np.abs(power_sum - 1.0)))
    minimum_power = min(float(np.min(reflectance)), float(np.min(transmittance)))
    if energy_error > 1.0e-10:
        raise RuntimeError(
            f"lossless energy-conservation error is too large: {energy_error:.3e}"
        )
    if minimum_power < -1.0e-12:
        raise RuntimeError(
            f"passive spectrum contains negative power: {minimum_power:.3e}"
        )

    return {
        "wavelength_um": wavelength,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "power_sum": power_sum,
        "maximum_energy_error": energy_error,
        "harmonics": float(rows["harmonics"][0]),
    }


def plot_spectrum(spectrum: dict[str, np.ndarray | float]) -> None:
    wavelength = np.asarray(spectrum["wavelength_um"])
    figure, axis = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    axis.plot(wavelength, spectrum["reflectance"], linewidth=1.8, label="R")
    axis.plot(wavelength, spectrum["transmittance"], linewidth=1.8, label="T")
    axis.plot(
        wavelength,
        spectrum["power_sum"],
        color="#2a9d55",
        linewidth=1.3,
        linestyle="--",
        label="R + T",
    )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel("Power fraction")
    axis.set_title("EMLab 1D dielectric grating spectrum")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, loc="center right")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, np.ndarray | float]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectrum = solve_spectrum()
    print(
        f"polarization={polarization}, harmonics={int(spectrum['harmonics'])}, "
        f"max |R+T-1|={spectrum['maximum_energy_error']:.3e}, "
        f"R range=[{np.min(spectrum['reflectance']):.6f}, "
        f"{np.max(spectrum['reflectance']):.6f}]"
    )
    plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
