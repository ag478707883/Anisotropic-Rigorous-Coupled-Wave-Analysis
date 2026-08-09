"""Reproduce Sun et al., Optics & Laser Technology 170 (2024) 110308, Fig. 2."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Paper Fig. 1 and Fig. 2.
period_um = 10.0
ridge_width_um = 5.0
grating_height_um = 0.6
ge_waveguide_height_um = 1.0
wsm_height_um = 0.3
ge_epsilon = 16.0

wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    # The scanned paper defines Ef but does not print its numerical value.
    "fermi_energy_ev": 0.15,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 1.0
phi_deg = 60.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)
orders = (40, 0)
workers = 0  # Automatic hardware-limited concurrency.
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = 0 if workers == 0 else max(1, workers // len(angles_deg))

wavelength_min_um = 10.0
wavelength_max_um = 12.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(wavelength_min_um, wavelength_max_um, wavelength_point_count)

paper_eta_peak_um = 10.42
paper_eta_peak = 0.66
paper_emission_reference_um = 10.50
paper_emission = 0.77
paper_absorptivity = 0.18
paper_eta_at_emission = 0.59
paper_markers = {"TE": (paper_eta_peak_um, paper_emission_reference_um)}

plot_path = project_root / "images" / "sun_2024_ge_zcg_wsm_ag_fig2.png"
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterial("Air", 1.0 + 0.0j)
    sim.SetMaterial("Ge", ge_epsilon + 0.0j)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Ag")
    sim.AddLayer("Ge_grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "Ge_grating",
        "Ge",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    sim.AddLayer("Ge_waveguide", ge_waveguide_height_um, "Ge")
    sim.AddLayer("WSM", wsm_height_um, "WSM")
    return sim


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    fig.suptitle(f"Ge ZCG / WSM / Ag, theta={theta_deg:g} deg, phi={phi_deg:g} deg")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve forward/reverse spectrum"):
        spectrum = solve_fr(
            rcwa=asyrcwa,
            build_simulation=build_simulation,
            wavelengths_um=wavelengths_um,
            polarizations=polarizations,
            angles_deg=angles_deg,
            phi_deg=phi_deg,
            workers_per_angle=workers_per_angle,
        )
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        peak = spectrum[int(np.argmax(spectrum["eta"]))]
        reference = spectrum[int(np.argmin(abs(spectrum["lambda_um"] - paper_emission_reference_um)))]
        print(
            f"computed max eta={peak['eta']:.4f} at {peak['lambda_um']:.4f} um "
            f"(paper {paper_eta_peak:.2f} at {paper_eta_peak_um:.2f} um)"
        )
        print(
            f"at {paper_emission_reference_um:.2f} um: "
            f"A/e/eta={reference['A']:.4f}/"
            f"{reference['emissivity_same_channel']:.4f}/{reference['eta']:.4f} "
            f"(paper {paper_absorptivity:.2f}/{paper_emission:.2f}/"
            f"{paper_eta_at_emission:.2f})"
        )
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
