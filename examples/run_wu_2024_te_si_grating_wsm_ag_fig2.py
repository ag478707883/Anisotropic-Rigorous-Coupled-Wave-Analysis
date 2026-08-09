"""Reproduce Wu and Qing, ICHMT 151 (2024) 107254, Fig. 2.

The one-dimensional Si grating uses the backend's analytic lamellar
Fourier-factorization route. The local RCWA backend computes the paper's TE
absorptivity, emissivity, and nonreciprocity under conical incidence.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Wu and Qing, Fig. 1, Fig. 2, Eqs. (1)-(4), and Section 3.
period_um = 4.80
ridge_width_um = 3.00
grating_height_um = 2.38
wsm_height_um = 0.63
ag_height_um = 1.00
silicon_epsilon = 11.9
substrate_refractive_index = 1.45

wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    # Fig. 1 uses paper y as the stack normal and paper z along the grooves.
    # Paper z therefore maps to solver y; -y selects its high-e(+56 deg) branch
    # under the backend's exp(-i*omega*t) convention.
    "node_separation_direction": (0.0, -1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    # The paper defines EF(T) but does not print its numerical value.
    "fermi_energy_ev": 0.15,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 56.0
phi_deg = 33.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)

# The paper does not report its retained Fourier order. This practical 1D
# setting keeps 81 harmonics and preserves the analytic Li inverse-rule route.
orders = (40, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 11.0
wavelength_max_um = 13.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_peak_um = 12.0
paper_absorptivity_upper_bound = 0.049
paper_emissivity_lower_bound = 0.998
paper_eta = 0.95
paper_markers = {"TE": (paper_peak_um,)}

plot_path = project_root / "images" / "wu_2024_te_si_grating_wsm_ag_fig2.png"
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < ridge_width_um < period_um:
        raise ValueError("ridge_width_um must lie between zero and period_um")

    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterial("Air", 1.0 + 0.0j)
    sim.SetMaterial("Si", silicon_epsilon + 0.0j)
    sim.SetMaterial("SiO2", substrate_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    sim.AddLayer("Si_grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "Si_grating",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    sim.AddLayer("WSM", wsm_height_um, "WSM")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


def solve_spectrum() -> np.ndarray:
    return solve_fr(
        rcwa=asyrcwa,
        build_simulation=build_simulation,
        wavelengths_um=wavelengths_um,
        polarizations=polarizations,
        angles_deg=angles_deg,
        phi_deg=phi_deg,
        workers_per_angle=workers_per_angle,
    )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    ax.set_title(
        rf"TE polarization, $\theta={theta_deg:g}^\circ$, "
        rf"$\phi={phi_deg:g}^\circ$"
    )
    fig.suptitle("Wu and Qing (2024), Fig. 2: Si grating / WSM / Ag")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Wu and Qing 107254 Fig. 2 TE spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        peak = spectrum[int(np.argmax(spectrum["eta"]))]
        print(
            f"computed peak at {peak['lambda_um']:.4f} um: "
            f"A/e/eta={peak['A']:.4f}/"
            f"{peak['emissivity_same_channel']:.4f}/{peak['eta']:.4f}; "
            f"paper near {paper_peak_um:g} um: "
            f"A<{paper_absorptivity_upper_bound:.3f}, "
            f"e>{paper_emissivity_lower_bound:.3f}, eta~{paper_eta:.2f}"
        )
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
