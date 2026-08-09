"""Reproduce Wu and Qing, PCCP 25 (2023) 9586-9591, Fig. 2."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


period_um = 2.96
duty_cycle = 0.61
grating_height_um = 5.60
dielectric_height_um = 2.77
graphene_height_um = 0.00034
wsm_height_um = 0.54
grating_index = 2.41
dielectric_index = 1.64
substrate_index = 2.41

graphene_parameters = {
    "fermi_energy_ev": 1.0,
    "tau_ps": 0.5,
    "thickness_nm": 0.34,
}
wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    # Solver +y reproduces the paper's high e(+66 deg), low alpha(+66 deg).
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}

theta_deg = 66.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)
orders = (9, 0)
workers = 10
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 7.0
wavelength_max_um = 9.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(wavelength_min_um, wavelength_max_um, wavelength_point_count)

paper_peak_um = 8.0
paper_absorptivity_max = 0.023
paper_emissivity_min = 0.9989
paper_eta_min = 0.976
paper_markers = {"TM": (paper_peak_um,)}

plot_path = project_root / "images" / "wu_2023_graphene_wsm_fig2.png"
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
    sim.SetMaterial("Grating", grating_index**2 + 0.0j)
    sim.SetMaterial("Dielectric", dielectric_index**2 + 0.0j)
    sim.SetMaterial("Substrate", substrate_index**2 + 0.0j)
    sim.SetMaterialModel("Graphene", "GrapheneKubo", graphene_parameters)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Substrate")
    sim.AddLayer("Grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "Grating",
        "Grating",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * duty_cycle * period_um, 0.5 * period_um),
    )
    sim.AddLayer("Dielectric", dielectric_height_um, "Dielectric")
    sim.AddLayer("Graphene", graphene_height_um, "Graphene")
    sim.AddLayer("WSM", wsm_height_um, "WSM")
    return sim


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    fig.suptitle(f"Dielectric grating / graphene / WSM, theta={theta_deg:g} deg")
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
        row = spectrum[int(np.argmin(abs(spectrum["lambda_um"] - paper_peak_um)))]
        print(
            f"at {paper_peak_um:g} um: A/e/eta={row['A']:.4f}/"
            f"{row['emissivity_same_channel']:.4f}/{row['eta']:.4f}; "
            f"paper <{paper_absorptivity_max:.3f}/>"
            f"{paper_emissivity_min:.4f}/>{paper_eta_min:.3f}"
        )
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
