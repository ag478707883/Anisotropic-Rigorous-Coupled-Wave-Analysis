"""Reproduce Wu and Qing, ICHMT 156 (2024) 107639, Fig. 2.

The one-dimensional Si high-contrast grating uses the backend's analytic
lamellar Fourier-factorization route. The local RCWA backend computes the
paper's TE absorptivity, emissivity, and nonreciprocity under conical incidence.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Wu and Qing, Fig. 1, Fig. 2, and Section 3.
period_um = 8.23
ridge_width_um = 4.30
grating_height_um = 1.40
dielectric_height_um = 1.40
wsm_height_um = 2.00
silicon_refractive_index = 3.45
dielectric_refractive_index = 1.64
substrate_refractive_index = 1.45

# Eq. (1) and Ref. 22. The target paper refers to Ref. 22 for the diagonal
# WSM response without reprinting its numerical constants. These are the
# repository's explicit Zhao et al. WSM parameters used by neighboring
# paper-reproduction examples. The backend uses exp(-i*omega*t), so -y selects
# the paper's high-emissivity +41-degree branch under its unspecified convention.
wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    "node_separation_direction": (0.0, -1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}

theta_deg = 41.0
phi_deg = 69.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)

# The paper does not report its retained Fourier order. This practical 1D
# setting keeps 81 harmonics and preserves the analytic Li inverse-rule route.
orders = (40, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.0
wavelength_max_um = 17.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_peak_um = 16.0
paper_absorptivity_upper_bound = 0.052
paper_emissivity = 0.993
paper_eta_lower_bound = 0.94
paper_markers = {"TE": (paper_peak_um,)}

plot_path = project_root / "images" / "wu_2024_te_dielectric_wsm_grating_fig2.png"
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
    sim.SetMaterial("Si", silicon_refractive_index**2 + 0.0j)
    sim.SetMaterial("Dielectric", dielectric_refractive_index**2 + 0.0j)
    sim.SetMaterial("Substrate", substrate_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Substrate")

    sim.AddLayer("Si_grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "Si_grating",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    sim.AddLayer("Dielectric", dielectric_height_um, "Dielectric")
    sim.AddLayer("WSM", wsm_height_um, "WSM")
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
    fig.suptitle("Wu and Qing (2024), Fig. 2: Si grating / dielectric / WSM")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Wu and Qing Fig. 2 TE spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        peak = spectrum[int(np.argmax(spectrum["eta"]))]
        print(
            f"computed peak at {peak['lambda_um']:.4f} um: "
            f"A/e/eta={peak['A']:.4f}/"
            f"{peak['emissivity_same_channel']:.4f}/{peak['eta']:.4f}; "
            f"paper near {paper_peak_um:g} um: "
            f"A<{paper_absorptivity_upper_bound:.3f}, "
            f"e={paper_emissivity:.3f}, eta>{paper_eta_lower_bound:.2f}"
        )
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
