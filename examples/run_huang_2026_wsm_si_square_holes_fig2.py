"""Reproduce Huang et al., IJHMT 267 (2026) 128958, Fig. 2(c,d)."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Huang et al., Fig. 1, Fig. 2, and Section 3.1.
period_um = 8.30
square_hole_width_um = 2.40
si_square_hole_height_um = 2.30
ge_height_um = 2.30
wsm_height_um = 2.00
ag_height_um = 0.50
si_refractive_index = 3.48
ge_refractive_index = 4.00

wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    # The paper uses +z out of the stack; this backend stacks layers along +z.
    # Keeping +x fixed in the right-handed coordinate transform maps +y to -y.
    "node_separation_direction": (0.0, -1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 5.0
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)
# RETICOLO V9 uses nx=ny=11 here: orders -5...5 on both axes, 121 modes.
harmonic_orders = (6, 6)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 16.4
wavelength_max_um = 17.6
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_emission_peaks_um = {
    "TE": (16.594, 17.073),
    "TM": (16.900, 17.491),
}
paper_absorption_peaks_um = {
    "TE": (16.758, 17.267),
    "TM": (16.769, 17.449),
}
paper_eta_peaks = {
    "TE": ((16.594, 0.9686), (17.073, 0.9157)),
    "TM": ((16.900, 0.9589), (17.449, 0.8093)),
}
paper_markers = {
    polarization: tuple(
        sorted(
            paper_emission_peaks_um[polarization]
            + paper_absorption_peaks_um[polarization]
        )
    )
    for polarization in polarizations
}

plot_path = project_root / "images" / "huang_2026_wsm_si_square_holes_fig2.png"
tm_spectrum_data_path = (
    project_root / "data" / "huang_2026_wsm_si_square_holes_fig2_tm.csv"
)
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < square_hole_width_um < period_um:
        raise ValueError("square_hole_width_um must lie between zero and period_um")

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=harmonic_orders,
    )
    sim.SetMaterial("Air", 1.0 + 0.0j)
    sim.SetMaterial("Si", si_refractive_index**2 + 0.0j)
    sim.SetMaterial("Ge", ge_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Air")

    sim.AddLayer("Si_square_holes", si_square_hole_height_um, "Si")
    sim.SetRegionRectangle(
        "Si_square_holes",
        "Air",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * square_hole_width_um, 0.5 * square_hole_width_um),
    )
    sim.AddLayer("Ge", ge_height_um, "Ge")
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


def export_tm_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TM"], order="lambda_um")
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete TM spectrum returned by backend")

    columns = [
        rows["lambda_um"],
        rows["lambda_um"] * 1000.0,
        rows["theta_deg"],
        np.full(len(rows), angles_deg[1], dtype=float),
        rows["phi_deg"],
        rows["harmonics"],
        rows["R"],
        rows["T"],
        rows["A"],
        rows["R_reverse"],
        rows["T_reverse"],
        rows["A_reverse"],
        rows["emissivity_same_channel"],
        rows["emissivity_reverse_channel"],
        rows["eta"],
        rows["conservation"],
        rows["reverse_conservation"],
    ]
    header = (
        "wavelength_um,wavelength_nm,theta_plus_deg,theta_minus_deg,phi_deg,"
        "harmonics,R_plus,T_plus,A_plus,R_minus,T_minus,A_minus,"
        "emissivity_same_channel,emissivity_reverse_channel,eta,"
        "conservation_plus,conservation_minus"
    )
    tm_spectrum_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        tm_spectrum_data_path,
        np.column_stack(columns),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10f",
    )
    print(f"Saved {tm_spectrum_data_path}.")


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    plot_fr_axes(axes, spectrum, polarizations, markers=paper_markers)
    for ax in axes:
        ax.set_xlim(wavelength_min_um, wavelength_max_um)
    fig.suptitle(
        "Si square-hole array / Ge / WSM / Ag, "
        f"RETICOLO li=1, Orders={harmonic_orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Huang et al. Fig. 2(c,d) spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        for polarization in polarizations:
            rows = spectrum[spectrum["pol"] == polarization]
            for wavelength_um, paper_eta in paper_eta_peaks[polarization]:
                row = rows[int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))]
                print(
                    f"{polarization}: eta({wavelength_um:.3f} um)="
                    f"{row['eta']:.4f}; paper {paper_eta:.4f}"
                )
    export_tm_spectrum(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
