"""Reproduce Shi et al., IJHMT 244 (2025) 126966, Fig. 2.

The TE spectrum is computed with the local RCWA backend for the analytic
cross-shaped Si unit cell, a uniform effective graphene layer, InAs, and an
opaque Al reflector.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Shi et al., Table 1 and Fig. 1.
period_um = 8.50
cross_arm_width_um = 4.25
cross_arm_length_um = period_um
si_cross_height_um = 9.52
inas_height_um = 8.57
al_height_um = 1.00
graphene_height_um = 0.00034

# The target paper does not print Si or SiO2 optical constants. The repository
# Si model is n=3.48; n=1.45 is an explicit SiO2 substrate assumption. The
# InAs carrier parameters are the standard values cited by the paper's refs.
sio2_refractive_index = 1.45
inas_parameters = {
    "magnetic_field_t": +4.0,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
graphene_parameters = {
    "fermi_energy_ev": 1.0,
    "temperature_k": 300.0,
    "tau_ps": 0.5,
    "thickness_nm": 0.34,
}
al_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

theta_deg = 0.65
phi_deg = 0.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)

# The article does not report the Fourier truncation. Circular (7, 7) retains
# 149 harmonics and resolves the cross-cell analytic polygon while remaining
# practical for a 0.5-nm wavelength grid; (5,5)->(7,7) is reported at runtime.
orders = (5, 5)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 11.40
wavelength_max_um = 11.65
wavelength_point_count = 501
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peaks_um = {"TE": (11.45, 11.55)}
paper_emission_peaks_um = {"TE": (11.46, 11.53)}
paper_peak_eta = {
    "TE": {
        11.45: 0.85,
        11.46: 0.85,
        11.53: 0.90,
        11.55: 0.75,
    }
}
paper_markers = {
    "TE": tuple(
        sorted(
            paper_absorption_peaks_um["TE"]
            + paper_emission_peaks_um["TE"]
        )
    )
}

plot_path = (
    project_root
    / "images"
    / "shi_2025_graphene_si_cross_inas_al_fig2.png"
)
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < cross_arm_width_um <= cross_arm_length_um <= period_um:
        raise ValueError("cross dimensions must fit within one period")

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("Graphene", "GrapheneKubo", graphene_parameters)
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    # Fig. 1(d) depicts graphene as a continuous 0.34-nm coating. The Si
    # cross itself is an analytic 12-vertex polygon inside the unit cell.
    sim.AddLayer("Graphene_coating", graphene_height_um, "Graphene")
    sim.AddLayer("Si_cross", si_cross_height_um, "Air")
    sim.SetRegionCross(
        "Si_cross",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        ArmLength=cross_arm_length_um,
        ArmWidth=cross_arm_width_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Al_reflector", al_height_um, "Al")
    return sim


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TE"]
    print("TE paper checkpoints (A/e/eta):")
    checkpoints = (
        *paper_absorption_peaks_um["TE"],
        *paper_emission_peaks_um["TE"],
    )
    for wavelength_um in sorted(set(checkpoints)):
        row = nearest_row(rows, wavelength_um)
        print(
            f"  {wavelength_um:.3f} um: "
            f"{row['A']:.6f}/{row['emissivity_same_channel']:.6f}/"
            f"{row['eta']:.6f}; "
            f"paper eta>{paper_peak_eta['TE'][wavelength_um]:.2f}"
        )

    absorption_peak = rows[int(np.argmax(rows["A"]))]
    emission_peak = rows[
        int(np.argmax(rows["emissivity_same_channel"]))
    ]
    print(
        f"computed max A={absorption_peak['A']:.6f} at "
        f"{absorption_peak['lambda_um']:.5f} um; "
        f"max e={emission_peak['emissivity_same_channel']:.6f} at "
        f"{emission_peak['lambda_um']:.5f} um"
    )
    paper_emissivity_gap = float(np.max(
        abs(1.0 - rows["R_reverse"] - rows["emissivity_same_channel"])
    ))
    max_transmission = max(
        float(np.max(rows["T"])),
        float(np.max(rows["T_reverse"])),
    )
    print(
        f"max T={max_transmission:.3e}; "
        f"max |1-R(-theta)-exact e|={paper_emissivity_gap:.3e}; "
        f"harmonics={int(rows['harmonics'][0])}"
    )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    ax.set_ylabel("Intensity")
    ax.set_title(
        "Shi et al. (2025): graphene / Si cross / InAs / Al, "
        f"TE, theta=+/-{theta_deg:g} deg, B={inas_parameters['magnetic_field_t']:g} T"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, cross width={cross_arm_width_um:g} um, "
        f"h1/h2/h3={si_cross_height_um:g}/{inas_height_um:g}/{al_height_um:g} um"
    )
    with asyrcwa.timed_step("solve Shi et al. Fig. 2 TE spectrum"):
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
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
