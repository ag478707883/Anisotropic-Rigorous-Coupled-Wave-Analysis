"""Reproduce Zou et al., Diamond Relat. Mater. 152 (2025) 112004, Fig. 2.

Article: "Strong nonreciprocal thermal graphene-based emitter under a 0.8 T
magnetic field for TE polarization".
DOI: 10.1016/j.diamond.2025.112004.

The paper defines the TE absorptivity/emissivity through the reflected
polarization-resolved scattering channels. This example uses the backend's
directional thermal-channel API, so the plotted emissivity is evaluated from
the exact reverse incidence channel rather than from a scalar 1 - R surrogate.
"""

from __future__ import annotations

import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Paper Fig. 1, Fig. 2, Table 1, and Eqs. (1)-(7).
period_um = 7.70
znse_ridge_width_um = 1.30
znse_ridge_height_um = 0.80
znse_waveguide_height_um = 0.10
graphene_height_um = 0.00034
inas_height_um = 1.55
ag_height_um = 0.20

znse_epsilon = 5.73

graphene_parameters = {
    "fermi_energy_ev": 0.10,
    "tau_ps": 0.5,
    "thickness_nm": 0.34,
}

# The article prints the InAs tensor form but refers to earlier material
# literature for the carrier constants; these are the repository's standard
# InAs parameters used by neighboring InAs nonreciprocal examples. The negative
# signed field reproduces the paper ordering alpha(+55 deg) before e(+55 deg).
inas_parameters = {
    "magnetic_field_t": -0.80,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}

ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 55.0
phi_deg = 65.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)

# The paper reports convergence for nn=30. Here Orders=(30, 0) keeps the 1D
# grating harmonics from -30...+30 and no y diffraction orders.
orders = (30, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.0
wavelength_max_um = 16.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peak_um = 15.416
paper_absorption_peak_value = 0.992
paper_emission_peak_um = 15.542
paper_emission_peak_value = 0.995
paper_eta_reference = 0.90

plot_path = project_root / "images" / "zou_2025_graphene_znse_inas_ag_fig2.png"
data_path = project_root / "data" / "zou_2025_graphene_znse_inas_ag_fig2_te.csv"
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    simulation = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("ZnSe", znse_epsilon + 0.0j)
    simulation.SetMaterialModel("Graphene", "GrapheneKubo", graphene_parameters)
    simulation.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_parameters)

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    simulation.AddLayer("ZnSe_grating", znse_ridge_height_um, "Air")
    simulation.SetRegionRectangle(
        "ZnSe_grating",
        "ZnSe",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * znse_ridge_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("ZnSe_waveguide", znse_waveguide_height_um, "ZnSe")
    simulation.AddLayer("Graphene", graphene_height_um, "Graphene")
    simulation.AddLayer("InAs", inas_height_um, "InAs")
    simulation.AddLayer("Ag_reflector", ag_height_um, "Ag")
    return simulation


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


def nearest_row(rows: np.ndarray, wavelength_um: float) -> np.void:
    return rows[int(np.argmin(np.abs(rows["lambda_um"] - wavelength_um)))]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TE"]
    absorption_peak = rows[int(np.argmax(rows["A"]))]
    emission_peak = rows[int(np.argmax(rows["emissivity_same_channel"]))]
    eta_peak = rows[int(np.argmax(rows["eta"]))]
    at_absorption = nearest_row(rows, paper_absorption_peak_um)
    at_emission = nearest_row(rows, paper_emission_peak_um)

    print(
        "  computed absorption peak: "
        f"{float(absorption_peak['lambda_um']):.4f} um, "
        f"alpha={float(absorption_peak['A']):.4f}; "
        f"paper {paper_absorption_peak_um:.3f} um, "
        f"alpha={paper_absorption_peak_value:.3f}"
    )
    print(
        "  computed emission peak: "
        f"{float(emission_peak['lambda_um']):.4f} um, "
        f"e={float(emission_peak['emissivity_same_channel']):.4f}; "
        f"paper {paper_emission_peak_um:.3f} um, "
        f"e={paper_emission_peak_value:.3f}"
    )
    print(
        f"  computed max eta={float(eta_peak['eta']):.4f} at "
        f"{float(eta_peak['lambda_um']):.4f} um "
        f"(paper about {paper_eta_reference:.2f})"
    )
    print(
        f"  at paper alpha peak: alpha/e/eta="
        f"{float(at_absorption['A']):.4f}/"
        f"{float(at_absorption['emissivity_same_channel']):.4f}/"
        f"{float(at_absorption['eta']):.4f}"
    )
    print(
        f"  at paper e peak: alpha/e/eta="
        f"{float(at_emission['A']):.4f}/"
        f"{float(at_emission['emissivity_same_channel']):.4f}/"
        f"{float(at_emission['eta']):.4f}"
    )
    print(
        f"  max forward/reverse transmittance="
        f"{max(float(np.max(rows['T'])), float(np.max(rows['T_reverse']))):.3e}"
    )


def export_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TE"], order="lambda_um")
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete TE spectrum returned by backend")

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(spectrum.dtype.names)
        for row in rows:
            writer.writerow(row[name].item() for name in spectrum.dtype.names)
    print(f"Saved {data_path}.")


def plot_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TE"], order="lambda_um")
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    axis.plot(
        rows["lambda_um"],
        rows["A"],
        color="#d62728",
        linewidth=2.0,
        label=r"Absorptivity, $\alpha$",
    )
    axis.plot(
        rows["lambda_um"],
        rows["emissivity_same_channel"],
        color="#1f4eaa",
        linestyle="-.",
        linewidth=1.9,
        label=r"Emissivity, $e$",
    )
    axis.plot(
        rows["lambda_um"],
        rows["eta"],
        color="#202020",
        linestyle=":",
        linewidth=1.9,
        label=r"Nonreciprocity, $\eta$",
    )
    for wavelength_um in (paper_absorption_peak_um, paper_emission_peak_um):
        axis.axvline(
            wavelength_um,
            color="#777777",
            linewidth=0.8,
            alpha=0.35,
        )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Wavelength (um)")
    axis.set_ylabel(r"Intensity ($\alpha$, $e$ and $\eta$)")
    axis.set_title(
        r"ZnSe grating / graphene / InAs / Ag, TE, "
        rf"$\theta={theta_deg:g}^\circ$, $\beta={phi_deg:g}^\circ$, "
        rf"$B={abs(inas_parameters['magnetic_field_t']):g}$ T"
    )
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False, loc="upper right")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"Orders={orders}, workers={workers}, theta=+/-{theta_deg:g} deg, "
        f"phi={phi_deg:g} deg, B_solver="
        f"{inas_parameters['magnetic_field_t']:g} T."
    )
    with asyrcwa.timed_step("solve Zou et al. Fig. 2 spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_paper_comparison(spectrum)
    export_spectrum(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
