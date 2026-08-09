"""Reproduce the TE nonreciprocal spectrum in Li and Wang (2024), Fig. 2.

Article: "Tunable nonreciprocal thermal emitter based on graphene/indium
arsenide/silver microstructure".
Journal: International Communications in Heat and Mass Transfer 157 (2024)
107772. DOI: 10.1016/j.icheatmasstransfer.2024.107772.

The paper defines alpha(+theta) = 1 - R(+theta) and
e(+theta) = 1 - R(-theta). The 0.2 um Ag reflector makes transmission
negligible, so the script follows those definitions directly.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Paper geometry: Fig. 1 and Table 1.
period_um = 10.25
graphene_ridge_width_um = 5.89
graphene_fill_factor = graphene_ridge_width_um / period_um
graphene_height_um = 0.00034
inas_height_um = 1.67
ag_height_um = 0.2

# Paper Eqs. (1)-(3). GrapheneKubo contains the paper's intraband term and a
# negligible interband correction for Ef=1 eV at these mid-IR wavelengths.
graphene_parameters = {
    "fermi_energy_ev": 1.0,
    "tau_ps": 1.0,
    "thickness_nm": 0.34,
}
ag_drude_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

# The paper prints the InAs tensor equations but refers to its cited material
# literature for these constants. These are the project's standard values for
# that InAs model. With the solver axes used here, +3 T reproduces the ordering
# of the paper's alpha(+27 deg) and e(+27 deg) resonance peaks.
inas_parameters = {
    "magnetic_field_t": 3.0,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}

theta_deg = 27.0
phi_deg = 72.0  # Paper azimuth beta, mapped to solver phi.
angles_deg = (+theta_deg, -theta_deg)
polarization = "TE"

# This is a genuinely 1D lamellar grating. AddLayer plus SetRegionRectangle
# uses the same ordered-Li patterned-layer path. Orders=(51, 0)
# gives peak positions within about 0.002 um of a (71, 0) check.
orders = (35, 0)
workers = 10
# Fourier convergence controls. "Default" preserves the original example result.

wavelength_min_um = 15.8
wavelength_max_um = 17.0
wavelength_point_count = 1201
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peak_um = 16.60
paper_emission_peak_um = 16.02
paper_nonreciprocity_at_16_6 = 0.91

plot_path = project_root / "images" / "li_2024_graphene_inas_ag_fig2.png"
figure_size = (7.2, 4.8)
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
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterialModel(
        "Graphene",
        "GrapheneKubo",
        graphene_parameters,
    )
    simulation.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_drude_parameters)

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    simulation.AddLayer("Graphene_grating", graphene_height_um, "Air")
    simulation.SetRegionRectangle(
        "Graphene_grating",
        "Graphene",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * graphene_fill_factor * period_um, 0.5 * period_um),
    )
    simulation.AddLayer("InAs", inas_height_um, "InAs")
    simulation.AddLayer("Ag_reflector", ag_height_um, "Ag")
    return simulation


def rows_for_angle(rows: np.ndarray, theta: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], theta, rtol=0.0, atol=1e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_fig2_spectrum() -> dict[str, np.ndarray | float]:
    simulation = build_simulation()
    start = perf_counter()
    rows = simulation.GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        [polarization],
        Phi=phi_deg,
        Workers=workers,
    )
    elapsed_seconds = perf_counter() - start

    forward = rows_for_angle(rows, +theta_deg)
    reverse = rows_for_angle(rows, -theta_deg)
    if len(forward) != wavelength_point_count or len(reverse) != wavelength_point_count:
        raise RuntimeError("incomplete forward/reverse wavelength spectrum")
    if not np.allclose(
        forward["lambda_um"],
        reverse["lambda_um"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("misaligned forward/reverse wavelengths")

    absorptivity = 1.0 - np.asarray(forward["R"], dtype=float)
    emissivity = 1.0 - np.asarray(reverse["R"], dtype=float)
    return {
        "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
        "absorptivity": absorptivity,
        "emissivity": emissivity,
        "nonreciprocity": np.abs(absorptivity - emissivity),
        "forward_transmittance": np.asarray(forward["T"], dtype=float),
        "reverse_transmittance": np.asarray(reverse["T"], dtype=float),
        "elapsed_seconds": elapsed_seconds,
    }


def print_summary(spectrum: dict[str, np.ndarray | float]) -> None:
    wavelength = np.asarray(spectrum["wavelength_um"])
    absorptivity = np.asarray(spectrum["absorptivity"])
    emissivity = np.asarray(spectrum["emissivity"])
    nonreciprocity = np.asarray(spectrum["nonreciprocity"])

    absorption_index = int(np.argmax(absorptivity))
    emission_index = int(np.argmax(emissivity))
    eta_index = int(np.argmax(nonreciprocity))
    paper_index = int(np.argmin(np.abs(wavelength - paper_absorption_peak_um)))
    max_transmittance = max(
        float(np.max(spectrum["forward_transmittance"])),
        float(np.max(spectrum["reverse_transmittance"])),
    )

    print(
        f"orders={orders}, points={wavelength_point_count}, workers={workers}, "
        f"elapsed={float(spectrum['elapsed_seconds']):.3f} s"
    )
    print(
        f"TE alpha(+{theta_deg:g} deg) peak: "
        f"{wavelength[absorption_index]:.4f} um, {absorptivity[absorption_index]:.4f} "
        f"(paper about {paper_absorption_peak_um:.2f} um)"
    )
    print(
        f"TE e(+{theta_deg:g} deg)=alpha(-{theta_deg:g} deg) peak: "
        f"{wavelength[emission_index]:.4f} um, {emissivity[emission_index]:.4f} "
        f"(paper about {paper_emission_peak_um:.2f} um)"
    )
    print(
        f"max eta={nonreciprocity[eta_index]:.4f} at "
        f"{wavelength[eta_index]:.4f} um; eta(16.60 um)="
        f"{nonreciprocity[paper_index]:.4f} (paper >{paper_nonreciprocity_at_16_6:.2f})"
    )
    print(f"maximum Ag-backed transmittance={max_transmittance:.3e}")


def plot_fig2_spectrum(spectrum: dict[str, np.ndarray | float]) -> None:
    wavelength = np.asarray(spectrum["wavelength_um"])
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    axis.plot(
        wavelength,
        spectrum["absorptivity"],
        color="#303030",
        linewidth=2.0,
        label=r"$\alpha(+\theta)$",
    )
    axis.plot(
        wavelength,
        spectrum["emissivity"],
        color="#ef4b43",
        linestyle="-.",
        linewidth=1.9,
        label=r"$e(+\theta)=\alpha(-\theta)$",
    )
    axis.plot(
        wavelength,
        spectrum["nonreciprocity"],
        color="#1765b0",
        linestyle="-.",
        linewidth=1.8,
        label=r"$\eta=|\alpha-e|$",
    )
    axis.axvline(
        paper_absorption_peak_um,
        color="#777777",
        linewidth=0.8,
        alpha=0.35,
    )
    axis.axvline(
        paper_emission_peak_um,
        color="#777777",
        linewidth=0.8,
        alpha=0.35,
    )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel(r"$\alpha$, $e$, $\eta$")
    axis.set_title(
        r"Graphene/InAs/Ag, TE, $\theta=27^\circ$, $\beta=72^\circ$, $B=3$ T"
    )
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, np.ndarray | float]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectrum = solve_fig2_spectrum()
    print_summary(spectrum)
    plot_fig2_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
