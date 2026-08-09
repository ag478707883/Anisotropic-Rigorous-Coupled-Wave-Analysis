"""Reproduce Yu et al., Phys. Scr. 101 (2026) 295509, Fig. 2.

The paper studies a cross-sector Si metasurface on magnetized InAs and an
opaque Al reflector.  The script computes absorptivity at +theta, emissivity
from the -theta spectrum, and their absolute difference with the local RCWA
backend.
"""

from __future__ import annotations

import math
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Yu et al., Fig. 1, Eqs. (1)-(5), and Section 3.
period_um = 8.8
outer_circle_diameter_ratio = 0.9
cross_groove_width_ratio = 0.2
outer_circle_diameter_um = outer_circle_diameter_ratio * period_um
outer_circle_radius_um = 0.5 * outer_circle_diameter_um
cross_groove_width_um = cross_groove_width_ratio * period_um
si_metasurface_height_um = 3.5
inas_height_um = 1.0
al_reflector_height_um = 0.2
silicon_refractive_index = 3.48

inas_base_parameters = {
    "eps_inf": 12.37,
    "carrier_density_cm3": 5.0e18,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
al_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

theta_deg = 9.0
phi_deg = 0.0
angles_deg = (+theta_deg, -theta_deg)
polarizations = ("TE", "TM")
paper_magnetic_fields_t = (0.0, 0.6)

# Paper +y is the outward layer normal while finite layers are added toward
# solver +z. Keeping +x fixed maps the paper's +z bias to solver +y. This sign
# also preserves the absorption/emission ordering reported in Fig. 2.
solver_magnetic_field_sign = +1.0

# The paper does not report its retained Fourier order. Circular (6, 6)
# retains 113 reciprocal vectors and resolves the four disconnected sectors
# while keeping the 0.2-nm wavelength grid practical.
orders = (6, 6)
sector_arc_vertex_count = 64
workers = 20

wavelength_min_um = 14.30
wavelength_max_um = 14.50
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_peak_values = {
    "TE": ((14.362, 0.826), (14.394, 0.839)),
    "TM": ((14.386, 0.877), (14.432, 0.902)),
}
plot_path = project_root / "images" / "yu_2026_cross_sector_si_inas_al_fig2.png"
figure_size = (13.0, 4.3)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rotate_vertices(
    vertices: list[tuple[float, float]],
    quarter_turns: int,
) -> list[tuple[float, float]]:
    angle = 0.5 * math.pi * quarter_turns
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        (cosine * x - sine * y, sine * x + cosine * y)
        for x, y in vertices
    ]


def first_quadrant_sector_vertices() -> list[tuple[float, float]]:
    """Return one Si sector bounded by the circular rim and cross groove."""
    groove_halfwidth_um = 0.5 * cross_groove_width_um
    start_angle = math.asin(groove_halfwidth_um / outer_circle_radius_um)
    end_angle = 0.5 * math.pi - start_angle
    arc_angles = np.linspace(
        start_angle,
        end_angle,
        sector_arc_vertex_count,
    )
    arc_vertices = [
        (
            outer_circle_radius_um * math.cos(angle),
            outer_circle_radius_um * math.sin(angle),
        )
        for angle in arc_angles
    ]
    return [
        (groove_halfwidth_um, groove_halfwidth_um),
        *arc_vertices,
    ]


def build_simulation(magnetic_field_t: float) -> object:
    simulation = asyrcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialModel("Si", "Si")
    simulation.SetMaterialModel(
        "InAs",
        "InAsMagneto",
        {
            **inas_base_parameters,
            "magnetic_field_t": (
                solver_magnetic_field_sign * magnetic_field_t
            ),
        },
    )
    simulation.SetMaterialModel("Al", "Drude", al_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    # Four non-overlapping analytic polygons reproduce the circular disk after
    # removal of the centered cross groove. Arc segmentation is geometric only;
    # the RCWA material Fourier coefficients remain analytic polygon integrals.
    simulation.AddLayer(
        "Si_cross_sectors",
        si_metasurface_height_um,
        "Air",
    )
    base_vertices = first_quadrant_sector_vertices()
    for sector_index in range(4):
        simulation.SetRegionPolygon(
            "Si_cross_sectors",
            "Si",
            Center=(0.0, 0.0),
            Angle=0.0,
            Vertices=rotate_vertices(base_vertices, sector_index),
        )
    simulation.AddLayer("InAs", inas_height_um, "InAs")
    simulation.AddLayer("Al_reflector", al_reflector_height_um, "Al")
    return simulation


def rows_at_angle(
    rows: np.ndarray,
    polarization: str,
    angle_deg: float,
) -> np.ndarray:
    selected = rows[
        (rows["pol"] == polarization)
        & np.isclose(rows["theta_deg"], angle_deg, rtol=0.0, atol=1.0e-12)
    ]
    return np.sort(selected, order="lambda_um")


def solve_case(magnetic_field_t: float) -> dict[str, dict[str, np.ndarray]]:
    rows = build_simulation(magnetic_field_t).GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        polarizations,
        Phi=phi_deg,
        Workers=workers,
    )
    spectra: dict[str, dict[str, np.ndarray]] = {}
    for polarization in polarizations:
        forward = rows_at_angle(rows, polarization, +theta_deg)
        reverse = rows_at_angle(rows, polarization, -theta_deg)
        if len(forward) != wavelength_point_count or len(reverse) != wavelength_point_count:
            raise RuntimeError("incomplete forward/reverse spectrum")
        absorptivity = 1.0 - forward["R"] - forward["T"]
        emissivity = 1.0 - reverse["R"] - reverse["T"]
        spectra[polarization] = {
            "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
            "absorptivity": np.asarray(absorptivity, dtype=float),
            "emissivity": np.asarray(emissivity, dtype=float),
            "nonreciprocity": np.abs(absorptivity - emissivity),
            "transmission": np.maximum(forward["T"], reverse["T"]),
        }
    return spectra


def nearest_index(wavelengths: np.ndarray, wavelength_um: float) -> int:
    return int(np.argmin(np.abs(wavelengths - wavelength_um)))


def print_summary(
    spectra: dict[float, dict[str, dict[str, np.ndarray]]],
) -> None:
    print(
        f"period={period_um:g} um, D={outer_circle_diameter_um:g} um, "
        f"groove={cross_groove_width_um:g} um, orders={orders}"
    )
    reciprocal_gap = max(
        float(np.max(np.abs(
            spectra[0.0][polarization]["absorptivity"]
            - spectra[0.0][polarization]["emissivity"]
        )))
        for polarization in polarizations
    )
    maximum_transmission = max(
        float(np.max(spectra[field_t][polarization]["transmission"]))
        for field_t in paper_magnetic_fields_t
        for polarization in polarizations
    )
    print(
        f"B=0 reciprocity max |alpha-e|={reciprocal_gap:.3e}; "
        f"max transmission={maximum_transmission:.3e}"
    )
    for polarization in polarizations:
        spectrum = spectra[0.6][polarization]
        wavelength = spectrum["wavelength_um"]
        nonreciprocity = spectrum["nonreciprocity"]
        peak_index = int(np.argmax(nonreciprocity))
        print(
            f"{polarization}: computed max eta={nonreciprocity[peak_index]:.6f} "
            f"at {wavelength[peak_index]:.6f} um"
        )
        for paper_wavelength_um, paper_eta in paper_peak_values[polarization]:
            index = nearest_index(wavelength, paper_wavelength_um)
            print(
                f"  paper {paper_wavelength_um:.3f} um, eta={paper_eta:.3f}; "
                f"computed alpha/e/eta="
                f"{spectrum['absorptivity'][index]:.6f}/"
                f"{spectrum['emissivity'][index]:.6f}/"
                f"{nonreciprocity[index]:.6f}"
            )


def plot_spectra(
    spectra: dict[float, dict[str, dict[str, np.ndarray]]],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, polarization, panel in zip(axes[:2], polarizations, ("(a)", "(b)")):
        reciprocal = spectra[0.0][polarization]
        biased = spectra[0.6][polarization]
        wavelength = biased["wavelength_um"]
        axis.plot(
            wavelength,
            reciprocal["absorptivity"],
            color="#222222",
            linewidth=1.6,
            label=r"0 T, $\alpha$ & $e$",
        )
        axis.plot(
            wavelength,
            biased["absorptivity"],
            color="#e54b4b",
            linewidth=1.5,
            linestyle="-.",
            label=r"0.6 T, $\alpha$",
        )
        axis.plot(
            wavelength,
            biased["emissivity"],
            color="#385fca",
            linewidth=1.5,
            linestyle="-.",
            label=r"0.6 T, $e$",
        )
        axis.text(0.04, 0.94, panel, transform=axis.transAxes, va="top")
        axis.text(0.08, 0.85, polarization, transform=axis.transAxes, ha="left")
        axis.legend(frameon=False, fontsize=7.5, loc="upper right")

    for polarization, color in (("TE", "#3a9d50"), ("TM", "#e39a32")):
        biased = spectra[0.6][polarization]
        axes[2].plot(
            biased["wavelength_um"],
            biased["nonreciprocity"],
            color=color,
            linewidth=1.8,
            label=polarization,
        )
        for wavelength_um, _ in paper_peak_values[polarization]:
            axes[2].axvline(
                wavelength_um,
                color=color,
                linewidth=0.7,
                alpha=0.20,
            )
    axes[2].text(0.04, 0.94, "(c)", transform=axes[2].transAxes, va="top")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")

    for axis in axes:
        axis.set_xlim(wavelength_min_um, wavelength_max_um)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel(r"Wavelength ($\mu$m)")
        axis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.65)
    axes[0].set_ylabel(r"$\alpha$ and $e$")
    axes[2].set_ylabel(r"$\eta=|\alpha-e|$")
    figure.suptitle(
        r"Yu et al. (2026), Fig. 2 - $\theta=9^\circ$, Si/InAs/Al"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[float, dict[str, dict[str, np.ndarray]]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectra = {
        magnetic_field_t: solve_case(magnetic_field_t)
        for magnetic_field_t in paper_magnetic_fields_t
    }
    print_summary(spectra)
    plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
