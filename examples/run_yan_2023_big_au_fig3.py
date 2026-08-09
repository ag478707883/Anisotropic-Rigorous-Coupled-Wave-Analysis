"""Reproduce Yan et al., Optics & Laser Technology 161 (2023) 109193, Fig. 3.

The four panels compare the Faraday rotation and zero-order transmittance of a
bare BIG/GGG film, a closed Au ring, a single-opening ring, and a double-opening
ring. The empirical BIG tensor is provided by ``asyrcwa.material_response`` and
all spectra are solved through the local RCWA Jones-amplitude interface.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.material_response import bismuth_iron_garnet_epsilon_tensor


# Yan et al., Figs. 1 and 3 and the paragraph below Fig. 3.
period_um = 0.600  # inferred exactly from the reported duty cycle s=0.6946
ring_outer_radius_um = 0.300
ring_width_um = 0.075
ring_inner_radius_um = ring_outer_radius_um - ring_width_um
opening_angle_deg = 20.0
single_opening_center_deg = -90.0
double_opening_centers_deg = (-45.0, 135.0)
annular_sector_arc_points = 72

ggg_thickness_um = 0.100
big_thickness_um = 0.200
au_thickness_um = 0.020
ggg_refractive_index = 2.0

# BIG Eqs. (8)-(10). The visible-range exponential gives g << machine scale
# in the 4-15 um range, which is incompatible with both Fig. 3 and the paper's
# cited g=0.01 long-wave checkpoint. A floor of 0.01 is therefore retained.
big_refractive_index_a = 2.36
big_refractive_index_b_nm = 413.0
big_extinction_a_nm = 1660.0
big_extinction_b = 15.2
big_gyration_prefactor = 100.0
big_gyration_decay_per_nm = 0.011
big_gyration_floor = 0.01
big_magnetization_sign = 1.0

# Mid-infrared Drude approximation to the Johnson-Christy Au data used by the
# paper. The article does not state the fitted database coefficients.
au_parameters = {
    "eps_inf": 9.54,
    "plasma_frequency_rad_s": 1.37e16,
    "damping_rate_rad_s": 1.075e14,
}

theta_deg = 0.0
phi_deg = 0.0
input_polarization = "TM"  # solver TM is x-polarized at normal incidence
orders = (3, 3)
workers = 10

wavelength_min_um = 4.0
wavelength_max_um = 15.0
wavelength_point_count = 221
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

structure_names = ("bare", "closed", "single", "double")
paper_rotation_max_deg = {
    "bare": -0.0109,
    "closed": 0.01445,
    "single": 27.64,
    "double": 45.6,
}
paper_transmission_max = {
    "bare": 0.94,
    "closed": 0.76,
    "single": 0.665,
    "double": 0.615,
}
paper_transmission_average = {
    "bare": 0.76,
    "closed": 0.408,
    "single": 0.54,
    "double": 0.52,
}

plot_path = project_root / "images" / "yan_2023_big_au_fig3.png"
figure_size = (12.0, 8.2)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def annular_sector_vertices(
    start_deg: float,
    end_deg: float,
) -> tuple[tuple[float, float], ...]:
    outer_angles = np.deg2rad(
        np.linspace(start_deg, end_deg, annular_sector_arc_points)
    )
    inner_angles = np.deg2rad(
        np.linspace(end_deg, start_deg, annular_sector_arc_points)
    )
    outer = np.column_stack(
        (ring_outer_radius_um * np.cos(outer_angles),
         ring_outer_radius_um * np.sin(outer_angles))
    )
    inner = np.column_stack(
        (ring_inner_radius_um * np.cos(inner_angles),
         ring_inner_radius_um * np.sin(inner_angles))
    )
    return tuple(map(tuple, np.vstack((outer, inner))))


def remaining_ring_sectors(structure: str) -> tuple[tuple[float, float], ...]:
    half_opening = 0.5 * opening_angle_deg
    if structure == "single":
        return ((
            single_opening_center_deg + half_opening,
            single_opening_center_deg + 360.0 - half_opening,
        ),)
    if structure == "double":
        first, second = sorted(angle % 360.0 for angle in double_opening_centers_deg)
        return (
            (first + half_opening, second - half_opening),
            (second + half_opening, first + 360.0 - half_opening),
        )
    return ()


def set_au_pattern(simulation: object, structure: str) -> None:
    if structure == "closed":
        simulation.SetRegionRing(
            "Au_pattern",
            "Au",
            "Air",
            Center=(0.0, 0.0),
            OuterRadius=ring_outer_radius_um,
            InnerRadius=ring_inner_radius_um,
        )
        return
    for index, (start_deg, end_deg) in enumerate(remaining_ring_sectors(structure)):
        simulation.SetRegionPolygon(
            "Au_pattern",
            "Au",
            Center=(0.0, 0.0),
            Angle=0.0,
            Vertices=annular_sector_vertices(start_deg, end_deg),
        )


def build_simulation(
    wavelength_um: float,
    structure: str,
) -> object:
    if structure == "bare":
        simulation = asyrcwa.New()
    else:
        simulation = asyrcwa.New(
            Lattice=(period_um, period_um),
            Orders=orders,
        )

    big_tensor = bismuth_iron_garnet_epsilon_tensor(
        wavelength_um,
        refractive_index_a=big_refractive_index_a,
        refractive_index_b_nm=big_refractive_index_b_nm,
        extinction_a_nm=big_extinction_a_nm,
        extinction_b=big_extinction_b,
        gyration_prefactor=big_gyration_prefactor,
        gyration_decay_per_nm=big_gyration_decay_per_nm,
        gyration_floor=big_gyration_floor,
        magnetization_sign=big_magnetization_sign,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("GGG", ggg_refractive_index**2 + 0.0j)
    simulation.SetMaterial("BIG", big_tensor)
    simulation.SetMaterialModel("Au", "Drude", au_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    if structure != "bare":
        simulation.AddLayer("Au_pattern", au_thickness_um, "Air")
        set_au_pattern(simulation, structure)
    simulation.AddLayer("BIG", big_thickness_um, "BIG")
    simulation.AddLayer("GGG", ggg_thickness_um, "GGG")
    return simulation


def complex_amplitude(
    rows: np.ndarray,
    out_polarization: str,
    in_polarization: str,
) -> complex:
    selected = rows[
        (rows["block"] == "T")
        & (rows["out_pol"] == out_polarization)
        & (rows["in_pol"] == in_polarization)
    ]
    if len(selected) != 1:
        raise RuntimeError("zero-order Jones row is missing or duplicated")
    return complex(float(selected[0]["value_re"]), float(selected[0]["value_im"]))


def solve_one_wavelength(
    structure: str,
    wavelength_um: float,
) -> tuple[float, float]:
    rows = build_simulation(wavelength_um, structure).GetZeroOrderAmplitudes(
        (wavelength_um,),
        Theta=theta_deg,
        Phi=phi_deg,
        Workers=1,
    )
    ex = complex_amplitude(rows, "TM", input_polarization)
    ey = complex_amplitude(rows, "TE", input_polarization)
    faraday_angle_deg = float(np.degrees(np.real(np.arctan(-ey / ex))))
    transmittance = float(abs(ex) ** 2 + abs(ey) ** 2)
    return faraday_angle_deg, transmittance


def solve_structure(structure: str) -> dict[str, np.ndarray]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(
            executor.map(
                lambda wavelength: solve_one_wavelength(structure, float(wavelength)),
                wavelengths_um,
            )
        )
    return {
        "theta_f_deg": np.asarray([value[0] for value in values], dtype=float),
        "T": np.asarray([value[1] for value in values], dtype=float),
    }


def draw_structure_inset(axis: object, structure: str) -> None:
    inset = inset_axes(axis, width="31%", height="31%", loc="lower right", borderpad=1.0)
    inset.add_patch(Rectangle((-0.38, -0.38), 0.76, 0.76, color="#8e1b0e"))
    inset.add_patch(Rectangle((-0.38, -0.48), 0.76, 0.10, color="#c6ad62"))
    if structure == "closed":
        inset.add_patch(Wedge((0.0, 0.0), 0.29, 0.0, 360.0, width=0.075, color="#d5a400"))
    elif structure in ("single", "double"):
        sectors = remaining_ring_sectors(structure)
        for start_deg, end_deg in sectors:
            inset.add_patch(
                Wedge(
                    (0.0, 0.0),
                    0.29,
                    start_deg,
                    end_deg,
                    width=0.075,
                    color="#d5a400",
                )
            )
    inset.set_xlim(-0.45, 0.45)
    inset.set_ylim(-0.52, 0.42)
    inset.set_aspect("equal")
    inset.axis("off")


def print_summary(spectra: dict[str, dict[str, np.ndarray]]) -> None:
    for structure in structure_names:
        theta_f = spectra[structure]["theta_f_deg"]
        transmission = spectra[structure]["T"]
        rotation_index = int(np.argmax(np.abs(theta_f)))
        print(
            f"{structure}: max|thetaF|={theta_f[rotation_index]:.4f} deg at "
            f"{wavelengths_um[rotation_index]:.3f} um "
            f"(paper max {paper_rotation_max_deg[structure]:.5g} deg); "
            f"T max/avg={float(np.max(transmission)):.4f}/"
            f"{float(np.mean(transmission)):.4f} "
            f"(paper {paper_transmission_max[structure]:.3f}/"
            f"{paper_transmission_average[structure]:.3f})"
        )


def plot_figure(spectra: dict[str, dict[str, np.ndarray]]) -> None:
    titles = {
        "bare": "(a) Bare BIG/GGG film",
        "closed": "(b) Closed Au ring",
        "single": "(c) Single-opening Au ring",
        "double": "(d) Double-opening Au ring",
    }
    figure, axes = plt.subplots(2, 2, figsize=figure_size, constrained_layout=True)
    for axis, structure in zip(axes.flat, structure_names):
        theta_f = spectra[structure]["theta_f_deg"]
        transmission = spectra[structure]["T"]
        transmission_axis = axis.twinx()
        axis.plot(
            wavelengths_um,
            theta_f,
            color="red",
            linewidth=2.0,
            linestyle=(0, (7, 3, 1, 3)),
            label=r"$\theta_F$",
        )
        transmission_axis.plot(
            wavelengths_um,
            transmission,
            color="blue",
            linewidth=2.0,
            linestyle=(0, (2, 2)),
            label="T",
        )
        axis.set_xlim(wavelength_min_um, wavelength_max_um)
        transmission_axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        axis.set_ylabel(r"$\theta_F$ (deg.)", color="red")
        transmission_axis.set_ylabel("T", color="blue")
        axis.tick_params(axis="y", colors="red")
        transmission_axis.tick_params(axis="y", colors="blue")
        axis.set_title(titles[structure])
        axis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.55)
        lines = axis.get_lines() + transmission_axis.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
        if structure != "bare":
            draw_structure_inset(axis, structure)

    figure.suptitle("Yan et al. (2023), Fig. 3: Faraday rotation and transmittance")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, dict[str, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, R={ring_outer_radius_um:g} um, "
        f"r={ring_inner_radius_um:g} um, opening={opening_angle_deg:g} deg, "
        f"orders={orders}, BIG gyration floor={big_gyration_floor:g}"
    )
    spectra: dict[str, dict[str, np.ndarray]] = {}
    for structure in structure_names:
        with asyrcwa.timed_step(f"solve Fig. 3 {structure}"):
            spectra[structure] = solve_structure(structure)
    print_summary(spectra)
    plot_figure(spectra)
    return spectra


if __name__ == "__main__":
    main()
