"""Reproduce Liu et al., J. Phys. D 49 (2016) 445104, Fig. 1(b).

The model is the paper's all-gold paired-patch array on a 100-nm Au
backplane.  Johnson-Christy optical constants are read from the repository's
materials directory.  The paper used FDTD; this script computes the same
normal-incidence spectrum with the local RCWA backend.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Liu et al., Fig. 1(a) and Section 2. All dimensions are in micrometres.
period_um = 1.200
patch_length_um = 0.600
large_patch_width_um = 0.300
small_patch_width_um = 0.200
gap_um = 0.050
patch_height_um = 0.050
backplane_height_um = 0.100

gold_data_path = project_root / "materials" / "Au_Johnson-Christy.txt"

# At normal incidence and phi=0, TM is x-polarized. Figure 1(a) places the
# electric field along the common 600-nm patch length (the solver x axis).
polarization = "TM"
theta_deg = 0.0
phi_deg = 0.0

# Metallic 2D patterns use the RETICOLO ordered-Li factorization on the common
# rectilinear cell geometry.
orders = (6, 6)
workers = 20

wavelength_min_um = 1.000
wavelength_max_um = 1.150
wavelength_point_count = 601  # 0.25-nm spacing resolves the 1.8-nm linewidth.
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_resonances_um = (1.064, 1.070)
paper_reflectance = (0.11, 0.21)

data_path = project_root / "data" / "liu_2016_all_metal_metasurface_fig1_tm.csv"
plot_path = project_root / "images" / "liu_2016_all_metal_metasurface_fig1.png"
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def build_simulation() -> object:
    simulation = asyrcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialNK("Au", gold_data_path)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    simulation.AddLayer("paired_patches", patch_height_um, "Air")
    large_center_y_um = -0.5 * (gap_um + large_patch_width_um)
    small_center_y_um = +0.5 * (gap_um + small_patch_width_um)
    simulation.SetRegionRectangle(
        "paired_patches",
        "Au",
        Center=(0.0, large_center_y_um),
        Angle=0.0,
        Halfwidths=(0.5 * patch_length_um, 0.5 * large_patch_width_um),
    )
    simulation.SetRegionRectangle(
        "paired_patches",
        "Au",
        Center=(0.0, small_center_y_um),
        Angle=0.0,
        Halfwidths=(0.5 * patch_length_um, 0.5 * small_patch_width_um),
    )
    simulation.AddLayer("Au_backplane", backplane_height_um, "Au")
    simulation.SetExcitationPlanewave((theta_deg, phi_deg))
    return simulation


def solve_spectrum() -> np.ndarray:
    rows = build_simulation().GetSpectrum(
        wavelengths_um,
        (polarization,),
        Workers=workers,
    )
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete spectrum returned by the RCWA backend")
    rows = np.sort(rows, order="lambda_um")
    if np.max(np.abs(rows["R"] + rows["T"] + rows["A"] - 1.0)) > 1.0e-8:
        raise RuntimeError("RCWA spectrum failed the energy-balance check")
    return rows


def local_reflection_minima(rows: np.ndarray) -> np.ndarray:
    reflection = np.asarray(rows["R"], dtype=float)
    indices = np.flatnonzero(
        (reflection[1:-1] < reflection[:-2])
        & (reflection[1:-1] <= reflection[2:])
    ) + 1
    return indices[np.argsort(reflection[indices])]


def print_comparison(rows: np.ndarray) -> None:
    print(
        f"harmonics={int(rows['harmonics'][0])}, RETICOLO li=1, "
        f"max T={float(np.max(rows['T'])):.3e}"
    )
    for wavelength_um, paper_r in zip(paper_resonances_um, paper_reflectance):
        index = int(np.argmin(abs(rows["lambda_um"] - wavelength_um)))
        print(
            f"paper checkpoint {wavelength_um:.3f} um: "
            f"computed R/A={rows['R'][index]:.4f}/{rows['A'][index]:.4f}; "
            f"paper R/A={paper_r:.2f}/{1.0 - paper_r:.2f}"
        )
    minima = local_reflection_minima(rows)
    if len(minima) == 0:
        print("No interior reflection minimum was found in the plotted band.")
    else:
        for index in minima[:4]:
            print(
                f"computed minimum: lambda={rows['lambda_um'][index]:.6f} um, "
                f"R={rows['R'][index]:.6f}, A={rows['A'][index]:.6f}"
            )


def export_spectrum(rows: np.ndarray) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack(
        (
            rows["lambda_um"],
            rows["R"],
            rows["T"],
            rows["A"],
            rows["R"] + rows["T"] + rows["A"],
        )
    )
    np.savetxt(
        data_path,
        values,
        delimiter=",",
        header="wavelength_um,R,T,A,R_plus_T_plus_A",
        comments="",
    )
    print(f"Saved {data_path}.")


def draw_unit_cell(axis: object) -> None:
    axis.add_patch(
        Rectangle(
            (-0.5 * period_um, -0.5 * period_um),
            period_um,
            period_um,
            facecolor="#f3f3f3",
            edgecolor="0.35",
            linewidth=0.8,
        )
    )
    for center_y, width in (
        (-0.5 * (gap_um + large_patch_width_um), large_patch_width_um),
        (+0.5 * (gap_um + small_patch_width_um), small_patch_width_um),
    ):
        axis.add_patch(
            Rectangle(
                (-0.5 * patch_length_um, center_y - 0.5 * width),
                patch_length_um,
                width,
                facecolor="#d5a11e",
                edgecolor="#7d5b00",
                linewidth=0.8,
            )
        )
    axis.annotate(
        "",
        xy=(0.42, -0.48),
        xytext=(-0.42, -0.48),
        arrowprops={"arrowstyle": "<->", "color": "#a32626", "lw": 1.2},
    )
    axis.text(0.0, -0.55, "E", color="#a32626", ha="center", va="top")
    axis.set_xlim(-0.62, 0.62)
    axis.set_ylim(-0.62, 0.62)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_spectrum(rows: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(rows["lambda_um"], rows["R"], "k--", linewidth=1.8, label="RCWA reflection")
    axis.plot(rows["lambda_um"], rows["A"], color="#d62728", linewidth=1.8, label="RCWA absorption")
    axis.scatter(
        paper_resonances_um,
        paper_reflectance,
        marker="o",
        facecolors="none",
        edgecolors="black",
        zorder=4,
        label="paper reflection checkpoints",
    )
    axis.scatter(
        paper_resonances_um,
        1.0 - np.asarray(paper_reflectance),
        marker="o",
        facecolors="none",
        edgecolors="#d62728",
        zorder=4,
        label="paper absorption checkpoints",
    )
    for wavelength_um in paper_resonances_um:
        axis.axvline(wavelength_um, color="0.55", linestyle=":", linewidth=0.8)
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel("Power fraction")
    axis.set_title("Liu et al. (2016), Fig. 1(b): all-metal paired-patch array")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.65)
    axis.legend(loc="center right", frameon=False, fontsize=8)

    inset = axis.inset_axes((0.05, 0.48, 0.30, 0.42))
    draw_unit_cell(inset)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"P={period_um:g} um, l={patch_length_um:g} um, "
        f"W/w/d={large_patch_width_um:g}/{small_patch_width_um:g}/{gap_um:g} um, "
        f"Au patch/backplane={patch_height_um:g}/{backplane_height_um:g} um"
    )
    with asyrcwa.timed_step("solve Liu et al. Fig. 1(b) TM spectrum"):
        rows = solve_spectrum()
        print_comparison(rows)
        export_spectrum(rows)
    with asyrcwa.timed_step("plot Liu et al. Fig. 1(b)"):
        plot_spectrum(rows)
    return rows


if __name__ == "__main__":
    main()
