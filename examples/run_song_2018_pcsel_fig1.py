"""Reproduce Song et al., Applied Physics Letters 113, 041106 (2018), Fig. 1.

The passive structure is a square-lattice dielectric slab suspended in air.
Panel (a) uses the local RCWA reflectance API along Gamma-X, panel (c) uses a
denser normal-incidence spectrum, and panel (b) evaluates the complete RCWA
S matrix on the complex-frequency plane at Gamma.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Song et al., paragraph preceding Fig. 1. The lattice constant a is the unit
# length, so normalized frequency f is reported in c/a.
lattice_constant_um = 1.0
slab_epsilon = 12.0
air_epsilon = 1.0
slab_thickness_um = 0.5 * lattice_constant_um
hole_radius_um = 0.2 * lattice_constant_um

polarization = "TE"
phi_deg = 0.0
# For the stated symmetric, lossless slab, the published Fig. 1(a,c) curves
# coincide with the backend's transmitted power T=1-R. The paper labels this
# plotted channel "reflection"; retain that label while making the port-
# convention mapping explicit here.
paper_plotted_power_field = "T"
reflection_orders = (3, 3)
pole_orders = (2, 2)
workers = 10

# Fig. 1(a): Gamma-X reflectance map. A real incidence angle samples
# kx/(2*pi/a) = f*sin(theta), automatically staying inside the light cone.
map_frequency_min = 0.25
map_frequency_max = 0.60
map_frequency_point_count = 181
map_frequencies = np.linspace(
    map_frequency_min,
    map_frequency_max,
    map_frequency_point_count,
)
map_angle_min_deg = 0.0
map_angle_max_deg = 89.0
map_angle_point_count = 91
map_angles_deg = np.linspace(
    map_angle_min_deg,
    map_angle_max_deg,
    map_angle_point_count,
)
kx_gamma_x_max = 0.5

# Fig. 1(b): paper coordinates use exp(+i*omega*t), whereas the backend uses
# exp(-i*omega*t). A paper frequency fr+i*fi is therefore evaluated with the
# backend scaling fr-i*fi. Every epsilon and mu, including exterior air, is
# multiplied by this scale at the real reference frequency c/a.
pole_real_frequency_min = 0.25
pole_real_frequency_max = 0.60
pole_real_frequency_point_count = 101
pole_real_frequencies = np.linspace(
    pole_real_frequency_min,
    pole_real_frequency_max,
    pole_real_frequency_point_count,
)
pole_imag_frequency_min = 0.0
pole_imag_frequency_max = 0.10
pole_imag_frequency_point_count = 61
pole_imag_frequencies = np.linspace(
    pole_imag_frequency_min,
    pole_imag_frequency_max,
    pole_imag_frequency_point_count,
)
pole_reference_wavelength_um = lattice_constant_um
pole_local_background_window = 9
pole_contrast_percentile = 99.5

# Fig. 1(c): normal-incidence spectrum needs a denser frequency grid to retain
# the narrow Fano resonances.
normal_frequency_min = 0.25
normal_frequency_max = 0.60
normal_frequency_point_count = 1401
normal_frequencies = np.linspace(
    normal_frequency_min,
    normal_frequency_max,
    normal_frequency_point_count,
)

paper_mode_a_frequency = 0.38
paper_low_fabry_perot_frequency = 0.31
paper_high_fabry_perot_frequency = 0.59

plot_path = project_root / "images" / "song_2018_pcsel_fig1.png"
figure_size = (8.0, 10.5)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def build_simulation(
    orders: tuple[int, int],
    frequency_scale: complex = 1.0 + 0.0j,
) -> object:
    simulation = asyrcwa.New(
        Lattice=(lattice_constant_um, lattice_constant_um),
        Orders=orders,
    )
    simulation.SetMaterial(
        "Air",
        air_epsilon * frequency_scale,
        Mu=frequency_scale,
    )
    simulation.SetMaterial(
        "Slab",
        slab_epsilon * frequency_scale,
        Mu=frequency_scale,
    )
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    simulation.AddLayer("photonic_crystal_slab", slab_thickness_um, "Slab")
    simulation.SetRegionCircle(
        "photonic_crystal_slab",
        "Air",
        Center=(0.0, 0.0),
        Radius=hole_radius_um,
    )
    return simulation


def solve_reflection_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = build_simulation(reflection_orders).GetSpectrumForAngles(
        lattice_constant_um / map_frequencies,
        map_angles_deg,
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    frequency = lattice_constant_um / rows["lambda_um"]
    kx = frequency * np.sin(np.deg2rad(rows["theta_deg"]))
    mask = kx <= kx_gamma_x_max + 1.0e-12
    return kx[mask], frequency[mask], np.asarray(
        rows[paper_plotted_power_field][mask],
        dtype=float,
    )


def solve_normal_reflection() -> tuple[np.ndarray, np.ndarray]:
    rows = build_simulation(reflection_orders).GetSpectrumForAngles(
        lattice_constant_um / normal_frequencies,
        (0.0,),
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    frequency = lattice_constant_um / rows["lambda_um"]
    order = np.argsort(frequency)
    return frequency[order], np.asarray(
        rows[paper_plotted_power_field][order],
        dtype=float,
    )


def full_smatrix(rows: np.ndarray) -> np.ndarray:
    harmonic_count = int(rows[0]["harmonics"])
    channel_count = 2 * harmonic_count
    block_size = channel_count * channel_count
    values = np.asarray(rows["value_re"], dtype=float) + 1j * np.asarray(
        rows["value_im"],
        dtype=float,
    )
    if len(values) != 4 * block_size:
        raise RuntimeError("S-parameter rows do not match the retained channel count")
    blocks = [
        values[index * block_size : (index + 1) * block_size].reshape(
            channel_count,
            channel_count,
        )
        for index in range(4)
    ]
    return np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])


def moving_average_2d(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("pole_local_background_window must be a positive odd integer")
    radius = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    smooth = np.apply_along_axis(
        lambda row: np.convolve(
            np.pad(row, (radius, radius), mode="edge"),
            kernel,
            mode="valid",
        ),
        1,
        values,
    )
    return np.apply_along_axis(
        lambda column: np.convolve(
            np.pad(column, (radius, radius), mode="edge"),
            kernel,
            mode="valid",
        ),
        0,
        smooth,
    )


def solve_pole_map() -> tuple[np.ndarray, np.ndarray]:
    log_abs_determinant = np.empty(
        (len(pole_imag_frequencies), len(pole_real_frequencies)),
        dtype=float,
    )
    for imag_index, imag_frequency in enumerate(pole_imag_frequencies):
        for real_index, real_frequency in enumerate(pole_real_frequencies):
            solver_frequency = complex(real_frequency, -imag_frequency)
            rows = build_simulation(
                pole_orders,
                solver_frequency,
            ).GetSParametersAnalyticContinuationAtGamma(
                pole_reference_wavelength_um
            )
            _, log_abs_determinant[imag_index, real_index] = np.linalg.slogdet(
                full_smatrix(rows)
            )

    local_background = moving_average_2d(
        log_abs_determinant,
        pole_local_background_window,
    )
    pole_contrast = np.maximum(local_background - log_abs_determinant, 0.0)
    normalization = max(
        float(np.percentile(pole_contrast, pole_contrast_percentile)),
        1.0e-12,
    )
    return log_abs_determinant, np.clip(pole_contrast / normalization, 0.0, 1.0)


def draw_structure_inset(axis: object) -> None:
    inset = inset_axes(axis, width="27%", height="27%", loc="lower right", borderpad=0.9)
    slab = Polygon(
        ((0.06, 0.30), (0.76, 0.12), (0.96, 0.38), (0.27, 0.57)),
        closed=True,
        facecolor="#9ecae1",
        edgecolor="none",
    )
    inset.add_patch(slab)
    for x, y in ((0.24, 0.39), (0.47, 0.33), (0.70, 0.27), (0.38, 0.47), (0.61, 0.41)):
        inset.add_patch(Circle((x, y), 0.055, facecolor="#f4f4f4", edgecolor="none"))
    inset.annotate("a", xy=(0.25, 0.18), xytext=(0.58, 0.08), arrowprops={"arrowstyle": "<->"})
    inset.annotate("r", xy=(0.47, 0.33), xytext=(0.56, 0.52), arrowprops={"arrowstyle": "->"})
    inset.annotate("d", xy=(0.08, 0.29), xytext=(0.01, 0.08), arrowprops={"arrowstyle": "<->"})
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 0.65)
    inset.axis("off")


def draw_brillouin_zone_inset(axis: object) -> None:
    inset = inset_axes(axis, width="17%", height="17%", loc="center right", borderpad=1.0)
    inset.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=False, edgecolor="black"))
    inset.plot((0.0, 1.0), (0.0, 0.0), color="#d32f2f", linewidth=2.0)
    inset.text(-0.10, -0.16, r"$\Gamma$", fontsize=8)
    inset.text(1.02, -0.14, "X", fontsize=8)
    inset.text(1.02, 0.90, "M", fontsize=8)
    inset.set_xlim(-1.15, 1.25)
    inset.set_ylim(-1.15, 1.15)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")


def print_summary(
    normal_frequency: np.ndarray,
    normal_reflection: np.ndarray,
    pole_contrast: np.ndarray,
) -> None:
    mode_index = int(np.argmin(np.abs(normal_frequency - paper_mode_a_frequency)))
    print(
        f"normal-incidence mode A marker: f={normal_frequency[mode_index]:.5f} c/a, "
        f"published-channel coefficient={normal_reflection[mode_index]:.5f}; "
        f"paper about {paper_mode_a_frequency:.2f} c/a"
    )
    for label, real_window, imag_window in (
        ("low Fabry-Perot pole", (0.29, 0.35), (0.04, 0.08)),
        ("mode-A/guided pole region", (0.36, 0.42), (0.00, 0.02)),
        ("high Fabry-Perot pole", (0.56, 0.60), (0.03, 0.07)),
    ):
        real_mask = (pole_real_frequencies >= real_window[0]) & (
            pole_real_frequencies <= real_window[1]
        )
        imag_mask = (pole_imag_frequencies >= imag_window[0]) & (
            pole_imag_frequencies <= imag_window[1]
        )
        submap = pole_contrast[np.ix_(imag_mask, real_mask)]
        imag_local, real_local = np.unravel_index(int(np.argmax(submap)), submap.shape)
        real_value = pole_real_frequencies[np.flatnonzero(real_mask)[real_local]]
        imag_value = pole_imag_frequencies[np.flatnonzero(imag_mask)[imag_local]]
        print(f"{label}: approximately {real_value:.4f}+{imag_value:.4f}i c/a")


def plot_figure(
    map_kx: np.ndarray,
    map_frequency: np.ndarray,
    map_reflection: np.ndarray,
    pole_contrast: np.ndarray,
    normal_frequency: np.ndarray,
    normal_reflection: np.ndarray,
) -> None:
    figure = plt.figure(figsize=figure_size, constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(3.2, 1.45, 1.35))
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[2, 0])

    image_a = axis_a.tripcolor(
        map_kx,
        map_frequency,
        map_reflection,
        shading="gouraud",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(image_a, ax=axis_a, label="Reflection")
    axis_a.set_xlim(0.0, kx_gamma_x_max)
    axis_a.set_ylim(map_frequency_min, map_frequency_max)
    axis_a.set_xlabel(r"$k_x$ ($2\pi/a$)")
    axis_a.set_ylabel(r"Frequency ($c/a$)")
    axis_a.set_title("(a) Reflection spectra along Gamma-X")
    axis_a.text(0.05, 0.385, "A", color="white", fontsize=12)
    axis_a.text(0.02, 0.305, "Fabry-Perot", color="#b96d00", fontsize=10)
    draw_structure_inset(axis_a)
    draw_brillouin_zone_inset(axis_a)

    image_b = axis_b.imshow(
        pole_contrast,
        extent=(
            pole_real_frequency_min,
            pole_real_frequency_max,
            pole_imag_frequency_min,
            pole_imag_frequency_max,
        ),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    figure.colorbar(image_b, ax=axis_b, label="Normalized pole contrast")
    axis_b.set_xlabel(r"$\Re(f)$ ($c/a$)")
    axis_b.set_ylabel(r"$\Im(f)$ ($c/a$)")
    axis_b.set_title(r"(b) Complete $S_\Gamma$ determinant on the complex-frequency plane")
    axis_b.annotate(
        "A",
        xy=(paper_mode_a_frequency, 0.003),
        xytext=(paper_mode_a_frequency, 0.032),
        color="white",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "white"},
    )

    axis_c.plot(normal_frequency, normal_reflection, color="#1f77b4", linewidth=1.6)
    axis_c.axvline(paper_mode_a_frequency, color="#777777", linewidth=0.8, alpha=0.35)
    axis_c.set_xlim(normal_frequency_min, normal_frequency_max)
    axis_c.set_ylim(0.0, 1.02)
    axis_c.set_xlabel(r"$\Re(f)$ ($c/a$)")
    axis_c.set_ylabel("Reflection")
    axis_c.set_title("(c) Reflection spectrum at Gamma (normal incidence)")
    axis_c.grid(True, color="#dddddd", linewidth=0.7, alpha=0.65)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"square lattice: eps={slab_epsilon:g}, d/a={slab_thickness_um / lattice_constant_um:g}, "
        f"r/a={hole_radius_um / lattice_constant_um:g}, reflection orders={reflection_orders}, "
        f"pole orders={pole_orders}"
    )
    with asyrcwa.timed_step("solve Fig. 1(a) Gamma-X reflection map"):
        map_kx, map_frequency, map_reflection = solve_reflection_map()
    with asyrcwa.timed_step("solve Fig. 1(c) normal-incidence reflection"):
        normal_frequency, normal_reflection = solve_normal_reflection()
    with asyrcwa.timed_step("solve Fig. 1(b) complex-frequency S determinant"):
        _, pole_contrast = solve_pole_map()
    print_summary(normal_frequency, normal_reflection, pole_contrast)
    plot_figure(
        map_kx,
        map_frequency,
        map_reflection,
        pole_contrast,
        normal_frequency,
        normal_reflection,
    )
    return map_reflection, pole_contrast, normal_reflection


if __name__ == "__main__":
    main()
