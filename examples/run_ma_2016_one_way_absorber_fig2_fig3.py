"""Reproduce Ma et al. (2016), Chinese Journal of Lasers 43, 0117001.

The paper's structure is a laterally uniform magneto-optical multilayer,
``(AB)^11 D C D (AB)^11``.  This example therefore uses the dedicated
``asyrcwa.New()`` interface rather than introducing an artificial
periodic lattice and retaining zero-order Fourier harmonics.

Figures 2 and 3 are reproduced with the paper's TM, +/-45 degree incidence
definition.  The paper does not print the Drude plasma frequency in the
parameter paragraph.  The value below is the value inferred from the two
lossless resonance locations and the reported peak transmissions; it is kept
explicit so that the assumption is easy to replace when a source value is
available.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Ma et al., Sec. 3 and Figs. 1-3. Lengths are in micrometres here.
period_count = 11
epsilon_a = 4.7
epsilon_b = 2.25
epsilon_d = 6.25
delta_d = 2.5
thickness_a_um = 0.052
thickness_b_um = 0.075
thickness_d_um = 0.025

# Eq. (1) uses eps_C = 1 - wp^2/(omega^2 + i omega gamma). The PDF does not
# state wp; 1.20e16 rad/s reproduces 410.39/419.22 nm and T=0.976/0.983.
metal_eps_inf = 1.0
metal_plasma_frequency_rad_s = 1.20e16  # inferred because the paper omits it
metal_lossless_damping_rad_s = 0.0
metal_lossy_damping_rad_s = 1.45e13  # paper Fig. 2(c,d)

theta_deg = 45.0
phi_deg = 0.0
polarization = "TM"
angles_deg = (+theta_deg, -theta_deg)
workers = 8

fig2_wavelength_min_um = 0.405
fig2_wavelength_max_um = 0.425
fig2_wavelength_point_count = 4001
fig2_wavelengths_um = np.linspace(
    fig2_wavelength_min_um,
    fig2_wavelength_max_um,
    fig2_wavelength_point_count,
)
fig2_metal_thickness_um = 0.010

fig3_wavelength_min_um = 0.408
fig3_wavelength_max_um = 0.412
fig3_wavelength_point_count = 2001
fig3_wavelengths_um = np.linspace(
    fig3_wavelength_min_um,
    fig3_wavelength_max_um,
    fig3_wavelength_point_count,
)
fig3_metal_thicknesses_nm = (10, 15, 20, 25, 30, 40, 50, 134)

paper_lossless_resonances_um = (0.41039, 0.41922)
paper_lossless_transmissions = (0.976, 0.983)
paper_lossy_values = {
    "+45 A": 0.3135,
    "+45 R": 0.126,
    "+45 T": 0.5605,
    "-45 A": 0.2642,
    "-45 R": 0.0828,
    "-45 T": 0.653,
}
paper_fig3_peak_absorptivity = 0.9761
paper_fig3_peak_reflectivity = 0.0239
paper_fig3_peak_wavelength_um = 0.409725
field_probe_wavelength_um = 0.409725
field_probe_metal_thickness_um = 0.134
field_z_min_um = -0.20
field_z_max_um = (
    2.0 * period_count * (thickness_a_um + thickness_b_um)
    + 2.0 * thickness_d_um
    + field_probe_metal_thickness_um
    + 0.20
)
field_z_point_count = 1601

combined_plot_path = project_root / "images" / "ma_2016_one_way_absorber_combined.png"
combined_figure_size = (14.0, 16.0)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(gamma_rad_s: float, metal_thickness_um: float) -> object:
    simulation = asyrcwa.New()

    magnetic_tensor = np.asarray(
        [
            [epsilon_d, 0.0, -1j * delta_d],
            [0.0, epsilon_d, 0.0],
            [1j * delta_d, 0.0, epsilon_d],
        ],
        dtype=complex,
    )

    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("A", epsilon_a + 0.0j)
    simulation.SetMaterial("B", epsilon_b + 0.0j)
    simulation.SetMaterial("D", magnetic_tensor)
    simulation.SetMaterialModel(
        "C",
        "Drude",
        {
            "eps_inf": metal_eps_inf,
            "plasma_frequency_rad_s": metal_plasma_frequency_rad_s,
            "damping_rate_rad_s": gamma_rad_s,
        },
    )
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    for index in range(period_count):
        simulation.AddLayer(f"A_left_{index + 1}", thickness_a_um, "A")
        simulation.AddLayer(f"B_left_{index + 1}", thickness_b_um, "B")
    simulation.AddLayer("D_left", thickness_d_um, "D")
    simulation.AddLayer("C", metal_thickness_um, "C")
    simulation.AddLayer("D_right", thickness_d_um, "D")
    for index in range(period_count):
        simulation.AddLayer(f"A_right_{index + 1}", thickness_a_um, "A")
        simulation.AddLayer(f"B_right_{index + 1}", thickness_b_um, "B")
    return simulation


def rows_at_angle(rows: np.ndarray, angle_deg: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], angle_deg, rtol=0.0, atol=1.0e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_spectrum(
    wavelengths_um: np.ndarray,
    gamma_rad_s: float,
    metal_thickness_um: float,
) -> dict[float, np.ndarray]:
    rows = build_simulation(gamma_rad_s, metal_thickness_um).GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    return {angle: rows_at_angle(rows, angle) for angle in angles_deg}


def peak_row(rows: np.ndarray, field: str) -> np.void:
    return rows[int(np.argmax(rows[field]))]


def print_fig2_comparison(
    lossless: dict[float, np.ndarray], lossy: dict[float, np.ndarray]
) -> None:
    print("Fig. 2 comparison (TM, theta=+/-45 deg):")
    for angle, paper_wavelength, paper_t in zip(
        angles_deg,
        paper_lossless_resonances_um,
        paper_lossless_transmissions,
    ):
        peak = peak_row(lossless[angle], "T")
        print(
            f"  gamma=0, theta={angle:+g}: lambda={float(peak['lambda_um']):.6f} um, "
            f"T={float(peak['T']):.6f}, R={float(peak['R']):.6f}, "
            f"paper lambda={paper_wavelength:.5f} um, T={paper_t:.3f}"
        )
    for angle in angles_deg:
        rows = lossy[angle]
        peak = peak_row(rows, "A")
        prefix = f"{angle:+g}"
        print(
            f"  gamma=1.45e13, theta={angle:+g}: lambda={float(peak['lambda_um']):.6f} um, "
            f"A={float(peak['A']):.6f}, R={float(peak['R']):.6f}, T={float(peak['T']):.6f}; "
            f"paper A/R/T={paper_lossy_values[prefix + ' A']:.4f}/"
            f"{paper_lossy_values[prefix + ' R']:.4f}/"
            f"{paper_lossy_values[prefix + ' T']:.4f}"
        )


def draw_fig2(
    axes: np.ndarray,
    lossless: dict[float, np.ndarray],
    lossy: dict[float, np.ndarray],
) -> None:
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    for row_index, (gamma_label, spectra) in enumerate(
        ((r"$\gamma=0$", lossless), (r"$\gamma=1.45\times10^{13}\,s^{-1}$", lossy))
    ):
        for column_index, angle in enumerate(angles_deg):
            axis = axes[row_index, column_index]
            rows = spectra[angle]
            axis.plot(rows["lambda_um"] * 1000.0, rows["A"], "k-", label="A")
            axis.plot(rows["lambda_um"] * 1000.0, rows["R"], "r--", label="R")
            axis.plot(rows["lambda_um"] * 1000.0, rows["T"], "b:", label="T")
            axis.set_xlim(fig2_wavelength_min_um * 1000.0, fig2_wavelength_max_um * 1000.0)
            axis.set_ylim(-0.02, 1.04)
            axis.set_xlabel(r"Wavelength $\lambda$ (nm)")
            axis.set_ylabel("A, R & T")
            axis.set_title(f"Fig. 2{panel_labels[row_index][column_index]}  {gamma_label}, theta={angle:+g} deg")
            axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, loc="center left", fontsize=8)


def draw_fig3(axes: np.ndarray, spectra: dict[int, np.ndarray]) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(fig3_metal_thicknesses_nm)))
    for thickness_nm, color in zip(fig3_metal_thicknesses_nm, colors):
        rows = spectra[thickness_nm]
        wavelength_nm = rows["lambda_um"] * 1000.0
        label = rf"$d_C={thickness_nm}\,$nm"
        axes[0].plot(wavelength_nm, rows["A"], color=color, label=label)
        axes[1].plot(wavelength_nm, rows["T"], color=color)
        axes[2].plot(wavelength_nm, rows["R"], color=color)

    for axis, ylabel, panel_label in zip(axes, ("A", "T", "R"), ("(a)", "(b)", "(c)")):
        axis.set_xlim(fig3_wavelength_min_um * 1000.0, fig3_wavelength_max_um * 1000.0)
        axis.set_ylim(-0.02, 1.04)
        axis.set_xlabel(r"Wavelength $\lambda$ (nm)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Fig. 3{panel_label}  {ylabel}, theta=+45 deg")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")


def solve_field_profile(angle_deg: float) -> np.ndarray:
    simulation = build_simulation(
        metal_lossy_damping_rad_s,
        field_probe_metal_thickness_um,
    )
    return simulation.GetFieldPlane(
        field_probe_wavelength_um,
        polarization,
        angle_deg,
        phi_deg,
        ["Hy"],
        "xz",
        0.0,
        0.0,
        1,
        field_z_min_um,
        field_z_max_um,
        field_z_point_count,
        0.0,
        Workers=workers,
    )


def draw_field_profiles(axis: object, profiles: dict[float, np.ndarray]) -> None:
    for angle, color in ((+theta_deg, "#c62828"), (-theta_deg, "#1565c0")):
        rows = np.sort(profiles[angle], order="z_um")
        axis.plot(
            rows["z_um"],
            rows["magnitude"],
            color=color,
            linewidth=1.7,
            label=rf"$\theta={angle:+g}^\circ$",
        )

    total_finite_thickness_um = field_z_max_um - 0.20
    central_d_left_um = period_count * (thickness_a_um + thickness_b_um)
    central_c_left_um = central_d_left_um + thickness_d_um
    central_c_right_um = central_c_left_um + field_probe_metal_thickness_um
    central_d_right_um = central_c_right_um + thickness_d_um
    for boundary, label in (
        (0.0, "stack front"),
        (central_d_left_um, "D/C"),
        (central_c_left_um, "C front"),
        (central_c_right_um, "C rear"),
        (central_d_right_um, "C/D"),
        (total_finite_thickness_um, "stack rear"),
    ):
        axis.axvline(boundary, color="#777777", linewidth=0.7, alpha=0.35)
        if boundary in (central_c_left_um, central_c_right_um):
            axis.text(
                boundary,
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="#555555",
            )

    axis.set_xlabel(r"Stack coordinate $z$ ($\mu$m)")
    axis.set_ylabel(r"$|H_y|$ (RCWA field units)")
    axis.set_title(
        rf"Fig. 4/5 field profile at $\lambda={field_probe_wavelength_um * 1000:.3f}\,$nm"
    )
    axis.set_xlim(field_z_min_um, field_z_max_um)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False)


def plot_combined_figure(
    lossless: dict[float, np.ndarray],
    lossy: dict[float, np.ndarray],
    fig3_spectra: dict[int, np.ndarray],
    field_profiles: dict[float, np.ndarray],
) -> None:
    layout = [
        ["fig2a", "fig2a", "fig2a", "fig2b", "fig2b", "fig2b"],
        ["fig2c", "fig2c", "fig2c", "fig2d", "fig2d", "fig2d"],
        ["fig3a", "fig3a", "fig3b", "fig3b", "fig3c", "fig3c"],
        ["field", "field", "field", "field", "field", "field"],
    ]
    figure, axes = plt.subplot_mosaic(
        layout,
        figsize=combined_figure_size,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0, 1.15)},
    )
    draw_fig2(
        np.asarray(
            [
                [axes["fig2a"], axes["fig2b"]],
                [axes["fig2c"], axes["fig2d"]],
            ]
        ),
        lossless,
        lossy,
    )
    draw_fig3(
        np.asarray([axes["fig3a"], axes["fig3b"], axes["fig3c"]]),
        fig3_spectra,
    )
    draw_field_profiles(axes["field"], field_profiles)
    figure.suptitle(
        "Ma et al. (2016): combined reproduction of Figs. 2-5",
        fontsize=18,
    )
    combined_plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(combined_plot_path, dpi=figure_dpi)
    print(f"Saved {combined_plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def print_fig3_comparison(
    spectra: dict[int, np.ndarray], reverse_134_nm: np.ndarray
) -> None:
    rows = spectra[134]
    peak = peak_row(rows, "A")
    print(
        "Fig. 3 at dC=134 nm: "
        f"lambda={float(peak['lambda_um']):.6f} um, "
        f"A={float(peak['A']):.6f}, R={float(peak['R']):.6f}, "
        f"T={float(peak['T']):.6e}; "
        f"paper lambda={paper_fig3_peak_wavelength_um:.6f} um, "
        f"A={paper_fig3_peak_absorptivity:.4f}, R={paper_fig3_peak_reflectivity:.4f}"
    )
    print(
        "  reverse theta=-45 deg over the Fig. 3 window: "
        f"max A={float(np.max(reverse_134_nm['A'])):.3e}, "
        f"max T={float(np.max(reverse_134_nm['T'])):.3e}, "
        f"min R={float(np.min(reverse_134_nm['R'])):.6f}"
    )


def main() -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[int, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"(AB)^{period_count}DCD(AB)^{period_count}, theta=+/-{theta_deg:g} deg, "
        f"wp={metal_plasma_frequency_rad_s:.3e} rad/s (inferred)"
    )
    with asyrcwa.timed_step("solve Fig. 2 lossless/lossy spectra"):
        lossless = solve_spectrum(
            fig2_wavelengths_um,
            metal_lossless_damping_rad_s,
            fig2_metal_thickness_um,
        )
        lossy = solve_spectrum(
            fig2_wavelengths_um,
            metal_lossy_damping_rad_s,
            fig2_metal_thickness_um,
        )
    print_fig2_comparison(lossless, lossy)

    with asyrcwa.timed_step("solve Fig. 3 metal-thickness sweep"):
        fig3_angle_spectra = {
            thickness_nm: solve_spectrum(
                fig3_wavelengths_um,
                metal_lossy_damping_rad_s,
                thickness_nm / 1000.0,
            )
            for thickness_nm in fig3_metal_thicknesses_nm
        }
        fig3_spectra = {
            thickness_nm: angle_spectra[+theta_deg]
            for thickness_nm, angle_spectra in fig3_angle_spectra.items()
        }
    print_fig3_comparison(fig3_spectra, fig3_angle_spectra[134][-theta_deg])
    with asyrcwa.timed_step("solve field profiles at the one-way wavelength"):
        field_profiles = {
            angle: solve_field_profile(angle)
            for angle in angles_deg
        }
    print(
        "Field-profile peak magnitudes: "
        + ", ".join(
            f"theta={angle:+g}: {float(np.max(rows['magnitude'])):.4f}"
            for angle, rows in field_profiles.items()
        )
    )
    plot_combined_figure(lossless, lossy, fig3_spectra, field_profiles)
    return lossless, lossy, fig3_spectra


if __name__ == "__main__":
    main()
