"""Reproduce the main InSb square-hole-cylinder spectra in Zhu et al. Fig. 3.

Article: "Nonreciprocal photothermal effect of InSb square-hole cylinders in
the mid-infrared region".
Authors: An Zhu and Han Wang.
Journal: Applied Physics A 132, 447 (2026).
DOI: 10.1007/s00339-026-09607-x.

The paper defines e(theta, lambda) = alpha(-theta, lambda), so the script
computes the two incidence directions together and does not request the more
expensive complete directional thermal-channel matrix.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Zhu and Wang, Table 1 and Figs. 1 and 3.
period_um = 2.0
substrate_height_um = 2.5
cylinder_height_um = 1.5
cylinder_diameter_um = 1.4
cylinder_radius_um = 0.5 * cylinder_diameter_um

# Table 1 gives d=0.3 um. Figure 3's in-panel annotation instead says 0.7 um,
# but that value is inconsistent with the plotted dual-band nonreciprocity:
# using 0.3 um reproduces the reported 0.282 first peak, whereas 0.7 um makes
# the second peak almost disappear. The table value is therefore used here.
square_hole_side_um = 0.3
square_hole_halfwidth_um = 0.5 * square_hole_side_um

# Paper coordinates: x/z are periodic, y is layer-normal, and B is along +z.
# Solver coordinates: x/y are periodic, z is layer-normal, and the built-in
# InSb model uses an xz Hall tensor for B along y. With the incidence-angle
# convention used below, solver +B_y reproduces the paper's enhanced
# absorptivity at +18 degrees. Reversing both B and theta is equivalent.
insb_parameters_b3 = {
    "magnetic_field_t": +3.0,
    "eps_inf": 15.68,
    "carrier_density_cm3": 5.8e17,
    "effective_mass_ratio": 0.014,
    "transverse_resonance_rad_s": 3.376e13,
    "damping_rate_rad_s": 2.017e12,
}
insb_parameters_b0 = dict(insb_parameters_b3, magnetic_field_t=0.0)

theta_deg = 18.0
phi_deg = 0.0
polarization = "TM"

# (5, 5) circular truncation retains 81 harmonics. A local convergence check
# gives eta_peak1=0.2778 here and 0.2848 at (7, 7), bracketing the paper's
# 0.282 while the (5, 5) spectrum is roughly an order of magnitude faster.
orders = (5, 5)
workers = 10
# Fourier convergence controls. "Default" preserves the original example result.

wavelength_min_um = 5.0
wavelength_max_um = 5.5
wavelength_point_count = 501
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_peak_wavelengths_um = (5.153, 5.430)
paper_peak_nonreciprocity = (0.282, 0.206)
paper_peak1_absorptivity = 0.954

plot_path = project_root / "images" / "zhu_2026_insb_square_hole_cylinders_fig3.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(insb_parameters: dict[str, float]) -> object:
    simulation = asyrcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterialModel("InSb", "InSbLorentzDrude", insb_parameters)

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    # This genuinely two-dimensional patterned layer uses ordered Li blocks.
    simulation.AddLayer("square_hole_cylinders", cylinder_height_um, "Air")
    simulation.SetRegionCircle(
        "square_hole_cylinders",
        "InSb",
        Center=(0.0, 0.0),
        Radius=cylinder_radius_um,
    )
    simulation.SetRegionRectangle(
        "square_hole_cylinders",
        "Air",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(square_hole_halfwidth_um, square_hole_halfwidth_um),
    )

    # The uniform InSb substrate remains on the compact analytic multilayer
    # path.
    simulation.AddLayer("InSb_substrate", substrate_height_um, "InSb")
    return simulation


def rows_for_angle(rows: np.ndarray, theta: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], theta, rtol=0.0, atol=1e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_fig3_spectra() -> dict[str, np.ndarray]:
    b3_rows = build_simulation(insb_parameters_b3).GetSpectrumForAngles(
        wavelengths_um,
        (+theta_deg, -theta_deg),
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    b0_rows = build_simulation(insb_parameters_b0).GetSpectrumForAngles(
        wavelengths_um,
        (+theta_deg,),
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )

    forward = rows_for_angle(b3_rows, +theta_deg)
    reverse = rows_for_angle(b3_rows, -theta_deg)
    reciprocal = rows_for_angle(b0_rows, +theta_deg)
    expected_count = len(wavelengths_um)
    if not all(len(rows) == expected_count for rows in (forward, reverse, reciprocal)):
        raise RuntimeError("incomplete wavelength spectrum returned by backend")
    if not (
        np.allclose(
            forward["lambda_um"], reverse["lambda_um"], rtol=0.0, atol=1e-12
        )
        and np.allclose(
            forward["lambda_um"], reciprocal["lambda_um"], rtol=0.0, atol=1e-12
        )
    ):
        raise RuntimeError("forward, reverse, and reciprocal wavelengths are misaligned")

    absorptivity = np.asarray(forward["A"], dtype=float)
    # Zhu and Wang Eq. (2): e(+theta, lambda) = alpha(-theta, lambda).
    emissivity = np.asarray(reverse["A"], dtype=float)
    return {
        "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
        "b0_absorptivity": np.asarray(reciprocal["A"], dtype=float),
        "absorptivity": absorptivity,
        "emissivity": emissivity,
        "nonreciprocity": np.abs(absorptivity - emissivity),
    }


def peak_in_window(
    wavelengths: np.ndarray,
    values: np.ndarray,
    lower_um: float,
    upper_um: float,
) -> tuple[float, float, int]:
    indices = np.flatnonzero((wavelengths >= lower_um) & (wavelengths <= upper_um))
    if len(indices) == 0:
        raise ValueError("peak window does not contain any wavelength samples")
    index = int(indices[np.argmax(values[indices])])
    return float(wavelengths[index]), float(values[index]), index


def print_summary(spectrum: dict[str, np.ndarray]) -> None:
    wavelength = spectrum["wavelength_um"]
    absorptivity = spectrum["absorptivity"]
    nonreciprocity = spectrum["nonreciprocity"]
    peak_windows_um = ((5.10, 5.22), (5.32, 5.48))

    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"orders={orders}, retained harmonics=81, points={len(wavelength)}, "
        f"TM theta=+/-{theta_deg:g} deg, B_solver=+3 T"
    )
    for peak_number, (window, paper_wavelength, paper_eta) in enumerate(
        zip(peak_windows_um, paper_peak_wavelengths_um, paper_peak_nonreciprocity),
        start=1,
    ):
        peak_um, peak_eta, peak_index = peak_in_window(
            wavelength,
            nonreciprocity,
            *window,
        )
        print(
            f"peak{peak_number}: eta={peak_eta:.4f} at {peak_um:.3f} um "
            f"(paper {paper_eta:.3f} at {paper_wavelength:.3f} um); "
            f"alpha={absorptivity[peak_index]:.4f}"
        )

    paper_index = int(np.argmin(np.abs(wavelength - paper_peak_wavelengths_um[0])))
    print(
        f"alpha at paper peak1 wavelength {wavelength[paper_index]:.3f} um: "
        f"{absorptivity[paper_index]:.4f} (paper {paper_peak1_absorptivity:.3f})"
    )


def plot_fig3_spectra(spectrum: dict[str, np.ndarray]) -> None:
    wavelength = spectrum["wavelength_um"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        constrained_layout=True,
    )

    axes[0].plot(
        wavelength,
        spectrum["b0_absorptivity"],
        color="#4a4a4a",
        linewidth=1.7,
        label=r"$B=0$ T, $\alpha=e$",
    )
    axes[0].plot(
        wavelength,
        spectrum["absorptivity"],
        color="#e53935",
        linewidth=1.9,
        label=rf"$B=3$ T, $\alpha(+{theta_deg:g}^\circ)$",
    )
    axes[0].plot(
        wavelength,
        spectrum["emissivity"],
        color="#e53935",
        linestyle="--",
        linewidth=1.8,
        label=rf"$B=3$ T, $e(+{theta_deg:g}^\circ)$",
    )
    axes[0].set_ylabel(r"Absorptivity $\alpha$ and emissivity $e$")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        wavelength,
        spectrum["nonreciprocity"],
        color="#e53935",
        linewidth=1.9,
        label=r"RCWA $\eta=|\alpha-e|$",
    )
    axes[1].scatter(
        paper_peak_wavelengths_um,
        paper_peak_nonreciprocity,
        marker="x",
        s=45,
        linewidths=1.5,
        color="#202020",
        label="Paper peaks",
        zorder=3,
    )
    axes[1].set_ylabel(r"Nonreciprocity $\eta$")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        for paper_wavelength in paper_peak_wavelengths_um:
            axis.axvline(
                paper_wavelength,
                color="#777777",
                linewidth=0.7,
                alpha=0.25,
            )
        axis.set_xlim(wavelength_min_um, wavelength_max_um)
        axis.set_ylim(-0.01, 1.01 if axis is axes[0] else 0.31)
        axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)

    fig.suptitle("InSb square-hole-cylinder spectrum - Zhu and Wang (2026) Fig. 3")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[str, np.ndarray]:
    spectrum = solve_fig3_spectra()
    print_summary(spectrum)
    plot_fig3_spectra(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
