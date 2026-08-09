"""Reproduce Wang et al., Optics Express 27 (2019) 25983, Fig. 2(a,b).

The local RCWA backend evaluates the normal-incidence Jones reflection matrix
of the copper / FR-4 / copper absorber.  The paper's Eq. (2) converts the
linear Jones matrix to circular reflection channels; Eqs. (3)-(5) then give
the LCP/RCP absorption and absorptive circular dichroism.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Wang et al., Fig. 1 and Sec. 2. Lengths are converted from mm to um.
period_um = 11_500.0
l1_um = 5_400.0
l2_um = 6_700.0
l3_um = 5_700.0
wire_width_um = 1_250.0
a_um = 2_900.0
b_um = 1_250.0
c_um = 1_000.0
d_um = 3_300.0
left_margin_e_um = 1_600.0
top_margin_f_um = 1_000.0
twist_angle_deg = 60.0
fr4_height_um = 2_500.0
backplane_height_um = 35.0

# The paper reports only the bottom-copper thickness.  Equal 35-um PCB copper
# on the patterned face is the explicit modeling assumption used here.
top_copper_height_um = 35.0

# Paper material data.  Positive Im(epsilon) is passive in the backend's
# exp(-i*omega*t) convention.
fr4_relative_permittivity = 4.3
fr4_loss_tangent = 0.025
fr4_epsilon = fr4_relative_permittivity * (1.0 + 1j * fr4_loss_tangent)
copper_conductivity_s_m = 5.8e7

# The repository has no direct Ohmic-conductivity model.  A Cu Drude plasma
# frequency is assumed and gamma is chosen so eps0*wp^2/gamma equals exactly
# the paper's DC conductivity; omega/gamma < 0.004 over 11-15 GHz.
vacuum_permittivity_f_m = 8.8541878128e-12
copper_plasma_frequency_rad_s = 1.39e16
copper_damping_rate_rad_s = (
    vacuum_permittivity_f_m
    * copper_plasma_frequency_rad_s**2
    / copper_conductivity_s_m
)
copper_drude_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": copper_plasma_frequency_rad_s,
    "damping_rate_rad_s": copper_damping_rate_rad_s,
}

theta_deg = 0.0
phi_deg = 0.0
polarizations = ("LCP", "RCP")

# CST rather than RCWA was used in the paper, so no Fourier order was stated.
# Circular (7,7) retains 149 harmonics and gives a practical full-spectrum run.
orders = (5, 5)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.

frequency_min_ghz = 11.0
frequency_max_ghz = 15.0
frequency_point_count = 401
frequencies_ghz = np.linspace(
    frequency_min_ghz,
    frequency_max_ghz,
    frequency_point_count,
)
speed_of_light_um_ghz = 299_792.458
wavelengths_um = speed_of_light_um_ghz / frequencies_ghz

paper_lcp_peaks = ((12.04, 0.9518), (14.22, 0.9177))
paper_rcp_at_lcp_peaks = ((12.04, 0.172), (14.22, 0.230))
paper_cd_at_peaks = ((12.04, 0.778), (14.22, 0.690))

# Exact simple-polygon union of L1, L2, and L3.  Fig. 1 fixes L1/L2 through
# e, f, a, b and places the upper long edge of the +30-degree L3 bar through
# the lower-left L2 corner.  c, d, and w close L3 to 5.743 mm; the printed
# L3=5.7 mm is rounded. A single concave polygon is converted to one common
# rectilinear cell geometry for both direct and ordered-Li blocks.
resonator_vertices_um = (
    (-4_150.000000, +4_750.000000),
    (+1_250.000000, +4_750.000000),
    (+1_250.000000, +3_500.000000),
    (+0.000000, +3_500.000000),
    (+0.000000, -2_297.890204),
    (+2_545.383832, -828.312164),
    (+3_170.383832, -1_910.843918),
    (-1_803.525404, -4_782.531755),
    (-2_428.525404, -3_700.000000),
    (-1_250.000000, -3_019.578041),
    (-1_250.000000, +3_500.000000),
    (-4_150.000000, +3_500.000000),
)

plot_path = (
    project_root
    / "images"
    / "wang_2019_chirality_selective_absorber_fig2.png"
)
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not np.isclose(a_um + wire_width_um + b_um, l1_um):
        raise ValueError("a + w + b must equal L1")

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterial("FR4", fr4_epsilon)
    sim.SetMaterialModel("Cu", "Drude", copper_drude_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Air")

    sim.AddLayer("I_resonator", top_copper_height_um, "Air")
    sim.SetRegionPolygon(
        "I_resonator",
        "Cu",
        Center=(0.0, 0.0),
        Angle=0.0,
        Vertices=resonator_vertices_um,
    )
    sim.AddLayer("FR4", fr4_height_um, "FR4")
    sim.AddLayer("Cu_backplane", backplane_height_um, "Cu")
    return sim


def complex_amplitude(
    rows: np.ndarray,
    block: str,
    out_polarization: str,
    in_polarization: str,
) -> np.ndarray:
    selected = rows[
        (rows["block"] == block)
        & (rows["out_pol"] == out_polarization)
        & (rows["in_pol"] == in_polarization)
    ]
    if len(selected) != frequency_point_count:
        raise RuntimeError(
            f"incomplete {block}_{out_polarization}{in_polarization} amplitude"
        )
    if not np.allclose(selected["lambda_um"], wavelengths_um, atol=1e-10):
        raise RuntimeError("zero-order amplitude frequencies are misaligned")
    return selected["value_re"] + 1j * selected["value_im"]


def solve_spectrum() -> dict[str, np.ndarray | float]:
    simulation = build_simulation(asyrcwa)
    rows = simulation.GetZeroOrderAmplitudes(
        wavelengths_um,
        Theta=theta_deg,
        Phi=phi_deg,
        Workers=workers,
    )

    r_te_te = complex_amplitude(rows, "R", "TE", "TE")
    r_te_tm = complex_amplitude(rows, "R", "TE", "TM")
    r_tm_te = complex_amplitude(rows, "R", "TM", "TE")
    r_tm_tm = complex_amplitude(rows, "R", "TM", "TM")

    # At normal incidence and phi=0, TE is Cartesian y.  The reflected TM
    # mode has the opposite Cartesian-x direction, hence the first-row signs.
    r_xx = -r_tm_tm
    r_xy = -r_tm_te
    r_yx = r_te_tm
    r_yy = r_te_te

    # Wang et al. Eq. (2): rows are reflected (+,-), columns incident (+,-).
    r_pp = 0.5 * (r_xx + r_yy + 1j * (r_xy - r_yx))
    r_pm = 0.5 * (r_xx - r_yy - 1j * (r_xy + r_yx))
    r_mp = 0.5 * (r_xx - r_yy + 1j * (r_xy + r_yx))
    r_mm = 0.5 * (r_xx + r_yy - 1j * (r_xy - r_yx))

    # The backend's exp(-i*omega*t) circular-state labels are conjugate to the
    # observer-facing handedness used by Wang et al.  Exchange +/- on both the
    # input and reflected states before applying the paper's channel labels.
    r_pp, r_pm, r_mp, r_mm = r_mm, r_mp, r_pm, r_pp

    reflectance_pp = abs(r_pp) ** 2
    reflectance_pm = abs(r_pm) ** 2
    reflectance_mp = abs(r_mp) ** 2
    reflectance_mm = abs(r_mm) ** 2
    absorption_lcp = 1.0 - reflectance_pm - reflectance_mm
    absorption_rcp = 1.0 - reflectance_mp - reflectance_pp
    cd_ab = absorption_lcp - absorption_rcp

    t_power_max = 0.0
    for out_pol in ("TE", "TM"):
        for in_pol in ("TE", "TM"):
            t = complex_amplitude(rows, "T", out_pol, in_pol)
            t_power_max = max(t_power_max, float(np.max(abs(t) ** 2)))

    return {
        "R_pp": reflectance_pp,
        "R_pm": reflectance_pm,
        "R_mp": reflectance_mp,
        "R_mm": reflectance_mm,
        "A_LCP": absorption_lcp,
        "A_RCP": absorption_rcp,
        "CD_ab": cd_ab,
        "max_T_channel": t_power_max,
        "harmonics": float(rows["harmonics"][0]),
    }


def nearest_index(frequency_ghz: float) -> int:
    return int(np.argmin(abs(frequencies_ghz - frequency_ghz)))


def print_paper_comparison(spectrum: dict[str, np.ndarray | float]) -> None:
    a_lcp = np.asarray(spectrum["A_LCP"])
    a_rcp = np.asarray(spectrum["A_RCP"])
    cd_ab = np.asarray(spectrum["CD_ab"])
    for (frequency_ghz, paper_lcp), (_, paper_rcp), (_, paper_cd) in zip(
        paper_lcp_peaks,
        paper_rcp_at_lcp_peaks,
        paper_cd_at_peaks,
    ):
        index = nearest_index(frequency_ghz)
        print(
            f"{frequency_ghz:.2f} GHz computed A_LCP/A_RCP/CD="
            f"{a_lcp[index]:.4f}/{a_rcp[index]:.4f}/{cd_ab[index]:.4f}; "
            f"paper={paper_lcp:.4f}/{paper_rcp:.4f}/{paper_cd:.4f}"
        )
    print(
        f"harmonics={int(spectrum['harmonics'])}, "
        f"max zero-order T channel={spectrum['max_T_channel']:.3e}"
    )


def plot_spectrum(spectrum: dict[str, np.ndarray | float]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        constrained_layout=True,
    )

    axes[0].plot(frequencies_ghz, spectrum["R_mp"], "k-", label=r"$R_{-+}$")
    axes[0].plot(frequencies_ghz, spectrum["R_pp"], "r-", label=r"$R_{++}$")
    axes[0].plot(frequencies_ghz, spectrum["R_mm"], "b-", label=r"$R_{--}$")
    axes[0].plot(frequencies_ghz, spectrum["R_pm"], color="magenta", label=r"$R_{+-}$")
    axes[0].set_ylabel("Reflectance")
    axes[0].set_title("(a) Circular reflection channels")
    axes[0].legend(frameon=False)

    axes[1].plot(frequencies_ghz, spectrum["A_RCP"], "k-", label="RCP")
    axes[1].plot(frequencies_ghz, spectrum["A_LCP"], "r-", label="LCP")
    axes[1].set_ylabel("Absorption")
    axes[1].set_title("(b) Absorption and circular dichroism")
    cd_axis = axes[1].twinx()
    cd_axis.plot(frequencies_ghz, spectrum["CD_ab"], "b-", label=r"$CD_{ab}$")
    cd_axis.set_ylabel(r"$CD_{ab}$", color="blue")
    cd_axis.tick_params(axis="y", colors="blue")

    handles, labels = axes[1].get_legend_handles_labels()
    cd_handles, cd_labels = cd_axis.get_legend_handles_labels()
    axes[1].legend(handles + cd_handles, labels + cd_labels, frameon=False)

    for axis in (*axes, cd_axis):
        axis.set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xlim(frequency_min_ghz, frequency_max_ghz)
        axis.set_xlabel("Frequency (GHz)")
        axis.grid(alpha=0.18)
        for frequency_ghz, _ in paper_lcp_peaks:
            axis.axvline(frequency_ghz, color="0.55", linestyle=":", linewidth=0.8)

    fig.suptitle(
        "Wang et al. (2019): chirality-selective copper / FR-4 absorber, "
        f"orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[str, np.ndarray | float]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"P={period_um / 1000:g} mm, FR-4={fr4_height_um / 1000:g} mm, "
        f"Cu top/back={top_copper_height_um:g}/{backplane_height_um:g} um, "
        f"orders={orders}"
    )
    with asyrcwa.timed_step("solve Wang et al. Fig. 2 Jones spectrum"):
        spectrum = solve_spectrum()
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
