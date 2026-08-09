"""Reproduce Wu and Qing, Materials Today Physics 32 (2023) 101025, Fig. 2.

The structure is a one-dimensional cascaded Ge grating / WSM grating above an
opaque Ag reflector.  The script computes the TM absorptivity, directional
emissivity, and nonreciprocity at theta=1 deg using the local RCWA backend.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Wu and Qing, Fig. 1, Fig. 2, Eqs. (1)-(7), and Section 3.
period_um = 15.18
ridge_width_um = 9.22
ge_grating_height_um = 6.556
wsm_grating_height_um = 0.77

# The Ag reflector is described as thick enough to block transmission, but its
# thickness is not printed.  This finite layer gives zero transmitted power in
# the plotted band with the local backend.
ag_height_um = 0.50

ge_refractive_index = 4.0
sio2_refractive_index = 1.45

ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}
wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    # Paper x is periodic, paper y is the stack normal, and paper z is along
    # the grooves.  That maps the printed xz gyrotropic tensor to solver +y.
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}

theta_deg = 1.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# The paper does not report its retained Fourier order.  This practical 1D
# setting keeps 57 harmonics and reproduces the near-15.5 um branch in Fig. 2.
orders = (28, 0)
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

paper_peak_um = 15.5
paper_absorptivity_lower_bound = 0.9997
paper_emissivity = 0.065
paper_eta = 0.935

data_path = project_root / "data" / "wu_2023_cascade_ge_wsm_grating_fig2_tm.csv"
plot_path = project_root / "images" / "wu_2023_cascade_ge_wsm_grating_fig2.png"
figure_size = (7.2, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def build_simulation(rcwa: object) -> object:
    if not 0.0 < ridge_width_um < period_um:
        raise ValueError("ridge_width_um must lie between zero and period_um")

    simulation = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("Ge", ge_refractive_index**2 + 0.0j)
    simulation.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("WSM", "WSM", wsm_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("SiO2")

    simulation.AddLayer("Ge_grating", ge_grating_height_um, "Air")
    simulation.SetRegionRectangle(
        "Ge_grating",
        "Ge",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("WSM_grating", wsm_grating_height_um, "Air")
    simulation.SetRegionRectangle(
        "WSM_grating",
        "WSM",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
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


def export_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TM"], order="lambda_um")
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete TM spectrum returned by backend")

    values = np.column_stack(
        (
            rows["lambda_um"],
            rows["lambda_um"] * 1000.0,
            rows["theta_deg"],
            np.full(len(rows), angles_deg[1], dtype=float),
            rows["phi_deg"],
            rows["harmonics"],
            rows["R"],
            rows["T"],
            rows["A"],
            rows["R_reverse"],
            rows["T_reverse"],
            rows["A_reverse"],
            rows["emissivity_same_channel"],
            rows["emissivity_reverse_channel"],
            rows["eta"],
            rows["conservation"],
            rows["reverse_conservation"],
        )
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_path,
        values,
        delimiter=",",
        header=(
            "wavelength_um,wavelength_nm,theta_plus_deg,theta_minus_deg,"
            "phi_deg,harmonics,R_plus,T_plus,A_plus,R_minus,T_minus,A_minus,"
            "emissivity_same_channel,emissivity_reverse_channel,eta,"
            "conservation_plus,conservation_minus"
        ),
        comments="",
        fmt="%.10f",
    )
    print(f"Saved {data_path}.")


def print_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TM"]
    absorption_peak = rows[int(np.argmax(rows["A"]))]
    eta_peak = rows[int(np.argmax(rows["eta"]))]
    reference = rows[int(np.argmin(abs(rows["lambda_um"] - paper_peak_um)))]
    transmission_maximum = max(
        float(np.max(rows["T"])),
        float(np.max(rows["T_reverse"])),
    )
    conservation_error = max(
        float(np.max(np.abs(rows["conservation"] - 1.0))),
        float(np.max(np.abs(rows["reverse_conservation"] - 1.0))),
    )
    print(
        f"absorption peak: lambda={absorption_peak['lambda_um']:.6f} um, "
        f"A={absorption_peak['A']:.6f}"
    )
    print(
        f"eta peak: lambda={eta_peak['lambda_um']:.6f} um, "
        f"A/e/eta={eta_peak['A']:.6f}/"
        f"{eta_peak['emissivity_same_channel']:.6f}/{eta_peak['eta']:.6f}"
    )
    print(
        f"paper checkpoint at {paper_peak_um:.1f} um: "
        f"A>{paper_absorptivity_lower_bound:.4f}, "
        f"e={paper_emissivity:.3f}, eta={paper_eta:.3f}"
    )
    print(
        f"computed at {paper_peak_um:.1f} um: "
        f"A/e/eta={reference['A']:.6f}/"
        f"{reference['emissivity_same_channel']:.6f}/{reference['eta']:.6f}"
    )
    print(
        f"max Ag transmission={transmission_maximum:.3e}; "
        f"max energy-balance error={conservation_error:.3e}"
    )


def draw_unit_cell(axis: object) -> None:
    left = -0.5 * period_um
    axis.add_patch(
        Rectangle(
            (left, 0.0),
            period_um,
            ag_height_um,
            facecolor="#c7d6c0",
            edgecolor="0.35",
            linewidth=0.7,
        )
    )
    axis.add_patch(
        Rectangle(
            (-0.5 * ridge_width_um, ag_height_um),
            ridge_width_um,
            wsm_grating_height_um,
            facecolor="#f2d95c",
            edgecolor="0.35",
            linewidth=0.7,
        )
    )
    axis.add_patch(
        Rectangle(
            (-0.5 * ridge_width_um, ag_height_um + wsm_grating_height_um),
            ridge_width_um,
            ge_grating_height_um,
            facecolor="#c99ac7",
            edgecolor="0.35",
            linewidth=0.7,
        )
    )
    axis.set_xlim(-0.55 * period_um, 0.55 * period_um)
    axis.set_ylim(0.0, ag_height_um + wsm_grating_height_um + ge_grating_height_um)
    axis.set_aspect("auto")
    axis.axis("off")


def plot_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TM"], order="lambda_um")
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    axis.plot(rows["lambda_um"], rows["A"], color="#1f5eff", linewidth=2.0, label=r"$\alpha$")
    axis.plot(
        rows["lambda_um"],
        rows["emissivity_same_channel"],
        color="#ff3030",
        linewidth=1.8,
        label=r"$e$",
    )
    axis.plot(rows["lambda_um"], rows["eta"], color="#19a832", linewidth=1.8, label=r"$\eta$")
    axis.scatter(
        [paper_peak_um],
        [paper_absorptivity_lower_bound],
        marker="o",
        facecolors="none",
        edgecolors="#1f5eff",
        zorder=4,
        label="paper checkpoint",
    )
    axis.scatter(
        [paper_peak_um],
        [paper_emissivity],
        marker="o",
        facecolors="none",
        edgecolors="#ff3030",
        zorder=4,
    )
    axis.scatter(
        [paper_peak_um],
        [paper_eta],
        marker="o",
        facecolors="none",
        edgecolors="#19a832",
        zorder=4,
    )
    axis.axvline(paper_peak_um, color="0.55", linestyle=":", linewidth=0.8)
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    axis.set_ylabel(r"$\alpha$, $e$, and $\eta$")
    axis.set_title(rf"TM spectrum at $\theta={theta_deg:g}^\circ$")
    axis.grid(True, color="#dddddd", linewidth=0.75, alpha=0.75)
    axis.legend(loc="upper left", frameon=False, fontsize=8)

    inset = axis.inset_axes((0.61, 0.50, 0.34, 0.38))
    draw_unit_cell(inset)

    figure.suptitle("Wu and Qing (2023), Fig. 2: cascaded Ge/WSM grating")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"d={period_um:g} um, w={ridge_width_um:g} um, "
        f"h_g={ge_grating_height_um:g} um, h={wsm_grating_height_um:g} um, "
        f"Ag={ag_height_um:g} um, Orders={orders}"
    )
    with asyrcwa.timed_step("solve Wu and Qing Fig. 2 TM spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_comparison(spectrum)
        export_spectrum(spectrum)
    with asyrcwa.timed_step("plot Wu and Qing Fig. 2"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
