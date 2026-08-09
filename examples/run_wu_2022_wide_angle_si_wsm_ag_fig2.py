"""Reproduce Wu et al., Case Studies in Thermal Engineering 40 (2022), Fig. 2.

The one-dimensional Si grating uses the backend's analytic lamellar
Fourier-factorization route. Both the wavelength spectrum and the fixed-
wavelength angular sweep are computed with the local exact thermal channels.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Wu et al., Fig. 1, Fig. 2, Eqs. (1)-(3), and Section 3.
period_um = 5.50
ridge_width_um = 3.355
grating_height_um = 0.65
wsm_height_um = 0.26
ag_height_um = 0.20
silicon_epsilon = 11.9
# The Ag mirror is opaque; the cited fused-quartz support is modeled by its
# standard mid-IR refractive index because the paper does not print a value.
substrate_refractive_index = 1.45
external_magnetic_field_t = 0.0

# Paper coordinates -> local solver coordinates:
#   paper x (periodic)     -> solver +x
#   paper y (stack normal) -> solver +z
#   paper z (grooves)      -> solver +y
# Thus the paper x-y incidence plane is solver x-z (phi=0), and its Weyl-node
# separation along paper +z maps to the local groove direction +y.
wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    # The paper delegates EF(T) to Refs. 29 and 30 and does not print it. This
    # explicit finite-temperature value reproduces the two Fig. 2(a) bands.
    "fermi_energy_ev": 0.082,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 30.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# The paper reports 60 harmonics. A centered Fourier basis contains the zero
# order plus symmetric pairs, so (30, 0) retains the nearest possible 61.
paper_harmonic_count = 60
orders = (paper_harmonic_count // 2, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 13.0
wavelength_max_um = 18.0
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

angle_wavelength_um = 15.98
angle_min_deg = 0.0
# The solver requires |theta| < 90 degrees; 89.5 is the last 0.5-degree point.
angle_max_deg = 89.5
angle_point_count = 180
angle_sweep_deg = np.linspace(angle_min_deg, angle_max_deg, angle_point_count)

paper_absorption_peak_um = 14.55
paper_absorption_peak = 0.985
paper_emission_peak_um = 15.98
paper_emission_peak = 0.998
paper_eta_at_emission_lower_bound = 0.935

plot_path = project_root / "images" / "wu_2022_wide_angle_si_wsm_ag_fig2.png"
figure_size = (8.0, 8.6)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANGLE_DTYPE = np.dtype([
    ("theta_deg", "f8"),
    ("A", "f8"),
    ("emissivity", "f8"),
    ("eta", "f8"),
])


def build_simulation(rcwa: object) -> object:
    if not 0.0 < ridge_width_um < period_um:
        raise ValueError("ridge_width_um must lie between zero and period_um")

    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterial("Air", 1.0 + 0.0j)
    sim.SetMaterial("Si", silicon_epsilon + 0.0j)
    sim.SetMaterial("SiO2", substrate_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("WSM", "WSM", wsm_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    sim.AddLayer("Si_grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "Si_grating",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    sim.AddLayer("WSM", wsm_height_um, "WSM")
    sim.AddLayer("Ag", ag_height_um, "Ag")
    return sim


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


def solve_angle_sweep() -> np.ndarray:
    sim = build_simulation(asyrcwa)
    signed_angles_deg = np.concatenate(
        (-angle_sweep_deg[:0:-1], angle_sweep_deg)
    )
    _, channels = sim.GetSpectrumAndDirectionalThermalChannelsForAngles(
        [angle_wavelength_um],
        signed_angles_deg,
        polarizations,
        Phi=phi_deg,
        Workers=workers,
    )
    tm_rows = channels[channels["pol"] == "TM"]

    output = np.empty(len(angle_sweep_deg), dtype=ANGLE_DTYPE)
    for index, angle_deg in enumerate(angle_sweep_deg):
        forward = tm_rows[np.isclose(
            tm_rows["theta_deg"], angle_deg, rtol=0.0, atol=1e-10
        )]
        reverse = tm_rows[np.isclose(
            tm_rows["theta_deg"], -angle_deg, rtol=0.0, atol=1e-10
        )]
        if len(forward) != 1 or len(reverse) != 1:
            raise RuntimeError(f"missing thermal-channel pair at {angle_deg:g} deg")
        absorptivity = float(forward[0]["absorptivity"])
        emissivity = float(reverse[0]["emissivity"])
        output[index] = (
            angle_deg,
            absorptivity,
            emissivity,
            abs(absorptivity - emissivity),
        )
    return output


def plot_figure(spectrum: np.ndarray, angle_rows: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TM"]
    colors = {"A": "#e02f2f", "e": "#2455d6", "eta": "#24a13a"}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        constrained_layout=True,
    )

    axes[0].plot(rows["lambda_um"], rows["A"], color=colors["A"], label=r"$\alpha$")
    axes[0].plot(
        rows["lambda_um"],
        rows["emissivity_same_channel"],
        color=colors["e"],
        label=r"$e$",
    )
    axes[0].plot(rows["lambda_um"], rows["eta"], color=colors["eta"], label=r"$\eta$")
    axes[0].axvline(paper_absorption_peak_um, color="0.55", linewidth=0.7, alpha=0.35)
    axes[0].axvline(paper_emission_peak_um, color="0.55", linewidth=0.7, alpha=0.35)
    axes[0].set_xlim(wavelength_min_um, wavelength_max_um)
    axes[0].set_xlabel(r"Wavelength ($\mu$m)")
    axes[0].set_ylabel(r"$\alpha$, $e$, $\eta$")
    axes[0].set_title(rf"(a) TM spectrum at $\theta={theta_deg:g}^\circ$")

    axes[1].plot(angle_rows["theta_deg"], angle_rows["A"], color=colors["A"], label=r"$\alpha$")
    axes[1].plot(
        angle_rows["theta_deg"],
        angle_rows["emissivity"],
        color=colors["e"],
        label=r"$e$",
    )
    axes[1].plot(angle_rows["theta_deg"], angle_rows["eta"], color=colors["eta"], label=r"$\eta$")
    axes[1].set_xlim(0.0, 90.0)
    axes[1].set_xlabel(r"Angle of incidence ($^\circ$)")
    axes[1].set_ylabel(r"$\alpha$, $e$, $\eta$")
    axes[1].set_title(rf"(b) Angular response at $\lambda={angle_wavelength_um:g}\ \mu$m")

    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Wu et al. (2022), Fig. 2: Si grating / WSM / Ag")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def print_comparison(spectrum: np.ndarray, angle_rows: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TM"]
    absorption_peak = rows[int(np.argmax(rows["A"]))]
    emission_peak = rows[int(np.argmax(rows["emissivity_same_channel"]))]
    reference = rows[int(np.argmin(abs(rows["lambda_um"] - paper_emission_peak_um)))]
    reference_angle = angle_rows[int(np.argmin(abs(angle_rows["theta_deg"] - theta_deg)))]
    print(
        f"absorption peak: {absorption_peak['lambda_um']:.4f} um, "
        f"A={absorption_peak['A']:.4f}; paper about "
        f"{paper_absorption_peak_um:.2f} um, A={paper_absorption_peak:.3f}"
    )
    print(
        f"emission peak: {emission_peak['lambda_um']:.4f} um, "
        f"e={emission_peak['emissivity_same_channel']:.4f}; paper "
        f"{paper_emission_peak_um:.2f} um, e={paper_emission_peak:.3f}"
    )
    print(
        f"at {paper_emission_peak_um:.2f} um: "
        f"A/e/eta={reference['A']:.4f}/"
        f"{reference['emissivity_same_channel']:.4f}/{reference['eta']:.4f}; "
        f"paper eta>{paper_eta_at_emission_lower_bound:.3f}"
    )
    print(
        f"angle sweep at {theta_deg:g} deg: "
        f"A/e/eta={reference_angle['A']:.4f}/"
        f"{reference_angle['emissivity']:.4f}/{reference_angle['eta']:.4f}"
    )


def main() -> tuple[np.ndarray, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        "coordinate map: paper (x periodic, y normal, z grooves) -> "
        "solver (x, z, y); phi=0 deg, grating/WSM direction=+y"
    )
    with asyrcwa.timed_step("solve Wu et al. Fig. 2(a) spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
    with asyrcwa.timed_step("solve Wu et al. Fig. 2(b) angular sweep"):
        angle_rows = solve_angle_sweep()
        print_comparison(spectrum, angle_rows)
    with asyrcwa.timed_step("plot Fig. 2"):
        plot_figure(spectrum, angle_rows)
    return spectrum, angle_rows


if __name__ == "__main__":
    main()
