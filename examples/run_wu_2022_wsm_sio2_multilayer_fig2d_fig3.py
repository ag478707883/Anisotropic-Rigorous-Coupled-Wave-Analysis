"""Reproduce Wu et al., Int. J. Therm. Sci. 181 (2022) 107788, Figs. 2(d) and 3.

The laterally uniform five-pair WSM/SiO2 stack is solved with the backend's
exact zero-order Berreman/TMM route.  The paper's spectrum and normalized
``|Hy|`` line profiles are combined in one output figure.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr


# Wu et al., Fig. 1, Eqs. (1)-(4), and the optimized dimensions below Fig. 2.
pair_count = 9
wsm_thickness_um = 0.050
sio2_thickness_um = 3.652
sio2_refractive_index = 1.45
substrate_refractive_index = 1.45

# The paper specifies an optically thick Ag reflector but not its thickness.
# 0.30 um is thick enough to make transmission negligible around 10 um and
# covers the Ag-side range displayed in the Fig. 3 boundary zoom.
ag_thickness_um = 0.30
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

# Wu et al. take the WSM model and all explicit parameters from Zhao et al.,
# Nano Lett. 20 (2020) 1923-1927, DOI 10.1021/acs.nanolett.9b05179.
# Paper +y node separation gives epsilon_xz=+i*epsilon_a in local coordinates.
wsm_parameters = {
    "node_separation_m_inv": 2.0e9,
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}

theta_deg = 30.0
phi_deg = 0.0
polarization = "TM"
polarizations = (polarization,)
angles_deg = (+theta_deg, -theta_deg)
workers = 12
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 9.0
wavelength_max_um = 11.0
wavelength_point_count = 2001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

field_wavelength_um = 10.0
multilayer_thickness_um = pair_count * (
    wsm_thickness_um + sio2_thickness_um
)
field_z_min_um = 0.0
field_z_max_um = multilayer_thickness_um + 0.20
field_z_point_count = 6001
field_zoom_min_um = multilayer_thickness_um - 0.13
field_zoom_max_um = multilayer_thickness_um + 0.19

# Published reference values are annotations only; they do not alter results.
paper_emissivity_peak_um = 10.0
paper_emissivity_peak = 1.0
paper_absorptivity_fwhm_um = 0.104
paper_emissivity_fwhm_um = 0.094
paper_nonreciprocity_fwhm_um = 0.075
paper_reverse_field_peak = 8.0

plot_path = (
    project_root
    / "images"
    / "wu_2022_wsm_sio2_multilayer_fig2d_fig3_combined.png"
)
figure_size = (10.5, 8.2)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def build_simulation(rcwa: object) -> object:
    simulation = rcwa.New()
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    simulation.SetMaterial(
        "Substrate",
        substrate_refractive_index**2 + 0.0j,
    )
    simulation.SetMaterialModel("WSM", "WSM", wsm_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Substrate")

    for pair_index in range(pair_count):
        simulation.AddLayer(
            f"WSM_{pair_index + 1}",
            wsm_thickness_um,
            "WSM",
        )
        simulation.AddLayer(
            f"SiO2_{pair_index + 1}",
            sio2_thickness_um,
            "SiO2",
        )
    simulation.AddLayer("Ag_reflector", ag_thickness_um, "Ag")
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


def solve_field_profile(angle_deg: float) -> np.ndarray:
    return build_simulation(asyrcwa).GetFieldPlane(
        field_wavelength_um,
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


def peak_row(rows: np.ndarray, field: str) -> np.void:
    return rows[int(np.argmax(rows[field]))]


def print_summary(
    spectrum: np.ndarray,
    field_profiles: dict[float, np.ndarray],
) -> None:
    absorption_peak = peak_row(spectrum, "A")
    emission_peak = peak_row(spectrum, "emissivity_same_channel")
    eta_peak = peak_row(spectrum, "eta")
    transmission_maximum = max(
        float(np.max(spectrum["T"])),
        float(np.max(spectrum["T_reverse"])),
    )
    conservation_error = max(
        float(np.max(np.abs(spectrum["conservation"] - 1.0))),
        float(np.max(np.abs(spectrum["reverse_conservation"] - 1.0))),
    )
    print(
        "Spectrum peaks for theta=+30 deg:\n"
        f"  alpha={float(absorption_peak['A']):.6f} at "
        f"{float(absorption_peak['lambda_um']):.6f} um\n"
        f"  e={float(emission_peak['emissivity_same_channel']):.6f} at "
        f"{float(emission_peak['lambda_um']):.6f} um "
        f"(paper about {paper_emissivity_peak:.1f} at "
        f"{paper_emissivity_peak_um:.1f} um)\n"
        f"  eta={float(eta_peak['eta']):.6f} at "
        f"{float(eta_peak['lambda_um']):.6f} um"
    )
    print(
        "Paper FWHM values: "
        f"alpha={paper_absorptivity_fwhm_um * 1000.0:.0f} nm, "
        f"e={paper_emissivity_fwhm_um * 1000.0:.0f} nm, "
        f"eta={paper_nonreciprocity_fwhm_um * 1000.0:.0f} nm"
    )
    print(
        "Field-profile peak magnitudes: "
        + ", ".join(
            f"theta={angle:+g}: {float(np.max(rows['magnitude'])):.6f}"
            for angle, rows in field_profiles.items()
        )
        + f"; paper theta=-30 about {paper_reverse_field_peak:.0f}"
    )
    print(
        f"max Ag transmission={transmission_maximum:.3e}; "
        f"max energy-balance error={conservation_error:.3e}"
    )


def draw_spectrum(axis: object, spectrum: np.ndarray) -> None:
    axis.plot(
        spectrum["lambda_um"],
        spectrum["A"],
        color="#2455d6",
        linewidth=1.8,
        label=r"$\alpha(+30^\circ)$",
    )
    axis.plot(
        spectrum["lambda_um"],
        spectrum["emissivity_same_channel"],
        color="#ef5350",
        linestyle="--",
        linewidth=1.8,
        label=r"$e(+30^\circ)$",
    )
    axis.plot(
        spectrum["lambda_um"],
        spectrum["eta"],
        color="#222222",
        linestyle="-.",
        linewidth=1.6,
        label=r"$\eta=|\alpha-e|$",
    )
    axis.axvline(
        paper_emissivity_peak_um,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
    )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    axis.set_ylabel(r"$\alpha$, $e$, and $\eta$")
    axis.set_title("(a) Spectral nonreciprocity, paper Fig. 2(d)")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False, loc="upper right")


def draw_field_profiles(
    axis: object,
    field_profiles: dict[float, np.ndarray],
) -> None:
    colors = {+theta_deg: "#ef5350", -theta_deg: "#2455d6"}
    for angle in angles_deg:
        rows = np.sort(field_profiles[angle], order="z_um")
        axis.plot(
            rows["z_um"],
            rows["magnitude"],
            color=colors[angle],
            linewidth=1.8,
            label=rf"$\theta={angle:+g}^\circ$",
        )

    for pair_index in range(pair_count + 1):
        boundary_um = pair_index * (wsm_thickness_um + sio2_thickness_um)
        axis.axvline(
            boundary_um,
            color="#999999",
            linewidth=0.55,
            alpha=0.40,
        )
    axis.axvline(
        multilayer_thickness_um,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
    )
    axis.text(
        multilayer_thickness_um,
        0.97,
        "SiO$_2$/Ag",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.set_xlim(field_z_min_um, multilayer_thickness_um)
    axis.set_ylim(0.0, 8.35)
    axis.set_xlabel(r"Stack coordinate $z$ ($\mu$m)")
    axis.set_ylabel(r"Normalized $|H_y|$")
    axis.set_title(
        rf"(b) Magnetic-field profile at $\lambda={field_wavelength_um:g}\,\mu$m, "
        "paper Fig. 3"
    )
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False, loc="upper left")

    zoom = inset_axes(axis, width="30%", height="48%", loc="upper right")
    for angle in angles_deg:
        rows = np.sort(field_profiles[angle], order="z_um")
        zoom.plot(
            rows["z_um"],
            rows["magnitude"],
            color=colors[angle],
            linewidth=1.35,
        )
    zoom.axvline(
        multilayer_thickness_um,
        color="#222222",
        linewidth=0.9,
    )
    zoom.set_xlim(field_zoom_min_um, field_zoom_max_um)
    zoom.set_ylim(0.0, 8.35)
    zoom.set_xticks(
        [
            round(field_zoom_min_um, 1),
            round(multilayer_thickness_um, 2),
            round(field_zoom_max_um, 1),
        ]
    )
    zoom.tick_params(labelsize=7)
    zoom.text(
        multilayer_thickness_um - 0.015,
        7.3,
        "SiO$_2$",
        ha="right",
        va="top",
        fontsize=8,
    )
    zoom.text(
        multilayer_thickness_um + 0.02,
        7.3,
        "Ag",
        ha="left",
        va="top",
        fontsize=8,
    )


def plot_combined_figure(
    spectrum: np.ndarray,
    field_profiles: dict[float, np.ndarray],
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.15)},
    )
    draw_spectrum(axes[0], spectrum)
    draw_field_profiles(axes[1], field_profiles)
    figure.suptitle(
        "Wu et al. (2022): WSM/SiO$_2$ multilayer spectrum and $H_y$ field",
        fontsize=14,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> tuple[np.ndarray, dict[float, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"Air/[WSM({wsm_thickness_um:g} um)/SiO2({sio2_thickness_um:g} um)]^"
        f"{pair_count}/Ag({ag_thickness_um:g} um)/substrate"
    )
    with asyrcwa.timed_step("solve Wu et al. Fig. 2(d) spectrum"):
        spectrum = solve_spectrum()
    with asyrcwa.timed_step("solve Wu et al. Fig. 3 Hy profiles"):
        field_profiles = {
            angle: solve_field_profile(angle)
            for angle in angles_deg
        }
    print_summary(spectrum, field_profiles)
    with asyrcwa.timed_step("plot combined spectrum and Hy figure"):
        plot_combined_figure(spectrum, field_profiles)
    return spectrum, field_profiles


if __name__ == "__main__":
    main()
