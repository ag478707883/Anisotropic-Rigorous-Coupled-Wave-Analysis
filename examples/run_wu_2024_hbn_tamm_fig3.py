"""Reproduce Wu et al., Physica Scripta 99 (2024) 075537, Fig. 3.

The structure is the uniform stack (TiO2/SiO2)^6 / air spacer / semi-infinite
hBN. Panels (a) and (b) evaluate the two mirror reflection coefficients from
the spacer and their Tamm phase condition. Panel (c) reconstructs the electric
field at the 7 um resonance with the repository's multilayer field solver.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Wu et al., Figs. 1-3 and Eqs. (1)-(5). All lengths are in micrometres.
pc_pair_count = 6
tio2_refractive_index = 2.4
sio2_refractive_index = 1.45
tio2_thickness_um = 0.698
sio2_thickness_um = 1.155
spacer_refractive_index = 1.0
spacer_thickness_um = 1.75

# hBN optic axis is solver z. Phonon parameters are quoted in cm^-1.
hbn_parameters = {
    "epsilon_inf": (4.87, 4.87, 2.95),
    "omega_to_cm_inv": (1370.0, 1370.0, 780.0),
    "omega_lo_cm_inv": (1610.0, 1610.0, 830.0),
    "damping_cm_inv": (5.0, 5.0, 4.0),
}

theta_deg = 0.0
phi_deg = 0.0
polarization = "TM"
workers = 10

wavelength_min_um = 6.0
wavelength_max_um = 7.5
wavelength_point_count = 1201
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)
resonance_wavelength_um = 7.0

pc_total_thickness_um = pc_pair_count * (
    tio2_thickness_um + sio2_thickness_um
)
field_z_min_um = -0.8
field_z_max_um = pc_total_thickness_um + spacer_thickness_um + 4.0
field_z_point_count = 2201

# Paper checkpoints at 7 um, in its exp(+i*omega*t) convention.
paper_rpc = complex(-0.877, -0.469)
paper_rhbn = complex(0.871, -0.464)
paper_pc_phase_rad = -2.56
paper_hbn_phase_rad = -0.48
paper_total_phase_rad = 0.01
paper_hbn_epsilon_perpendicular = complex(-15.980, 0.892)
paper_hbn_epsilon_parallel = complex(2.78, 0.001)

plot_path = project_root / "images" / "wu_2024_hbn_tamm_fig3.png"
figure_size = (11.0, 8.2)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_pc_from_air_side(simulation: object) -> None:
    for index in range(pc_pair_count):
        simulation.AddLayer(f"TiO2_{index + 1}", tio2_thickness_um, "TiO2")
        simulation.AddLayer(f"SiO2_{index + 1}", sio2_thickness_um, "SiO2")


def add_pc_from_spacer_side(simulation: object) -> None:
    # Looking into the PC from the spacer reverses the layer sequence.
    for index in range(pc_pair_count):
        simulation.AddLayer(f"SiO2_{index + 1}", sio2_thickness_um, "SiO2")
        simulation.AddLayer(f"TiO2_{index + 1}", tio2_thickness_um, "TiO2")


def set_common_materials(simulation: object) -> None:
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("TiO2", tio2_refractive_index**2 + 0.0j)
    simulation.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("hBN", "PhononPolariton", hbn_parameters)


def build_pc_reflector() -> object:
    simulation = asyrcwa.New()
    set_common_materials(simulation)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    add_pc_from_spacer_side(simulation)
    return simulation


def build_hbn_reflector() -> object:
    simulation = asyrcwa.New()
    set_common_materials(simulation)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("hBN")
    return simulation


def build_complete_structure() -> object:
    simulation = asyrcwa.New()
    set_common_materials(simulation)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("hBN")
    add_pc_from_air_side(simulation)
    simulation.AddLayer("air_spacer", spacer_thickness_um, "Air")
    return simulation


def reflection_amplitude(simulation: object) -> np.ndarray:
    rows = simulation.GetZeroOrderAmplitudes(
        wavelengths_um,
        Theta=theta_deg,
        Phi=phi_deg,
        Workers=workers,
    )
    selected = rows[
        (rows["block"] == "R")
        & (rows["out_pol"] == polarization)
        & (rows["in_pol"] == polarization)
    ]
    selected = np.sort(selected, order="lambda_um")
    if len(selected) != wavelength_point_count:
        raise RuntimeError("incomplete zero-order reflection spectrum")
    if not np.allclose(
        selected["lambda_um"],
        wavelengths_um,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("reflection wavelengths are not aligned")
    backend_amplitude = np.asarray(selected["value_re"], dtype=float) + 1j * np.asarray(
        selected["value_im"],
        dtype=float,
    )
    # The paper uses exp(+i*omega*t); the backend uses exp(-i*omega*t).
    return np.conj(backend_amplitude)


def solve_reflection_conditions() -> dict[str, np.ndarray]:
    r_pc = reflection_amplitude(build_pc_reflector())
    r_hbn = reflection_amplitude(build_hbn_reflector())
    spacer_phase = (
        2.0
        * np.pi
        * spacer_refractive_index
        * spacer_thickness_um
        / wavelengths_um
    )
    total_factor = r_hbn * r_pc * np.exp(2j * spacer_phase)
    return {
        "r_pc": r_pc,
        "r_hbn": r_hbn,
        "phase_pc": np.angle(r_pc),
        "phase_hbn": np.angle(r_hbn),
        "phase_total": np.angle(total_factor),
    }


def solve_field_profile() -> np.ndarray:
    rows = build_complete_structure().GetFieldPlane(
        resonance_wavelength_um,
        polarization,
        theta_deg,
        phi_deg,
        ["E"],
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
    return np.sort(rows, order="z_um")


def nearest_index(wavelength_um: float) -> int:
    return int(np.argmin(np.abs(wavelengths_um - wavelength_um)))


def print_comparison(conditions: dict[str, np.ndarray], field: np.ndarray) -> None:
    index = nearest_index(resonance_wavelength_um)
    r_pc = conditions["r_pc"][index]
    r_hbn = conditions["r_hbn"][index]
    phase_pc = conditions["phase_pc"][index]
    phase_hbn = conditions["phase_hbn"][index]
    phase_total = conditions["phase_total"][index]
    epsilon = asyrcwa.MaterialModelTensors(
        "PhononPolariton",
        resonance_wavelength_um,
        hbn_parameters,
    )["epsilon"]
    print(
        f"At {resonance_wavelength_um:g} um: rPC={r_pc.real:+.4f}{r_pc.imag:+.4f}i "
        f"(paper {paper_rpc.real:+.3f}{paper_rpc.imag:+.3f}i), "
        f"rhBN={r_hbn.real:+.4f}{r_hbn.imag:+.4f}i "
        f"(paper {paper_rhbn.real:+.3f}{paper_rhbn.imag:+.3f}i)"
    )
    print(
        f"  phases PC/hBN/total={phase_pc:.4f}/{phase_hbn:.4f}/{phase_total:.4f} rad; "
        f"paper {paper_pc_phase_rad:.2f}/{paper_hbn_phase_rad:.2f}/"
        f"{paper_total_phase_rad:.2f} rad"
    )
    print(
        f"  epsilon_perp={epsilon[0, 0]:.4f} "
        f"(paper {paper_hbn_epsilon_perpendicular}), "
        f"epsilon_parallel={epsilon[2, 2]:.4f} "
        f"(paper {paper_hbn_epsilon_parallel})"
    )
    peak = field[int(np.argmax(field["magnitude"]))]
    print(
        f"  max |E|/|E0|={float(peak['magnitude']):.4f} at "
        f"z={float(peak['z_um']):.4f} um"
    )


def plot_figure(conditions: dict[str, np.ndarray], field: np.ndarray) -> None:
    figure = plt.figure(figsize=figure_size, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])

    axis_a.plot(
        wavelengths_um,
        np.abs(conditions["r_hbn"]),
        color="#ef8a17",
        linewidth=2.0,
        label=r"$|r_{hBN}|$",
    )
    axis_a.plot(
        wavelengths_um,
        np.abs(conditions["r_pc"]),
        color="#bd1e2d",
        linewidth=2.0,
        label=r"$|r_{PC}|$",
    )
    axis_a.set_xlim(wavelength_min_um, wavelength_max_um)
    axis_a.set_ylim(0.0, 1.05)
    axis_a.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    axis_a.set_ylabel("Modulus")
    axis_a.set_title("(a) Reflection-coefficient modulus")
    axis_a.legend(frameon=False, loc="lower center")

    axis_b.plot(
        wavelengths_um,
        conditions["phase_hbn"],
        color="#bd1e2d",
        linewidth=2.0,
        label=r"$\phi_{hBN}$",
    )
    axis_b.plot(
        wavelengths_um,
        conditions["phase_pc"],
        color="#174a9c",
        linewidth=2.0,
        label=r"$\phi_{PC}$",
    )
    axis_b.plot(
        wavelengths_um,
        conditions["phase_total"],
        color="black",
        linewidth=1.8,
        label=r"$\phi$",
    )
    axis_b.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.5)
    axis_b.set_xlim(wavelength_min_um, wavelength_max_um)
    axis_b.set_ylim(-np.pi, np.pi)
    axis_b.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
    axis_b.set_ylabel("Phase (rad)")
    axis_b.set_title("(b) Tamm phase condition")
    axis_b.legend(frameon=False, fontsize=8, loc="lower center")

    z_um = np.asarray(field["z_um"], dtype=float)
    field_magnitude = np.asarray(field["magnitude"], dtype=float)
    axis_c.plot(z_um, field_magnitude, color="#d7191c", linewidth=1.8)
    pc_boundaries = [0.0]
    position = 0.0
    for _ in range(pc_pair_count):
        position += tio2_thickness_um
        pc_boundaries.append(position)
        position += sio2_thickness_um
        pc_boundaries.append(position)
    spacer_end_um = pc_total_thickness_um + spacer_thickness_um
    for boundary in pc_boundaries:
        axis_c.axvline(boundary, color="#888888", linewidth=0.45, linestyle="--", alpha=0.5)
    axis_c.axvline(
        spacer_end_um,
        color="#333333",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )
    axis_c.axvspan(
        pc_total_thickness_um,
        spacer_end_um,
        color="#f5e6a8",
        alpha=0.35,
        label="air spacer",
    )
    axis_c.axvspan(
        spacer_end_um,
        field_z_max_um,
        color="#b8d8b8",
        alpha=0.25,
        label="semi-infinite hBN",
    )
    axis_c.set_xlim(field_z_min_um, field_z_max_um)
    axis_c.set_ylim(bottom=0.0)
    axis_c.set_xlabel(r"Stack coordinate $z$ ($\mu$m)")
    axis_c.set_ylabel(r"$|E|/|E_0|$")
    axis_c.set_title(r"(c) Electric-field profile at $\lambda=7\,\mu$m")
    axis_c.legend(frameon=False, loc="upper right")

    for axis in (axis_a, axis_b, axis_c):
        axis.grid(True, color="#dddddd", linewidth=0.65, alpha=0.65)

    figure.suptitle("Wu et al. (2024), Fig. 3: hBN Tamm phonon-polariton")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> tuple[dict[str, np.ndarray], np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"(TiO2/SiO2)^{pc_pair_count}, dA={tio2_thickness_um:g} um, "
        f"dB={sio2_thickness_um:g} um, spacer={spacer_thickness_um:g} um"
    )
    with asyrcwa.timed_step("solve reflection amplitude and phase"):
        conditions = solve_reflection_conditions()
    with asyrcwa.timed_step("solve 7 um electric-field profile"):
        field = solve_field_profile()
    print_comparison(conditions, field)
    plot_figure(conditions, field)
    return conditions, field


if __name__ == "__main__":
    main()
