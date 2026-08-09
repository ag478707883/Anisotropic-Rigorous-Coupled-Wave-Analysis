"""Reproduce Tang and Fan, EPL 142 (2023) 35001, Fig. 3(b).

The SiC strips are invariant along solver y, periodic along solver x, and the
stack follows solver +z. The local RCWA backend evaluates the paper's three
magnetic-field cases with exact forward/reverse thermal channels.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr


# Tang and Fan, Fig. 2, Fig. 3(b), and Eqs. (3)-(8).
period_um = 10.0
ridge_width_um = 5.0
grating_height_um = 0.39
sic_film_height_um = 4.0
inas_film_height_um = 1.6
bilayer_height_um = 5.6
bilayer_count = 1

# The paper and solver use the same axes: x is periodic, y is the strip and
# magnetic-bias direction, and +z follows the finite-layer stack.
theta_deg = 60.0
phi_deg = 0.0
polarization = "TM"
polarizations = (polarization,)
angles_deg = (+theta_deg, -theta_deg)
paper_magnetic_fields_t = (1.0, 1.5, 2.0)
# The paper and backend use the same +y bias direction; this gives the Fig. 3(b)
# ordering with the absorption resonance above the emission resonance.
solver_magnetic_field_sign = +1.0

inas_base_parameters = {
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}

speed_of_light_cm_s = 2.99792458e10
angular_frequency_to_cm_inv = 1.0 / (2.0 * np.pi * speed_of_light_cm_s)
sic_epsilon_inf = 6.7
sic_omega_lo_rad_s = 1.83e14
sic_omega_to_rad_s = 1.49e14
# The PDF glyph is easy to misread; Fig. 1(c,d) confirms 8.97e11 rad/s
# (n=3.98-4.18 and k=0.015-0.020 over 18.8-19.6 THz).
sic_damping_rad_s = 8.97e11
sic_parameters = {
    "epsilon_inf": (sic_epsilon_inf,) * 3,
    "omega_to_cm_inv": (
        sic_omega_to_rad_s * angular_frequency_to_cm_inv,
    ) * 3,
    "omega_lo_cm_inv": (
        sic_omega_lo_rad_s * angular_frequency_to_cm_inv,
    ) * 3,
    "damping_cm_inv": (
        sic_damping_rad_s * angular_frequency_to_cm_inv,
    ) * 3,
}
al_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

# The paper uses COMSOL and does not report its spatial/Fourier resolution.
# This centered 1D setting retains 81 analytic lamellar harmonics.
orders = (40, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

frequency_min_thz = 18.90
frequency_max_thz = 19.70
frequency_point_count = 801
frequencies_thz = np.linspace(
    frequency_min_thz,
    frequency_max_thz,
    frequency_point_count,
)
speed_of_light_um_thz = 299.792458
wavelengths_um = speed_of_light_um_thz / frequencies_thz

paper_absorption_peak_thz = 19.25
paper_emission_peak_thz = 19.03
paper_peak_level = 0.98

plot_path = project_root / "images" / "tang_2023_sic_inas_fig3.png"
figure_size = (8.0, 5.2)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object, magnetic_field_t: float) -> object:
    if not 0.0 < ridge_width_um < period_um:
        raise ValueError("ridge_width_um must lie between zero and period_um")
    if not np.isclose(
        sic_film_height_um + inas_film_height_um,
        bilayer_height_um,
    ):
        raise ValueError("SiC and InAs thicknesses must equal one bilayer")

    inas_parameters = {
        **inas_base_parameters,
        "magnetic_field_t": solver_magnetic_field_sign * magnetic_field_t,
    }
    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("SiC", "PhononPolariton", sic_parameters)
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Al")

    sim.AddLayer("SiC_grating", grating_height_um, "Air")
    sim.SetRegionRectangle(
        "SiC_grating",
        "SiC",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    for index in range(bilayer_count):
        sim.AddLayer(f"SiC_film_{index + 1}", sic_film_height_um, "SiC")
        sim.AddLayer(f"InAs_film_{index + 1}", inas_film_height_um, "InAs")
    return sim


def solve_case(magnetic_field_t: float) -> np.ndarray:
    return solve_fr(
        rcwa=asyrcwa,
        build_simulation=lambda module: build_simulation(module, magnetic_field_t),
        wavelengths_um=wavelengths_um,
        polarizations=polarizations,
        angles_deg=angles_deg,
        phi_deg=phi_deg,
        workers_per_angle=workers_per_angle,
    )


def frequency_ordered_rows(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = spectrum[spectrum["pol"] == polarization]
    frequency_thz = speed_of_light_um_thz / rows["lambda_um"]
    order = np.argsort(frequency_thz)
    return frequency_thz[order], rows[order]


def print_comparison(spectra: dict[float, np.ndarray]) -> None:
    for magnetic_field_t in paper_magnetic_fields_t:
        frequency_thz, rows = frequency_ordered_rows(spectra[magnetic_field_t])
        absorption = rows["A"]
        emission = rows["emissivity_same_channel"]
        absorption_index = int(np.argmax(absorption))
        emission_index = int(np.argmax(emission))
        print(
            f"B={magnetic_field_t:g} T: absorption peak "
            f"{frequency_thz[absorption_index]:.4f} THz, "
            f"alpha={absorption[absorption_index]:.4f}; emission peak "
            f"{frequency_thz[emission_index]:.4f} THz, "
            f"e={emission[emission_index]:.4f}"
        )
        if magnetic_field_t == 1.0:
            print(
                f"  paper: absorption about {paper_absorption_peak_thz:.2f} THz, "
                f"emission about {paper_emission_peak_thz:.2f} THz, "
                f"both >{paper_peak_level:.2f}"
            )


def plot_spectra(spectra: dict[float, np.ndarray]) -> None:
    colors = {
        1.0: ("#303030", "#e23b32"),
        1.5: ("#2e65d1", "#23a04a"),
        2.0: ("#8b62c6", "#d6a52a"),
    }
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    for magnetic_field_t in paper_magnetic_fields_t:
        frequency_thz, rows = frequency_ordered_rows(spectra[magnetic_field_t])
        absorption_color, emission_color = colors[magnetic_field_t]
        ax.plot(
            frequency_thz,
            rows["A"],
            color=absorption_color,
            linewidth=1.8,
            label=f"{magnetic_field_t:g} T absorption",
        )
        ax.plot(
            frequency_thz,
            rows["emissivity_same_channel"],
            color=emission_color,
            linewidth=1.8,
            label=f"{magnetic_field_t:g} T emission",
        )

    ax.axvline(paper_absorption_peak_thz, color="0.55", linewidth=0.7, alpha=0.35)
    ax.axvline(paper_emission_peak_thz, color="0.55", linewidth=0.7, alpha=0.35)
    ax.set_xlim(frequency_min_thz, frequency_max_thz)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Absorption / emission")
    ax.set_title(
        r"Tang and Fan (2023), Fig. 3(b): SiC grating / SiC / InAs / Al, "
        r"$\theta=60^\circ$"
    )
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[float, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        "coordinates: x periodic, y grating/bias direction, z stack; "
        f"theta=+/-{theta_deg:g} deg, phi={phi_deg:g} deg"
    )
    spectra: dict[float, np.ndarray] = {}
    for magnetic_field_t in paper_magnetic_fields_t:
        with asyrcwa.timed_step(f"solve B={magnetic_field_t:g} T"):
            spectra[magnetic_field_t] = solve_case(magnetic_field_t)
    print_comparison(spectra)
    with asyrcwa.timed_step("plot Tang and Fan Fig. 3(b)"):
        plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
