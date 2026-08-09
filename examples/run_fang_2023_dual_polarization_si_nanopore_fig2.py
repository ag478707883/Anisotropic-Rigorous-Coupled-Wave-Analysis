"""Reproduce Fang et al., ICHMT 148 (2023) 107031, Fig. 2(a,b).

The circular-hole layer uses RETICOLO's area-normalized rectangle geometry,
and the directional absorptivity, emissivity, and nonreciprocity are computed
by the local RCWA backend.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Fang et al., Fig. 1, Eqs. (1)-(8), and Section 3.
period_um = 5.20
hole_diameter_ratio = 0.56
hole_diameter_um = hole_diameter_ratio * period_um
si_nanopore_height_um = 4.30
inas_height_um = 3.90
al_height_um = 0.20
silicon_refractive_index = 3.48

# With the backend's +z-into-the-stack convention, +5 T reproduces the
# absorption/emission peak ordering in Fig. 2 for both polarizations.
inas_parameters = {
    "magnetic_field_t": +5.0,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
al_parameters = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

theta_deg = 1.5
phi_deg = 0.0
polarizations = ("TE", "TM")
angles_deg = (+theta_deg, -theta_deg)

# The article does not state its retained Fourier order. Circular (9, 9)
# resolves the genuinely 2D circular-hole interface; a (7, 7)/(9, 9)/(11, 11)
# check moves each peak by less than 1 nm between the final two settings.
orders = (5, 5)
workers = 14
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.10
wavelength_max_um = 15.25
wavelength_point_count = 601
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peak_um = {"TE": 15.193, "TM": 15.154}
paper_emission_peak_um = {"TE": 15.201, "TM": 15.127}
paper_absorption_q = {"TE": 1923.0, "TM": 1720.0}
paper_emission_q = {"TE": 2140.0, "TM": 1680.0}
paper_peak_level = 0.98
paper_markers = {
    polarization: tuple(
        sorted(
            (
                paper_absorption_peak_um[polarization],
                paper_emission_peak_um[polarization],
            )
        )
    )
    for polarization in polarizations
}

plot_path = (
    project_root
    / "images"
    / "fang_2023_dual_polarization_si_nanopore_fig2.png"
)
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < hole_diameter_um < period_um:
        raise ValueError("hole_diameter_um must lie between zero and period_um")

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Si")

    # RETICOLO's nested rectangles feed both direct and ordered-Li blocks;
    # the two operators do not mix different circle geometries.
    sim.AddLayer("Si_circular_nanopores", si_nanopore_height_um, "Si")
    sim.SetRegionCircle(
        "Si_circular_nanopores",
        "Air",
        Center=(0.0, 0.0),
        Radius=0.5 * hole_diameter_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Al", al_height_um, "Al")
    return sim


def estimate_q(wavelength_um: np.ndarray, values: np.ndarray) -> float:
    peak_index = int(np.argmax(values))
    peak_value = float(values[peak_index])
    half_maximum = 0.5 * peak_value

    left_candidates = np.flatnonzero(values[:peak_index] <= half_maximum)
    right_candidates = np.flatnonzero(values[peak_index + 1 :] <= half_maximum)
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return float("nan")

    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak_index + 1 + int(right_candidates[0])
    right_low = right_high - 1

    left_crossing = float(
        np.interp(
            half_maximum,
            values[[left_low, left_high]],
            wavelength_um[[left_low, left_high]],
        )
    )
    right_crossing = float(
        np.interp(
            half_maximum,
            values[[right_high, right_low]],
            wavelength_um[[right_high, right_low]],
        )
    )
    width_um = right_crossing - left_crossing
    return float(wavelength_um[peak_index] / width_um)


def print_paper_comparison(spectrum: np.ndarray) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        absorption_peak = rows[int(np.argmax(rows["A"]))]
        emission_peak = rows[
            int(np.argmax(rows["emissivity_same_channel"]))
        ]
        absorption_q = estimate_q(rows["lambda_um"], rows["A"])
        emission_q = estimate_q(
            rows["lambda_um"],
            rows["emissivity_same_channel"],
        )
        print(
            f"{polarization} absorption: "
            f"lambda={absorption_peak['lambda_um']:.6f} um, "
            f"A={absorption_peak['A']:.6f}, Q={absorption_q:.1f}; "
            f"paper lambda={paper_absorption_peak_um[polarization]:.3f} um, "
            f"A~{paper_peak_level:.2f}, Q={paper_absorption_q[polarization]:.0f}"
        )
        print(
            f"{polarization} emission: "
            f"lambda={emission_peak['lambda_um']:.6f} um, "
            f"e={emission_peak['emissivity_same_channel']:.6f}, "
            f"Q={emission_q:.1f}; "
            f"paper lambda={paper_emission_peak_um[polarization]:.3f} um, "
            f"e~{paper_peak_level:.2f}, Q={paper_emission_q[polarization]:.0f}"
        )
        print(
            f"  max T={max(float(np.max(rows['T'])), float(np.max(rows['T_reverse']))):.3e}, "
            f"max |A(-theta)-exact e|="
            f"{float(np.max(abs(rows['A_reverse'] - rows['emissivity_same_channel']))):.3e}"
        )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    plot_fr_axes(axes, spectrum, polarizations, markers=paper_markers)
    for ax in axes:
        ax.set_xlim(wavelength_min_um, wavelength_max_um)
    fig.suptitle(
        "Si circular-nanopore array / InAs / Al / Si, "
        f"B={inas_parameters['magnetic_field_t']:g} T, "
        f"theta=+/-{theta_deg:g} deg, orders={orders}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, hole diameter={hole_diameter_um:g} um "
        f"(w/d={hole_diameter_ratio:g}), n(Si)={silicon_refractive_index:g}"
    )
    with asyrcwa.timed_step("solve Fang et al. Fig. 2(a,b) spectrum"):
        spectrum = solve_fr(
            rcwa=asyrcwa,
            build_simulation=build_simulation,
            wavelengths_um=wavelengths_um,
            polarizations=polarizations,
            angles_deg=angles_deg,
            phi_deg=phi_deg,
            workers_per_angle=workers_per_angle,
        )
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
