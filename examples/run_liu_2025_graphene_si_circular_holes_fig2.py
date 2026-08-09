"""Recompute Liu et al., Diamond & Related Materials 159 (2025) 112839, Fig. 2."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Liu et al., Fig. 1 and Table 1.
period_um = 5.31
hole_fill_ratio = 0.40
hole_diameter_um = hole_fill_ratio * period_um
si_height_um = 9.60
graphene_height_um = 0.00034
inas_height_um = 2.06
al_height_um = 1.00

graphene_parameters = {
    "fermi_energy_ev": 0.5,
    "tau_ps": 0.5,
    "thickness_nm": 0.34,
}
# The paper gives eps_inf and damping but refers to Refs. 39-41 for wp and
# m*. These are the repository's standard n-InAs values (wp=2.743e14 rad/s).
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
sio2_refractive_index = 1.45

theta_deg = 13.0
phi_deg = 0.0
angles_deg = (+theta_deg, -theta_deg)
polarizations = ("TE", "TM")
# Circular (7, 7) retains 149 harmonics. A (5, 5)/(7, 7)/(9, 9) audit
# rejects the paper-like but nonconverged (3, 3) TM peak; (7, 7) is the
# practical converged spectrum, while (9, 9) is about twice as expensive.
orders = (5, 5)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_point_count = 401
wavelengths_um = {
    "TE": np.linspace(14.2, 14.6, wavelength_point_count),
    "TM": np.linspace(14.8, 15.2, wavelength_point_count),
}
paper_emission_peak_um = {"TE": 14.350, "TM": 14.910}
paper_absorption_peak_um = {"TE": 14.408, "TM": 15.030}
paper_peak_eta = {"TE": 0.978, "TM": 0.951}
paper_markers = {
    pol: (paper_emission_peak_um[pol], paper_absorption_peak_um[pol])
    for pol in polarizations
}

plot_path = project_root / "images" / "liu_2025_graphene_si_circular_holes_fig2.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("Graphene", "GrapheneKubo", graphene_parameters)
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    # RETICOLO's area-normalized nested rectangles feed both the material
    # convolution and ordered Li blocks; no point-sample DFT is mixed in.
    sim.AddLayer("Si_circular_holes", si_height_um, "Si")
    sim.SetRegionCircle(
        "Si_circular_holes",
        "Air",
        Center=(0.0, 0.0),
        Radius=0.5 * hole_diameter_um,
    )
    sim.AddLayer("Graphene", graphene_height_um, "Graphene")
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Al_reflector", al_height_um, "Al")
    return sim


def solve_spectrum() -> np.ndarray:
    spectra = []
    for pol in polarizations:
        spectra.append(
            solve_fr(
                rcwa=asyrcwa,
                build_simulation=build_simulation,
                wavelengths_um=wavelengths_um[pol],
                polarizations=(pol,),
                angles_deg=angles_deg,
                phi_deg=phi_deg,
                workers_per_angle=workers_per_angle,
            )
        )
    return np.concatenate(spectra)


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    plot_fr_axes(axes, spectrum, polarizations, markers=paper_markers)
    for ax, pol in zip(axes, polarizations):
        ax.set_xlim(float(wavelengths_um[pol][0]), float(wavelengths_um[pol][-1]))
    fig.suptitle(f"Si circular-hole array / graphene / InAs / Al, orders={orders}")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Liu et al. Fig. 2 TE/TM spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        for pol in polarizations:
            rows = spectrum[spectrum["pol"] == pol]
            wavelength = paper_absorption_peak_um[pol]
            row = rows[int(np.argmin(abs(rows["lambda_um"] - wavelength)))]
            print(f"{pol}: eta({wavelength:.3f} um)={row['eta']:.4f}; paper {paper_peak_eta[pol]:.3f}")
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
