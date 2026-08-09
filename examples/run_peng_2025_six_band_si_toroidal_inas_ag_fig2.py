"""Reproduce Peng and Wang, Appl. Phys. Lett. 126 (2025) 253907, Fig. 2.

DOI: 10.1063/5.0273263. The square array contains one hollow silicon
cylinder per cell above magnetized InAs and an opaque Ag reflector. The ring
uses exact nested-circle Fourier coefficients rather than a sampled mask.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Peng and Wang, Fig. 1 and Fig. 2. The DOI abstract explicitly gives
# B=2.5 T and theta=5.9 degrees. The AIP subscription preview does not expose
# the dimension table; the provisional dimensions below retain the scale used
# in the authors' closely related InAs/Ag emitter and remain explicit so they
# can be replaced directly when the article PDF is available.
period_um = 7.84
ring_outer_diameter_um = 7.74
ring_inner_diameter_um = 5.74
ring_outer_radius_um = 0.5 * ring_outer_diameter_um
ring_inner_radius_um = 0.5 * ring_inner_diameter_um
ring_height_um = 7.00
inas_height_um = 6.50
ag_height_um = 0.40

# Paper coordinates use +z toward the incident half-space. Local layers are
# added from Air toward solver +z, so paper +y maps to solver -y while paper x
# remains solver x. The incidence plane is local x-z (phi=0 degrees).
inas_parameters = {
    "magnetic_field_t": -2.5,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 5.9
phi_deg = 0.0
angles_deg = (+theta_deg, -theta_deg)
polarizations = ("TE", "TM")

# The article does not state its retained Fourier order in the public preview.
# Circular truncation with (5,5) keeps the example practical for the four
# directional/polarization channels while resolving the analytic annulus.
orders = (5, 5)
workers = 20
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.10
wavelength_max_um = 16.30
wavelength_point_count = 1201
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

# The abstract reports six strong nonreciprocal bands with eta > 80% under
# dual polarization. In the two-panel presentation this is treated as the
# three strongest TE and three strongest TM bands.
paper_peak_count_per_polarization = 3
paper_peak_eta_lower_bound = 0.80

plot_path = (
    project_root
    / "images"
    / "peng_2025_six_band_si_toroidal_inas_ag_fig2.png"
)
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 <= ring_inner_radius_um < ring_outer_radius_um:
        raise ValueError("ring radii must satisfy outer > inner >= 0")
    if ring_outer_diameter_um >= period_um:
        raise ValueError("the isolated ring must fit inside one unit cell")

    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("Si", "Si")
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Ag", "Drude", ag_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Air")

    sim.AddLayer("Si_toroidal_cylinders", ring_height_um, "Air")
    sim.SetRegionRing(
        "Si_toroidal_cylinders",
        "Si",
        "Air",
        Center=(0.0, 0.0),
        OuterRadius=ring_outer_radius_um,
        InnerRadius=ring_inner_radius_um,
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Ag_reflector", ag_height_um, "Ag")
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


def strongest_local_peaks(
    wavelengths: np.ndarray,
    values: np.ndarray,
    count: int,
) -> list[tuple[float, float]]:
    indices = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    ranked = indices[np.argsort(values[indices])[::-1]][:count]
    ranked = ranked[np.argsort(wavelengths[ranked])]
    return [
        (float(wavelengths[index]), float(values[index]))
        for index in ranked
    ]


def print_paper_comparison(spectrum: np.ndarray) -> None:
    for polarization in polarizations:
        rows = spectrum[spectrum["pol"] == polarization]
        peaks = strongest_local_peaks(
            rows["lambda_um"],
            rows["eta"],
            paper_peak_count_per_polarization,
        )
        print(f"{polarization} strongest nonreciprocal bands:")
        for wavelength_um, eta in peaks:
            print(f"  {wavelength_um:.4f} um: eta={eta:.4f}")
        strong_count = sum(eta >= paper_peak_eta_lower_bound for _, eta in peaks)
        print(
            f"  {strong_count}/{paper_peak_count_per_polarization} selected "
            f"bands have eta >= {paper_peak_eta_lower_bound:.2f}"
        )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    for axis, polarization in zip(axes, polarizations):
        rows = spectrum[spectrum["pol"] == polarization]
        axis.plot(
            rows["lambda_um"],
            rows["A"],
            color="#d62728",
            linewidth=1.7,
            label=r"$\alpha(+\theta)$",
        )
        axis.plot(
            rows["lambda_um"],
            rows["emissivity_same_channel"],
            color="#1f4eaa",
            linewidth=1.7,
            label=r"$e(+\theta)$",
        )
        axis.plot(
            rows["lambda_um"],
            rows["eta"],
            color="#202020",
            linestyle="--",
            linewidth=1.4,
            label=r"$\eta=|\alpha-e|$",
        )
        axis.axhline(
            paper_peak_eta_lower_bound,
            color="#777777",
            linestyle=":",
            linewidth=0.9,
        )
        axis.set_xlim(wavelength_min_um, wavelength_max_um)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"{polarization} polarization")
        axis.set_xlabel("Wavelength (um)")
        axis.set_ylabel(r"$\alpha$, $e$, $\eta$")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
        axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Peng and Wang (2025): Si toroidal cylinders / InAs / Ag, "
        rf"$\theta={theta_deg:g}^\circ$, $B=2.5$ T"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        "coordinate map: paper (x, y, z_up) -> solver (x, -y, z_stack); "
        f"theta=+/-{theta_deg:g} deg, phi={phi_deg:g} deg"
    )
    with asyrcwa.timed_step("solve Peng and Wang six-band spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot Peng and Wang spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
