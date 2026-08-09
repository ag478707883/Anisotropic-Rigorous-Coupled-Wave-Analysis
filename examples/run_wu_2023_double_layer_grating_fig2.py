"""Reproduce Wu and Qing, Adv. Compos. Hybrid Mater. 6 (2023) 87, Fig. 2.

The aligned Al/InAs double-layer grating is invariant along solver y,
periodic along solver x, and stacked along solver +z. Both patterned slices
use the backend's analytic one-dimensional lamellar Fourier route.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Wu and Qing, Fig. 1, Fig. 2, and the baseline values in Fig. 6.
period_um = 6.20
ridge_fill_factor = 0.76
ridge_width_um = ridge_fill_factor * period_um
al_cap_height_um = 0.24
inas_grating_height_um = 3.30
inas_base_height_um = 5.62
# Fig. 1 shows an opaque Al mirror on SiO2 but the freely accessible figures
# do not label its thickness. 0.20 um is already opaque in this band; replacing
# it by a semi-infinite Al half-space gives the same spectrum numerically.
al_mirror_height_um = 0.20
sio2_refractive_index = 1.45

# Local and paper coordinates coincide: x is periodic, y follows the grating
# ridges and magnetic bias, and +z follows the layer stack from Air to Al.
theta_deg = 1.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# The subscription preview does not expose the sentence specifying the main
# bias. The 3-T value is inferred from the article's smaller-field comparison
# and the standard InAs benchmark used by the authors. Solver +y gives the
# Fig. 2 absorptivity/emissivity ordering; reversing B swaps those channels.
inas_parameters = {
    "magnetic_field_t": 3.0,
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

# The paper uses a numerical full-wave solver and does not report its Fourier
# order. This centered 1D basis retains 81 analytic lamellar harmonics.
orders = (40, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.50
wavelength_max_um = 16.50
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peaks_um = (15.88, 16.09)
paper_emission_peak_um = 16.00
paper_emission_peak = 0.99
paper_eta_peak = 0.90
paper_markers = {
    "TM": tuple(sorted(paper_absorption_peaks_um + (paper_emission_peak_um,)))
}

plot_path = project_root / "images" / "wu_2023_double_layer_grating_fig2.png"
figure_size = (7.4, 5.0)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < ridge_fill_factor < 1.0:
        raise ValueError("ridge_fill_factor must lie between zero and one")

    sim = rcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    sim.SetMaterialModel("Air", "Air")
    sim.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("SiO2")

    sim.AddLayer("Al_cap_grating", al_cap_height_um, "Air")
    sim.SetRegionRectangle(
        "Al_cap_grating",
        "Al",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_fill_factor * period_um, 0.5 * period_um),
    )
    sim.AddLayer("InAs_grating", inas_grating_height_um, "Air")
    sim.SetRegionRectangle(
        "InAs_grating",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_fill_factor * period_um, 0.5 * period_um),
    )
    sim.AddLayer("InAs_base", inas_base_height_um, "InAs")
    sim.AddLayer("Al_mirror", al_mirror_height_um, "Al")
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


def print_paper_comparison(spectrum: np.ndarray) -> None:
    rows = spectrum[spectrum["pol"] == "TM"]
    emission_peak = rows[int(np.argmax(rows["emissivity_same_channel"]))]
    eta_peak = rows[int(np.argmax(rows["eta"]))]
    local_absorption_indices = np.flatnonzero(
        (rows["A"][1:-1] > rows["A"][:-2])
        & (rows["A"][1:-1] >= rows["A"][2:])
    ) + 1
    ranked = local_absorption_indices[
        np.argsort(rows["A"][local_absorption_indices])[::-1]
    ][:2]
    ranked = ranked[np.argsort(rows["lambda_um"][ranked])]
    for index, paper_wavelength_um in zip(ranked, paper_absorption_peaks_um):
        print(
            f"absorption peak {rows['lambda_um'][index]:.4f} um, "
            f"alpha={rows['A'][index]:.4f}; paper about "
            f"{paper_wavelength_um:.2f} um"
        )
    print(
        f"emission peak {emission_peak['lambda_um']:.4f} um, "
        f"e={emission_peak['emissivity_same_channel']:.4f}; paper "
        f"{paper_emission_peak_um:.2f} um, e~{paper_emission_peak:.2f}"
    )
    print(
        f"max eta={eta_peak['eta']:.4f} at {eta_peak['lambda_um']:.4f} um; "
        f"paper about {paper_eta_peak:.2f}"
    )


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    ax.set_title(
        r"Wu and Qing (2023), Fig. 2: Al/InAs double grating, "
        rf"$\theta={theta_deg:g}^\circ$, $B={inas_parameters['magnetic_field_t']:g}$ T"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"coordinates: x periodic, y ridges/bias, z stack; period={period_um:g} um, "
        f"ridge width={ridge_width_um:g} um"
    )
    with asyrcwa.timed_step("solve Wu and Qing Fig. 2 spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_paper_comparison(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
