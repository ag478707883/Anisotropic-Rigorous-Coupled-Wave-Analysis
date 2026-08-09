"""Recompute Lao et al., IJHMT 267 (2026) 128918, Fig. 2 with Li factorization."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


period_um = 5.04
central_si_width_um = 2.94
ring_groove_width_um = 0.49
ring_outer_width_um = central_si_width_um + 2.0 * ring_groove_width_um
si_height_um = 4.8
inas_height_um = 2.2
al_height_um = 0.5
graphene_height_um = 0.00034

# The paper omits the Si optical constant; 3.42 is the standard mid-IR value.
si_refractive_index = 3.48
inas_base_parameters = {
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

theta_deg = 36.0
phi_deg = 0.0
polarizations = ("TE",)
angles_deg = (+theta_deg, -theta_deg)
orders = (5, 5)
workers = 20
# Fourier convergence controls. "Default" preserves the original example result.
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 13.0
wavelength_max_um = 13.8
wavelength_point_count = 401
wavelengths_um = np.linspace(wavelength_min_um, wavelength_max_um, wavelength_point_count)

cases = (
    {
        "label": "B=4 T, Ef=1.0 eV",
        "magnetic_field_t": -4.0,
        "fermi_energy_ev": 1.0,
        "paper_peaks_um": (13.452, 13.587),
        "paper_eta": (0.98, 0.93),
    },
    {
        "label": "B=0.8 T, Ef=0.1 eV",
        "magnetic_field_t": -0.8,
        "fermi_energy_ev": 0.1,
        "paper_peaks_um": (13.560, 13.589),
        "paper_eta": (0.86, 0.79),
    },
)

plot_path = project_root / "images" / "lao_2026_graphene_si_square_ring_fig2.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object, case: dict[str, object]) -> object:
    inas = dict(inas_base_parameters)
    inas["magnetic_field_t"] = case["magnetic_field_t"]
    graphene = {
        "fermi_energy_ev": case["fermi_energy_ev"],
        "tau_ps": 0.5,
        "thickness_nm": 0.34,
    }
    sim = rcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )
    sim.SetMaterial("Air", 1.0 + 0.0j)
    sim.SetMaterial("Si", si_refractive_index**2 + 0.0j)
    sim.SetMaterialModel("Graphene", "GrapheneKubo", graphene)
    sim.SetMaterialModel("InAs", "InAsMagneto", inas)
    sim.SetMaterialModel("Al", "Drude", al_parameters)
    sim.SetSuperstrate("Air")
    sim.SetSubstrate("Air")
    sim.AddLayer("Graphene", graphene_height_um, "Graphene")
    sim.AddLayer("Si_square_ring_groove", si_height_um, "Si")
    # Exact nested-square cells feed RETICOLO's ordered Li factorization.
    sim.SetRegionRectangle(
        "Si_square_ring_groove",
        "Air",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ring_outer_width_um, 0.5 * ring_outer_width_um),
    )
    sim.SetRegionRectangle(
        "Si_square_ring_groove",
        "Si",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * central_si_width_um, 0.5 * central_si_width_um),
    )
    sim.AddLayer("InAs", inas_height_um, "InAs")
    sim.AddLayer("Al_reflector", al_height_um, "Al")
    return sim


def solve_case(case: dict[str, object]) -> np.ndarray:
    return solve_fr(
        rcwa=asyrcwa,
        build_simulation=lambda rcwa: build_simulation(rcwa, case),
        wavelengths_um=wavelengths_um,
        polarizations=polarizations,
        angles_deg=angles_deg,
        phi_deg=phi_deg,
        workers_per_angle=workers_per_angle,
    )


def plot_spectra(spectra: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    for ax, spectrum, case in zip(axes, spectra, cases):
        plot_fr_axes(
            np.asarray([ax]),
            spectrum,
            polarizations,
            markers={"TE": case["paper_peaks_um"]},
        )
        ax.set_xlim(wavelength_min_um, wavelength_max_um)
        ax.set_title(str(case["label"]))
    fig.suptitle(f"Graphene / Si square-ring groove / InAs / Al, orders={orders}")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> list[np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectra = []
    with asyrcwa.timed_step("solve both Fig. 2 cases"):
        for case in cases:
            spectrum = solve_case(case)
            spectra.append(spectrum)
            print(case["label"])
            summarize_fr(spectrum, polarizations, include_peak_values=True)
            for wavelength, paper_eta in zip(case["paper_peaks_um"], case["paper_eta"]):
                row = spectrum[int(np.argmin(abs(spectrum["lambda_um"] - wavelength)))]
                print(f"  eta({wavelength:.3f} um)={row['eta']:.4f}; paper {paper_eta:.2f}")
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
