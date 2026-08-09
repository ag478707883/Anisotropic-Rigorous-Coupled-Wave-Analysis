"""Reproduce TE/TM spectra and nonreciprocity from one published article.

Article: "Near-infrared violation of Kirchhoff's law of thermal radiation at
near-zero angle".
Authors: Yuqing Xu, Bo Wang, and Jing Ye.
Journal: International Journal of Thermal Sciences 211 (2025) 109752.
DOI: 10.1016/j.ijthermalsci.2025.109752.

This script computes only the directional TE/TM absorptivity, emissivity, and
nonreciprocity eta = abs(alpha - emissivity) needed for the paper spectrum.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Xu, Wang, and Ye, Fig. 2 and Tables 1-2.
#
# The incidence plane is x-z (beta=0). The paper's layer-normal +z points
# toward Air, opposite to solver +z, so its +y axial magnetization maps to
# solver -y while finite layers follow AddLayer order toward the substrate.
period_um = 1.2
ring_outer_diameter_ratio = 0.60
ring_inner_diameter_ratio = 0.24
ring_outer_diameter_um = ring_outer_diameter_ratio * period_um
ring_inner_diameter_um = ring_inner_diameter_ratio * period_um
ring_outer_radius_um = 0.5 * ring_outer_diameter_um
ring_inner_radius_um = 0.5 * ring_inner_diameter_um

ring_height_um = 0.10
ceyig_height_um = 0.60
ag_height_um = 0.20

si_refractive_index = 3.48
ceyig_diagonal_epsilon = 4.0
ceyig_offdiagonal_b = 0.10
ceyig_magnetization_direction = (0.0, -1.0, 0.0)
ceyig_parameters = {
    "diagonal_epsilon": 4,
    "offdiagonal_gyration": 0.1,
    "magnetization_direction": ceyig_magnetization_direction,
}
ag_drude_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 2.0
phi_deg = 0.0
angles_deg = (+theta_deg, -theta_deg)
polarizations = ("TE", "TM")
orders = (6, 6)
workers = 10
# Fourier convergence controls. "Default" preserves the original example result.

wavelength_min_um = 1.640
wavelength_max_um = 1.740
wavelength_point_count = 1001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

# Paper Table 2 checkpoints. Figure 1(b) draws w1 and w2 across the complete
# outer and inner openings, so the tabulated f1*d and f2*d are treated as
# diameters even though the surrounding prose calls them radii.
paper_absorption_peaks_um = {
    "TE": (1.6734, 1.6968),
    "TM": (1.6708, 1.6943),
}
paper_emission_peaks_um = {
    "TE": (1.6590, 1.7109),
    "TM": (1.6565, 1.7084),
}

plot_path = project_root / "images" / "xu_2025_ceyig_cylindrical_ring_fig2.png"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation() -> object:
    simulation = asyrcwa.New(
        Lattice=(period_um, period_um),
        Orders=orders,
    )

    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("Si", si_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("CeYIG", "YIG", ceyig_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_drude_parameters)

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Si")

    simulation.AddLayer("cylindrical_ring", ring_height_um, "Air")
    simulation.SetRegionCircle(
        "cylindrical_ring",
        "Si",
        Center=(0.0, 0.0),
        Radius=ring_outer_radius_um,
    )
    simulation.SetRegionCircle(
        "cylindrical_ring",
        "Air",
        Center=(0.0, 0.0),
        Radius=ring_inner_radius_um,
    )
    simulation.AddLayer("CeYIG", ceyig_height_um, "CeYIG")
    simulation.AddLayer("Ag_reflector", ag_height_um, "Ag")
    return simulation


def rows_for_channel(
    rows: np.ndarray,
    theta: float,
    polarization: str,
) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], theta, rtol=0.0, atol=1e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_fig2_spectra() -> dict[str, dict[str, np.ndarray]]:
    simulation = build_simulation()
    _, thermal_rows = (
        simulation.GetSpectrumAndDirectionalThermalChannelsForAngles(
            wavelengths_um,
            angles_deg,
            polarizations,
            Phi=phi_deg,
            Workers=workers,
        )
    )

    spectra: dict[str, dict[str, np.ndarray]] = {}
    for polarization in polarizations:
        forward = rows_for_channel(thermal_rows, +theta_deg, polarization)
        reverse = rows_for_channel(thermal_rows, -theta_deg, polarization)
        if len(forward) != len(reverse) or len(forward) != wavelength_point_count:
            raise RuntimeError(f"incomplete {polarization} spectrum returned by backend")
        if not np.allclose(
            forward["lambda_um"],
            reverse["lambda_um"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"misaligned {polarization} forward/reverse wavelengths")

        absorptivity = np.asarray(forward["absorptivity"], dtype=float)
        emissivity = np.asarray(reverse["emissivity"], dtype=float)
        spectra[polarization] = {
            "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
            "absorptivity": absorptivity,
            "emissivity": emissivity,
            "nonreciprocity": np.abs(absorptivity - emissivity),
        }
    return spectra


def strongest_local_peaks(
    wavelengths: np.ndarray,
    values: np.ndarray,
    count: int,
) -> list[tuple[float, float]]:
    indices = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    ranked = indices[np.argsort(values[indices])[::-1]]
    selected = sorted(ranked[:count], key=lambda index: wavelengths[index])
    return [(float(wavelengths[index]), float(values[index])) for index in selected]


def print_summary(spectra: dict[str, dict[str, np.ndarray]]) -> None:
    print(
        f"orders={orders}, retained wavelength points={wavelength_point_count}, "
        f"theta=+/-{theta_deg:g} deg, phi={phi_deg:g} deg"
    )
    for polarization in polarizations:
        spectrum = spectra[polarization]
        wavelength = spectrum["wavelength_um"]
        absorptivity = spectrum["absorptivity"]
        emissivity = spectrum["emissivity"]
        nonreciprocity = spectrum["nonreciprocity"]
        print(f"{polarization}:")
        print(
            "  strongest absorptivity peaks: "
            + ", ".join(
                f"{peak_um * 1000.0:.1f} nm ({value:.4f})"
                for peak_um, value in strongest_local_peaks(
                    wavelength, absorptivity, 2
                )
            )
        )
        print(
            "  strongest emissivity peaks: "
            + ", ".join(
                f"{peak_um * 1000.0:.1f} nm ({value:.4f})"
                for peak_um, value in strongest_local_peaks(
                    wavelength, emissivity, 2
                )
            )
        )
        max_eta_index = int(np.argmax(nonreciprocity))
        print(
            f"  max eta={nonreciprocity[max_eta_index]:.4f} at "
            f"{wavelength[max_eta_index] * 1000.0:.1f} nm"
        )


def add_reference_markers(axis: plt.Axes, polarization: str) -> None:
    for wavelength_um in paper_absorption_peaks_um[polarization]:
        axis.axvline(
            wavelength_um * 1000.0,
            color="#202020",
            linewidth=0.7,
            alpha=0.22,
        )
    for wavelength_um in paper_emission_peaks_um[polarization]:
        axis.axvline(
            wavelength_um * 1000.0,
            color="#d62728",
            linewidth=0.7,
            alpha=0.22,
        )


def plot_fig2_spectra(spectra: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )

    for axis, polarization in zip(axes, polarizations):
        spectrum = spectra[polarization]
        wavelength_nm = spectrum["wavelength_um"] * 1000.0
        axis.plot(
            wavelength_nm,
            spectrum["absorptivity"],
            color="#202020",
            linewidth=1.8,
            label=rf"$\alpha(+{theta_deg:g}^\circ)$",
        )
        axis.plot(
            wavelength_nm,
            spectrum["emissivity"],
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=rf"$e(+{theta_deg:g}^\circ)$",
        )
        axis.plot(
            wavelength_nm,
            spectrum["nonreciprocity"],
            color="#ff8c00",
            linestyle=":",
            linewidth=1.7,
            label=rf"$\eta=|\alpha-e|$",
        )
        add_reference_markers(axis, polarization)
        axis.axhline(0.95, color="#7b2cbf", linestyle=":", linewidth=1.0)
        axis.set_title(f"{polarization} polarization")
        axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlim(wavelength_min_um * 1000.0, wavelength_max_um * 1000.0)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Intensity")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)

    fig.suptitle("TE/TM spectra and nonreciprocity - Xu et al. (2025) Fig. 2")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[str, dict[str, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectra = solve_fig2_spectra()
    print_summary(spectra)
    plot_fig2_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
