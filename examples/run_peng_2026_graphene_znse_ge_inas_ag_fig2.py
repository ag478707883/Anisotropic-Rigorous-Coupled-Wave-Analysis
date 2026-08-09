"""Reproduce TE/TM spectra and nonreciprocity from one published article.

Article: "Dynamical modulation for graphene-based highly nonreciprocal
emitter with TE-four/TM-ten bands".
Authors: Wenkun Peng, Bo Wang, and Yong He.
Journal: International Journal of Heat and Mass Transfer 261 (2026) 128546.
DOI: 10.1016/j.ijheatmasstransfer.2026.128546.

This script computes only the directional TE/TM absorptivity, emissivity, and
nonreciprocity eta = abs(alpha - emissivity) used for Fig. 2 and Table 2.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Paper geometry: Table 1 and Fig. 1. The paper's layer-normal +z points
# toward the incident side, opposite to solver +z; therefore its +y axial
# magnetic bias maps to solver -y for the InAs model.
period_um = 7.84
znse_cylinder_diameter_um = period_um
ge_cylinder_diameter_um = 7.74
znse_cylinder_radius_um = 0.5 * znse_cylinder_diameter_um
ge_cylinder_radius_um = 0.5 * ge_cylinder_diameter_um

graphene_height_um = 0.00034
znse_array_height_um = 7.0
ge_znse_composite_height_um = 2.0
inas_height_um = 6.5
ag_height_um = 0.4

znse_epsilon = 5.73
ge_refractive_index = 4.0

graphene_parameters = {
    "fermi_energy_ev": 0.1,
    "tau_ps": 1.0,
    "thickness_nm": 0.34,
}

# The article gives the InAs tensor equations and cites Refs. [44-46] for
# material constants, but does not print those constants in the paper. These
# explicit values are the project's standard InAsMagneto parameter set.
inas_parameters = {
    "magnetic_field_t": -1.5,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}

ag_drude_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 3.2
phi_deg = 0.0
angles_deg = (+theta_deg, -theta_deg)
polarizations = ("TE", "TM")
orders = (6, 6)
workers = 14
# Fourier convergence controls. "Default" preserves the original example result.

wavelength_min_um = 15.10
wavelength_max_um = 16.30
wavelength_point_count = 1201
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_absorption_peaks_um = {
    "TE": (15.725, 15.995),
    "TM": (15.254, 15.542, 15.752, 16.033, 16.246),
}
paper_emission_peaks_um = {
    "TE": (15.673, 16.023),
    "TM": (15.285, 15.511, 15.776, 16.086, 16.200),
}
paper_reference_peaks_um = {
    "TE": paper_absorption_peaks_um["TE"] + paper_emission_peaks_um["TE"],
    "TM": paper_absorption_peaks_um["TM"] + paper_emission_peaks_um["TM"],
}

plot_path = project_root / "images" / "peng_2026_graphene_znse_ge_inas_ag_fig2.png"
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
    simulation.SetMaterial("ZnSe", znse_epsilon + 0.0j)
    simulation.SetMaterial("Ge", ge_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("Graphene", "GrapheneKubo", graphene_parameters)
    simulation.SetMaterialModel("InAs", "InAsMagneto", inas_parameters)
    simulation.SetMaterialModel("Ag", "Drude", ag_drude_parameters)

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    # A finite effective layer represents the monolayer graphene used by the
    # paper's epsilon_g = 1 + i*sigma_g/(epsilon_0*omega*h) model.
    simulation.AddLayer("Graphene", graphene_height_um, "Graphene")

    # ZnSe cylinders have the paper's diameter P, so adjacent square-lattice
    # cylinders touch at the cell boundaries but leave corner gaps.
    simulation.AddLayer("ZnSe_cylinders", znse_array_height_um, "Air")
    simulation.SetRegionCircle(
        "ZnSe_cylinders",
        "ZnSe",
        Center=(0.0, 0.0),
        Radius=znse_cylinder_radius_um,
    )

    simulation.AddLayer("Ge_in_ZnSe", ge_znse_composite_height_um, "ZnSe")
    simulation.SetRegionCircle(
        "Ge_in_ZnSe",
        "Ge",
        Center=(0.0, 0.0),
        Radius=ge_cylinder_radius_um,
    )

    simulation.AddLayer("InAs", inas_height_um, "InAs")
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
    _, thermal_rows = simulation.GetSpectrumAndDirectionalThermalChannelsForAngles(
        wavelengths_um,
        angles_deg,
        polarizations,
        Phi=phi_deg,
        Workers=workers,
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
        f"theta=+/-{theta_deg:g} deg, phi={phi_deg:g} deg, "
        f"InAs B_solver={inas_parameters['magnetic_field_t']:g} T"
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
                f"{peak_um:.3f} um ({value:.4f})"
                for peak_um, value in strongest_local_peaks(
                    wavelength, absorptivity, len(paper_absorption_peaks_um[polarization])
                )
            )
        )
        print(
            "  strongest emissivity peaks: "
            + ", ".join(
                f"{peak_um:.3f} um ({value:.4f})"
                for peak_um, value in strongest_local_peaks(
                    wavelength, emissivity, len(paper_emission_peaks_um[polarization])
                )
            )
        )
        eta_index = int(np.argmax(nonreciprocity))
        print(
            f"  max eta={nonreciprocity[eta_index]:.4f} at "
            f"{wavelength[eta_index]:.3f} um"
        )


def add_reference_markers(axis: plt.Axes, polarization: str) -> None:
    for wavelength_um in paper_reference_peaks_um[polarization]:
        axis.axvline(
            wavelength_um,
            color="#777777",
            linewidth=0.7,
            alpha=0.28,
        )


def plot_fig2_spectra(spectra: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    x_limits = {"TE": (15.60, 16.3), "TM": (15.20, 16.30)}

    for axis, polarization in zip(axes, polarizations):
        spectrum = spectra[polarization]
        axis.plot(
            spectrum["wavelength_um"],
            spectrum["absorptivity"],
            color="#d62728",
            linewidth=1.8,
            label=r"$\alpha(+\theta)$",
        )
        axis.plot(
            spectrum["wavelength_um"],
            spectrum["emissivity"],
            color="#1f4eaa",
            linewidth=1.8,
            label=r"$e(+\theta)$",
        )
        axis.plot(
            spectrum["wavelength_um"],
            spectrum["nonreciprocity"],
            color="#202020",
            linestyle="--",
            linewidth=1.5,
            label=r"$\eta=|\alpha-e|$",
        )
        add_reference_markers(axis, polarization)
        axis.set_xlim(*x_limits[polarization])
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"{polarization} polarization")
        axis.set_xlabel("Wavelength (um)")
        axis.set_ylabel("Intensity")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
        axis.legend(frameon=False, fontsize=8)

    fig.suptitle("TE/TM spectra and nonreciprocity - Peng et al. (2026) Fig. 2")
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
