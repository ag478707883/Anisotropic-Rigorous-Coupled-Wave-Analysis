"""Reproduce Wang and Qi, IJHMT 135 (2019) 142-148, Fig. 6(a) and Fig. 7(a).

Article: "Omnidirectional infrared nonreciprocal absorbers based on CdTe
gratings".
DOI: 10.1016/j.ijheatmasstransfer.2019.01.126.

The paper used FDTD with a metal boundary and converted the CdTe gyrotropic
response to equivalent anisotropic refractive indices. This example uses the
current project's full tensor CdTeLorentzDrude material in RCWA, with a finite
0.2-um Ag reflector under the CdTe planar layer.
"""

from __future__ import annotations

import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Paper Fig. 5 and Section 3.
period_um = 7.24
ridge_width_um = 3.258
grating_height_um = 4.0
base_height_um = 0.6
metal_height_um = 0.2

# Paper optical parameters for N=3e17 cm^-3.
cdte_base_parameters = {
    "eps_inf": 7.1,
    "carrier_density_cm3": 3.0e17,
    "effective_mass_ratio": 0.09,
    "transverse_resonance_rad_s": 2.652e13,
    "damping_rate_rad_s": 1.24e12,
}

ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 45.0
phi_deg = 0.0
polarization = "TM"
angles_deg = (+theta_deg, -theta_deg)
magnetic_fields_t = (0.0, 3.0, 6.0)

# This 1D lamellar grating retains 71 harmonics.
orders = (35, 0)
workers = 14

wavelength_min_um = 18.0
wavelength_max_um = 23.0
wavelength_point_count = 401
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_monitor_wavelength_um = 18.3
plot_path = project_root / "images" / "wang_2019_cdte_grating_fig6_fig7.png"
data_path = project_root / "data" / "wang_2019_cdte_grating_fig6_fig7.csv"
figure_size = (12.0, 4.8)
figure_dpi = 240
show_plot = False

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(magnetic_field_t: float) -> object:
    simulation = asyrcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialModel(
        "CdTe",
        "CdTeLorentzDrude",
        {**cdte_base_parameters, "magnetic_field_t": magnetic_field_t},
    )
    simulation.SetMaterialModel("Ag", "Drude", ag_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    simulation.AddLayer("CdTe_grating", grating_height_um, "Air")
    simulation.SetRegionRectangle(
        "CdTe_grating",
        "CdTe",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("CdTe_planar_layer", base_height_um, "CdTe")
    simulation.AddLayer("Ag_reflector", metal_height_um, "Ag")
    return simulation


def rows_for_angle(rows: np.ndarray, theta: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], theta, rtol=0.0, atol=1.0e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_case(magnetic_field_t: float) -> dict[float, np.ndarray]:
    rows = build_simulation(magnetic_field_t).GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    spectra: dict[float, np.ndarray] = {}
    for angle in angles_deg:
        angle_rows = rows_for_angle(rows, angle)
        if len(angle_rows) != wavelength_point_count:
            raise RuntimeError(f"incomplete spectrum for theta={angle:g} deg")
        spectra[angle] = angle_rows
    return spectra


def absorptivity(rows: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(rows["R"], dtype=float) - np.asarray(rows["T"], dtype=float)


def solve_spectra() -> dict[float, dict[float, np.ndarray]]:
    return {
        magnetic_field_t: solve_case(magnetic_field_t)
        for magnetic_field_t in magnetic_fields_t
    }


def print_summary(spectra: dict[float, dict[float, np.ndarray]]) -> None:
    print(
        f"K={period_um:g} um, W={ridge_width_um:g} um, "
        f"d1={grating_height_um:g} um, d2={base_height_um:g} um, "
        f"Ag={metal_height_um:g} um, orders={orders}, "
        f"points={wavelength_point_count}"
    )
    for magnetic_field_t in magnetic_fields_t:
        for angle in angles_deg:
            rows = spectra[magnetic_field_t][angle]
            absorption = absorptivity(rows)
            peak_index = int(np.argmax(absorption))
            monitor_index = int(
                np.argmin(np.abs(rows["lambda_um"] - paper_monitor_wavelength_um))
            )
            print(
                f"B={magnetic_field_t:g} T, theta={angle:+g} deg: "
                f"peak {float(rows['lambda_um'][peak_index]):.3f} um, "
                f"A={float(absorption[peak_index]):.4f}; "
                f"A({paper_monitor_wavelength_um:.1f} um)="
                f"{float(absorption[monitor_index]):.4f}; "
                f"max T={float(np.max(rows['T'])):.2e}"
            )


def export_csv(spectra: dict[float, dict[float, np.ndarray]]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "wavelength_um",
                "A_B0_theta_plus",
                "A_B0_theta_minus",
                "A_B3_theta_plus",
                "A_B3_theta_minus",
                "A_B6_theta_plus",
                "A_B6_theta_minus",
            ]
        )
        for index, wavelength_um in enumerate(wavelengths_um):
            writer.writerow(
                [
                    wavelength_um,
                    absorptivity(spectra[0.0][+theta_deg])[index],
                    absorptivity(spectra[0.0][-theta_deg])[index],
                    absorptivity(spectra[3.0][+theta_deg])[index],
                    absorptivity(spectra[3.0][-theta_deg])[index],
                    absorptivity(spectra[6.0][+theta_deg])[index],
                    absorptivity(spectra[6.0][-theta_deg])[index],
                ]
            )
    print(f"Saved {data_path}.")


def plot_panel(
    axis: plt.Axes,
    spectra: dict[float, dict[float, np.ndarray]],
    magnetic_field_t: float,
    title: str,
) -> None:
    zero_rows = spectra[0.0][+theta_deg]
    plus_rows = spectra[magnetic_field_t][+theta_deg]
    minus_rows = spectra[magnetic_field_t][-theta_deg]
    axis.plot(
        zero_rows["lambda_um"],
        absorptivity(zero_rows),
        color="#202020",
        linewidth=1.8,
        label=r"$\theta=+45^\circ$, $B=0$ T",
    )
    axis.plot(
        plus_rows["lambda_um"],
        absorptivity(plus_rows),
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=rf"$\theta=+45^\circ$, $B={magnetic_field_t:g}$ T",
    )
    axis.plot(
        minus_rows["lambda_um"],
        absorptivity(minus_rows),
        color="#1f4eaa",
        linestyle=":",
        linewidth=1.9,
        label=rf"$\theta=-45^\circ$, $B={magnetic_field_t:g}$ T",
    )
    axis.axvline(
        paper_monitor_wavelength_um,
        color="#2ca02c",
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        label="monitoring point",
    )
    axis.set_title(title)
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Wavelength (um)")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False, fontsize=8, loc="lower left")


def plot_spectra(spectra: dict[float, dict[float, np.ndarray]]) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        sharey=True,
        constrained_layout=True,
    )
    plot_panel(axes[0], spectra, 3.0, "(a) Magnetic field B=3 T")
    plot_panel(axes[1], spectra, 6.0, "(b) Magnetic field B=6 T")
    axes[0].set_ylabel("Absorptivity")
    figure.suptitle(
        "Wang and Qi (2019): CdTe grating nonreciprocal absorption, TM"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[float, dict[float, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Wang and Qi CdTe grating spectra"):
        spectra = solve_spectra()
        print_summary(spectra)
    export_csv(spectra)
    with asyrcwa.timed_step("plot Fig. 6(a)/Fig. 7(a)"):
        plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
