"""Reproduce Zou et al., Optical Materials 83 (2018) 28-33, Fig. 3.

Article: "Enhancement of absorption of molybdenum disulfide monolayer on
low-index contrast dielectric grating in the visible regions".
DOI: 10.1016/j.optmat.2018.05.081.

The paper uses RCWA with 103 diffraction orders. This example models the
monolayer MoS2 as a uniform 0.65-nm dispersive film above the lamellar
low-index-contrast grating and compares the TE/TM spectra with and without
that film.
"""

from __future__ import annotations

import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Zou et al., Fig. 1 and Section 2.
period_um = 0.40
ridge_width_um = 0.16
grating_height_um = 0.18
si_spacer_height_um = 0.04
mos2_height_um = 0.00065

sf11_refractive_index = 1.78
si3n4_refractive_index = 2.03
si_spacer_refractive_index = 3.96
sio2_substrate_refractive_index = 1.45

theta_deg = 0.0
phi_deg = 0.0
polarizations = ("TE", "TM")

orders = (51, 0)
workers = 20

wavelength_min_um = 0.40
wavelength_max_um = 0.80
wavelength_point_count = 1000
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_resonant_wavelength_um = {
    "TE": 0.59,
    "TM": 0.69,
}

plot_path = project_root / "images" / "zou_2018_mos2_low_index_grating_fig3.png"
data_path = project_root / "data" / "zou_2018_mos2_low_index_grating_fig3.csv"
figure_size = (9.0, 6.8)
figure_dpi = 240

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(with_mos2: bool) -> object:
    simulation = asyrcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("SF11", sf11_refractive_index**2 + 0.0j)
    simulation.SetMaterial("Si3N4", si3n4_refractive_index**2 + 0.0j)
    simulation.SetMaterial("Si", si_spacer_refractive_index**2 + 0.0j)
    simulation.SetMaterial("SiO2", sio2_substrate_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel("MoS2", "MoS2")
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("SiO2")

    if with_mos2:
        simulation.AddLayer("MoS2_monolayer", mos2_height_um, "MoS2")

    simulation.AddLayer("low_index_contrast_grating", grating_height_um, "SF11")
    simulation.SetRegionRectangle(
        "low_index_contrast_grating",
        "Si3N4",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * ridge_width_um, 0.5 * period_um),
    )
    simulation.AddLayer("Si_spacer", si_spacer_height_um, "Si")
    simulation.SetExcitationPlanewave((theta_deg, phi_deg))
    return simulation


def rows_for_polarization(rows: np.ndarray, polarization: str) -> np.ndarray:
    selected = rows[rows["pol"] == polarization]
    selected = np.sort(selected, order="lambda_um")
    if len(selected) != wavelength_point_count:
        raise RuntimeError(f"incomplete {polarization} spectrum")
    return selected


def solve_spectra() -> dict[str, dict[bool, np.ndarray]]:
    spectra: dict[str, dict[bool, np.ndarray]] = {
        polarization: {} for polarization in polarizations
    }
    for with_mos2 in (True, False):
        rows = build_simulation(with_mos2).GetSpectrum(
            wavelengths_um,
            polarizations,
            Workers=workers,
        )
        for polarization in polarizations:
            spectra[polarization][with_mos2] = rows_for_polarization(
                rows,
                polarization,
            )
    return spectra


def absorptance(rows: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(rows["R"], dtype=float) - np.asarray(rows["T"], dtype=float)


def print_summary(spectra: dict[str, dict[bool, np.ndarray]]) -> None:
    print(
        f"p={period_um:g} um, w={ridge_width_um:g} um, "
        f"h={grating_height_um:g} um, t={si_spacer_height_um:g} um, "
        f"MoS2={mos2_height_um * 1000:g} nm, orders={orders}, "
        f"points={wavelength_point_count}"
    )
    for polarization in polarizations:
        with_rows = spectra[polarization][True]
        without_rows = spectra[polarization][False]
        absorption = absorptance(with_rows)
        reflectance_without = np.asarray(without_rows["R"], dtype=float)
        absorption_peak = int(np.argmax(absorption))
        reflection_peak = int(np.argmax(reflectance_without))
        paper_wavelength = paper_resonant_wavelength_um[polarization]
        paper_index = int(np.argmin(abs(with_rows["lambda_um"] - paper_wavelength)))
        print(
            f"{polarization}: with MoS2 peak A="
            f"{float(absorption[absorption_peak]):.4f} at "
            f"{float(with_rows['lambda_um'][absorption_peak]):.4f} um; "
            f"A({paper_wavelength:.2f} um)="
            f"{float(absorption[paper_index]):.4f}"
        )
        print(
            f"{polarization}: without MoS2 max R="
            f"{float(reflectance_without[reflection_peak]):.4f} at "
            f"{float(without_rows['lambda_um'][reflection_peak]):.4f} um; "
            f"max A={float(np.max(absorptance(without_rows))):.2e}"
        )


def export_csv(spectra: dict[str, dict[bool, np.ndarray]]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        header = ["wavelength_um"]
        for polarization in polarizations:
            for case_name in ("with_mos2", "without_mos2"):
                header.extend(
                    [
                        f"R_{polarization}_{case_name}",
                        f"T_{polarization}_{case_name}",
                        f"A_{polarization}_{case_name}",
                    ]
                )
        writer.writerow(header)

        for index, wavelength_um in enumerate(wavelengths_um):
            row: list[float] = [float(wavelength_um)]
            for polarization in polarizations:
                for with_mos2 in (True, False):
                    rows = spectra[polarization][with_mos2]
                    row.extend(
                        [
                            float(rows["R"][index]),
                            float(rows["T"][index]),
                            float(absorptance(rows)[index]),
                        ]
                    )
            writer.writerow(row)
    print(f"Saved {data_path}.")


def plot_panel(
    axis: plt.Axes,
    rows: np.ndarray,
    title: str,
    *,
    show_legend: bool = False,
) -> None:
    wavelength = np.asarray(rows["lambda_um"], dtype=float)
    axis.plot(wavelength, rows["R"], color="#1f4eaa", linewidth=1.7, label="R")
    axis.plot(wavelength, rows["T"], color="#2ca02c", linewidth=1.7, label="T")
    axis.plot(
        wavelength,
        absorptance(rows),
        color="#d62728",
        linewidth=1.7,
        label="A",
    )
    axis.set_title(title)
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    if show_legend:
        axis.legend(frameon=False, loc="upper right")


def plot_spectra(spectra: dict[str, dict[bool, np.ndarray]]) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (axes[0, 0], "TE", True, "(a) TE with MoS2"),
        (axes[0, 1], "TE", False, "(b) TE without MoS2"),
        (axes[1, 0], "TM", True, "(c) TM with MoS2"),
        (axes[1, 1], "TM", False, "(d) TM without MoS2"),
    ]
    for axis, polarization, with_mos2, title in panels:
        plot_panel(
            axis,
            spectra[polarization][with_mos2],
            title,
            show_legend=(polarization == "TE" and not with_mos2),
        )

    axes[0, 0].set_ylabel("Intensity")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].set_xlabel("Wavelength (um)")
    axes[1, 1].set_xlabel("Wavelength (um)")
    figure.suptitle(
        "Zou et al. (2018): MoS2 on low-index-contrast dielectric grating"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.close(figure)


def main() -> dict[str, dict[bool, np.ndarray]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    with asyrcwa.timed_step("solve Zou et al. Fig. 3 spectra"):
        spectra = solve_spectra()
        print_summary(spectra)
    export_csv(spectra)
    with asyrcwa.timed_step("plot Fig. 3"):
        plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
