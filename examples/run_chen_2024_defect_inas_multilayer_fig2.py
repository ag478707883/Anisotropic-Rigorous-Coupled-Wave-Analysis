"""Reproduce Chen et al., Sci. China Tech. Sci. 67 (2024), Fig. 2(a,b)."""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Chen et al., Fig. 1, Fig. 2, Eqs. (1)-(7), and Section 3.
inas_height_um = 1.7
sio2_height_um = 3.1
defect_height_um = 7.4
al_height_um = 0.2
sio2_refractive_index = 1.45
pair_count = 6

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

theta_deg = 60.0
phi_deg = 0.0
polarization = "TM"
angles_deg = (+theta_deg, -theta_deg)
magnetic_fields_t = (0.0, 3.0)
workers = 8

wavelength_min_um = 14.0
wavelength_max_um = 17.0
wavelength_point_count = 3001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_reference_wavelengths_um = {
    "without_defect": (15.760, 16.035),
    "with_defect": (14.750, 16.295),
}
plot_path = project_root / "images" / "chen_2024_defect_inas_multilayer_fig2.png"
figure_size = (10.5, 4.4)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_pair(simulation: object, index: int) -> None:
    simulation.AddLayer(f"A_SiO2_{index}", sio2_height_um, "SiO2")
    simulation.AddLayer(f"C_InAs_{index}", inas_height_um, "InAs")


def build_simulation(with_defect: bool, magnetic_field_t: float) -> object:
    simulation = asyrcwa.New()
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterial("SiO2", sio2_refractive_index**2 + 0.0j)
    simulation.SetMaterialModel(
        "InAs",
        "InAsMagneto",
        {**inas_base_parameters, "magnetic_field_t": magnetic_field_t},
    )
    simulation.SetMaterialModel("Al", "Drude", al_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    leading_pairs = pair_count - 1 if with_defect else pair_count
    for index in range(leading_pairs):
        add_pair(simulation, index)
    if with_defect:
        simulation.AddLayer("D_InAs_defect", defect_height_um, "InAs")
        add_pair(simulation, pair_count - 1)
    simulation.AddLayer("M_Al", al_height_um, "Al")
    return simulation


def rows_at_angle(rows: np.ndarray, angle_deg: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], angle_deg, atol=1.0e-12, rtol=0.0)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_case(with_defect: bool, magnetic_field_t: float) -> dict[str, np.ndarray]:
    rows = build_simulation(with_defect, magnetic_field_t).GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    forward = rows_at_angle(rows, +theta_deg)
    reverse = rows_at_angle(rows, -theta_deg)
    if len(forward) != wavelength_point_count or len(reverse) != wavelength_point_count:
        raise RuntimeError("incomplete forward/reverse spectrum")
    maximum_transmission = max(float(np.max(forward["T"])), float(np.max(reverse["T"])))
    if maximum_transmission > 1.0e-8:
        raise RuntimeError("Al-backed structure is not numerically opaque")
    return {
        "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
        "absorptivity": 1.0 - np.asarray(forward["R"], dtype=float),
        "emissivity": 1.0 - np.asarray(reverse["R"], dtype=float),
    }


def print_comparison(label: str, spectrum: dict[str, np.ndarray]) -> None:
    wavelength = spectrum["wavelength_um"]
    contrast = np.abs(spectrum["absorptivity"] - spectrum["emissivity"])
    print(label)
    for reference_um in paper_reference_wavelengths_um[label]:
        index = int(np.argmin(np.abs(wavelength - reference_um)))
        print(
            f"  {wavelength[index]:.3f} um: alpha={spectrum['absorptivity'][index]:.4f}, "
            f"e={spectrum['emissivity'][index]:.4f}, |alpha-e|={contrast[index]:.4f}"
        )


def plot_spectra(spectra: dict[str, dict[float, dict[str, np.ndarray]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=figure_size, sharey=True, constrained_layout=True)
    titles = {
        "without_defect": r"(a) $(AC)^6M$",
        "with_defect": r"(b) $(AC)^5D(AC)M$",
    }
    styles = (
        (0.0, "absorptivity", "#555555", r"$\alpha$, 0 T"),
        (0.0, "emissivity", "#ef6c6c", r"$e$, 0 T"),
        (3.0, "absorptivity", "#2f6ee5", r"$\alpha$, 3 T"),
        (3.0, "emissivity", "#35a45b", r"$e$, 3 T"),
    )
    for axis, label in zip(axes, ("without_defect", "with_defect")):
        for field_t, observable, color, legend_label in styles:
            spectrum = spectra[label][field_t]
            axis.plot(
                spectrum["wavelength_um"],
                spectrum[observable],
                color=color,
                linewidth=1.6,
                label=legend_label,
            )
        axis.set_xlim(wavelength_min_um, wavelength_max_um)
        axis.set_ylim(-0.01, 1.02)
        axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        axis.set_title(titles[label])
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel(r"Absorptivity $\alpha$ / emissivity $e$")
    figure.suptitle(r"Chen et al. (2024), Fig. 2 — TM, $\theta=60^\circ$")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, dict[float, dict[str, np.ndarray]]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    spectra = {
        label: {
            field_t: solve_case(label == "with_defect", field_t)
            for field_t in magnetic_fields_t
        }
        for label in ("without_defect", "with_defect")
    }
    print_comparison("without_defect", spectra["without_defect"][3.0])
    print_comparison("with_defect", spectra["with_defect"][3.0])
    plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
