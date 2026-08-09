"""Reproduce Liu et al., JQSRT 300 (2023) 108528, Fig. 3.

Article: "Realization of the nonreciprocal thermal radiation by the front
and back sides of the Weyl semimetal-dielectric multilayer structures".
DOI: 10.1016/j.jqsrt.2023.108528.

The structure is uniform in x and y, so the local full-tensor RCWA backend
reduces exactly to its zero-order Berreman/TMM route.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import solve_fr, summarize_fr


# Liu et al., Fig. 1, Eqs. (1)-(4), and Section 3.
high_index = 4.1
low_index = 1.2
high_layer_height_um = 0.2439
low_layer_height_um = 1.25
wsm_layer_height_um = 1.0

# The paper writes the stack as U W U^N W U, where U=(H L).  Its optimized
# main spectrum is the N=8 case identified explicitly again in Fig. 11.
stack_period_count = 8

temperature_k = 300.0
theta_deg = 45.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# No lateral pattern exists.  Retaining only G=(0,0) is the exact uniform
# multilayer problem, not a Fourier-order approximation.
orders = (0, 0)
lattice_um = (1.0, 1.0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 3.75
wavelength_max_um = 5.00
wavelength_point_count = 5001
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_wsm_parameters = {
    "node_separation_m_inv": 8.5e8,
    "node_separation_direction": (0.0, 1.0, 0.0),
    "fermi_velocity_m_s": 0.85e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": temperature_k,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    # The paper states E_F=0.30 eV at T=300 K.  It is therefore supplied as
    # the finite-temperature chemical potential used in Eqs. (2)-(3), rather
    # than reinterpreted as E_F(0) in Eq. (4).
    "fermi_energy_ev": 0.30,
}

# The rounded paper parameters do not reproduce every labelled feature at
# once.  Keep each reproducibility mode explicit: "paper_stated" uses only
# printed values; "gamma_calibrated" inverse-fits the Fig. 2 maximum; and
# "spectrum_calibrated" uses the same effective v_F as the independent
# PyLlama reproduction to align the main Fig. 3 resonances.  Neither
# calibrated mode is claimed to contain the paper's unrounded inputs.
wsm_parameter_mode = "spectrum_calibrated"
gamma_calibrated_fermi_energy_ev = 0.300835620
gamma_calibrated_tau_fs = 956.790988
spectrum_calibrated_fermi_velocity_m_s = 0.835e5

wsm_parameters = dict(paper_wsm_parameters)
if wsm_parameter_mode == "gamma_calibrated":
    wsm_parameters.update(
        {
            "fermi_energy_ev": gamma_calibrated_fermi_energy_ev,
            "tau_fs": gamma_calibrated_tau_fs,
        }
    )
elif wsm_parameter_mode == "spectrum_calibrated":
    wsm_parameters["fermi_velocity_m_s"] = (
        spectrum_calibrated_fermi_velocity_m_s
    )
elif wsm_parameter_mode != "paper_stated":
    raise ValueError(
        "wsm_parameter_mode must be paper_stated, gamma_calibrated, "
        "or spectrum_calibrated"
    )
wsm_parameter_labels = {
    "paper_stated": "paper-stated parameters",
    "gamma_calibrated": "Fig. 2 gamma-calibrated local model",
    "spectrum_calibrated": "Fig. 3 spectrum-calibrated local model",
}
wsm_parameter_label = wsm_parameter_labels[wsm_parameter_mode]

# Figure 3 annotations.  The values are not used by the solver.
paper_features = {
    "front": {
        "emission": ((3.890, 0.915),),
        "absorption": ((4.692, 0.989),),
        "eta": ((3.890, 0.908), (4.692, 0.931)),
    },
    "back": {
        "emission": ((4.692, 0.999),),
        "absorption": ((3.896, 0.990),),
        "eta": ((3.896, 0.959), (4.692, 0.953), (4.832, 0.953)),
    },
}
peak_search_halfwidth_um = 0.04

plot_filenames = {
    "paper_stated": "liu_2023_wsm_dielectric_multilayer_fig3.png",
    "gamma_calibrated": (
        "liu_2023_wsm_dielectric_multilayer_fig3_gamma_calibrated.png"
    ),
    "spectrum_calibrated": (
        "liu_2023_wsm_dielectric_multilayer_fig3_spectrum_calibrated.png"
    ),
}
plot_filename = plot_filenames[wsm_parameter_mode]
plot_path = project_root / "images" / plot_filename
figure_size = (12.0, 8.0)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_unit_cell(
    simulation: object,
    side: str,
    unit_index: int,
) -> None:
    materials = ("H", "L") if side == "front" else ("L", "H")
    thicknesses_um = {
        "H": high_layer_height_um,
        "L": low_layer_height_um,
    }
    for position, material in enumerate(materials):
        simulation.AddLayer(
            f"U{unit_index}_{position}_{material}",
            thicknesses_um[material],
            material,
        )


def build_simulation(rcwa: object, side: str) -> object:
    if side not in {"front", "back"}:
        raise ValueError("side must be 'front' or 'back'")
    if stack_period_count < 0:
        raise ValueError("stack_period_count must be non-negative")

    simulation = rcwa.New(
        Lattice=lattice_um,
        Orders=orders,
    )
    simulation.SetMaterial("Air", 1.0 + 0.0j)
    simulation.SetMaterial("H", high_index**2 + 0.0j)
    simulation.SetMaterial("L", low_index**2 + 0.0j)
    simulation.SetMaterialModel("W", "WSM", wsm_parameters)
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")

    add_unit_cell(simulation, side, 0)
    simulation.AddLayer("W0", wsm_layer_height_um, "W")
    for unit_index in range(1, stack_period_count + 1):
        add_unit_cell(simulation, side, unit_index)
    simulation.AddLayer("W1", wsm_layer_height_um, "W")
    add_unit_cell(simulation, side, stack_period_count + 1)
    return simulation


def solve_side_spectrum(side: str) -> np.ndarray:
    return solve_fr(
        rcwa=asyrcwa,
        build_simulation=lambda module: build_simulation(module, side),
        wavelengths_um=wavelengths_um,
        polarizations=polarizations,
        angles_deg=angles_deg,
        phi_deg=phi_deg,
        workers_per_angle=workers_per_angle,
    )


def peak_near(
    spectrum: np.ndarray,
    observable: str,
    reference_wavelength_um: float,
) -> np.void:
    mask = (
        np.abs(spectrum["lambda_um"] - reference_wavelength_um)
        <= peak_search_halfwidth_um
    )
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        raise RuntimeError("paper peak search window contains no wavelength samples")
    return spectrum[candidates[int(np.argmax(spectrum[observable][mask]))]]


def print_material_check() -> None:
    material_wavelengths_um = np.linspace(
        wavelength_min_um,
        wavelength_max_um,
        wavelength_point_count,
    )
    gamma = np.empty(len(material_wavelengths_um), dtype=float)
    for index, wavelength_um in enumerate(material_wavelengths_um):
        epsilon = asyrcwa.MaterialModelTensors(
            "WSM",
            float(wavelength_um),
            wsm_parameters,
        )["epsilon"]
        gamma[index] = abs(epsilon[0, 2]) / abs(epsilon[0, 0])
    peak_index = int(np.argmax(gamma))
    print(
        f"WSM parameters: {wsm_parameter_label}; "
        f"E_F={wsm_parameters['fermi_energy_ev']:.9f} eV, "
        f"tau={wsm_parameters['tau_fs']:.6f} fs, "
        f"v_F={wsm_parameters['fermi_velocity_m_s']:.6g} m/s\n"
        "WSM gamma peak: "
        f"{gamma[peak_index]:.6f} at "
        f"{material_wavelengths_um[peak_index]:.6f} um; "
        "paper 40.297 at 4.432 um"
    )


def print_comparison(side: str, spectrum: np.ndarray) -> None:
    observable_fields = {
        "absorption": "A",
        "emission": "emissivity_same_channel",
        "eta": "eta",
    }
    print(f"{side} side:")
    for observable, references in paper_features[side].items():
        field = observable_fields[observable]
        for paper_wavelength_um, paper_value in references:
            peak = peak_near(spectrum, field, paper_wavelength_um)
            print(
                f"  {observable}: {float(peak[field]):.6f} at "
                f"{float(peak['lambda_um']):.6f} um; paper "
                f"{paper_value:.4f} at {paper_wavelength_um:.3f} um"
            )
    maximum_transmission = max(
        float(np.max(spectrum["T"])),
        float(np.max(spectrum["T_reverse"])),
    )
    emission_proxy_error = float(
        np.max(
            np.abs(
                spectrum["emissivity_same_channel"]
                - spectrum["A_reverse"]
            )
        )
    )
    print(
        f"  max T={maximum_transmission:.3e}; "
        f"max |exact e-alpha(-theta)|={emission_proxy_error:.3e}"
    )


def plot_spectra(spectra: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    side_titles = {"front": "Front side", "back": "Back side"}
    for column, side in enumerate(("front", "back")):
        spectrum = spectra[side]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(
            spectrum["lambda_um"],
            spectrum["A"],
            color="#e53935",
            linewidth=1.8,
            label=rf"$\alpha(+{theta_deg:g}^\circ)$",
        )
        top.plot(
            spectrum["lambda_um"],
            spectrum["emissivity_same_channel"],
            color="#2455d6",
            linewidth=1.8,
            label=rf"$e(+{theta_deg:g}^\circ)$",
        )
        bottom.plot(
            spectrum["lambda_um"],
            spectrum["eta"],
            color="#f06292",
            linewidth=1.8,
            label=r"$\eta=|\alpha-e|$",
        )
        for observable, references in paper_features[side].items():
            axis = bottom if observable == "eta" else top
            for wavelength_um, value in references:
                axis.scatter(
                    [wavelength_um],
                    [value],
                    marker="x",
                    s=42,
                    linewidths=1.3,
                    color="#202020",
                    zorder=3,
                )
        top.set_title(side_titles[side])
        top.set_ylabel(r"$\alpha$ and $e$")
        bottom.set_ylabel(r"Nonreciprocity $\eta$")
        bottom.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        top.legend(frameon=False, fontsize=8)
        bottom.legend(frameon=False, fontsize=8)
        for axis in (top, bottom):
            axis.set_xlim(wavelength_min_um, wavelength_max_um)
            axis.set_ylim(-0.03, 1.03)
            axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)

    fig.suptitle(
        "Weyl-semimetal/dielectric multilayer, "
        f"N={stack_period_count}, TM, theta=+/-{theta_deg:g} deg\n"
        f"{wsm_parameter_label}"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> dict[str, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print_material_check()
    spectra: dict[str, np.ndarray] = {}
    for side in ("front", "back"):
        with asyrcwa.timed_step(f"solve Liu et al. Fig. 3 {side} side"):
            spectrum = solve_side_spectrum(side)
        spectra[side] = spectrum
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_comparison(side, spectrum)
    with asyrcwa.timed_step("plot Liu et al. Fig. 3"):
        plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
