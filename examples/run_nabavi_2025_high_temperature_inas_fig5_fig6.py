"""Reproduce Nabavi et al. (arXiv:2502.07245v2), Figs. 5 and 6.

The paper considers a laterally uniform, opaque 525-um n-doped InAs slab in
air.  The theoretical curves are evaluated with the local RCWA multilayer
backend.  The temperature-dependent InAs parameters are the explicit model
from Eqs. (8)-(16) of the paper; the experimental points are not recreated
because their underlying measured data are not supplied with the PDF.

The paper's coordinates have the layer normal along ``y`` and the magnetic
field along ``z``.  The project convention uses layers along ``z`` and the
InAsMagneto model uses a Hall tensor in the ``x-z`` plane for a signed field
along ``+y``.  The in-plane rotation therefore maps the paper directly to
the project convention: ``theta`` is the x-z incidence angle and
``magnetic_field_t=+1`` represents the paper's +1 T case.  Reversing the sign
only swaps the +theta and -theta curves.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Nabavi et al., Eqs. (8)-(16), Fig. 5, and Fig. 6.
sample_thickness_um = 525.0
theta_deg = 30.0
phi_deg = 0.0
magnetic_field_t = 1.0
polarization = "TM"
angles_deg = (theta_deg, -theta_deg)
workers = 8

sample_doping_cm3 = {
    "ND1": 3.24e18,
    "ND2": 1.40e18,
}
temperatures_k = (448.0, 498.0, 548.0, 598.0)
reference_temperature_k = 598.0

# Temperature model constants from the paper.  Nc and Nv are in cm^-3,
# mobility is in cm^2/(V s), and the resulting damping rate is in rad/s.
bandgap_0_ev = 0.415
bandgap_varshni_a_ev_per_k = 2.76e-4
bandgap_varshni_b_k = 83.0
conduction_dos_prefactor_cm3_k32 = 1.68e13
valence_dos_prefactor_cm3_k32 = 1.27e15
boltzmann_ev_per_k = 8.617333262e-5
electron_mass_kg = 9.1093837015e-31
electron_charge_c = 1.602176634e-19
mobility_min_cm2_vs = 1000.0
mobility_max_cm2_vs = 34000.0
mobility_reference_density_cm3 = 1.1e18
mobility_psi1 = 1.57
mobility_psi2 = 3.0
mobility_zeta = 0.32
eps_inf_slope_per_k = 0.0018
eps_inf_intercept = 13.54

# Fig. 5 and Fig. 6 wavelength windows.  The paper uses a separate window for
# each doping level, so the broad temperature curves remain easy to inspect.
fig5_windows_um = {
    "ND1": (10.0, 15.0),
    "ND2": (14.5, 20.6),
}
fig6_windows_um = fig5_windows_um
spectral_point_count = 401

combined_plot_path = project_root / "images" / "nabavi_2025_inas_fig5_fig6.png"
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def temperature_parameters(doping_cm3: float, temperature_k: float) -> dict[str, float]:
    """Return InAsMagneto parameters from the paper's Eqs. (8)-(16)."""

    bandgap_ev = bandgap_0_ev - bandgap_varshni_a_ev_per_k * temperature_k**2 / (
        temperature_k + bandgap_varshni_b_k
    )
    conduction_dos_cm3 = conduction_dos_prefactor_cm3_k32 * temperature_k**1.5
    valence_dos_cm3 = valence_dos_prefactor_cm3_k32 * temperature_k**1.5
    intrinsic_density_cm3 = np.sqrt(conduction_dos_cm3 * valence_dos_cm3) * np.exp(
        -bandgap_ev / (2.0 * boltzmann_ev_per_k * temperature_k)
    )
    carrier_density_cm3 = doping_cm3 + intrinsic_density_cm3**2 / doping_cm3

    effective_mass_ratio = (
        0.0310
        - 0.0218 * bandgap_ev
        + 0.1 * bandgap_ev**2
        - 0.046 * bandgap_ev**3
    )
    mobility_cm2_vs = mobility_min_cm2_vs + (
        mobility_max_cm2_vs * (380.0 / temperature_k) ** mobility_psi1
        - mobility_min_cm2_vs
    ) / (
        1.0
        + (
            carrier_density_cm3
            / (mobility_reference_density_cm3 * (temperature_k / 300.0) ** mobility_psi2)
        )
        ** mobility_zeta
    )
    effective_mass_kg = effective_mass_ratio * electron_mass_kg
    damping_rate_rad_s = electron_charge_c / (
        mobility_cm2_vs * 1.0e-4 * effective_mass_kg
    )

    return {
        "magnetic_field_t": magnetic_field_t,
        "eps_inf": eps_inf_slope_per_k * temperature_k + eps_inf_intercept,
        "carrier_density_cm3": float(carrier_density_cm3),
        "effective_mass_ratio": float(effective_mass_ratio),
        "damping_rate_rad_s": float(damping_rate_rad_s),
    }


def build_simulation(doping_cm3: float, temperature_k: float) -> object:
    simulation = asyrcwa.New()
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialModel(
        "InAs",
        "InAsMagneto",
        temperature_parameters(doping_cm3, temperature_k),
    )
    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Air")
    simulation.AddLayer("InAs", sample_thickness_um, "InAs")
    return simulation


def wavelength_grid(window_um: tuple[float, float]) -> np.ndarray:
    return np.linspace(window_um[0], window_um[1], spectral_point_count)


def rows_at_angle(rows: np.ndarray, angle_deg: float) -> np.ndarray:
    selected = rows[
        np.isclose(rows["theta_deg"], angle_deg, rtol=0.0, atol=1.0e-12)
        & (rows["pol"] == polarization)
    ]
    return np.sort(selected, order="lambda_um")


def solve_spectrum(doping_cm3: float, temperature_k: float, window_um: tuple[float, float]) -> dict[str, np.ndarray]:
    wavelengths_um = wavelength_grid(window_um)
    rows = build_simulation(doping_cm3, temperature_k).GetSpectrumForAngles(
        wavelengths_um,
        angles_deg,
        (polarization,),
        Phi=phi_deg,
        Workers=workers,
    )
    forward = rows_at_angle(rows, theta_deg)
    reverse = rows_at_angle(rows, -theta_deg)
    if len(forward) != spectral_point_count or len(reverse) != spectral_point_count:
        raise RuntimeError("incomplete forward/reverse spectrum")
    if not np.allclose(forward["lambda_um"], reverse["lambda_um"], atol=1.0e-12):
        raise RuntimeError("forward/reverse wavelengths are misaligned")
    if max(float(np.max(forward["T"])), float(np.max(reverse["T"]))) > 1.0e-8:
        raise RuntimeError("525-um InAs slab is not numerically opaque")

    # The opaque-sample relations used by the paper are alpha=1-r(theta) and
    # epsilon(theta)=1-r(-theta).  T is retained above as an opacity check.
    return {
        "wavelength_um": np.asarray(forward["lambda_um"], dtype=float),
        "absorptivity": 1.0 - np.asarray(forward["R"], dtype=float),
        "emissivity": 1.0 - np.asarray(reverse["R"], dtype=float),
    }


def solve_all() -> tuple[dict[str, dict[float, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]]:
    fig6_spectra: dict[str, dict[float, dict[str, np.ndarray]]] = {"ND1": {}, "ND2": {}}
    for sample, doping_cm3 in sample_doping_cm3.items():
        for temperature_k in temperatures_k:
            fig6_spectra[sample][temperature_k] = solve_spectrum(
                doping_cm3,
                temperature_k,
                fig6_windows_um[sample],
            )
    fig5_spectra = {
        sample: fig6_spectra[sample][reference_temperature_k]
        for sample in sample_doping_cm3
    }
    return fig6_spectra, fig5_spectra


def draw_spectrum(axis: object, spectrum: dict[str, np.ndarray], *, offset: float = 0.0, labels: bool = True) -> None:
    wavelength = spectrum["wavelength_um"]
    axis.plot(
        wavelength,
        spectrum["emissivity"] + offset,
        color="#1f4ed8",
        linewidth=1.8,
        label=r"Theory $\varepsilon$" if labels else None,
    )
    axis.plot(
        wavelength,
        spectrum["absorptivity"] + offset,
        color="#df2f2f",
        linewidth=1.8,
        label=r"Theory $\alpha$" if labels else None,
    )


def doping_label(sample: str) -> str:
    if sample == "ND1":
        return r"$N_{D1}=3.24\times10^{18}\,\mathrm{cm}^{-3}$"
    return r"$N_{D2}=1.4\times10^{18}\,\mathrm{cm}^{-3}$"


def plot_combined(fig5_spectra: dict[str, dict[str, np.ndarray]], fig6_spectra: dict[str, dict[float, dict[str, np.ndarray]]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    for axis, sample in zip(axes[0], ("ND1", "ND2")):
        draw_spectrum(axis, fig5_spectra[sample])
        axis.set_xlim(*fig5_windows_um[sample])
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        axis.set_ylabel(r"$\varepsilon$, $\alpha$")
        axis.set_title(rf"Fig. 5, sample {sample[-1]} ($T=598$ K)")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    for axis, sample in zip(axes[1], ("ND1", "ND2")):
        for index, temperature_k in enumerate(temperatures_k):
            spectrum = fig6_spectra[sample][temperature_k]
            offset = 0.5 * (index - 2)
            axis.plot(spectrum["wavelength_um"], spectrum["emissivity"] + offset, color="#1f4ed8", linewidth=1.4)
            axis.plot(spectrum["wavelength_um"], spectrum["absorptivity"] + offset, color="#df2f2f", linewidth=1.4)
        axis.set_xlim(*fig6_windows_um[sample])
        axis.set_ylim(-0.55, 1.72)
        axis.set_xlabel(r"Wavelength $\lambda$ ($\mu$m)")
        axis.set_ylabel(r"shifted $\varepsilon$, $\alpha$")
        axis.set_title(rf"Fig. 6, {doping_label(sample)} (offset 0.5 per temperature)")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    figure.suptitle("Nabavi et al. (2025), Figs. 5 and 6: RCWA theory")
    combined_plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(combined_plot_path, dpi=figure_dpi)
    print(f"Saved {combined_plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def print_summary(fig6_spectra: dict[str, dict[float, dict[str, np.ndarray]]]) -> None:
    for sample in sample_doping_cm3:
        print(f"{sample}: ND={sample_doping_cm3[sample]:.3e} cm^-3")
        for temperature_k in temperatures_k:
            spectrum = fig6_spectra[sample][temperature_k]
            contrast = np.abs(spectrum["absorptivity"] - spectrum["emissivity"])
            index = int(np.argmax(contrast))
            print(
                f"  T={temperature_k:g} K: max |alpha-epsilon|="
                f"{contrast[index]:.4f} at {spectrum['wavelength_um'][index]:.4f} um"
            )


def main() -> tuple[dict[str, dict[float, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"InAs slab={sample_thickness_um:g} um, TM, theta=+/-{theta_deg:g} deg, "
        f"B={magnetic_field_t:g} T, temperatures={temperatures_k}"
    )
    with asyrcwa.timed_step("solve Nabavi Figs. 5 and 6"):
        fig6_spectra, fig5_spectra = solve_all()
    print_summary(fig6_spectra)
    plot_combined(fig5_spectra, fig6_spectra)
    return fig6_spectra, fig5_spectra


if __name__ == "__main__":
    main()
