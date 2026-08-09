"""Reproduce Wu et al., ICHMT 172 (2026) 110653, Fig. 2(a).

The local RCWA backend evaluates the normal-incidence emissivity of the
VO2 grating / BaF2 spacer / opaque Ag reflector in metallic and insulating
VO2 states for TM and TE polarization.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa


# Wu et al., Section 3 and Figs. 1-2. Dimensions are in micrometres.
period_um = 7.0
grating_width_um = 5.0
grating_fill_factor = grating_width_um / period_um
vo2_height_um = 0.062
baf2_height_um = 1.5
baf2_data_path = project_root / "materials" / "BaF2_Querry.txt"

# Metallic VO2, paper Eq. (2):
# epsilon = -epsilon_inf*omega_p^2/(omega^2+i*gamma*omega).
# The backend's ordinary Drude form is equivalent when eps_inf=0 and its
# plasma frequency is sqrt(paper epsilon_inf)*paper omega_p.
paper_vo2_epsilon_inf = 9.0
paper_vo2_plasma_frequency_rad_s = 1.51e15
paper_vo2_damping_rate_rad_s = 1.88e15
metallic_vo2_parameters = {
    "eps_inf": 0.0,
    "plasma_frequency_rad_s": (
        np.sqrt(paper_vo2_epsilon_inf) * paper_vo2_plasma_frequency_rad_s
    ),
    "damping_rate_rad_s": paper_vo2_damping_rate_rad_s,
}

# The paper cites Barker et al. (1966) for insulating VO2 but does not print
# or embed the tabulated ordinary/extraordinary optical constants. These
# low-loss LWIR values are an explicit representative approximation; replace
# them with the cited table for a quantitative Q-factor comparison.
insulating_vo2_epsilon_o = 9.0 + 0.10j
insulating_vo2_epsilon_e = 9.0 + 0.10j
insulating_vo2_tensor = np.diag(
    (
        insulating_vo2_epsilon_o,
        insulating_vo2_epsilon_o,
        insulating_vo2_epsilon_e,
    )
).astype(complex)

# Standard mid-infrared Ag Drude fit used by the repository. The article cites
# a Drude model but does not print its coefficients.
ag_parameters = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}

theta_deg = 0.0
phi_deg = 0.0
polarizations = ("TM", "TE")
orders = (40, 0)
workers = 20

wavelength_min_um = 8.0
wavelength_max_um = 14.0
wavelength_point_count = 1201
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_narrowband_peak_um = 8.29
paper_narrowband_peak_emissivity = 0.99
paper_narrowband_q = 619.0

data_path = project_root / "data" / "wu_2026_vo2_grating_fig2.csv"
plot_path = project_root / "images" / "wu_2026_vo2_grating_fig2.png"
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(state: str) -> object:
    simulation = asyrcwa.New(
        Lattice=period_um,
        Orders=orders,
    )
    simulation.SetMaterialModel("Air", "Air")
    simulation.SetMaterialNK("BaF2", baf2_data_path)
    simulation.SetMaterialModel("Ag", "Drude", ag_parameters)
    if state == "metallic":
        simulation.SetMaterialModel(
            "VO2",
            "Drude",
            metallic_vo2_parameters,
        )
    elif state == "insulating":
        simulation.SetMaterial("VO2", insulating_vo2_tensor)
    else:
        raise ValueError(f"unknown VO2 state: {state}")

    simulation.SetSuperstrate("Air")
    simulation.SetSubstrate("Ag")
    simulation.AddLayer("VO2_grating", vo2_height_um, "Air")
    simulation.SetRegionRectangle(
        "VO2_grating",
        "VO2",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * grating_fill_factor * period_um, 0.5 * period_um),
    )
    simulation.AddLayer("BaF2_spacer", baf2_height_um, "BaF2")
    simulation.SetExcitationPlanewave((theta_deg, phi_deg))
    return simulation


def solve_state(state: str) -> np.ndarray:
    rows = build_simulation(state).GetSpectrum(
        wavelengths_um,
        polarizations,
        Workers=workers,
    )
    expected_rows = wavelength_point_count * len(polarizations)
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete {state} spectrum")
    if np.max(np.abs(rows["R"] + rows["T"] + rows["A"] - 1.0)) > 1.0e-8:
        raise RuntimeError(f"{state} spectrum failed energy conservation")
    return rows


def polarization_rows(rows: np.ndarray, polarization: str) -> np.ndarray:
    return np.sort(rows[rows["pol"] == polarization], order="lambda_um")


def peak_summary(rows: np.ndarray, state: str, polarization: str) -> None:
    selected = polarization_rows(rows, polarization)
    peak = selected[int(np.argmax(selected["A"]))]
    print(
        f"{state} {polarization}: max emissivity={peak['A']:.6f} "
        f"at {peak['lambda_um']:.5f} um; "
        f"band average={float(np.mean(selected['A'])):.6f}"
    )


def export_spectra(spectra: dict[str, np.ndarray]) -> None:
    columns = [wavelengths_um]
    names = ["wavelength_um"]
    for state in ("metallic", "insulating"):
        for polarization in polarizations:
            rows = polarization_rows(spectra[state], polarization)
            columns.extend((rows["R"], rows["T"], rows["A"]))
            prefix = f"{state}_{polarization.lower()}"
            names.extend((f"{prefix}_R", f"{prefix}_T", f"{prefix}_emissivity"))
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    print(f"Saved {data_path}.")


def plot_spectra(spectra: dict[str, np.ndarray]) -> None:
    styles = {
        ("insulating", "TM"): ("#1769aa", "-", r"$i$-VO$_2$ TM"),
        ("metallic", "TM"): ("#e24a33", "-", r"$m$-VO$_2$ TM"),
        ("metallic", "TE"): ("#e24a33", "--", r"$m$-VO$_2$ TE"),
        ("insulating", "TE"): ("#2a9d55", "--", r"$i$-VO$_2$ TE"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for state, polarization in styles:
        rows = polarization_rows(spectra[state], polarization)
        color, linestyle, label = styles[(state, polarization)]
        axis.plot(
            rows["lambda_um"],
            rows["A"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    axis.scatter(
        (paper_narrowband_peak_um,),
        (paper_narrowband_peak_emissivity,),
        marker="o",
        facecolors="none",
        edgecolors="#1769aa",
        zorder=4,
        label="paper TM peak",
    )
    axis.axvline(
        paper_narrowband_peak_um,
        color="0.55",
        linestyle=":",
        linewidth=0.8,
    )
    axis.set_xlim(wavelength_min_um, wavelength_max_um)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Wavelength ($\mu$m)")
    axis.set_ylabel("Emissivity")
    axis.set_title("Wu et al. (2026), Fig. 2(a): VO$_2$ grating emitter")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.65)
    axis.legend(loc="center right", frameon=False)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(figure)


def main() -> dict[str, np.ndarray]:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"P={period_um:g} um, w={grating_width_um:g} um, "
        f"d1/d2={vo2_height_um:g}/{baf2_height_um:g} um, orders={orders}"
    )
    spectra: dict[str, np.ndarray] = {}
    for state in ("metallic", "insulating"):
        with asyrcwa.timed_step(f"solve Fig. 2(a) {state} VO2"):
            spectra[state] = solve_state(state)
        for polarization in polarizations:
            peak_summary(spectra[state], state, polarization)
    print(
        f"Paper insulating-TM checkpoint: {paper_narrowband_peak_emissivity:.2f} "
        f"at {paper_narrowband_peak_um:.2f} um, Q={paper_narrowband_q:.0f}."
    )
    export_spectra(spectra)
    plot_spectra(spectra)
    return spectra


if __name__ == "__main__":
    main()
