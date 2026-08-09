"""Reproduce Wu and Qing, ICHMT 144 (2023) 106794, Fig. 2.

The paper demonstrates near-complete nonreciprocal radiation at a one-degree
incidence angle. The unit cell is a conformally Al-coated InAs grating on a
continuous InAs film, backed by an opaque Al mirror and a SiO2 substrate.
Only the TM channel is considered, as in the paper.
"""

from __future__ import annotations

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

import matplotlib
import numpy as np

import asyrcwa
from asyrcwa.spectrum import plot_fr_axes, solve_fr, summarize_fr


# Wu and Qing, Section 3 and Fig. 1. All dimensions are in micrometres.
period_um = 6.015
grating_width_um = 4.798
grating_fill_factor = grating_width_um / period_um
inas_grating_height_um = 3.012
inas_base_height_um = 2.836
al_cap_height_um = 0.502
al_mirror_height_um = 1.0
sio2_refractive_index = 1.45

# Paper +z points out of the stack, whereas solver +z follows the stack from
# Air to SiO2. Preserving a right-handed frame maps the paper's +y bias to
# solver -y. This sign reproduces the paper's alpha/e channel ordering.
inas_parameters = {
    "magnetic_field_t": -2.0,
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

theta_deg = 1.0
phi_deg = 0.0
polarizations = ("TM",)
angles_deg = (+theta_deg, -theta_deg)

# The paper uses a finite-element reference and does not state its numerical
# truncation. The conformal Al coating converges slowly; this centered 1D basis
# retains 161 analytic lamellar harmonics. Lower orders underresolve the peak.
orders = (30, 0)
workers = 14
workers_per_angle = max(1, workers // len(angles_deg))

wavelength_min_um = 15.0
wavelength_max_um = 16.0
wavelength_point_count = 501
wavelengths_um = np.linspace(
    wavelength_min_um,
    wavelength_max_um,
    wavelength_point_count,
)

paper_peak_um = 15.492
paper_absorptivity = 1.0
paper_emissivity_upper_bound = 0.087
paper_eta = 0.912
paper_markers = {"TM": (paper_peak_um,)}

plot_path = project_root / "images" / "wu_2023_extreme_small_angle_fig2.png"
data_path = project_root / "data" / "wu_2023_extreme_small_angle_fig2_tm.csv"
figure_size = (7.4, 5.0)
figure_dpi = 240
show_plot = True

if not show_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_simulation(rcwa: object) -> object:
    if not 0.0 < grating_fill_factor < 1.0:
        raise ValueError("grating_width_um must lie between zero and period_um")

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

    # Approximate the conformal top film in Fig. 1 with three lamellar slices:
    # the Al cap is first because layers are added from Air toward SiO2.
    sim.AddLayer("Al_cap_grating", al_cap_height_um, "Air")
    sim.SetRegionRectangle(
        "Al_cap_grating",
        "Al",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * grating_fill_factor * period_um, 0.5 * period_um),
    )
    sim.AddLayer("InAs_grating", inas_grating_height_um - al_cap_height_um, "Air")
    sim.SetRegionRectangle(
        "InAs_grating",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * grating_fill_factor * period_um, 0.5 * period_um),
    )
    sim.AddLayer("Al_groove_floor", al_cap_height_um, "Al")
    sim.SetRegionRectangle(
        "Al_groove_floor",
        "InAs",
        Center=(0.0, 0.0),
        Angle=0.0,
        Halfwidths=(0.5 * grating_fill_factor * period_um, 0.5 * period_um),
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
    row = rows[int(np.argmin(abs(rows["lambda_um"] - paper_peak_um)))]
    max_eta = rows[int(np.argmax(rows["eta"]))]
    print(
        f"at {paper_peak_um:.3f} um: alpha={row['A']:.4f}, "
        f"e={row['emissivity_same_channel']:.4f}, "
        f"eta={row['eta']:.4f}; paper approximately "
        f"alpha={paper_absorptivity:.2f}, e<{paper_emissivity_upper_bound:.3f}, "
        f"eta={paper_eta:.3f}"
    )
    print(
        f"maximum eta={max_eta['eta']:.4f} at "
        f"{max_eta['lambda_um']:.4f} um"
    )


def export_spectrum(spectrum: np.ndarray) -> None:
    rows = np.sort(spectrum[spectrum["pol"] == "TM"], order="lambda_um")
    if len(rows) != wavelength_point_count:
        raise RuntimeError("incomplete TM spectrum returned by backend")
    columns = np.column_stack(
        (
            rows["lambda_um"],
            rows["R"],
            rows["T"],
            rows["A"],
            rows["R_reverse"],
            rows["T_reverse"],
            rows["A_reverse"],
            rows["emissivity_same_channel"],
            rows["eta"],
            rows["conservation"],
            rows["reverse_conservation"],
        )
    )
    header = (
        "wavelength_um,R_plus,T_plus,A_plus,R_minus,T_minus,A_minus,"
        "emissivity_same_channel,eta,conservation_plus,conservation_minus"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_path, columns, delimiter=",", header=header, comments="")
    print(f"Saved {data_path}.")


def plot_spectrum(spectrum: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    plot_fr_axes(np.asarray([ax]), spectrum, polarizations, markers=paper_markers)
    ax.set_xlim(wavelength_min_um, wavelength_max_um)
    ax.set_title(
        "Wu and Qing (2023), Fig. 2 | "
        rf"TM, $|B|={abs(inas_parameters['magnetic_field_t']):g}$ T, "
        rf"$theta={theta_deg:g}$ deg"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=figure_dpi)
    print(f"Saved {plot_path}.")
    plt.show() if show_plot else plt.close(fig)


def main() -> np.ndarray:
    print(f"Using asyrcwa backend from {asyrcwa.module_directory()}.")
    print(
        f"period={period_um:g} um, width={grating_width_um:g} um, "
        f"fill factor={grating_fill_factor:.6f}, TM theta=+/-{theta_deg:g} deg"
    )
    with asyrcwa.timed_step("solve Wu and Qing Fig. 2 spectrum"):
        spectrum = solve_spectrum()
        summarize_fr(spectrum, polarizations, include_peak_values=True)
        print_paper_comparison(spectrum)
    export_spectrum(spectrum)
    with asyrcwa.timed_step("plot spectrum"):
        plot_spectrum(spectrum)
    return spectrum


if __name__ == "__main__":
    main()
