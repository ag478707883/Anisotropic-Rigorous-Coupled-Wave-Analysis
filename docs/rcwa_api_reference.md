# ASYRCWA 完整接口文档

本文档描述 ASYRCWA 当前公开的 Python 与 C++ 应用接口。除参数名另有说明外，
所有长度使用微米（um），角度使用度（deg），频率接口采用归一化频率
`Frequency = 1 / wavelength_um`。电磁场统一采用 `exp(-i omega t)` 时间约定。

生产配置使用双精度 MKL LAPACKE/CBLAS，并在采样 Fourier 路径中使用 FFTW。
光谱默认通过 Redheffer S 矩阵级联，也可选择 Moharam 等人在
JOSA A 12, 1077-1086 (1995) 中的增强透射矩阵递推。

本文档是公开应用接口的维护位置：构建与启动步骤见 `../README.md`，Li Fourier
分解推导见 `li_fourier_factorization.md`。`include/rcwa` 中供后端实现使用的矩阵、
本征模和 Fourier 组装函数在文末按头文件索引；稳定的用户入口是 Python
`asyrcwa.Simulation` 与 C++ `rcwa::RcwaSolver`。

## 快速索引

| Section | Use |
| --- | --- |
| C++ API | Direct use of `rcwa::RcwaSolver` from C++. |
| C++ S-Parameter API | Complex scattering-amplitude matrices for retained channels. |
| C++ Field API | Near-field plane requests and field result layout. |
| Python API | RETICOLO-aligned `asyrcwa.New(...)` workflow. |
| Python Materials | Scalar, tensor, dispersive, and built-in material model inputs. |
| Python Geometry Regions | Layer, grating, and `SetRegion...` geometry entry points. |
| Python Spectrum API | `GetSpectrum`, `GetSpectrumForAngles`, and structured-array fields. |
| Python S-Parameter API | `GetSParameters`, `GetZeroOrderAmplitudes`, and structured-array fields. |
| Python Field API | `GetFieldPlane` signature and structured-array fields. |
| 完整 Python 签名索引 | 当前绑定的全部构造函数和 67 个 `Simulation` 方法。 |
| Python 辅助模块 | `asyrcwa.spectrum`、`material_response` 与运行时工具。 |
| C++ 完整高层接口 | `RcwaSolver`、输入结构和结果结构的完整方法清单。 |
| C++ 头文件索引 | `include/rcwa` 各公开头文件的职责与稳定性边界。 |
| Supported Boundary | Explicitly supported and rejected configurations. |

## C++ API

Include the solver header:

```cpp
#include "rcwa/solver.hpp"
```

### Basic Spectrum Solve

Use `rcwa::RcwaSolver` for a single wavelength and polarization:

```cpp
rcwa::RcwaSolver solver;
solver.setWavelength(1.0);
solver.setIncidence(5.0, 0.0);
solver.setHarmonicCount(81, rcwa::LatticeTruncation::Circular);
solver.setSuperstrate(rcwa::Material::air());
solver.setSubstrate(rcwa::Material::air());
solver.setPolarization(rcwa::Polarization::TE);

const rcwa::SpectrumResult spectrum = solver.solveSpectrumOnly();
```

`solve()` is a compatibility alias for `solveSpectrumOnly()`.

Global convention: fields have time dependence `exp(-i omega t)`. Finite
layers are stacked along solver `+z`. The
superstrate is the incident half-space at `z < 0`; by default this is `Air`,
not `Vacuum`. The first finite layer starts at `z = 0`, additional layers
follow in `AddLayer` order toward `+z`, and the substrate starts after the
last finite layer. `theta` is measured from solver `+z`; `phi` is the in-plane
azimuth from `+x` toward `+y`, so the incident zero order obeys
`kx/k0 = n_inc sin(theta) cos(phi)` and
`ky/k0 = n_inc sin(theta) sin(phi)`. The incident superstrate is required to
be isotropic, lossless, and to have a positive real refractive index, so
`n_inc` is real. Injection from a lossy, active, or anisotropic half-space is
rejected because TE/TM far-field incident power is not uniquely normalized.
The substrate and finite layers may still be lossy or anisotropic. Passive constitutive models therefore
have positive-semidefinite loss (`Im(epsilon) >= 0` in the scalar case), and a
downward evanescent mode has `Im(kz) > 0`. Material tensor components, magnetic
field directions, source angles, fields, and S-parameters all use this one
coordinate convention; paper coordinates must be converted when parameters
are entered, not by changing solver signs internally.

`theta_deg` must be strictly between `-90` and `90`. Negative theta is a
signed-angle convenience used by the nonreciprocal examples; `(-theta, phi)`
represents the same transverse wavevector direction as
`(+theta, phi + 180 deg)`. `phi_deg` is periodic and is reduced modulo
`360 deg` before trigonometric evaluation.

### Solver Configuration

`RcwaSolver` stores one stack definition and one current solve state.

| Method | Description |
| --- | --- |
| `setWavelength(lambda_um)` | Sets the active wavelength. |
| `setIncidence(theta_deg, phi_deg = 0.0)` | Sets finite incident angles with `-90 < theta_deg < 90`; `phi_deg` is periodic. |
| `setHarmonicCount(harmonic_count, truncation = Circular)` | Builds a basis containing approximately `harmonic_count` retained orders; circular selection uses the physical reciprocal-lattice metric. |
| `setHarmonicOrders(order_x, order_y, truncation = Circular)` | Builds a basis from explicit order limits; the circular reciprocal-space disk is inscribed in those limits. |
| `setPolarization(pol)` | Sets `TE`, `TM`, `Both`, `LCP`, or `RCP` for single-solve entry points. |
| `setStackingAlgorithm(algorithm)` | Selects `ScatteringMatrix` (default) or `EnhancedTransmittanceMatrix`. |
| `stackingAlgorithm()` | Returns the active finite-layer stacking algorithm. |
| `setSuperstrate(material)` | Sets the incident half space. |
| `setSubstrate(material)` | Sets the transmission half space. |
| `addUniformLayer(material, thickness_um)` | Adds a finite uniform layer. |
| `addGratingLayer(grating)` | Adds a 1D binary lamellar grating layer. |
| `addAdaptiveTaperedGratingLayer(layer)` | Expands a z-tapered 1D grating into adaptive lamellar midpoint slices. |
| `addAdaptiveProfiledGratingLayer(layer)` | Expands an arbitrary piecewise-linear fill-factor profile into adaptive lamellar slices. |
| `addPeriodicLayer2d(layer)` | Adds a sampled 2D periodic layer. |
| `addPatternedLayer2d(layer)` | Adds a shape-based 2D patterned layer. |
| `addPrecomputedPeriodicLayer2d(layer)` | Adds a precomputed Fourier-block 2D periodic layer. |
| `clearLayers()` | Removes all finite layers. |

Harmonic count and explicit harmonic orders are mutually exclusive. Calling one
mode resets the other.

`Polarization::Both` solves equal-amplitude TE and TM incidence and reports the
combined power response. Use explicit `{TE, TM}` batches when separate TE and
TM rows are needed.

`Polarization::LCP` and `Polarization::RCP` use coherent zero-order circular
states in the solver's TE/TM basis:

```text
LCP = (TE + i TM) / sqrt(2)
RCP = (TE - i TM) / sqrt(2)
```

Circular spectra are computed from the complex S-matrix amplitudes before
power summation, so TE/TM conversion and interference terms are retained.

### Materials

Common material constructors:

```cpp
auto air = rcwa::Material::constant("Air", {1.0, 0.0});
auto si = rcwa::Material::fromIndex("Si", {3.48, 0.0});
auto tensor = rcwa::Material::anisotropic(
    "uniaxial",
    rcwa::Tensor3::diagonal({2.25, 0.0}, {2.25, 0.0}, {2.56, 0.0}));
```

Use `Material::dispersive()` or `Material::anisotropicDispersive()` when the
permittivity depends on wavelength.

### Layer Types

Finite layers are stacked along `z`. Periodicity is in the `x/y` plane.

| Type | Use |
| --- | --- |
| `UniformLayer` | Uniform isotropic or anisotropic finite layer. |
| `GratingLayer` | 1D binary grating with ridge/groove materials, fill factor, period, optional `ridge_offset_um`, and thickness. |
| `AdaptiveTaperedGratingLayer` | VarRCWA-inspired helper for sidewall-sloped or trapezoidal 1D gratings; expands to `GratingLayer` slices and preserves `ridge_offset_um`. |
| `AdaptiveProfiledGratingLayer` | Generalized VarRCWA-inspired helper for measured or nonlinear sidewall profiles; expands to `GratingLayer` slices and preserves `ridge_offset_um`. |
| `PeriodicLayer2D` | Explicit sampled material cells on one `period_x_um` by `period_y_um` lattice. |
| `PatternedLayer2D` | Background plus circle, rectangle, or polygon regions. |
| `PrecomputedPeriodicLayer2D` | Cached tensor Fourier matrices for repeated solves. |

All patterned layers in one solve must use the same `x/y` lattice. Mixed
periods are rejected.

For `PatternedLayer2D`, later regions override earlier regions.

The `Default` backend builds ordinary material convolutions and ordered Li
blocks from one common piecewise-constant geometry. Axis-aligned rectangles
use exact discontinuity boundaries for scalar and tensor materials. Circles
use RETICOLO V9's area-normalized nested rectangles; the rectangle count is
explicitly controllable. Rotated rectangles and polygons use a rectilinear
cell approximation whose cell integrals are exact. Regions crossing a unit-cell
edge are split and wrapped periodically instead of being clipped. Later
regions override earlier regions after this periodic construction.

### Batch Spectra

Use the batch wavelength API for high-throughput spectra:

```cpp
std::vector<double> wavelengths{16.4, 16.5, 16.6};
std::vector<rcwa::Polarization> pols{
    rcwa::Polarization::TE,
    rcwa::Polarization::TM,
};

std::vector<rcwa::SpectrumResult> rows =
    solver.solveSpectrumBatchOnly(wavelengths, pols, 20);
```

The result order is wavelength-major. For each wavelength, results appear in
the same order as `pols`. Internally each wavelength builds one prepared stack
and shares it across the requested polarizations.

For one wavelength with multiple polarizations, call
`solveSpectrumOnlyForPolarizations(pols)`.

For sparse angle/polarization requests, use the pair batch API. This is useful
for circular-emission maps where one handedness is sampled on one half-angle
range and the opposite handedness is sampled on the other:

```cpp
std::vector<rcwa::SpectrumAnglePolarization> requests{
    {-10.0, rcwa::Polarization::LCP},
    {-5.0, rcwa::Polarization::LCP},
    {0.0, rcwa::Polarization::LCP},
    {0.0, rcwa::Polarization::RCP},
    {5.0, rcwa::Polarization::RCP},
    {10.0, rcwa::Polarization::RCP},
};

std::vector<rcwa::SpectrumTotalsResult> rows =
    solver.solveSpectrumBatchTotalsForAnglePolarizations(
        requests,
        90.0,
        wavelengths,
        20);
```

`SpectrumAnglePolarization` contains `theta_deg` and `polarization`. Result
order is request-major, then wavelength-major. Internally, equal `theta_deg`
requests are grouped, so the example above solves the normal direction once
and computes LCP/RCP from the same prepared stack and TE/TM incident channels.
The same grouping is available with diffraction-order output through
`solveSpectrumBatchForAnglePolarizations()`.

### SpectrumResult

`SpectrumResult` contains diffraction-order and total power data:

| Field | Meaning |
| --- | --- |
| `R_orders`, `T_orders` | Reflection and transmission efficiencies for retained orders. |
| `center_idx` | Index of the zero diffraction order. |
| `R0`, `T0` | Zero-order reflection and transmission efficiencies. |
| `R_total`, `T_total` | Sum over propagating diffraction orders. |
| `conservation` | Power conservation diagnostic returned by the solver. |
| `wavelength_um`, `theta_deg`, `phi_deg` | Solve state used for the row. |
| `N` | Number of retained harmonics. |

Absorption is usually computed by callers as `1 - R_total - T_total`.

## C++ S-Parameter API

Use `solveSParameters()` when complex scattering amplitudes are needed
without running the spectrum or field-plane entry points:

```cpp
const rcwa::SParameterResult s = solver.solveSParameters();
```

The returned `S11`, `S12`, `S21`, and `S22` matrices use the usual two-port
scattering convention:

```text
[outgoing top, outgoing bottom]^T =
[[S11, S12], [S21, S22]] [incident top, incident bottom]^T
```

Each block has shape `(2 * N, 2 * N)`, where `N` is the number of retained
harmonics. Channels `0..N-1` are TE for `orders[channel]`; channels `N..2N-1`
are TM for `orders[channel - N]`. Values are complex amplitude coefficients,
not power efficiencies. Power spectra still use the existing flux-normalized
`solve_spectrum_*` APIs.

## C++ Field API

Near fields use one public entry point:

```cpp
rcwa::FieldPlaneRequest request;
request.plane = rcwa::FieldPlane::XZ;
request.u_min_um = -4.15;
request.u_max_um = 4.15;
request.u_points = 161;
request.v_min_um = 0.0;
request.v_max_um = 7.1;
request.v_points = 181;
request.fixed_um = 0.0;
request.components = {rcwa::FieldComponent::E, rcwa::FieldComponent::H};
request.workers = 8;

const rcwa::FieldPlaneResult field = solver.solveFieldPlane(request);
```

The request uses global coordinates, with `z = 0` at the top of the first
finite layer. Negative `z` samples the incident half space. `z` beyond the
finite stack samples the transmission half space.

Plane coordinate mapping:

| Plane | `u` axis | `v` axis | `fixed_um` |
| --- | --- | --- | --- |
| `XY` | `x` | `y` | `z` |
| `XZ` | `x` | `z` | `y` |
| `YZ` | `y` | `z` | `x` |

Valid components are `Ex`, `Ey`, `Ez`, `E`, `Hx`, `Hy`, `Hz`, and `H`.
Aggregate `E` and `H` are magnitudes of their vector components.

`FieldPlaneResult.samples` is ordered by `iv`, then `iu`, then requested
component order.

## Field Scale

The solver's native modal state uses normalized magnetic fields.

- `E`, `Ex`, `Ey`, and `Ez` return electric-field components in the solver's
  normalized amplitude scale.
- `H`, `Hx`, `Hy`, and `Hz` return the normalized magnetic components used by
  the modal solver.

Do not put `|E|` and `|H|` on one shared plot scale unless the magnetic field
has been intentionally converted for that post-processing workflow.

## Python API

Import the pybind module:

```python
import numpy as np
import asyrcwa
```

Use the single top-level interface for uniform multilayers, gratings, and
metasurfaces:

```python
sim = asyrcwa.New(
    Lattice=(8.3, 8.3),
    Orders=(6, 6),
)
```

Every finite layer is created by `AddLayer(Name, Thickness, Material)`. A layer
with no regions is uniform and is sent to the TMM/Berreman backend. Calling one
or more `SetRegion...` methods on that same layer makes it patterned and sends
it to RCWA. There are no separate Multilayer, Grating, or Metasurface front-end
classes, factories, or construction gates.

`New()` without arguments creates a zeroth-order setup suitable for a uniform
multilayer. Periodic structures pass `Lattice` and explicit `Orders` to the
same factory.

`Lattice` may be a scalar period or a two-value `(period_x_um, period_y_um)`
sequence. `Orders=(Nx, Ny)` is the RETICOLO rectangular order range and is the
only exposed harmonic selection mode.

Configuration methods can be called after construction:

| Method | Description |
| --- | --- |
| `SetWavelength(Wavelength)` | Sets the active wavelength for single-state methods. |
| `SetFrequency(Frequency)` | Sets the active wavelength to `1 / Frequency`. |
| `SetHarmonicOrders(OrderX, OrderY)` | Switches the simulation to explicit order limits. |
The Python frontend fixes the RETICOLO choices internally: rectangular
`li=1` ordered factorization, no Lanczos smoothing, and the stable scattering
matrix stack recurrence. Geometry-cell resolution is selected internally from
the retained orders; users only increase `Orders` and the circle `Rectangles`
count for convergence.

### Python Stack Setup

The construction rule is identical for every structure. A uniform multilayer
uses `New()` and `AddLayer` only:

```python
sim = asyrcwa.New()
sim.SetMaterial("Air", 1.0 + 0.0j)
sim.SetMaterial("Film", 2.25 + 0.01j)
sim.SetSuperstrate("Air")
sim.SetSubstrate("Air")
sim.AddLayer("film", 0.5, "Film")
rows = sim.GetSpectrumForAngles(
    np.linspace(0.8, 1.2, 101),
    Angles=(30.0, -30.0),
    Polarizations=("TE", "TM"),
    Workers=4,
)
```

For a 2D patterned layer, pass the lattice and orders to the same `New()`
factory. Add its background with `AddLayer`, then add regions:

```python
sim = asyrcwa.New(Lattice=(8.3, 8.3), Orders=(6, 6))
sim.SetMaterial("Air", 1.0 + 0.0j)
sim.SetMaterial("Si", 3.48**2 + 0.0j)

sim.SetSuperstrate("Air")
sim.SetSubstrate("Air")
sim.AddLayer("Si_nanopore", 2.3, "Si")
sim.SetRegionRectangle(
    "Si_nanopore",
    "Air",
    Center=(0.0, 0.0),
    Angle=0.0,
    Halfwidths=(1.2, 1.2),
)
sim.SetExcitationPlanewave((5.0, 0.0))
```

For circular incidence, pass equal-magnitude s/p amplitudes with a `+90` or
`-90` degree relative phase. The Python bridge maps these to the native
`LCP`/`RCP` coherent paths:

```python
sim.SetExcitationPlanewave(
    (5.0, 0.0),
    sAmplitude=1 / np.sqrt(2),
    pAmplitude=1j / np.sqrt(2),   # LCP
)
```

A 1D binary lamellar grating uses exactly the same construction. Add the groove
material as the layer background and describe the ridge with an axis-aligned
rectangle spanning the y period:

```python
sim = asyrcwa.New(
    Lattice=5.5,
    Orders=(20, 0),
    )
sim.SetMaterial("Air", 1.0 + 0.0j)
sim.SetMaterial("Si", 3.48**2 + 0.0j)
sim.AddLayer("Si_grating", 0.65, "Air")
sim.SetRegionRectangle(
    "Si_grating",
    "Si",
    Center=(0.0, 0.0),
    Angle=0.0,
    Halfwidths=(0.5 * 3.355, 0.5 * 5.5),
)
```

The rectangle center is the former ridge offset. Tapered or profiled structures
are expressed explicitly as multiple `AddLayer` slices, each followed by its
own `SetRegionRectangle` or `SetRegionCircle`. This keeps one visible stack
construction model instead of hidden structure-specific builders.

### Python Materials

`SetMaterial(Name, Epsilon, Mu=None)` accepts:

| Input | Meaning |
| --- | --- |
| complex scalar | Constant isotropic permittivity. |
| complex scalar plus `Mu` | Constant isotropic permittivity and permeability. |
| `3 x 3` NumPy array | Constant anisotropic permittivity tensor. |
| `3 x 3` epsilon plus `3 x 3` `Mu` | Constant anisotropic epsilon and mu tensors. |
| dict | One of the material dictionaries below. |

Supported material dictionaries:

```python
{"kind": "vacuum"}
{"kind": "constant", "epsilon": 12.0 + 0.0j, "mu": 1.0 + 0.0j}
{"kind": "index", "n": 3.48 + 0.0j}
{"kind": "tabulated_index", "wavelength_um": wavelength, "n": n, "k": k, "extrapolation": "error"}
{"kind": "tensor", "epsilon": eps_3x3, "mu": mu_3x3}
{"kind": "builtin", "model": "Drude", "parameters": {"eps_inf": 3.4, "wp": 1.39e16, "gamma": 2.7e13}}
{"kind": "builtin", "model": "PhononPolariton", "parameters": {"epsilon_inf": [4.0, 5.2, 2.4], "omega_to_cm_inv": [818.0, 545.0, 962.0], "omega_lo_cm_inv": [974.0, 851.0, 1010.0], "damping_cm_inv": [4.0, 4.0, 2.0]}}
{"kind": "builtin", "model": "YIG", "parameters": {"diagonal_epsilon": 4.0, "offdiagonal_gyration": 0.1, "magnetization_direction": [0.0, 1.0, 0.0]}}
{"kind": "builtin", "model": "WSM", "parameters": {"fermi_energy_ev": 0.15}}
{"kind": "builtin", "model": "BP", "parameters": {"carrier_density_cm2": 5.0e13, "scattering_rate_mev": 10.0}}
{"kind": "builtin", "model": "GrapheneKubo", "parameters": {"fermi_energy_ev": 0.3}}
{"kind": "builtin", "model": "InAsMagneto", "parameters": {"magnetic_field_t": 3.0}}
{"kind": "builtin", "model": "CdTeLorentzDrude", "parameters": {"magnetic_field_t": 3.0}}
{"kind": "builtin", "model": "InSbLorentzDrude", "parameters": {"magnetic_field_t": 3.0}}
```

`tabulated_index` accepts equal-length, strictly increasing one-dimensional
arrays and linearly interpolates the complex refractive index `n + i*k` before
forming epsilon. Wavelengths are in microns. `n` and `k` must be finite and
non-negative. The default `extrapolation="error"` rejects wavelengths outside
the supplied table; `"clamp"` explicitly holds the nearest endpoint value.

For an nk text file, register the material directly without loading or parsing
it in Python:

```python
sim.SetMaterialNK("Au", "materials/Au_Johnson-Christy.txt")
sim.SetMaterialNK("sample", "path/to/material_nk_nm.txt", WavelengthUnit="nm")
```

Every file line is exactly three whitespace-separated numeric columns,
`wavelength n k`, starting on the first line. Headers and column names are not
used. Wavelengths must be strictly increasing and contain at least two
samples. `WavelengthUnit` is `"um"` by default and also accepts `"nm"`;
`Extrapolation` is `"error"` by default and also accepts `"clamp"`. File
parsing, validation, interpolation of `n + i*k`, and conversion to epsilon are
all performed by the C++ backend.

`SetMaterialModel(Name, Model, Parameters=None)` is the built-in material
entry point. `Parameters` must be a dictionary whenever the selected built-in
model has variable physical parameters. The old scalar `parameter` key and
scalar third argument were removed.

The scalar Drude model accepts `damping_rate_rad_s=0` for the lossless limit
used in idealized paper comparisons. Negative damping is rejected.

Parameter dictionaries are strict: an unknown key, a non-finite numeric
value, two aliases for the same quantity, or the same physical quantity in
two unit systems raises `ValueError`. This is intentional so a misspelled
paper parameter cannot be silently replaced by a model default.

Built-in materials are grouped by calculation model, not by one-off material
names. Use the same model name and change the parameter dictionary for
different parameter sets:

| Category | Model names | Notes |
| --- | --- | --- |
| Fixed isotropic | `Air`, `Vacuum`, `Si`, `Silicon`, `Ge`, `Germanium` | no parameter dictionary needed |
| Metal Drude | `Drude`, `MetalDrude`, `Metal_Drude` | use parameters for Ag, Al, or any other Drude metal; element-specific Drude model names from older revisions are not supported |
| Anisotropic phonon polariton | `PhononPolariton`, `Phonon_Polariton`, `LorentzPhonon`, `Lorentz_Phonon` | three-axis Lorentz oscillator specified by x/y/z TO, LO, damping wavenumbers, with optional in-plane tensor rotation |
| Magneto-optical garnet | `YIG`, `CeYIG`, `Ce:YIG`, `MagnetoOpticalYIG`, `MagnetoOpticYIG` | constant gyrotropic tensor with complex diagonal permittivity, complex gyration, and arbitrary magnetization direction |
| Weyl semimetal | `WSM`, `Weyl_Semimetal`, `Weyl-Semimetal` | tensor orientation is fixed to the solver's xz gyrotropy convention |
| Black phosphorus Drude sheet | `BP`, `BlackPhosphorus`, `BlackPhosphorusDrude`, `Phosphorene`, `PhosphoreneDrude` | anisotropic BP surface conductivity from the Zanbouri et al. 2025 absorber paper, converted to an effective thin-layer tensor |
| Magnetized InAs | `InAsMagneto`, `InAs_Magneto`, `InAs` | magnetization is fixed along the y axis; paper-specific n-InAs variants from older revisions were removed |
| Magnetized CdTe | `CdTeLorentzDrude`, `CdTe_Lorentz_Drude`, `CdTeMagneto`, `CdTe` | Lorentz-Drude tensor with transverse optical resonance; magnetization is fixed along the y axis |
| Magnetized InSb | `InSbLorentzDrude`, `InSb_Lorentz_Drude`, `InSbMagneto`, `InSb` | Lorentz-Drude tensor with transverse optical resonance; magnetization is fixed along the y axis |
| Graphene Kubo | `GrapheneKubo` | full local Kubo conductivity model |

All three carrier magneto-optical models use the same signed field convention.
For positive `magnetic_field_t` (`+B_y`) and positive carrier density, their
xz Hall block is oriented as `epsilon_xz = -i*g` and
`epsilon_zx = +i*g` (with complex dispersive `g` as defined by the selected
model). Reversing `magnetic_field_t` reverses both off-diagonal entries. This
statement is part of the public coordinate convention; examples must not
reinterpret the sign differently for InAs, CdTe, or InSb.

The YIG model uses
`epsilon = diagonal_epsilon*I - i*offdiagonal_gyration*[m_hat]_x`, where
`[m_hat]_x` is the cross-product matrix of the normalized solver-coordinate
magnetization direction. For `magnetization_direction=(0, +1, 0)`, this gives
`epsilon_xz=-i*g` and `epsilon_zx=+i*g`. Reversing the direction reverses all
off-diagonal entries. Both dielectric parameters may be complex.

Example Drude parameter dictionaries:

```python
AG_DRUDE_PARAMETERS = {
    "eps_inf": 3.4,
    "plasma_frequency_rad_s": 1.39e16,
    "damping_rate_rad_s": 2.7e13,
}
AL_DRUDE_PARAMETERS = {
    "eps_inf": 1.0,
    "plasma_frequency_rad_s": 2.24e16,
    "damping_rate_rad_s": 1.24e14,
}

sim.SetMaterialModel("Ag", "Drude", AG_DRUDE_PARAMETERS)
sim.SetMaterialModel("Al", "Drude", AL_DRUDE_PARAMETERS)
```

Example alpha-MoO3 phonon-polariton parameter dictionary:

```python
MOO3_PARAMETERS = {
    "epsilon_inf": (4.0, 5.2, 2.4),
    "omega_to_cm_inv": (818.0, 545.0, 962.0),
    "omega_lo_cm_inv": (974.0, 851.0, 1010.0),
    "damping_cm_inv": (4.0, 4.0, 2.0),
    "twist_deg": 0.0,
}

sim.SetMaterialModel("MoO3", "PhononPolariton", MOO3_PARAMETERS)
```

Example Ce:YIG parameter dictionary:

```python
CEYIG_PARAMETERS = {
    "diagonal_epsilon": 4.0,
    "offdiagonal_gyration": 0.1,
    "magnetization_direction": (0.0, 1.0, 0.0),
}

sim.SetMaterialModel("CeYIG", "YIG", CEYIG_PARAMETERS)
```

Example WSM and y-magnetized InAs parameter dictionaries:

```python
WSM_PARAMETERS = {
    "node_separation_m_inv": 2.0e9,
    "fermi_velocity_m_s": 0.83e5,
    "cutoff_xi": 3.0,
    "background_epsilon": 6.2,
    "temperature_k": 300.0,
    "degeneracy": 2.0,
    "tau_fs": 1000.0,
    "fermi_energy_ev": 0.15,
}
INAS_PARAMETERS = {
    "magnetic_field_t": 4.0,
    "eps_inf": 12.37,
    "carrier_density_cm3": 7.8e17,
    "effective_mass_ratio": 0.033,
    "damping_rate_rad_s": 1.55e11,
}

sim.SetMaterialModel("WSM", "WSM", WSM_PARAMETERS)
sim.SetMaterialModel("InAs", "InAsMagneto", INAS_PARAMETERS)
```

Example black-phosphorus parameter dictionary:

```python
BP_PARAMETERS = {
    # Zanbouri et al. 2025 use N = 1e13 to 10e13 cm^-2 and eta = 10 meV.
    "carrier_density_cm2": 5.0e13,
    "scattering_rate_mev": 10.0,
    # Effective thin-layer thickness used to convert surface conductivity to epsilon.
    "thickness_nm": 1.0,
    "twistDeg": 0.0,
}

sim.SetMaterialModel("BP", "BP", BP_PARAMETERS)
```

Example y-magnetized CdTe Lorentz-Drude parameter dictionary:

```python
CDTE_PARAMETERS = {
    "magnetic_field_t": 3.0,
    "eps_inf": 7.1,
    "carrier_density_cm3": 3.0e17,
    "effective_mass_ratio": 0.09,
    "transverse_resonance_rad_s": 2.652e13,
    "damping_rate_rad_s": 1.24e12,
}

sim.SetMaterialModel("CdTe", "CdTeLorentzDrude", CDTE_PARAMETERS)
```

Example y-magnetized InSb Lorentz-Drude parameter dictionary:

```python
INSB_PARAMETERS = {
    "magnetic_field_t": 3.0,
    "eps_inf": 15.68,
    "carrier_density_m3": 5.8e23,
    "effective_mass_ratio": 0.014,
    "transverse_resonance_rad_s": 3.376e13,
    "damping_rate_rad_s": 2.017e12,
}

sim.SetMaterialModel("InSb", "InSbLorentzDrude", INSB_PARAMETERS)
```

Graphene uses the complete local Kubo model. Custom settings must be passed
as a named dictionary:

```python
sim.SetMaterialModel("Graphene", "GrapheneKubo", {
    "fermi_energy_ev": 0.3,
    "temperature_k": 300.0,
    "tau_ps": 1.0,
    "thickness_nm": 0.34,
})

sim.SetMaterial("Graphene", {
    "kind": "builtin",
    "model": "GrapheneKubo",
    "parameters": {
        "fermi_energy_ev": 0.3,
        "tau_ps": 1.0,
        "thickness_nm": 0.34,
    },
})
```

Graphene parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Fermi energy / chemical potential | `fermi_energy_ev`, `fermi_ev`, `ef_ev`, `chemical_potential_ev`, `mu_c_ev`, `chemical_potential`, `mu_c` | eV | `1.0` |
| Temperature | `temperature_k`, `temperature` | K | `300` |
| Relaxation time | `relaxation_time_s`, `tau_s`, `tau`, `relaxation_time_ps`, `tau_ps`, `relaxation_time_fs`, `tau_fs` | s, ps, or fs by key | `0.5 ps` |
| Effective thickness | `thickness_m`, `graphene_thickness_m`, `thickness_nm`, `graphene_thickness_nm`, `thickness_um`, `graphene_thickness_um` | m, nm, or um by key | `0.34 nm` |

Drude parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| High-frequency permittivity | `eps_inf`, `epsilon_inf`, `eps_infinity` | dimensionless | `1.0` |
| Plasma angular frequency | `plasma_frequency_rad_s`, `omega_p_rad_s`, `wp_rad_s`, `wp` | rad/s | required |
| Damping angular frequency | `damping_rate_rad_s`, `gamma_rad_s`, `gamma`, `relaxation_rate_rad_s` | rad/s | required |

PhononPolariton parameter keys (each axis tuple is ordered x/y/z):

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| High-frequency permittivity | `epsilon_inf`, `eps_inf` | dimensionless | required |
| Transverse optical phonon wavenumber | `omega_to_cm_inv`, `transverse_wavenumber_cm_inv` | cm^-1 | required |
| Longitudinal optical phonon wavenumber | `omega_lo_cm_inv`, `longitudinal_wavenumber_cm_inv` | cm^-1 | required |
| Phonon damping wavenumber | `damping_cm_inv`, `gamma_cm_inv` | cm^-1 | required |
| In-plane dielectric-tensor rotation | `twist_deg`, `rotation_deg` | degrees, counterclockwise from solver +x toward +y | `0.0` |

YIG parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Diagonal relative permittivity | `diagonal_epsilon`, `epsilon_diagonal`, `epsilon`, `eps` | dimensionless, real or complex | `4.0` |
| Off-diagonal gyration | `offdiagonal_gyration`, `off_diagonal_gyration`, `offdiagonal_b`, `off_diagonal_b`, `gyration`, `b` | dimensionless, real or complex | `0.0` |
| Magnetization direction | `magnetization_direction`, `magnetic_field_direction`, `bias_direction` | normalized internally, solver x/y/z | `(0, 1, 0)` |

WSM parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Weyl-node separation | `node_separation_m_inv`, `b_m_inv`, `b` | 1/m | `2.0e9` |
| Fermi velocity | `fermi_velocity_m_s`, `v_f_m_s`, `vf_m_s`, `v_f` | m/s | `0.83e5` |
| Cutoff | `cutoff_xi`, `xi_c`, `xi` | dimensionless | `3.0` |
| Background permittivity | `background_epsilon`, `eps_b`, `epsilon_b` | dimensionless | `6.2` |
| Temperature | `temperature_k`, `temperature` | K | `300` |
| Degeneracy | `degeneracy`, `g` | dimensionless | `2.0` |
| Relaxation time | `relaxation_time_s`, `tau_s`, `tau`, `relaxation_time_fs`, `tau_fs` | s or fs by key | `1000 fs` |
| Fermi energy | `fermi_energy_ev`, `fermi_ev`, `ef_ev` | eV | `0.15` |

Black phosphorus parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| 2D carrier density | `carrier_density_cm2`, `density_cm2`, `n_cm2`, `N_cm2`, `N`, `carrier_density_m2`, `density_m2`, `n_m2` | cm^-2 or m^-2 by key | `1.0e13 cm^-2` |
| Scattering rate | `scattering_rate_ev`, `eta_ev`, `damping_ev`, `scattering_rate_mev`, `eta_mev`, `damping_mev` | eV or meV by key | `10 meV` |
| Effective thickness | `thickness_m`, `bp_thickness_m`, `thickness_nm`, `bp_thickness_nm`, `thickness_um`, `bp_thickness_um` | m, nm, or um by key | `1.0 nm` |
| Bandgap Delta | `bandgap_ev`, `delta_ev`, `Delta_ev` | eV | `2.0` |
| Lattice constant a | `lattice_constant_m`, `a_m`, `lattice_constant_nm`, `a_nm` | m or nm by key | `0.223 nm` |
| In-plane dielectric-tensor twist | `twist_deg`, `rotation_deg` (`twistDeg` retained for compatibility) | degrees, counterclockwise from solver +x toward +y | `0.0` |

InAsMagneto parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Magnetic field along y | `magnetic_field_t`, `b_t`, `B_t`, `B` | T | `0.0` |
| High-frequency permittivity | `eps_inf`, `epsilon_inf`, `eps_infinity` | dimensionless | `12.37` |
| Carrier density | `carrier_density_cm3`, `density_cm3`, `n_cm3`, `carrier_density_m3`, `density_m3`, `n_m3` | cm^-3 or m^-3 by key | `7.8e17 cm^-3` |
| Effective mass ratio | `effective_mass_ratio`, `m_eff_over_me`, `mstar` | m*/me | `0.033` |
| Damping angular frequency | `damping_rate_rad_s`, `gamma_rad_s`, `gamma`, `relaxation_rate_rad_s` | rad/s | `1.55e11` |

CdTeLorentzDrude parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Magnetic field along y | `magnetic_field_t`, `b_t`, `B_t`, `B` | T | `0.0` |
| High-frequency permittivity | `eps_inf`, `epsilon_inf`, `eps_infinity` | dimensionless | `7.1` |
| Carrier density | `carrier_density_cm3`, `density_cm3`, `n_cm3`, `carrier_density_m3`, `density_m3`, `n_m3` | cm^-3 or m^-3 by key | `3.0e17 cm^-3` |
| Effective mass ratio | `effective_mass_ratio`, `m_eff_over_me`, `mstar` | m*/me | `0.09` |
| Transverse resonance angular frequency | `transverse_resonance_rad_s`, `omega_t_rad_s`, `omega_t` | rad/s | `2.652e13` |
| Damping/collision angular frequency | `damping_rate_rad_s`, `collision_frequency_rad_s`, `gamma_rad_s`, `gamma`, `relaxation_rate_rad_s` | rad/s | `1.24e12` |

InSbLorentzDrude parameter keys:

| Parameter | Accepted keys | Unit | Default |
| --- | --- | --- | --- |
| Magnetic field along y | `magnetic_field_t`, `b_t`, `B_t`, `B` | T | `0.0` |
| High-frequency permittivity | `eps_inf`, `epsilon_inf`, `eps_infinity` | dimensionless | `15.68` |
| Carrier density | `carrier_density_cm3`, `density_cm3`, `n_cm3`, `carrier_density_m3`, `density_m3`, `n_m3` | cm^-3 or m^-3 by key | `5.8e17 cm^-3` |
| Effective mass ratio | `effective_mass_ratio`, `m_eff_over_me`, `mstar` | m*/me | `0.014` |
| Transverse resonance angular frequency | `transverse_resonance_rad_s`, `omega_t_rad_s`, `omega_t` | rad/s | `3.376e13` |
| Damping/collision angular frequency | `damping_rate_rad_s`, `collision_frequency_rad_s`, `gamma_rad_s`, `gamma`, `relaxation_rate_rad_s` | rad/s | `2.017e12` |

### Python Geometry Regions

The region methods and built-in pattern structures are documented in this
section. Runnable patterned-layer constructions are available in the repository's
`../examples/` directory.

Regions are added to an existing layer and replace the layer background inside
the region. Later regions override earlier regions where they overlap.

| Method | Main parameters |
| --- | --- |
| `SetRegionCircle` | `Center`, `Radius`, `Rectangles=10` |
| `SetRegionRectangle` | `Center`, `Angle`, `Halfwidths` |
| `SetRegionPolygon` | `Center`, `Angle`, `Vertices` |
| `SetRegionEllipse` | `Center`, `Angle`, `Radii`, `Vertices=96` |
| `SetRegionRegularPolygon` | `Center`, `Angle`, `Sides`, `Radius` |
| `SetRegionTriangle` | `Center`, `Angle`, `Radius` |
| `SetRegionCross` | `Center`, `Angle`, `ArmLength`, `ArmWidth` |
| `SetRegionRing` | `Center`, `OuterRadius`, `InnerRadius`, `HoleMaterial` |
| `SetRegionRoundedRectangle` | `Center`, `Angle`, `Halfwidths`, `CornerRadius`, `VerticesPerCorner=16` |
| `SetRegionCapsule` | `Center`, `Angle`, `Length`, `Radius`, `VerticesPerCap=32` |
| `SetRegionStripe` | `Center`, `Angle`, `Width` |

Angles are in degrees. Polygon vertices are local coordinates before rotation
and translation by `Center`.

`SetRegionCircle(..., Rectangles=N)` follows the RETICOLO V9 `res0` default:
`N >= 2` equal-angle nested rectangles approximate the circle and are
normalized so their union has exactly the circle area. Increasing `N` is the
circle-geometry convergence control and does not change the retained basis.

Tapered and profiled structures are built explicitly as ordinary layers. For
each z slice call `AddLayer`, then attach its rectangle, circle, or polygon with
the corresponding `SetRegion...` method.

Full-height or full-width scalar lamellar rectangles use analytic 1D Li
factorization. A single full-height or full-width tensor rectangle also
delegates to the corresponding x-normal or y-normal analytic tensor lamellar
Li route. Genuinely 2D axis-aligned rectangle tensors use exact cell Fourier
integrals and RETICOLO `li=1` y-then-x factorization. Circles use the same path
after RETICOLO's area-normalized nested-rectangle construction. Rotated
rectangles and oblique polygons use one uniform rectangle-cell geometry for
both ordinary convolution and Li factorization; no analytic-direct/sampled-Li
mixture remains. For these arbitrary shapes, increase the layer cell grid and
harmonic orders together and verify convergence of R/T/A and requested fields.

### Python Spectrum API

```python
rows = sim.GetSpectrum(
    np.linspace(16.4, 17.6, 500),
    ["TE", "TM"],
    Workers=20,
)
```

`GetSpectrum(Wavelengths, Polarizations=None, Workers=1)` returns a NumPy
structured array. `Wavelengths` must be one-dimensional. `Polarizations` may be
`None`, a string, or a sequence containing `"TE"`, `"TM"`, `"Both"`, `"LCP"`,
or `"RCP"`. `None` uses the current `SetExcitationPlanewave` polarization;
the initial default is `["TE"]`.

Rows are wavelength-major and preserve the requested polarization order.

For high-throughput forward/reverse angle sweeps, prefer the multi-angle batch
API:

```python
rows = sim.GetSpectrumForAngles(
    np.linspace(16.4, 17.6, 500),
    [+5.0, -5.0],
    ["TE", "TM"],
    Phi=0.0,
    Workers=20,
)
```

`GetSpectrumForAngles(Wavelengths, Angles, Polarizations=None, Phi=0.0,
Workers=1)` returns the same structured dtype as `GetSpectrum`. Rows are
angle-major, then wavelength-major, then polarization-major. The method keeps
one C++ work queue for all angle/wavelength jobs and lets one `Simulation`
share cached patterned-layer Fourier blocks across the requested angles.

When both spectra and exact directional thermal channels are required, use
the fused backend call:

```python
spectrum, thermal = sim.GetSpectrumAndDirectionalThermalChannelsForAngles(
    wavelengths,
    [+5.0, -5.0],
    ["TE", "TM"],
    Phi=0.0,
    Workers=20,
)
```

This returns the same spectrum and thermal dtypes as the standalone APIs, but
prepares each wavelength/angle stack once and performs one exact scattering
cascade. The backend retains every internal diffraction/Maxwell mode and
solves only the exterior radiative incident columns and observable outgoing
rows required by the requested spectrum and complete two-port thermal
row/column deficits. This is an exact projection of the finite-order RCWA
system; it does not discard an internal mode or replace emissivity by a
reverse-angle approximation. The high-level `solve_fr` helper uses this fused
path exclusively.

For per-angle polarization requests, use:

```python
angles = np.r_[np.linspace(-10.0, 0.0, 21), np.linspace(0.0, 10.0, 21)]
pols = ["LCP"] * 21 + ["RCP"] * 21

rows = sim.GetSpectrumForAnglePolarizationPairs(
    np.linspace(5.15, 5.50, 151),
    angles,
    pols,
    Phi=90.0,
    Workers=20,
)
```

`GetSpectrumForAnglePolarizationPairs(Wavelengths, Angles, Polarizations,
Phi=0.0, Workers=1)` accepts one polarization string for all angles or one
polarization per angle. Rows are request-major, then wavelength-major. Matching
angles are solved once with the union of requested polarizations, which avoids
duplicate normal-incidence solves in LCP/RCP half-angle scans. This is a
general batch API; it is not tied to any particular example.

Structured-array fields:

| Field | Meaning |
| --- | --- |
| `lambda_um` | Wavelength. |
| `pol` | Polarization label. |
| `theta_deg`, `phi_deg` | Incidence angles. |
| `harmonics` | Number of retained harmonics. |
| `R0`, `T0` | Zero-order reflection and transmission. |
| `R`, `T` | Total reflection and transmission. |
| `A` | `1 - R - T`. |
| `conservation` | Solver power-conservation diagnostic. |

`GetSpectrum` calls the C++ batch wavelength API. The Python bridge caches
precomputed patterned-layer Fourier matrices by material dependency: changing
a wavelength-dependent uniform layer such as `WSM`, `InAs`, or `Ag` does not
force unrelated patterned geometry layers such as `Si/Air` pores to rebuild
their sampled Fourier blocks.

`GetPowerFlux(Layer=None, zOffset=0.0)` is a compatibility helper. In the
current binding it ignores `Layer` and `zOffset` and returns a two-element
array `[R, T]` for the active wavelength, incidence, and polarization. Use
`GetSpectrum` for wavelength-resolved spectra and `GetFieldPlane` for spatial
field sampling.

For circular-dichroism spectra, request both handednesses explicitly:

```python
rows = sim.GetSpectrum(
    np.linspace(16.4, 17.6, 500),
    ["LCP", "RCP"],
    Workers=20,
)
cd = rows[rows["pol"] == "LCP"]["A"] - rows[rows["pol"] == "RCP"]["A"]
```

For angle-resolved circular dichroism where LCP and RCP are sampled at
different angle sets, prefer `GetSpectrumForAnglePolarizationPairs` so shared
angles reuse one S-matrix solve.

For diffraction-order-resolved output, use the matching order APIs:

```python
orders = sim.GetDiffractionOrders(wavelengths, ["TE", "TM"], Workers=20)
orders_by_angle = sim.GetDiffractionOrdersForAngles(
    wavelengths,
    angles,
    ["TE"],
    Phi=0.0,
    Workers=20,
)
orders_by_pair = sim.GetDiffractionOrdersForAnglePolarizationPairs(
    wavelengths,
    angles,
    polarizations,
    Phi=0.0,
    Workers=20,
)
```

These methods use the same solve ordering as the corresponding total-spectrum
methods, but expand each solve row into one row per retained diffraction order.
Returned fields are `lambda_um`, `pol`, `theta_deg`, `phi_deg`, `harmonics`,
`order_index`, `m`, `n`, `R`, and `T`.

The minimal spectrum example in `../README.md` and the paper-reproduction
scripts in `../examples/` show complete structure construction and result
post-processing.

### Python peak-search API

For an independent RCWA peak in a narrow, single-maximum bracket, use:

```python
peak = sim.FindAbsorptivityPeak(
    (15.54, 15.63),
    "TE",
    Theta=-8.0,
    Phi=0.0,
    GridPoints=13,
    Refinements=3,
    Workers=14,
)
```

The backend performs successively refined wavelength batches and returns the
final `R`, `T`, `A`, and wavelength. A boundary maximum is rejected instead of
being silently reported as an interior resonance.

### Python S-Parameter API

```python
s = sim.GetSParameters(16.9, Theta=5.0, Phi=0.0)
```

`GetSParameters(Wavelength, Theta=None, Phi=None)` returns one structured row
per complex matrix entry in `S11`, `S12`, `S21`, and `S22`. If `Theta` or `Phi`
is omitted, the current `SetExcitationPlanewave` angles are used.

`GetSParametersAnalyticContinuationAtGamma(Wavelength)` provides the same full
amplitude matrix at the Gamma point while intentionally bypassing real-power
normalization of the exterior channels. It is restricted to zero tangential
wavevector and is intended for complex-frequency pole maps. For a
nondispersive system, choose a real reference frequency and multiply every
region's relative epsilon and mu, including both exterior half spaces, by the
complex frequency ratio before calling this method. Ordinary spectra must keep
using `GetSParameters` or `GetSpectrum`.

Structured-array fields:

| Field | Meaning |
| --- | --- |
| `lambda_um`, `theta_deg`, `phi_deg` | Solve state. |
| `harmonics` | Number of retained harmonics. |
| `block` | `S11`, `S12`, `S21`, or `S22`. |
| `out_port`, `in_port` | Port numbers, with `1` as the superstrate/top side and `2` as the substrate/bottom side. |
| `out_m`, `out_n`, `out_pol` | Output diffraction order and TE/TM channel. |
| `in_m`, `in_n`, `in_pol` | Input diffraction order and TE/TM channel. |
| `value_re`, `value_im` | Complex amplitude coefficient. |
| `magnitude`, `phase_rad` | Magnitude and phase of the coefficient. |

These values are amplitudes. To compare with `GetSpectrum`, convert amplitudes
to power with the appropriate incident/reflected/transmitted flux weights; do
not treat `|S|^2` as a transmission efficiency when the two ports have
different media.

For wavelength sweeps that only need the zero diffraction order, use the lighter
batch helper:

```python
a = sim.GetZeroOrderAmplitudes(
    np.linspace(1.0, 3.5, 251),
    Theta=0.0,
    Phi=0.0,
    Workers=20,
)
```

`GetZeroOrderAmplitudes(Wavelengths, Theta=None, Phi=None, Workers=1)` returns
one structured row per zero-order complex amplitude in the `R` and `T` 2x2
TE/TM blocks. Rows are wavelength-major, then block-major, then output/input
polarization. The block convention is row=`out_pol`, column=`in_pol`, with
polarization order `TE`, `TM`.

Structured-array fields:

| Field | Meaning |
| --- | --- |
| `lambda_um`, `theta_deg`, `phi_deg` | Solve state. |
| `harmonics` | Number of retained harmonics. |
| `block` | `R` for reflected zero order or `T` for transmitted zero order. |
| `out_pol`, `in_pol` | Output and input TE/TM channel labels. |
| `value_re`, `value_im` | Complex amplitude coefficient. |
| `magnitude`, `phase_rad` | Magnitude and phase of the coefficient. |

This helper is intended for Jones-matrix spectra, circular conversion plots,
phase retardance, and paper reproduction workflows. It avoids materializing the
full `S11/S12/S21/S22` matrices for every wavelength.

For directional thermal radiation, use the power-normalized complete two-port
channel deficits rather than substituting reverse-angle absorptivity for
emissivity:

```python
channels = sim.GetDirectionalThermalChannels(
    np.linspace(10.0, 12.0, 401),
    Theta=-1.0,
    Phi=60.0,
    Workers=20,
    Polarizations=["TE", "TM", "LCP", "RCP"],
)
```

`GetDirectionalThermalChannels(Wavelengths, Theta=None, Phi=None, Workers=1,
Polarizations=None)` returns the upward zero-order channels requested through
`Polarizations`. The backwards-compatible default is `TE` and `TM`; request
`LCP` or `RCP` explicitly when needed. Circular rows retain the coherent TE-TM
cross term; they are not an incoherent average. `absorptivity` is the column deficit and `emissivity` is the row
deficit of the flux-normalized complete scattering matrix. Both sums include
every propagating diffraction and polarization channel at both lossless
exterior ports. A lossy semi-infinite substrate is part of the absorbing body,
so its complex-`kz` modes are excluded from the radiative-port sum.

The thermal APIs require a passive, local, linear stack. Under the project-wide
`exp(-i*omega*t)` convention, every material is checked using the complete
Hermitian loss operators `(epsilon-epsilon^H)/(2i)` and
`(mu-mu^H)/(2i)`, including off-diagonal tensor terms. A materially negative
eigenvalue raises an error naming the material; ordinary spectrum and
S-parameter APIs continue to support active media. Opaque manually constructed
`PrecomputedPeriodicLayer2D` blocks are rejected by thermal APIs unless their
`sourceMaterials` are retained by `precomputePatternedLayer2d`.

`I-S^H S` and `I-S S^H` describe one passive body (or a stack whose emitting
constituents share one temperature). They are not a substitute for summing the
separate fluctuating-current correlations of layers held at different
temperatures. This scope follows Beenakker's scattering formulation of thermal
emission (DOI `10.1103/PhysRevLett.81.1829`), the general fluctuational-
electrodynamics trace formulation (DOI `10.1103/PhysRevB.86.115423`), and the
adjoint Kirchhoff law for nonreciprocal emitters (DOI
`10.1103/PhysRevX.12.021023`).

Returned fields are `lambda_um`, `pol`, `theta_deg`, `phi_deg`, `harmonics`,
`absorptivity`, `emissivity`, `incident_scattering`, and
`outgoing_scattering`. The identities are
`absorptivity = 1 - incident_scattering` and
`emissivity = 1 - outgoing_scattering`, up to roundoff. For a target outgoing
direction `(+theta, phi)`, evaluate the emissivity rows at the time-reversed
wavevector, represented by `(-theta, phi)` or equivalently
`(+theta, phi + 180 deg)`.

The high-level `asyrcwa.spectrum.solve_fr` helper performs these forward and
reverse calls automatically. Its `A` and `emissivity_same_channel` fields are
therefore thermal channel deficits, not the generally invalid shortcut
`A(-theta)`. The paired `A_reverse` and `emissivity_reverse_channel` fields
expose the exact opposite physical channel without reconstructing it from
reflectance. For a semi-infinite absorbing substrate, `solve_fr` reports only
radiative far-field transmission; power entering that substrate contributes to
`A`.

### Python Field API

`GetFieldPlane` returns a structured array of field samples:

```python
field = sim.GetFieldPlane(
    16.9,
    "TM",
    5.0,
    0.0,
    ["E", "H"],
    "xz",
    -4.15,
    4.15,
    161,
    0.0,
    7.1,
    181,
    0.0,
    Workers=8,
)
```

Signature:

```python
sim.GetFieldPlane(
    Wavelength,
    Polarization,
    Theta,
    Phi,
    Components,
    Plane,
    u_min_um,
    u_max_um,
    u_points,
    v_min_um,
    v_max_um,
    v_points,
    fixed_um,
    Workers=1,
)
```

Valid planes are `"xy"`, `"xz"`, and `"yz"`. Valid components are `"E"`,
`"Ex"`, `"Ey"`, `"Ez"`, `"H"`, `"Hx"`, `"Hy"`, and `"Hz"`.
Valid polarizations are `"TE"`, `"TM"`, `"Both"`, `"LCP"`, and `"RCP"`.

Example `xy` plane:

```python
field_xy = sim.GetFieldPlane(
    0.66,
    "TM",
    0.0,
    0.0,
    ["E"],
    "xy",
    -0.175,
    0.175,
    161,
    -0.175,
    0.175,
    161,
    1.60,
    Workers=8,
)
```

Field structured-array fields:

| Field | Meaning |
| --- | --- |
| `lambda_um`, `pol`, `theta_deg`, `phi_deg` | Solve state. |
| `plane`, `component` | Requested plane and component. |
| `iu`, `iv` | Grid indices. |
| `layer_index` | `-1` in the superstrate, `0..n-1` in finite layers, `n` in the substrate. |
| `x_um`, `y_um`, `z_um` | Global coordinates. |
| `z_local_um` | Local depth in the containing region. |
| `value_re`, `value_im` | Complex component value. |
| `magnitude` | Absolute value of the returned component. |

### Utility Functions

`asyrcwa.wavelength_grid(start_um, stop_um, points)` returns an evenly spaced
NumPy array. `points` must be positive.

`asyrcwa.MaterialModelTensorsBatch(Model, Wavelengths_um, Parameters=None)`
evaluates one built-in material instance for a one-dimensional wavelength
array. It returns `epsilon` and `mu` arrays with shape `(wavelengths, 3, 3)`
and avoids rebuilding the model for every sample.

## 完整 Python 签名索引

本节逐项列出 `asyrcwa` 当前导出的应用接口。签名来自实际编译后的 pybind11
模块；大写参数名与绑定保持一致。`Simulation` 没有公开构造器，必须通过下列工厂
函数创建。修改材料、几何或谐波阶数会使相关层缓存失效；RETICOLO Fourier
策略由后端固定，不作为 Python 输入参数。

### 顶层构造与函数

| 签名 | 返回值与用途 |
| --- | --- |
| `New(Lattice=1.0, Orders=None)` | 唯一仿真工厂；均匀多层、光栅和二维超表面均返回同一个 `Simulation`。`Orders` 使用 RETICOLO 矩形阶数范围。 |
| `MaterialModelTensors(Model, Wavelength_um, Parameters=None)` | 计算内置色散模型，返回 `{"epsilon": complex[3,3], "mu": complex[3,3]}`。 |
| `wavelength_grid(start_um, stop_um, points)` | 返回包含两端点的等间距 `float64` 一维数组。 |

`Orders` 传入时应写成 `(OrderX, OrderY)`；后端固定使用 RETICOLO 的矩形索引范围。

### 状态、材料与激励

| `Simulation` 方法 | 说明 |
| --- | --- |
| `SetMaterial(Name, Epsilon, Mu=None)` | 注册或替换材料。标量为各向同性值；NumPy `3x3` 复数组为张量；`Mu` 缺省时使用材料对象中的值或单位磁导率。 |
| `SetMaterialNK(Name, Path, WavelengthUnit="um", Extrapolation="error")` | 直接注册三列 `wavelength n k` 文本文件；文件从第一行开始就是数值数据，不需要表头。读取、校验、插值和介电常数转换均由 C++ 后端完成。 |
| `SetMaterialModel(Name, Model, Parameters=None)` | 以可选参数字典注册内置色散材料。未知模型、未知参数键和缺失必填物理参数会抛出 `ValueError`。 |
| `SetSuperstrate(Material)` | 设置入射半空间；材料必须已注册。远场功率接口要求其各向同性、无损且折射率实部为正。 |
| `SetSubstrate(Material)` | 设置透射半空间；材料必须已注册，可为有损或各向异性材料。 |
| `SetWavelength(Wavelength)` | 设置单状态接口使用的正波长，单位 um。 |
| `SetFrequency(Frequency)` | 设置归一化频率并令 `wavelength_um = 1 / Frequency`；不是 Hz 接口。 |
| `SetExcitationPlanewave(Angles, sAmplitude=1+0j, pAmplitude=0j, Order=0)` | 设置 `(theta_deg, phi_deg)` 和 TE(s)/TM(p) 振幅；目前仅支持 `Order=0`。任意非零 s、p 会按 TE、TM、Both 或标准 LCP/RCP 对识别。 |
| `SetHarmonicOrders(OrderX, OrderY)` | 切换到显式非负阶数模式并清空图案层缓存。 |

`SetMaterial` 的常用输入如下：

```python
sim.SetMaterial("Air", 1.0)
sim.SetMaterial("lossy", 3.2 + 0.15j, 1.0)
sim.SetMaterial("uniaxial", np.diag([2.25, 2.25, 2.56]).astype(complex))
sim.SetMaterial("magneto", epsilon_3x3, mu_3x3)
```

材料名是结构中的引用键。重新注册同名材料后，依赖它的预计算图案层会按色散性和
依赖关系失效；已经添加的层不需要重新添加。

### 统一层与图案结构

| `Simulation` 方法 | 说明 |
| --- | --- |
| `AddLayer(Name, Thickness, Material)` | 追加有限层并指定背景材料；没有区域时自动走均匀层 TMM。 |

所有层都先用 `AddLayer` 创建。随后对某层调用任意 `SetRegion...`，该层就自动成为
RCWA 图案层；不调用区域方法则保持 TMM 均匀层。光栅用跨越一个 y 周期的矩形区域
表示。锥形或轮廓结构显式拆成多个 `AddLayer + SetRegion...` 切片。

### 二维区域

后添加的区域覆盖先添加区域。`Center=(x_um, y_um)`；`Angle` 为绕局部原点的逆时针
角度；多边形 `Vertices` 是应用旋转和平移前的局部坐标。

| `Simulation` 方法 | 说明 |
| --- | --- |
| `SetRegionCircle(Layer, Material, Center, Radius, Rectangles=10)` | 添加圆，`Radius > 0`；`Rectangles` 为 RETICOLO V9 默认等角嵌套矩形数，范围 `[2,4096]`。 |
| `SetRegionRectangle(Layer, Material, Center, Angle, Halfwidths)` | 添加旋转矩形，`Halfwidths=(hx, hy)`。 |
| `SetRegionPolygon(Layer, Material, Center, Angle, Vertices)` | 添加简单多边形；顶点至少 3 个，不应自相交。 |
| `SetRegionEllipse(Layer, Material, Center, Angle, Radii, Vertices=96)` | 以正多边形近似椭圆，`Radii=(rx, ry)`。 |
| `SetRegionRegularPolygon(Layer, Material, Center, Angle, Sides, Radius)` | 添加正多边形，`Sides >= 3`。 |
| `SetRegionTriangle(Layer, Material, Center, Angle, Radius)` | 正三角形便捷接口。 |
| `SetRegionCross(Layer, Material, Center, Angle, ArmLength, ArmWidth)` | 添加由两矩形组成的十字区域。 |
| `SetRegionRing(Layer, Material, HoleMaterial, Center, OuterRadius, InnerRadius)` | 依次添加外圆和孔材料内圆；要求 `0 < InnerRadius < OuterRadius`。 |
| `SetRegionRoundedRectangle(Layer, Material, Center, Angle, Halfwidths, CornerRadius, VerticesPerCorner=16)` | 添加圆角矩形多边形近似。 |
| `SetRegionCapsule(Layer, Material, Center, Angle, Length, Radius, VerticesPerCap=32)` | 添加胶囊形多边形近似。 |
| `SetRegionStripe(Layer, Material, Center, Angle, Width)` | 添加跨越周期单胞的条带。 |

### Fourier 收敛

Python 前端固定 RETICOLO `li=1` ordered-Li、无 Lanczos 平滑和内部几何网格策略；
用户只需联合增加 `Orders` 与圆形 `Rectangles` 数检查收敛。

### 诊断、Fourier 块与缓存

| `Simulation` 方法 | 返回值 |
| --- | --- |
| `GetPowerFlux(Layer=None, zOffset=0.0)` | 返回 `[R_total, T_total]`。当前兼容实现忽略 `Layer` 和 `zOffset`，不代表指定深度的局部 Poynting 通量。 |
| `GetHarmonicBasis()` | 结构化数组，字段为 `order_index, m, n`。 |
| `GetWavevectorMatrices(Wavelength=None, Theta=None, Phi=None)` | 返回求解状态、`orders`、复数 `Kx/Ky` 对角矩阵及 `kx_diag/ky_diag`。 |
| `GetLayerFourierMatrices(Layer, Wavelength=None)` | 返回普通材料卷积块、Li Q 块、zz 逆块及因子分解元数据。 |
| `GetLayerModes(Layer, Wavelength=None, Theta=None, Phi=None)` | 返回层模态矩阵 `W`、传播常数 `gamma`、传播相位 `phase` 与模态路径标志。 |
| `PrecomputeLayerCache(Layer=None, Wavelength=None)` | 预热一个或全部图案层缓存，并返回与 `GetLayerCacheInfo` 相同的列表。 |
| `GetLayerCacheInfo(Layer=None)` | 返回一个或全部图案层的缓存状态列表；均匀层不在列表中。 |

`GetWavevectorMatrices` 字典字段：

| 字段 | 类型与含义 |
| --- | --- |
| `wavelength_um`, `theta_deg`, `phi_deg` | 实际使用的状态。 |
| `period_x_um`, `period_y_um`, `harmonics` | 晶格周期与保留谐波数 `N`。 |
| `orders` | 长度 `N` 的 `(order_index,m,n)` 结构化数组。 |
| `Kx`, `Ky` | `complex128[N,N]` 归一化切向波矢对角矩阵。 |
| `kx_diag`, `ky_diag` | 对角元素的一维视图副本。 |

`GetLayerFourierMatrices` 字典字段：

| 字段 | 类型与含义 |
| --- | --- |
| `layer`, `wavelength_um`, `harmonics`, `orders` | 层名、状态与谐波基。 |
| `eps`, `mu` | 普通卷积块字典；键为 `xx,xy,xz,yx,yy,yz,zx,zy,zz`，值为 `complex128[N,N]`。 |
| `li_eps_q`, `li_mu_q` | Li 正确乘积规则后的 Q 块，键和尺寸同上。 |
| `li_eps_q_zz_inverse`, `li_mu_q_zz_inverse` | 纵向消元使用的 `complex128[N,N]` 逆块。 |
| `metadata` | `valid`、`harmonic_count`、`factorization_complete`、标量/对角快路径、采样适用性及 formulation 信息。 |

`GetLayerModes` 的 `kind` 可能为 `UniformBerreman`、`LamellarRcwa`、
`PatternedRcwa` 等后端路径名。`W` 的列是模态状态向量，`gamma` 是对应传播常数，
`phase` 是穿过该层厚度后的相位因子；这些量是诊断数据，模态列次序不承诺跨版本
稳定。

缓存信息字典包含 `layer`、`kind`、`region_count`、
`has_precomputed_tensor_cache`、`has_geometry_cache`、`sample_count_x/y`、
`material_slot_count` 和 `cached_material_dependencies`。解析 Fourier 路径可能直接
建立张量缓存而不建立采样几何缓存，因此 `has_geometry_cache=False` 不等于未缓存。

### 求解方法完整清单

| `Simulation` 方法 | 返回值与行顺序 |
| --- | --- |
| `GetSpectrum(Wavelengths, Polarizations=None, Workers=1)` | 总功率结构化数组；波长主序、偏振次序。 |
| `GetDiffractionOrders(Wavelengths, Polarizations=None, Workers=1)` | 逐衍射级次功率；每个光谱行展开为 `N` 行。 |
| `GetSpectrumForAngles(Wavelengths, Angles, Polarizations=None, Phi=0.0, Workers=1)` | 角度主序、波长次序、偏振末序。 |
| `GetDiffractionOrdersForAngles(Wavelengths, Angles, Polarizations=None, Phi=0.0, Workers=1)` | 与上一接口相同的求解顺序，再按级次展开。 |
| `GetSpectrumForAnglePolarizationPairs(Wavelengths, Angles, Polarizations, Phi=0.0, Workers=1)` | 请求主序、波长次序；`Polarizations` 可为单字符串或与 `Angles` 等长。 |
| `GetDiffractionOrdersForAnglePolarizationPairs(Wavelengths, Angles, Polarizations, Phi=0.0, Workers=1)` | 配对请求的逐级次版本。 |
| `GetSpectrumAndDirectionalThermalChannelsForAngles(Wavelengths, Angles, Polarizations=None, Phi=0.0, Workers=1)` | 返回 `(spectrum, thermal)`，共享一次准备和完整散射级联。 |
| `FindAbsorptivityPeak(WavelengthBracket, Polarization, Theta=None, Phi=None, GridPoints=17, Refinements=3, Workers=1)` | 返回峰值字典；每轮网格缩小，边界最大值会报错。 |
| `GetSParameters(Wavelength, Theta=None, Phi=None)` | 四个两端口 S 块的全部复振幅元素。 |
| `GetSParametersAnalyticContinuation(Wavelength, BlochWavevector)` | 在 `(kx*ax/2pi, ky*ay/2pi)` 直接 Bloch 波矢处返回未功率归一化 S 矩阵。 |
| `GetSParametersAnalyticContinuationAtGamma(Wavelength)` | 上一接口在 Gamma 点的便捷形式。 |
| `GetZeroOrderAmplitudes(Wavelengths, Theta=None, Phi=None, Workers=1)` | 每波长 8 行：R/T 各自的 `2x2` TE/TM Jones 振幅。 |
| `GetDirectionalThermalChannels(Wavelengths, Theta=None, Phi=None, Workers=1, Polarizations=None)` | 完整功率归一化 S 矩阵的列/行亏损；默认返回 TE、TM。 |
| `GetFieldPlane(Wavelength, Polarization, Theta, Phi, Components, Plane, u_min_um, u_max_um, u_points, v_min_um, v_max_um, v_points, fixed_um, Workers=1)` | 返回指定平面的复电磁场结构化数组。 |

`Workers` 必须为正整数。批量接口在 C++ 中释放 Python GIL；为了避免外层任务线程
与 MKL/OMP 内层线程过度订阅，默认运行时把 BLAS 线程数设为 1，可用
`RCWA_BLAS_THREADS` 显式调整。

### 统一错误约定

- 形状、有限性、正值、范围或枚举字符串错误映射为 Python `ValueError`。
- 未注册材料、未知层名、错误结构族调用和不受支持的物理配置同样以带上下文的
  `ValueError`/`RuntimeError` 抛出，不会静默选择近似算法。
- `theta` 必须满足 `-90 < theta < 90`；`phi` 按 360 度周期归一化。
- 空波长或角度数组返回具有正确 dtype 的空数组；这不是求解失败。
- 返回的 NumPy 数组和字典矩阵拥有独立存储，修改它们不会反向修改求解器状态。

## Python 辅助模块

### `asyrcwa` 运行时工具

这些函数由 `asyrcwa.__all__` 导出，主要用于定位扩展和控制示例脚本：

| 签名 | 说明 |
| --- | --- |
| `apply_runtime_environment()` | 配置 DLL 搜索目录以及 MKL/OMP 线程环境。导入 `asyrcwa` 时已自动调用。 |
| `build_directory()` | 返回当前选择的 CMake 构建目录 `pathlib.Path`。优先读取 `RCWA_BUILD_DIR`。 |
| `module_directory()` | 返回包含 `rcwa_cpp` 扩展的目录。`RCWA_EXTENSION_DIR` 优先级最高。 |
| `load_rcwa_cpp()` | 加载并检查编译扩展；扩展过旧或缺少必需方法时抛出 `RuntimeError`。 |
| `timed_step(label)` | 上下文管理器，退出时打印 `[time] label: ...s`。 |

命令 `python -m asyrcwa` 打印实际加载的后端文件和模块目录，适合排查加载了错误
构建产物的问题。

### `asyrcwa.spectrum`

| 签名 | 说明 |
| --- | --- |
| `measure_peak(wavelengths_um, values, *, window_um=None)` | 二次插值峰位并线性插值半高宽；返回 `PeakMetrics`。 |
| `solve_forward_reverse_spectrum(*, rcwa, build_simulation, wavelengths_um, polarizations, angles_deg, phi_deg, workers_per_angle=0)` | 一次构建结构并联合求解正反角光谱与精确热通道；`0` 表示按硬件自动并发。别名为 `solve_fr`。 |
| `combine_forward_reverse_spectrum(raw, forward_channels, reverse_channels, *, forward_angle_deg, reverse_angle_deg)` | 校验并合并已有结构化数组。 |
| `summarize_forward_reverse_spectrum(spectrum, polarizations, *, include_peak_values=False)` | 打印每种偏振最大 `|A-e|`；别名为 `summarize_fr`。 |
| `plot_forward_reverse_axes(axes, spectrum, polarizations, *, markers=None)` | 在已有 Matplotlib axes 上绘制 A、e 和 eta；别名为 `plot_fr_axes`。 |

`PeakMetrics` 是只读 dataclass，字段为 `wavelength_um`、`value`、`fwhm_um`、
`left_half_max_um`、`right_half_max_um`。峰位位于窗口边界、样本少于 3、波长非严格
递增或两侧没有半高交点时，`measure_peak` 会拒绝返回误导性线宽。

正反向合并数组字段为 `lambda_um, pol, theta_deg, phi_deg, harmonics, R, T, A,
R_reverse, T_reverse, A_reverse, emissivity_same_channel,
emissivity_reverse_channel, eta, conservation, reverse_conservation`，其中
`eta = abs(A - emissivity_same_channel)`。

### `asyrcwa.material_response`

| 签名 | 说明 |
| --- | --- |
| `material_tensor_component_ratio(rcwa, model, wavelengths_um, parameters, numerator=(0,2), denominator=(0,0))` | 逐波长返回 `abs(epsilon[numerator]) / abs(epsilon[denominator])`。 |
| `wsm_gamma(rcwa, wavelengths_um, parameters)` | 上一函数对 `WSM` 模型的便捷封装。 |
| `inas_magneto_gamma(rcwa, wavelengths_um, parameters)` | 上一函数对 `InAsMagneto` 模型的便捷封装。 |
| `bismuth_iron_garnet_epsilon_tensor(wavelength_um, *, refractive_index_a=2.36, refractive_index_b_nm=413.0, extinction_a_nm=1660.0, extinction_b=15.2, gyration_prefactor=100.0, gyration_decay_per_nm=0.011, gyration_floor=0.0, magnetization_sign=1.0)` | 返回 Yan 等人 (2023) 使用的 BIG `complex128[3,3]` 经验张量。 |

## C++ 完整高层接口

应用代码通常只需包含：

```cpp
#include "rcwa/solver.hpp"
```

### `rcwa::RcwaSolver` 配置与结构方法

| 方法 | 说明 |
| --- | --- |
| `setWavelength(Real lambdaUm)` | 设置正波长。 |
| `setIncidence(Real thetaDeg, Real phiDeg=0.0)` | 设置入射角。 |
| `setHarmonicCount(int harmonicCount, LatticeTruncation truncation=Circular)` | 使用目标谐波数。 |
| `setHarmonicOrders(int orderX, int orderY, LatticeTruncation truncation=Circular)` | 使用显式阶数。 |
| `setPolarization(Polarization pol)` | 设置 `TE/TM/Both/LCP/RCP`。 |
| `setStackingAlgorithm(StackingAlgorithm algorithm)` | 设置 S 矩阵或增强 T 矩阵。 |
| `stackingAlgorithm() const` | 读取当前级联算法。 |
| `setFourierConvergenceOptions(FourierConvergenceOptions options)` | 设置完整 Fourier 收敛选项。 |
| `fourierConvergenceOptions() const` | 返回当前选项的 const 引用。 |
| `setSuperstrate(Material m)` / `setSubstrate(Material m)` | 设置两个半空间。 |
| `addLayer(LayerSpec layer)` | 追加任一 `LayerSpec` 变体。 |
| `addUniformLayer(Material mat, Real thicknessUm)` | 追加均匀层。 |
| `addGratingLayer(GratingLayer g)` | 追加一维层状光栅。 |
| `addAdaptiveTaperedGratingLayer(AdaptiveTaperedGratingLayer layer)` | 展开并追加线性锥度光栅。 |
| `addAdaptiveProfiledGratingLayer(AdaptiveProfiledGratingLayer layer)` | 展开并追加分段轮廓光栅。 |
| `addPeriodicLayer2d(PeriodicLayer2D layer)` | 追加显式采样二维层。 |
| `addPatternedLayer2d(PatternedLayer2D layer)` | 追加形状描述二维层。 |
| `addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2D layer)` | 按值追加预计算 Fourier 层。 |
| `addPrecomputedPeriodicLayer2d(PrecomputedPeriodicLayer2DPtr layer)` | 共享指针版本，适合批量重复使用缓存。 |
| `clearLayers()` | 清空全部有限层，不改变半空间和当前求解状态。 |

### `rcwa::RcwaSolver` 求解方法

| 方法 | 返回值 |
| --- | --- |
| `solve() const` | `SpectrumResult`，兼容入口。 |
| `solveSpectrumOnly() const` | 当前状态的 `SpectrumResult`。 |
| `solveSpectrumOnlyForPolarizations(polarizations) const` | 同一准备栈上的多个 `SpectrumResult`。 |
| `solveSpectrumBatchOnly(wavelengthsUm, polarizations[, workers]) const` | 波长主序、偏振次序的完整级次结果。 |
| `solveSpectrumBatchForAngles(thetaDegs, phiDeg, wavelengthsUm, polarizations[, workers]) const` | 角度主序批量完整结果。 |
| `solveSpectrumBatchForAnglePolarizations(requests, phiDeg, wavelengthsUm[, workers]) const` | 请求主序的角度/偏振配对结果。 |
| `solveSpectrumBatchTotalsOnly(wavelengthsUm, polarizations[, workers]) const` | 只返回零阶与总功率，减少内存。 |
| `solveSpectrumBatchTotalsForAngles(thetaDegs, phiDeg, wavelengthsUm, polarizations[, workers]) const` | 多角度 totals 版本。 |
| `solveSpectrumBatchTotalsForAnglePolarizations(requests, phiDeg, wavelengthsUm[, workers]) const` | 配对请求 totals 版本。 |
| `solveSParameters() const` | 完整两端口 `SParameterResult`。 |
| `solveSParametersAnalyticContinuation(blochKxReduced, blochKyReduced) const` | 直接 Bloch 波矢未归一化 S 参数。 |
| `solveSParametersAnalyticContinuationAtGamma() const` | Gamma 点解析延拓。 |
| `findAbsorptivityPeak(wavelengthMinUm, wavelengthMaxUm, polarization, gridPoints=17, refinements=3, workers=1) const` | 自适应网格吸收峰搜索。 |
| `solveZeroOrderAmplitudes() const` | 当前状态 `ZeroOrderAmplitudeResult`。 |
| `solveZeroOrderAmplitudesBatchOnly(wavelengthsUm[, workers]) const` | 零阶 Jones 振幅批量结果。 |
| `solveDirectionalThermalChannels() const` | 当前状态 `DirectionalThermalChannelResult`。 |
| `solveDirectionalThermalChannelsBatchOnly(wavelengthsUm[, workers]) const` | 定向热通道批量结果。 |
| `solveSpectrumThermalBatchForAngles(thetaDegs, phiDeg, wavelengthsUm, polarizations, workers) const` | 联合返回 `SpectrumThermalBatchResult`。 |
| `solveFieldPlane(const FieldPlaneRequest& request) const` | 返回 `FieldPlaneResult`。 |

方括号中的 `[, workers]` 表示头文件同时提供无 workers 和显式 workers 两个重载。
所有批量返回顺序与同名 Python 接口一致。

### C++ 主要输入与结果类型

| 类型 | 关键字段/职责 |
| --- | --- |
| `Tensor3` | 行主序 `3x3` 复张量；提供 `diagonal`、`isotropic` 和 `(r,c)` 访问。 |
| `Material` | `constant`、`fromIndex`、`tabulatedIndex`、`fromNkFile`、`dispersive`、`anisotropic`、`anisotropicDispersive`、`air`、`vacuum` 工厂，以及 epsilon/mu/tensor/passivity 查询。 |
| `UniformLayer` | `material`, `thicknessUm`。 |
| `GratingLayer` | ridge/groove 材料、fill、x/y 周期、偏移和厚度。 |
| `AdaptiveTaperedGratingLayer` | 顶/底填充率、自适应误差和切片范围。 |
| `AdaptiveProfiledGratingLayer` | `FillFactorProfilePoint` 列表与自适应切片参数。 |
| `PeriodicLayer2D` | 单胞周期、采样数、材料单元与厚度。 |
| `PatternedLayer2D` | background、`PatternRegion` 列表、周期、采样控制与厚度。 |
| `PrecomputedPeriodicLayer2D` | 完整 `TensorFourierMatrices` 与来源材料，用于复用。 |
| `SpectrumAnglePolarization` | `thetaDeg`, `polarization`。 |
| `SpectrumResult` | `rOrders`, `tOrders`, `centerIdx`, `R0`, `T0`, `rTotal`, `tTotal`, `conservation`, 状态和 `N`。 |
| `SpectrumTotalsResult` | 不含逐级次数组的轻量 `SpectrumResult`。 |
| `SParameterResult` | `S11/S12/S21/S22`、`orders`、`centerIdx`、状态和 `N`。 |
| `ZeroOrderAmplitudeResult` | `R/T` 两个 `2x2` TE/TM 复振幅矩阵及状态。 |
| `DirectionalThermalChannelResult` | TE/TM/LCP/RCP 的吸收、发射、入射/出射散射功率及状态。 |
| `SpectrumThermalBatchResult` | `spectra` 与 `thermalChannels` 两个向量。 |
| `FieldPlaneRequest` | 平面、u/v 范围与点数、固定坐标、分量列表、workers。 |
| `FieldPlaneResult` | 平面元数据、坐标轴、分量和按 `iv -> iu -> component` 排列的样本。 |

### C++ 材料模型函数

`Material` 的完整高层方法如下：

| 方法 | 说明 |
| --- | --- |
| `Material()` | 构造名为 `unnamed` 的单位 epsilon/mu 材料。 |
| `Material::constant(name, epsilon, mu={1,0})` | 常数各向同性材料。 |
| `Material::fromIndex(name, nComplex)` | 由复折射率建立 `epsilon=n^2, mu=1` 的材料。 |
| `Material::tabulatedIndex(name, wavelengthsUm, refractiveIndices, extrapolation=Error)` | 由严格递增的波长与复折射率样本建立线性插值材料。 |
| `Material::fromNkFile(name, path, wavelengthScaleToUm=1, extrapolation=Error)` | 在 C++ 后端读取无表头三列 `wavelength n k` 文件并建立插值材料。 |
| `Material::dispersive(name, DispersionFn)` | 回调按波长返回标量 `(epsilon, mu)`。 |
| `Material::anisotropic(name, epsilon, mu=I)` | 常数 `Tensor3` 材料。 |
| `Material::anisotropicDispersive(name, TensorDispersionFn)` | 回调按波长返回 `(epsilonTensor, muTensor)`。 |
| `Material::vacuum()` / `Material::air()` | 单位介质便捷工厂。 |
| `name() const` | 返回材料名。 |
| `epsilon(wavelengthUm=1)` / `mu(wavelengthUm=1)` | 返回标量值；仅适用于标量材料。 |
| `tensors(wavelengthUm=1)` | 返回 `(Tensor3 epsilon, Tensor3 mu)`。 |
| `epsilonTensor(wavelengthUm=1)` / `muTensor(wavelengthUm=1)` | 分别返回完整张量。 |
| `passivity(wavelengthUm=1)` | 返回 `MaterialPassivityResult`。 |
| `refractiveIndex(wavelengthUm=1)` | 返回标量材料的复折射率。 |
| `isDispersive() const` | 是否包含标量或张量色散回调。 |

`material.hpp` 为每个可参数化模型提供 `...ModelParameters`、
`default...ModelParameters(model)` 和 `builtinMaterialModel(name, model,
parameters)` 重载。模型族为 `GrapheneKubo`、`Drude`、
`PhononPolariton`、`YIG`、`WSM`、`BlackPhosphorusDrude`、`InAsMagneto`、
`CdTeLorentzDrude`、`InSbLorentzDrude`；无参数内置材料还包括 `Air`、`Si` 和
`Ge`。模型别名与 Python 参数键以前文“Python Materials”表为准。

`materialPassivity(epsilon, mu)` 和 `Material::passivity(wavelengthUm)` 返回
`MaterialPassivityResult`，包含 epsilon/mu 损耗算子的最小本征值、数值容差和
`passive` 标志。

## C++ 头文件索引

| 头文件 | 公开内容 | 稳定性建议 |
| --- | --- | --- |
| `rcwa/types.hpp` | `Real/Complex`、向量、谐波索引、偏振、截断枚举及批量工作调度。 | 基础类型稳定。 |
| `rcwa/matrix.hpp` | `Matrix`、`DiagonalOperator`、BLAS/LAPACK 运算、本征分解。 | 后端低层接口。 |
| `rcwa/material.hpp` | 张量、材料、耗散检查和内置色散模型。 | 应用接口。 |
| `rcwa/fourier.hpp` | 谐波基、卷积矩阵、Fourier formulation 与 `TensorFourierMatrices`。 | 高级/算法接口。 |
| `rcwa/li_factorization.hpp` | 一维、二维及张量 Li Q 块的唯一分解入口。 | 算法接口；不要在应用层重复分解。 |
| `rcwa/pattern.hpp` | 采样层、预计算层、图案区域和形状生成器。 | 几何应用接口。 |
| `rcwa/geometry.hpp` | 层结构、`LayerSpec`、栈构建与定位工具。 | `LayerSpec` 为应用接口，其余偏后端。 |
| `rcwa/wavevector.hpp` | 归一化切向波矢和对角算子构建。 | 后端低层接口。 |
| `rcwa/physics.hpp` | 各向同性波、Fresnel 和薄膜参考解。 | 验证/低层接口。 |
| `rcwa/layer.hpp` | 均匀、层状和周期层本征模求解。 | 后端低层接口。 |
| `rcwa/smatrix.hpp` | S 矩阵级联、衍射效率、零阶振幅和热通道。 | 高级接口。 |
| `rcwa/scalar_stack.hpp` | 均匀标量栈快速路径检测与求解。 | 内部优化接口。 |
| `rcwa/field.hpp` | 场分量/平面枚举、请求与结果结构。 | 应用接口。 |
| `rcwa/engine.hpp` | `StackEngine`，统一准备、散射和效率编排。 | 后端编排接口。 |
| `rcwa/solver.hpp` | `RcwaSolver` 及全部高层结果类型。 | 首选稳定 C++ 应用入口。 |

若应用只需要建模和求解，不应直接组合 `matrix.hpp`、`fourier.hpp`、`layer.hpp`
和 `engine.hpp`；这些接口暴露是为了算法测试、诊断和扩展，内部维度与模态排序可随
实现优化而变化。

## Supported Boundary

Current material and patterning support:

- Uniform layers support isotropic and anisotropic epsilon/mu tensors.
- 1D and 2D patterned layers support isotropic or anisotropic epsilon/mu
  tensors. Tensor permeability is Fourier-factorized and solved by the full
  periodic tensor eigenproblem when the scalar-mu fast path is not valid.
- Scalar 1D lamellar layers use Li inverse-rule blocks only in the interface
  normal direction.
- `PatternedLayer2D` constructs direct and factorized blocks from the same
  piecewise-constant cell partition. Axis-aligned tensor rectangles are exact;
  circles use area-normalized RETICOLO rectangles; arbitrary shapes converge
  with the configured rectilinear grid. Ordered y-then-x factorization is
  completed before a circular harmonic basis is projected. The inverse of the
  factorized `Qzz` block is used for longitudinal-field elimination.
- Full-height/full-width lamellar gratings retain the directional Li inverse
  rule and uniform multilayers retain their compact analytic propagation path.
- TE, TM, LCP, and RCP spectra at one wavelength/angle share the same prepared
  Li-factorized modal stack. High-contrast arbitrary shapes should still be
  checked by increasing the internal rectilinear cell resolution and harmonic
  order together.
- Sampled 2D layers require enough samples to cover the retained
  order-difference range.
- Curved or oblique anisotropic interfaces are supported as rectilinear convergence
  paths, not as unconditional finite-order strict equivalents. Their metadata
  reports `CurvedOrOblique` and `requires_joint_convergence=true`.
- `PrecomputedPeriodicLayer2D` inputs must provide complete factorized epsilon
  and mu Q blocks plus their zz-inverse blocks. Missing blocks raise an error;
  the solver no longer substitutes ordinary material convolution implicitly.

These restrictions are part of the public contract: unsupported configurations
raise explicit errors instead of silently using an incorrect factorization.
