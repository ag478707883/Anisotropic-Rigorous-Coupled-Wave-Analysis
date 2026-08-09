# Li Fourier Factorization Used by This Solver

This note fixes the mathematical contract implemented by the Li factorization
code. It uses the project convention

    exp(-i omega t + i k0 k . r),

and the Fourier series

    f(x,y) = sum_(m,n) f_(m,n) exp(i G_(m,n) . r).

Consequently, a convolution matrix is indexed as

    [[f]]_(row,column) = f_(m_row-m_column,n_row-n_column).

This is also the index order used by `toeplitzConvMatrix` and `convMatrix2d`.

## 1. Direct and inverse rules

Let h = f g and let `[f]` denote the truncated Toeplitz convolution matrix of
f.

1. Type 1: f and g have no concurrent jumps. Use the direct (Laurent) rule:

       h = [f] g.

2. Type 2: f and g have pairwise complementary jumps, so h is continuous.
   Rewrite g = (1/f) h, apply the direct rule to that product, and solve:

       h = [1/f]^-1 g.

3. Type 3: f and g jump together but the jumps are not complementary. Neither
   finite rule is valid. Maxwell's constitutive relation must first be
   rearranged so that type-3 products disappear.

The two finite matrices `[f]` and `[1/f]^-1` have the same infinite-order
limit, but they are not interchangeable after truncation.

## 2. Why one-dimensional TE and TM differ

Consider an isotropic lamellar layer whose vertical material interfaces have
normal x. Across an interface without surface charge or current:

    E_y, E_z, H_y, H_z, D_x, and B_x are continuous.

For the electric constitutive relation D = eps E:

* TE has tangential E_y. The product eps E_y is type 1, hence

      D_y = [eps] E_y.

* TM has normal E_x while D_x is continuous. The factors eps and E_x have
  complementary jumps, hence

      D_x = [1/eps]^-1 E_x.

Thus the scalar factorized tensor for an x-normal lamellar layer is

    Q_eps = diag([1/eps]^-1, [eps], [eps]).

For a y-normal interface the first two entries are exchanged. The magnetic
tensor obeys the exact dual rule:

    Q_mu = diag([1/mu]^-1, [mu], [mu])

for an x-normal interface.

These are not two material models selected from the incident-polarization
enum. They are different blocks of one Maxwell operator. Classical 1D TE and
TM decouple and use the tangential and normal blocks respectively. Conical
incidence, anisotropy, and genuinely 2D patterns mix those blocks.

## 3. One-dimensional anisotropic tensor

For an x-normal interface, first isolate the continuous normal displacement:

    D_x = eps_xx (
        E_x + (eps_xy/eps_xx) E_y + (eps_xz/eps_xx) E_z).

The outer product is type 2; the products inside the parentheses are type 1.
Therefore

    Q_xx = [1/eps_xx]^-1,
    Q_xj = Q_xx [eps_xj/eps_xx],
    Q_ix = [eps_ix/eps_xx] Q_xx,

and, for i,j != x,

    Q_ij = [eps_ij - eps_ix eps_xj/eps_xx]
           + [eps_ix/eps_xx] Q_xx [eps_xj/eps_xx].

Every expression inside one pair of brackets is Fourier transformed as a
composite pointwise function. It must not be assembled from separately
truncated factors. Replacing eps by mu gives the magnetic Q tensor. Selecting
y as the pivot gives the y-normal formula without a second derivation.

Equivalently, define Li's block transforms for pivot p:

    (l_p^+/- A)_pp = A_pp^-1,
    (l_p^+/- A)_pj = A_pp^-1 A_pj,
    (l_p^+/- A)_ip = A_ip A_pp^-1,
    (l_p^+/- A)_ij = A_ij +/- A_ip A_pp^-1 A_pj.

Then the one-dimensional tensor operator is

    Q_p = l_p^+ F_p l_p^- (eps).

## 4. Ordered two-dimensional factorization

For coordinate-aligned discontinuities in x and y, the two partial Fourier
factorizations do not commute at finite order. This solver deliberately uses
RETICOLO V9's default `res1.li=1` y-first, then x ordering

    Q = L_x L_y(eps)
      = l_x^+ F_x l_x^- l_y^+ F_y l_y^- (eps).

The y transform and its block inversion are completed for every x cell. Only
then are those matrix entries Fourier transformed and assembled along x. A
single 2D convolution followed by one inversion is not equivalent.

For scalar eps this reduces to

    Q_xx = F_y( [1/eps]_x^-1 ),
    Q_yy = F_y( [eps]_x^-1 )^-1,
    Q_zz = F_y( [eps]_x ),

with zero off-diagonal material blocks. The complete rectangular harmonic
envelope is factorized before a circular harmonic subset is projected.

## 5. Longitudinal elimination

The layer state is `[E_x,E_y,H_x,H_y]^T`. On the default uniform, lamellar,
direct, and ordered-Li routes, longitudinal E_z and H_z are eliminated with
`Q_eps,zz^-1` and `Q_mu,zz^-1`. Their stored contract is

    liEpsQZzInverse = inverse(liEpsQ[2][2]),
    liMuQZzInverse  = inverse(liMuQ[2][2]).

Uniform media are an exact limit: Q equals the original tensor multiplied by
the harmonic identity.

The explicitly selected S4-compatible PolBasis formulations are a separate
normal-vector extension. They store a separately Fourier-factorized inverse
constitutive operator in the same solver slot; at finite order that operator
is deliberately not required to equal the algebraic inverse of `Q_zz`.

## 6. Scope and source audit

The formula sources supplied with this repository request have different
roles:

* `li1998.pdf`, PDF pages 11-12: 1D anisotropic rearrangement and the full Q
  tensor, especially Eq. (32).
* `li2003.pdf`, PDF page 5: block operators and ordered `L2 L1`, Eqs. (13)-(20).
* `phd-thesis_andre-junker.pdf`, PDF pages 53-58 and 60-61: direct/inverse
  rules, 2D partial rules, and the continuity argument for Maxwell fields.
* `JOSAA.19.000325.pdf`, PDF page 2: Laurent and inverse rules, Eqs. (11)-(14),
  plus the warning about TM convergence.
* `azu_etd_11457_sip1_m.pdf`, PDF pages 112-113: anisotropic application and
  the direct/inverse formulas.
* `2510.05973v1.pdf`, PDF pages 3-4: the same Li block operators and their
  extension to bi-anisotropic constitutive laws. This solver currently supports
  eps and mu tensors, not magneto-electric tensors.
* `Li96.pdf` is the S/R recursive-matrix paper, not Li's 1996 Fourier-series
  factorization paper.
* `JOSAA.24.002313.pdf` is a parallel scattering-matrix paper, not Schuster's
  normal-vector paper at JOSA A 24, 2880 (2007).
* `JOSAA.12.001077.pdf`, `JOSA.62.000502.pdf`, and
  `photonics-12-00943.pdf` support stable propagation, the 4x4 state reduction,
  and modern 2D RCWA context respectively; they do not redefine Li's product
  rules.
* `1406.0313v1.pdf` describes integral grating methods and is not used as a
  source for the Fourier factorization operator.

Normal-vector and subpixel formulations are convergence extensions for
arbitrary interface orientation. They must not be described as the ordered
coordinate-aligned Li operator above, and they do not change the TE/TM
continuity argument.

## 7. Implementation boundary

The implementation has one Li algorithm module:

* `include/rcwa/li_factorization.hpp` declares the internal factorization
  contract and its uniform, direct, lamellar, and ordered-2D entry points.
* `src/rcwa/li_factorization.cpp` is the only file that implements the
  pointwise `l^-` transform, block `l^+/-` transforms, Schur complements, and
  ordered `L_x L_y` (y first, then x) partial Fourier sequence.
* `src/rcwa/fourier.cpp` only validates inputs, constructs Fourier convolution
  matrices, detects uniform/1D/2D dependence, and routes them to that module.
* `src/rcwa/pattern.cpp` builds direct and ordered-Li blocks from one common
  piecewise-constant cell partition. Axis-aligned rectangles use their exact
  boundaries; circles use RETICOLO's area-normalized nested rectangles;
  arbitrary shapes use a convergent rectilinear approximation. The Python
  frontend exposes only this RETICOLO-aligned route; alternative convergence
  formulations remain C++ algorithm-test infrastructure and are not user
  selectable through `asyrcwa`.
