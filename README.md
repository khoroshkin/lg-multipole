# lg-multipole

Research code for calculating the multipolar interaction of a circularly
polarized Laguerre-Gaussian (LG) beam with molecular transition moments.

The interaction contains electric-dipole (E1), magnetic-dipole (M1), and
electric-quadrupole (E2) contributions. The code evaluates the local
interaction matrix element, its numerical integral over the transverse beam
profile, the corresponding analytical expression, and averages over molecular
orientations.

## Scope

The implementation is intentionally limited to the model used in the
manuscript:

- paraxial Laguerre-Gaussian beam;
- non-negative topological charge `n`;
- multipole interaction through E1, M1, and E2 terms;
- numerical field gradients evaluated by central finite differences;
- analytical transverse integral corresponding to Eq. (12) of the manuscript.

The integrated quantity is the transverse integral of the squared modulus of
the interaction matrix element. It is used here to compare the dependence of
the molecular absorption response on topological charge and beam parameters;
it is not introduced as an independently normalized experimental observable.

## Installation

```bash
git clone <repository-url>
cd lg-multipole
python -m pip install .
```

## Repository layout

```text
src/lg_multipole/
    models.py          data structures
    field.py           scalar and vector LG field E, gradient(E), and B
    interaction.py     matrix element at a point
    integration.py     numerical and analytical transverse integrals 
    orientation.py     rotations and orientational averaging
    systems.py         structures data
    visualization.py   visualization of the LG field, interaction, B, and gradient(E)

scripts/
    visualize.py              run the field and interaction visualizations
    reproduce_figure_1.py     reproduce Figure 1
    reproduce_figure_2.py     reproduce Figure 2
    validate_analytic.py      compare analytical and numerical integration

```

## Quick examples
This example calculates the analytical integrated matrix element for circulene at a topological charge of n=5 and a beam waist of 1μm.

```python
from lg_multipole import (
    LGBeam,
    CIRCULENE,
    integrated_matrix_element_analytic,
)

beam = LGBeam(
    topological_charge=5,
    wavelength_m=CIRCULENE.wavelength_m,
    waist_m=1.0e-6,
)

value = integrated_matrix_element_analytic(beam, CIRCULENE.moments)
print(value)
```

Calculate the interaction at a single point

```python
from lg_multipole import LGBeam, CIRCULENE
from lg_multipole.interaction import interaction_matrix_element

beam = LGBeam(
    topological_charge=1,
    wavelength_m=415e-9,
    waist_m=1e-6,
)

M_if = interaction_matrix_element(
    x=0.5e-6,
    y=0.2e-6,
    z=0.0,
    beam=beam,
    moments=CIRCULENE.moments,
)

print(M_if)
```
This example calculates the local interaction matrix element M at a selected point (x,y,z).

## Reproduce the manuscript calculations

Fixed molecular orientations (Z-oriented and Y-oriented):

```bash
python scripts/reproduce_figure_1.py
```

Orientation-averaged curves using 1000 reproducible random rotations:

```bash
python scripts/reproduce_figure_2.py
```

By default the scripts write figures to `figures/`.

## Analytical / numerical cross-check

The analytical expression and direct numerical integration are independent
implementations of the same transverse integral. A small validation script is
included:

```bash
python scripts/validate_analytic.py
```

For unit tests a reduced quadrature is used to keep the suite fast.

## Conventions and units

Coordinates supplied to the field functions are in metres.

The transition moments are supplied in atomic units. Spatial derivatives are
converted from SI inverse metres to inverse Bohr before entering the
multipole interaction.

The LG modes are normalized consistently across topological charges, but no
absolute incident-field amplitude is specified. Consequently, the code is
intended for consistent comparison of the integrated response within the
model rather than for assigning an absolute experimental absorption cross
section.

## Molecular data

`src/lg_multipole/systems.py` contains the transition wavelength and multipole
moments listed in Table 1 of the manuscript for isophlorin and circulene.
The quadrupole tensor is stored explicitly as a symmetric Cartesian tensor.
