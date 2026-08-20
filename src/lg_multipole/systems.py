"""Molecular transition data used in the manuscript."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import TransitionMoments


@dataclass(frozen=True)
class MolecularTransition:
    name: str
    wavelength_m: float
    moments: TransitionMoments


def _symmetric_q_xy(value: float) -> np.ndarray:
    q = np.zeros((3, 3), dtype=np.complex128)
    q[0, 1] = value
    q[1, 0] = value
    return q


ISOPHLORIN = MolecularTransition(
    name="isophlorin",
    wavelength_m=1548e-9,
    moments=TransitionMoments(
        electric_dipole=np.zeros(3, dtype=np.complex128),
        magnetic_dipole=np.array([0.0, 0.0, 4.86], dtype=np.complex128),
        electric_quadrupole=_symmetric_q_xy(0.09),
    ),
)


CIRCULENE = MolecularTransition(
    name="circulene",
    wavelength_m=415e-9,
    moments=TransitionMoments(
        electric_dipole=np.zeros(3, dtype=np.complex128),
        magnetic_dipole=np.array([0.0, 0.0, 3.38], dtype=np.complex128),
        electric_quadrupole=np.zeros((3, 3), dtype=np.complex128),
    ),
)
