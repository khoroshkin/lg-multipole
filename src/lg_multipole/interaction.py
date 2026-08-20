"""Local multipole light-molecule interaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import electric_field, electric_field_gradient, magnetic_field
from .models import LGBeam, TransitionMoments


@dataclass(frozen=True)
class LocalInteractionTerms:
    """E1, M1, and E2 contributions to the local matrix element."""

    electric_dipole: complex
    magnetic_dipole: complex
    electric_quadrupole: complex

    @property
    def total(self) -> complex:
        return self.electric_dipole + self.magnetic_dipole + self.electric_quadrupole


def local_interaction_terms(
    x: float,
    y: float,
    z: float,
    beam: LGBeam,
    moments: TransitionMoments,
) -> LocalInteractionTerms:
    """Evaluate the three multipole contributions at one position."""
    e_field = electric_field(x, y, z, beam)
    b_field = magnetic_field(x, y, z, beam)
    grad_e = electric_field_gradient(x, y, z, beam)

    mu = moments.electric_dipole
    m = moments.magnetic_dipole
    q = moments.electric_quadrupole

    e1 = -np.dot(mu, e_field)
    m1 = -np.dot(m, b_field)
    e2 = -(1.0 / 6.0) * np.einsum("ab,ba->", q, grad_e)

    return LocalInteractionTerms(
        electric_dipole=complex(e1),
        magnetic_dipole=complex(m1),
        electric_quadrupole=complex(e2),
    )


def interaction_matrix_element(
    x: float,
    y: float,
    z: float,
    beam: LGBeam,
    moments: TransitionMoments,
) -> complex:
    """Return the total local E1 + M1 + E2 interaction matrix element."""
    return local_interaction_terms(x, y, z, beam, moments).total
