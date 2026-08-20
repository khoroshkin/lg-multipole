"""Data structures used by the LG-multipole model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


DEFAULT_CIRCULAR_POLARIZATION = (
    1.0 / np.sqrt(2.0),
    1.0j / np.sqrt(2.0),
    0.0 + 0.0j,
)


@dataclass(frozen=True)
class LGBeam:
    """Parameters of the paraxial LG beam used in the manuscript.

    Parameters
    ----------
    topological_charge:
        Non-negative LG topological charge ``n``.
    wavelength_m:
        Vacuum wavelength in metres.
    waist_m:
        Beam waist ``w0`` in metres.
    polarization:
        Complex Cartesian polarization vector.
    finite_difference_step_m:
        Step used for central finite differences of the electric field.
    bohr_radius_m:
        Bohr radius in metres.
    speed_of_light_au:
        Speed of light in atomic units.
    """

    topological_charge: int = 0
    wavelength_m: float = 415e-9
    waist_m: float = 1.0e-6
    polarization: tuple[complex, complex, complex] = DEFAULT_CIRCULAR_POLARIZATION
    finite_difference_step_m: float = 1.0e-11
    bohr_radius_m: float = 5.29177210903e-11
    speed_of_light_au: float = 137.036

    def __post_init__(self) -> None:
        n = self.topological_charge
        if not isinstance(n, (int, np.integer)):
            raise TypeError("topological_charge must be an integer")
        if n < 0:
            raise ValueError(
                "This implementation of the manuscript model is restricted "
                "to non-negative topological charge n."
            )
        if self.wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        if self.waist_m <= 0:
            raise ValueError("waist_m must be positive")
        if self.finite_difference_step_m <= 0:
            raise ValueError("finite_difference_step_m must be positive")
        if len(self.polarization) != 3:
            raise ValueError("polarization must have three Cartesian components")

    def with_topological_charge(self, n: int) -> "LGBeam":
        """Return a copy of the beam with topological charge ``n``."""
        return replace(self, topological_charge=n)


@dataclass(frozen=True)
class TransitionMoments:
    """Cartesian molecular transition multipole moments in atomic units."""

    electric_dipole: np.ndarray
    magnetic_dipole: np.ndarray
    electric_quadrupole: np.ndarray

    def __post_init__(self) -> None:
        mu = np.asarray(self.electric_dipole, dtype=np.complex128).copy()
        m = np.asarray(self.magnetic_dipole, dtype=np.complex128).copy()
        q = np.asarray(self.electric_quadrupole, dtype=np.complex128).copy()

        if mu.shape != (3,):
            raise ValueError("electric_dipole must have shape (3,)")
        if m.shape != (3,):
            raise ValueError("magnetic_dipole must have shape (3,)")
        if q.shape != (3, 3):
            raise ValueError("electric_quadrupole must have shape (3, 3)")

        mu.setflags(write=False)
        m.setflags(write=False)
        q.setflags(write=False)

        object.__setattr__(self, "electric_dipole", mu)
        object.__setattr__(self, "magnetic_dipole", m)
        object.__setattr__(self, "electric_quadrupole", q)
