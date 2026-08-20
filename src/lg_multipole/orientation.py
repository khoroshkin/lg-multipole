"""Molecular rotations and orientation averaging."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .integration import integrated_matrix_element_analytic
from .models import LGBeam, TransitionMoments


def rotate_transition_moments(
    rotation_matrix: np.ndarray,
    moments: TransitionMoments,
) -> TransitionMoments:
    """Rotate vector moments as R v and the quadrupole as R Q R^T."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")

    mu_rot = rotation_matrix @ moments.electric_dipole
    m_rot = rotation_matrix @ moments.magnetic_dipole
    q_rot = rotation_matrix @ moments.electric_quadrupole @ rotation_matrix.T

    return TransitionMoments(mu_rot, m_rot, q_rot)


def random_rotation_matrices(
    number_of_orientations: int = 1000,
    *,
    seed: int = 12345,
) -> np.ndarray:
    """Generate a reproducible set of uniformly distributed SO(3) rotations."""
    if number_of_orientations <= 0:
        raise ValueError("number_of_orientations must be positive")

    rng = np.random.default_rng(seed)
    return Rotation.random(number_of_orientations, random_state=rng).as_matrix()


def orientation_average(
    beam: LGBeam,
    moments: TransitionMoments,
    *,
    rotations: np.ndarray | None = None,
    number_of_orientations: int = 1000,
    seed: int = 12345,
) -> float:
    """Average the analytical integrated matrix element over molecular orientation."""
    if rotations is None:
        rotations = random_rotation_matrices(
            number_of_orientations=number_of_orientations,
            seed=seed,
        )

    values = np.empty(len(rotations), dtype=float)

    for i, rotation_matrix in enumerate(rotations):
        rotated = rotate_transition_moments(rotation_matrix, moments)
        values[i] = integrated_matrix_element_analytic(beam, rotated)

    return float(np.mean(values))
