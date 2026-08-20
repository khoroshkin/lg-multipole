import numpy as np
from scipy.spatial.transform import Rotation

from lg_multipole import TransitionMoments, rotate_transition_moments


def test_rotation_preserves_vector_and_tensor_norms():
    q = np.array(
        [
            [0.1, 0.02, -0.03],
            [0.02, -0.05, 0.01],
            [-0.03, 0.01, -0.05],
        ],
        dtype=complex,
    )
    moments = TransitionMoments(
        electric_dipole=np.array([0.2, -0.1, 0.3]),
        magnetic_dipole=np.array([1.0, 2.0, -0.5]),
        electric_quadrupole=q,
    )

    rotation = Rotation.from_euler("xyz", [17, -38, 61], degrees=True).as_matrix()
    rotated = rotate_transition_moments(rotation, moments)

    assert np.isclose(
        np.linalg.norm(rotated.electric_dipole),
        np.linalg.norm(moments.electric_dipole),
    )
    assert np.isclose(
        np.linalg.norm(rotated.magnetic_dipole),
        np.linalg.norm(moments.magnetic_dipole),
    )
    assert np.isclose(
        np.linalg.norm(rotated.electric_quadrupole),
        np.linalg.norm(moments.electric_quadrupole),
    )
    assert np.allclose(
        rotated.electric_quadrupole,
        rotated.electric_quadrupole.T,
    )
