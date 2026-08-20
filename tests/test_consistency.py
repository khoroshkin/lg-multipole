import numpy as np
import pytest

from lg_multipole import (
    LGBeam,
    TransitionMoments,
    integrated_matrix_element_analytic,
    integrated_matrix_element_numeric,
)


def _q_xy(value: float):
    q = np.zeros((3, 3), dtype=np.complex128)
    q[0, 1] = q[1, 0] = value
    return q


CASES = [
    TransitionMoments(
        electric_dipole=np.zeros(3),
        magnetic_dipole=np.array([0.0, 0.0, 3.38]),
        electric_quadrupole=np.zeros((3, 3)),
    ),
    TransitionMoments(
        electric_dipole=np.zeros(3),
        magnetic_dipole=np.array([1.3, -0.7, 0.0]),
        electric_quadrupole=np.zeros((3, 3)),
    ),
    TransitionMoments(
        electric_dipole=np.zeros(3),
        magnetic_dipole=np.zeros(3),
        electric_quadrupole=_q_xy(0.09),
    ),
    TransitionMoments(
        electric_dipole=np.array([0.12, -0.04j, 0.0]),
        magnetic_dipole=np.array([0.6, -0.25, 0.9]),
        electric_quadrupole=np.array(
            [
                [0.08, 0.02, 0.01j],
                [0.02, -0.04, 0.03],
                [0.01j, 0.03, -0.04],
            ],
            dtype=np.complex128,
        ),
    ),
]


@pytest.mark.parametrize("moments", CASES)
@pytest.mark.parametrize("n", [0, 2, 5])
def test_analytic_matches_numeric(moments, n):
    beam = LGBeam(
        topological_charge=n,
        wavelength_m=800e-9,
        waist_m=1.0e-6,
    )

    analytic = integrated_matrix_element_analytic(beam, moments)
    numeric = integrated_matrix_element_numeric(
        beam,
        moments,
        radial_nodes=12,
        azimuthal_samples=48,
    )

    assert np.isclose(numeric, analytic, rtol=2e-6, atol=1e-12)


def test_axial_magnetic_case_is_linear_in_n_plus_one():
    moments = CASES[0]

    values = []
    for n in range(5):
        beam = LGBeam(
            topological_charge=n,
            wavelength_m=415e-9,
            waist_m=1.0e-6,
        )
        values.append(integrated_matrix_element_analytic(beam, moments))

    first_differences = np.diff(values)
    assert np.allclose(first_differences, first_differences[0], rtol=1e-12, atol=1e-15)
