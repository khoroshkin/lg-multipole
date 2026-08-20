"""Numerical and analytical integration of the interaction matrix element."""

import numpy as np
from scipy.special import roots_laguerre

from .interaction import interaction_matrix_element
from .models import LGBeam, TransitionMoments


def integrated_matrix_element_numeric(
    beam: LGBeam,
    moments: TransitionMoments,
    radial_nodes: int = 120,
    azimuthal_samples: int = 720,
):
    """Calculate the integrated matrix element numerically."""

    t, weights = roots_laguerre(radial_nodes)
    phi = np.linspace(0, 2 * np.pi, azimuthal_samples, endpoint=False)
    dphi = 2 * np.pi / azimuthal_samples

    integral = 0.0

    for t_i, weight_i in zip(t, weights):
        rho = np.sqrt(t_i / 2)
        angular_sum = 0.0

        for phi_i in phi:
            x = beam.waist_m * rho * np.cos(phi_i)
            y = beam.waist_m * rho * np.sin(phi_i)

            matrix_element = interaction_matrix_element(
                x,
                y,
                0.0,
                beam,
                moments,
            )

            angular_sum += np.abs(matrix_element) ** 2

        integral += weight_i * np.exp(t_i) * angular_sum * dphi

    return 0.25 * integral


def integrated_matrix_element_analytic(
    beam: LGBeam,
    moments: TransitionMoments,
):
    """Calculate the integrated matrix element using the analytical expression."""

    n = beam.topological_charge

    wavelength = beam.wavelength_m / beam.bohr_radius_m
    w0 = beam.waist_m / beam.bohr_radius_m

    k = 2 * np.pi / wavelength
    z_R = np.pi * w0**2 / wavelength
    omega = beam.speed_of_light_au * k

    mu = moments.electric_dipole
    m = moments.magnetic_dipole
    Q = moments.electric_quadrupole

    K1 = k - (n + 1) / (2 * z_R)

    K2 = (
        k**2
        - k * (n + 1) / z_R
        + (n + 1) * (n + 2) / (4 * z_R**2)
    )

    mQ = (
        (m[1] - 1j * m[0]) / omega
        + (1j * Q[0, 2] - Q[1, 2]) / 6
    )

    I1 = (
        0.5 * np.abs(mu[0] + 1j * mu[1]) ** 2
        + K1 * np.real(
            np.conj(mu[0] + 1j * mu[1]) * mQ
        )
        + 0.5 * K2 * np.abs(mQ) ** 2
    )

    I2 = (
        (n + 1) / (2 * w0**2)
        * (
            np.abs(
                np.sqrt(2) * m[2] / omega
                + (Q[0, 0] + Q[1, 1]) / (6 * np.sqrt(2))
            ) ** 2
            + np.abs(
                (Q[0, 0] - Q[1, 1] + 2j * Q[0, 1])
                / (6 * np.sqrt(2))
            ) ** 2
        )
    )

    return float(np.real(I1+I2))
