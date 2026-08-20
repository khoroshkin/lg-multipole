"""Paraxial Laguerre-Gaussian field."""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import eval_genlaguerre

from .models import LGBeam


def scalar_lg_field(x, y, z, beam: LGBeam):
    """Return the normalized scalar LG amplitude for radial index zero."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = beam.topological_charge
    wavelength = beam.wavelength_m
    w0 = beam.waist_m

    k = 2.0 * np.pi / wavelength
    z_rayleigh = np.pi * w0**2 / wavelength
    zeta = z / z_rayleigh

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    w_z = w0 * np.sqrt(1.0 + zeta**2)
    p = 0 # radial index
    gouy_phase = (2 * p + n + 1) * np.arctan(zeta)
    laguerre = eval_genlaguerre(p, n, 2.0 * rho**2 / w_z**2)

    normalization = (np.sqrt(2.0) ** (n + 1)) * np.sqrt(
        factorial(p) / (factorial(p + n) * np.pi)
    )

    return (
        normalization
        * np.exp(-1.0j * n * phi)
        * (rho / w_z) ** n
        * laguerre
        * np.exp(1.0j * k * z)
        * np.exp(-rho**2 / (w0**2 * (1.0 + 1.0j * zeta)))
        * np.exp(-1.0j * gouy_phase)
        * (w0 / w_z)
    )


def electric_field(x, y, z, beam: LGBeam) -> np.ndarray:
    """Return the complex Cartesian electric field vector."""
    scalar = scalar_lg_field(x, y, z, beam)
    px, py, pz = beam.polarization
    return np.array(
        [px * scalar, py * scalar, pz * scalar],
        dtype=np.complex128,
    )


def electric_field_gradient(x, y, z, beam: LGBeam) -> np.ndarray:
    """ Return g[i, j] = dE_j / dx_i """
    h = beam.finite_difference_step_m

    d_dx = (
        electric_field(x + h, y, z, beam) - electric_field(x - h, y, z, beam)
    ) / (2.0 * h)
    d_dy = (
        electric_field(x, y + h, z, beam) - electric_field(x, y - h, z, beam)
    ) / (2.0 * h)
    d_dz = (
        electric_field(x, y, z + h, beam) - electric_field(x, y, z - h, beam)
    ) / (2.0 * h)

    gradient_per_m = np.array([d_dx, d_dy, d_dz], dtype=np.complex128)

    return gradient_per_m * beam.bohr_radius_m


def magnetic_field(x, y, z, beam: LGBeam) -> np.ndarray:
    """Return the complex magnetic field from Faraday's law."""
    gradient = electric_field_gradient(x, y, z, beam)

    d_ex_dx, d_ey_dx, d_ez_dx = gradient[0]
    d_ex_dy, d_ey_dy, d_ez_dy = gradient[1]
    d_ex_dz, d_ey_dz, d_ez_dz = gradient[2]

    curl_e = np.array(
        [
            d_ez_dy - d_ey_dz,
            d_ex_dz - d_ez_dx,
            d_ey_dx - d_ex_dy,
        ],
        dtype=np.complex128,
    )

    wavelength_au = beam.wavelength_m / beam.bohr_radius_m
    omega_au = 2.0 * np.pi * beam.speed_of_light_au / wavelength_au

    return curl_e / (1.0j * omega_au)
