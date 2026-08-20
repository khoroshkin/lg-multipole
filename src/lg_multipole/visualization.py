"""Visualization of the LG field and light-molecule interaction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .field import (
    scalar_lg_field,
    electric_field_gradient,
    magnetic_field,
)
from .interaction import interaction_matrix_element
from .models import LGBeam, TransitionMoments


def visualize_scalar_field(
    beam: LGBeam,
    z: float = 0.0,
    xy_limit: float = 2.0e-6,
    npts: int = 400,
):
    """Visualize the scalar LG field in the transverse plane."""

    x = np.linspace(-xy_limit, xy_limit, npts)
    y = np.linspace(-xy_limit, xy_limit, npts)
    X, Y = np.meshgrid(x, y, indexing="xy")

    field = scalar_lg_field(X, Y, z, beam)

    h = beam.finite_difference_step_m

    dfield_dx = (
        scalar_lg_field(X + h, Y, z, beam)
        - scalar_lg_field(X - h, Y, z, beam)
    ) / (2.0 * h)

    dfield_dy = (
        scalar_lg_field(X, Y + h, z, beam)
        - scalar_lg_field(X, Y - h, z, beam)
    ) / (2.0 * h)

    gradient = np.sqrt(
        np.abs(dfield_dx) ** 2
        + np.abs(dfield_dy) ** 2
    )

    amplitude = np.abs(field)
    phase = np.angle(field)

    length_scale = amplitude / (gradient + 1.0e-30)

    extent = [
        x[0] * 1.0e6,
        x[-1] * 1.0e6,
        y[0] * 1.0e6,
        y[-1] * 1.0e6,
    ]

    plots = [
        (np.real(field), "Re(E)"),
        (np.imag(field), "Im(E)"),
        (amplitude, "|E|"),
        (phase, "arg(E)"),
        (gradient, "|∇E|"),
        (length_scale * 1.0e9, "|E|/|∇E| (nm)"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
    )

    for ax, (data, title) in zip(axes.flat, plots):
        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
        )

        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

        fig.colorbar(image, ax=ax)

    plt.show()


def visualize_interaction(
    beam: LGBeam,
    moments: TransitionMoments,
    z: float = 0.0,
    xy_limit: float = 3.0e-6,
    step: float = 2.0e-7,
):
    """Visualize the magnitude of the interaction matrix element."""

    x = np.arange(
        -xy_limit,
        xy_limit + step,
        step,
    )

    y = np.arange(
        -xy_limit,
        xy_limit + step,
        step,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    matrix_element = np.empty_like(
        X,
        dtype=np.complex128,
    )

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):

            matrix_element[j, i] = interaction_matrix_element(
                X[j, i],
                Y[j, i],
                z,
                beam,
                moments,
            )

    fig, ax = plt.subplots(
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        X * 1.0e6,
        Y * 1.0e6,
        np.abs(matrix_element),
        shading="nearest",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("|M_if|")

    fig.colorbar(image, ax=ax)

    plt.show()


def visualize_magnetic_field(
    beam: LGBeam,
    z: float = 0.0,
    xy_limit: float = 4.0e-6,
    npts: int = 400,
):
    """Visualize magnetic-field components in the transverse plane."""

    x = np.linspace(
        -xy_limit,
        xy_limit,
        npts,
    )

    y = np.linspace(
        -xy_limit,
        xy_limit,
        npts,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    Bx = np.empty_like(
        X,
        dtype=np.complex128,
    )

    Bz = np.empty_like(
        X,
        dtype=np.complex128,
    )

    for j in range(npts):
        for i in range(npts):

            B_field = magnetic_field(
                X[j, i],
                Y[j, i],
                z,
                beam,
            )

            Bx[j, i] = B_field[0]
            Bz[j, i] = B_field[2]

    extent = [
        x[0] * 1.0e6,
        x[-1] * 1.0e6,
        y[0] * 1.0e6,
        y[-1] * 1.0e6,
    ]

    plots = [
        (np.real(Bx), "Re(Bx)"),
        (np.imag(Bz), "Im(Bz)"),
        (np.real(Bz), "Re(Bz)"),
        (np.abs(Bz), "|Bz|"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        constrained_layout=True,
    )

    for ax, (data, title) in zip(axes.flat, plots):

        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
        )

        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

        fig.colorbar(image, ax=ax)

    plt.show()


def visualize_field_gradient(
    beam: LGBeam,
    z: float = 0.0,
    xy_limit: float = 7.0e-6,
    npts: int = 300,
):
    """Visualize dEx/dy and dEy/dx in the transverse plane."""

    x = np.linspace(
        -xy_limit,
        xy_limit,
        npts,
    )

    y = np.linspace(
        -xy_limit,
        xy_limit,
        npts,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    dEx_dy = np.empty_like(
        X,
        dtype=np.complex128,
    )

    dEy_dx = np.empty_like(
        X,
        dtype=np.complex128,
    )

    for j in range(npts):
        for i in range(npts):

            gradient = electric_field_gradient(
                X[j, i],
                Y[j, i],
                z,
                beam,
            )

            # gradient[direction, component]
            # gradient[1, 0] = dEx/dy
            # gradient[0, 1] = dEy/dx

            dEx_dy[j, i] = gradient[1, 0]
            dEy_dx[j, i] = gradient[0, 1]

    extent = [
        x[0] * 1.0e6,
        x[-1] * 1.0e6,
        y[0] * 1.0e6,
        y[-1] * 1.0e6,
    ]

    plots = [
        (
            np.real(dEx_dy),
            "Re(dEx/dy)",
        ),
        (
            np.imag(dEx_dy),
            "Im(dEx/dy)",
        ),
        (
            np.abs(dEx_dy),
            "|dEx/dy|",
        ),
        (
            np.real(dEy_dx),
            "Re(dEy/dx)",
        ),
        (
            np.imag(dEy_dx),
            "Im(dEy/dx)",
        ),
        (
            np.abs(dEy_dx),
            "|dEy/dx|",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 12),
        constrained_layout=True,
    )

    for ax, (data, title) in zip(axes.flat, plots):

        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
        )

        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

        fig.colorbar(image, ax=ax)

    plt.show()




