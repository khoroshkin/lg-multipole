"""Check conformity between analytical and numerical implementations.

The test cases below are synthetic and are NOT molecular data from the manuscript.
They are chosen only to exercise different terms of the analytical expression.
"""

from __future__ import annotations

import numpy as np

from lg_multipole import (
    LGBeam,
    TransitionMoments,
    integrated_matrix_element_analytic,
    integrated_matrix_element_numeric,
)

def make_transition_moments():
    """Return arbitrary transition moments used only for the conformity check."""

    mu = np.array(
        [0.12, -0.08, 0.05],
        dtype=np.complex128,
    )

    m = np.array(
        [0.50, -0.20, 0.70],
        dtype=np.complex128,
    )

    Q = np.array(
        [
            [0.08, 0.03, -0.02],
            [0.03, -0.05, 0.04],
            [-0.02, 0.04, -0.03],
        ],
        dtype=np.complex128,
    )

    return TransitionMoments(
        electric_dipole=mu,
        magnetic_dipole=m,
        electric_quadrupole=Q,
    )


def main():
    moments = make_transition_moments()

    print(
        f"{'n':>3} "
        f"{'analytic':>22} "
        f"{'numeric':>22} "
        f"{'relative error':>18}"
    )

    for n in range(6):
        beam = LGBeam(
            topological_charge=n,
            wavelength_m=800e-9,
            waist_m=1.0e-6,
        )

        analytic = integrated_matrix_element_analytic(
            beam,
            moments,
        )

        numeric = integrated_matrix_element_numeric(
            beam,
            moments,
        )

        relative_error = abs(numeric - analytic) / abs(analytic)

        print(
            f"{n}\t"
            f"{analytic:.12e}\t"
            f"{numeric:.12e}\t"
            f"{relative_error:.3e}"
        )


if __name__ == "__main__":
    main()
