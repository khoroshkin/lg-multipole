"""Reproduce the fixed-orientation OAM sweeps corresponding to manuscript Fig. 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from lg_multipole import (
    CIRCULENE,
    ISOPHLORIN,
    LGBeam,
    integrated_matrix_element_analytic,
    rotate_transition_moments,
)


def calculate_curves(transition, n_max: int, waist_m: float):
    charges = np.arange(n_max + 1)

    z_oriented = transition.moments

    # Rotate the molecular principal z axis onto the laboratory y axis.
    rotation_to_y = Rotation.from_euler("x", -90.0, degrees=True).as_matrix()
    y_oriented = rotate_transition_moments(rotation_to_y, transition.moments)

    z_values = []
    y_values = []

    for n in charges:
        beam = LGBeam(
            topological_charge=int(n),
            wavelength_m=transition.wavelength_m,
            waist_m=waist_m,
        )
        z_values.append(integrated_matrix_element_analytic(beam, z_oriented))
        y_values.append(integrated_matrix_element_analytic(beam, y_oriented))

    return charges, np.asarray(z_values), np.asarray(y_values)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--waist-m", type=float, default=1.0e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/figure_1_reproduction.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    waist_m = args.waist_m

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, transition in zip(axes, (ISOPHLORIN, CIRCULENE)):
        n, z_values, y_values = calculate_curves(transition, args.n_max, waist_m)

        ax.plot(n, z_values, marker="o", markersize=3, label="Z-oriented frame")
        ax.plot(n, y_values, marker="o", markersize=3, label="Y-oriented frame")
        ax.set_title(transition.name.capitalize())
        ax.set_xlabel("Topological charge n")
        ax.set_ylabel("Integrated matrix element")
        ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)

    if args.show:
        plt.show()

    print(args.output)


if __name__ == "__main__":
    main()
