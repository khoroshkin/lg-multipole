"""Reproduce the orientation-averaged OAM sweeps corresponding to manuscript Fig. 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lg_multipole import (
    CIRCULENE,
    ISOPHLORIN,
    LGBeam,
    orientation_average,
    random_rotation_matrices,
)


def calculate_curve(
    transition,
    *,
    n_max: int,
    waist_m: float,
    number_of_orientations: int,
    seed: int,
):
    charges = np.arange(n_max + 1)
    rotations = random_rotation_matrices(number_of_orientations, seed=seed)

    values = np.empty(len(charges), dtype=float)

    for i, n in enumerate(charges):
        beam = LGBeam(
            topological_charge=int(n),
            wavelength_m=transition.wavelength_m,
            waist_m=waist_m,
        )
        values[i] = orientation_average(
            beam,
            transition.moments,
            rotations=rotations,
        )

    return charges, values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--waist-m", type=float, default=1.0e-6)
    parser.add_argument("--orientations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/figure_2_reproduction.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    waist_m = args.waist_m

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, transition in zip(axes, (ISOPHLORIN, CIRCULENE)):
        n, values = calculate_curve(
            transition,
            n_max=args.n_max,
            waist_m=waist_m,
            number_of_orientations=args.orientations,
            seed=args.seed,
        )

        ax.plot(n, values, marker="o", markersize=3)
        ax.set_title(transition.name.capitalize())
        ax.set_xlabel("Topological charge n")
        ax.set_ylabel("Orientation-averaged integrated matrix element")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)

    if args.show:
        plt.show()

    print(args.output)


if __name__ == "__main__":
    main()
