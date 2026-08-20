from lg_multipole import LGBeam, CIRCULENE
from lg_multipole.visualization import (
    visualize_scalar_field,
    visualize_interaction,
    visualize_magnetic_field,
    visualize_field_gradient,
)


def main():
    beam = LGBeam(
        topological_charge=1,
        wavelength_m=415e-9,
        waist_m=1.0e-6,
    )

    moments = CIRCULENE.moments

    visualize_scalar_field(beam)
    visualize_interaction(beam, moments)
    visualize_magnetic_field(beam)
    visualize_field_gradient(beam)


if __name__ == "__main__":
    main()