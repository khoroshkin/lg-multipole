import numpy as np
import pytest

from lg_multipole import LGBeam, TransitionMoments


def test_negative_topological_charge_is_rejected():
    with pytest.raises(ValueError):
        LGBeam(topological_charge=-1)


def test_transition_moment_shapes_are_checked():
    with pytest.raises(ValueError):
        TransitionMoments(
            electric_dipole=np.zeros(2),
            magnetic_dipole=np.zeros(3),
            electric_quadrupole=np.zeros((3, 3)),
        )
