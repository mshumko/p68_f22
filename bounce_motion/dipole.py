import numpy as np

class Dipole:
    """
    A class for the dipole magnetic field
    """
    def __init__(self, X, m) -> None:
        self.X = X
        self.m = m

    def __abs__(self):
        # np.linalg.norm()
        return