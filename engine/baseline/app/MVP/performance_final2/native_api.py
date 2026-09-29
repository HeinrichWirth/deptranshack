from .native import _c4_sprint as native
import numpy as np

def startup():
    np.testing.assert_array_equal(native.exact_nn_bruteforce(np.array([[1.,2.]]),np.array([[1.,1.]])),[1.])
    return native
