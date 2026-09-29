from pathlib import Path
import importlib.util
import numpy as np


def load_native():
    directory=Path(__file__).parent/'native'
    files=list(directory.glob('_final_c4_native*.pyd'))+list(directory.glob('_final_c4_native*.so'))
    if len(files)!=1:raise RuntimeError('C4_BACKEND=native requested but exactly one compiled module is required; use C4_BACKEND=python explicitly')
    spec=importlib.util.spec_from_file_location('_final_c4_native',files[0]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.abi_version!=1:raise RuntimeError('Native ABI version mismatch')
    np.testing.assert_array_equal(module.distances(np.array([[3.,4.]]),np.zeros(2),np.zeros((1,2))),np.array([5.]))
    return module
