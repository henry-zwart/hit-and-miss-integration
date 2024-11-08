import numpy as np


def mean_and_ci(arr, z=1.96, ddof=1, axis=0):
    return (
        arr.mean(axis=axis),
        z * arr.std(axis=axis, ddof=ddof) / np.sqrt(arr.shape[axis]),
    )
