import numpy as np


def mean_and_ci(arr, z=1.96, ddof=1):
    print(z, ddof)
    return (arr.mean(), z * arr.std(ddof=ddof) / np.sqrt(arr.shape[0]))
