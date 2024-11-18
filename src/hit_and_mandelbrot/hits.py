"""
    Course: Stochastic Simulation
    Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
    Student IDs: 15719227, 15393879, 13392425 
    Assignement: Hit and Miss Integration

    File description:
        This file contains many different kinds of function that returns the amount of hits within the Mandelbrot set area.
"""


import numba
import numpy as np


@numba.njit
def calculate_hits(c: np.ndarray, iterations: int):
    """
    Calculate hits without storing intermediary information.
    Returns: Integer
    """
    return calculate_sample_hits(c, iterations).sum(axis=-1)


@numba.njit
def calculate_sample_hits(c: np.ndarray, iterations: int):
    """
    Calculate hits per-sample.
    Returns: Bool array
    """
    z = c.copy()
    bounded = np.abs(c) <= 2

    for _ in range(1, iterations + 1):
        for r in range(c.shape[0]):
            for s in range(c.shape[1]):
                if bounded[r, s]:
                    z[r, s] = z[r, s] ** 2 + c[r, s]
                    if np.abs(z[r, s]) > 2:
                        bounded[r, s] = False

    return np.abs(z) <= 2


@numba.njit
def calculate_iter_hits(c: np.ndarray, iterations: np.ndarray):
    """
    Calculate total hits per-iteration.
    Returns: Integer array
    """
    record_iter_idx = 0
    max_iters = iterations[-1]

    z = c.copy()
    bounded = np.abs(c) <= 2

    # Initialise hits array, calculate hits at iter 0 if in iters array
    hits = np.zeros((c.shape[0], iterations.shape[0]), dtype=np.int64)
    if iterations[record_iter_idx] == 0:
        hits[:, 0] = bounded.sum(axis=-1)
        record_iter_idx += 1

    for i in range(1, max_iters + 1):
        for r in range(c.shape[0]):
            for s in range(c.shape[1]):
                if bounded[r, s]:
                    z[r, s] = z[r, s] ** 2 + c[r, s]
                    if np.abs(z[r, s]) > 2:
                        bounded[r, s] = False

        # If this is an iteration we want to record, record the hits
        if i == iterations[record_iter_idx]:
            hits[:, i] = bounded.sum(axis=-1)
            record_iter_idx += 1

    return hits


@numba.njit
def calculate_sample_iter_hits(c: np.ndarray, iterations: np.ndarray):
    """
    For each iteration and each sample, determines whether the sample is a hit.
    Returns: 3D bool array
    """
    max_iters = iterations[-1]

    z = c.copy()

    # Keep a boolean record of the samples which are still within bounds
    bounded = np.abs(c) <= 2

    # Record the iteration at which each point exceeds threshold
    unbounded_at = np.full(c.shape, max_iters + 1, dtype=np.int64)

    # Initialise the record for samples which are unbounded at iteration 0
    for r in range(c.shape[0]):
        for s in range(c.shape[1]):
            if not bounded[r, s]:
                unbounded_at[r, s] = 0

    # For each iteration,
    for i in range(1, max_iters + 1):
        # ... in each repeat of the estimation:
        for r in range(c.shape[0]):
            # ... for each sample,
            for s in range(c.shape[1]):
                # If the sample was previously within the bounds.
                # Note, this avoids extra computation on already-excluded samples.
                if bounded[r, s]:
                    # Iterate on the sample
                    z[r, s] = z[r, s] ** 2 + c[r, s]
                    # If it is now outside the bounds, record that.
                    if np.abs(z[r, s]) > 2:
                        bounded[r, s] = False
                        unbounded_at[r, s] = i

    # Prepare boolean array to say which samples were included, and at which iterations
    hits = np.zeros((c.shape[0], iterations.shape[0], c.shape[1]), dtype=np.bool)
    for r in range(c.shape[0]):
        for iter_idx, iters in enumerate(iterations):
            for s in range(c.shape[1]):
                hits[r, iter_idx, s] = unbounded_at[r, s] > iters

    return hits
