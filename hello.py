"""
Course: Stochastic Simulation
Assignement 1: Hit and Miss Integration
Names: Tika van Bennekum, ..., ...
Student numbers: 13392425, ..., ...

Description:
    ...

next steps:
- rewrite calculate_area to keep track of hits
- calculate difference for different nr of samples

- use this information to perform experiments with balanced i and s

- improve model (perhaps baysian estimation)

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


def mandelbrot(x, y, iterations):
    """
    This function takes a sample point (x, y) and calculates using the nr of
    iterations wheter it falls within the area of the Mandel brot set or not.
    If it does, 1 is returned and if not 0 is returned.
    """
    c0 = complex(x, y)  # The complex number that is the starting point

    z = 0
    for _ in range(1, iterations):
        if abs(z) > 2:  # a circle with radius 2, goes around mandelbrot set
            return 0
        z = z * z + c0
    return 1


def random_sampling(n):
    """Function for the pure random sampling."""
    return np.reshape(np.random.uniform(-2, 2, n * 2), shape=(n, 2))


def hypercube_sampling(n):
    """Function for the latin hyercube sampling."""
    sampler = qmc.LatinHypercube(d=2, strength=1)
    samples = sampler.random(n)
    samples = (samples - 0.5) * 4  # rescaling from 0 to 1 to -2 to 2
    return samples


def orthognonal_sampling(n):
    """Function for the orthogonal sampling."""
    sampler = qmc.LatinHypercube(d=2, strength=2)
    samples = sampler.random(n)
    samples = (samples - 0.5) * 4  # rescaling from 0 to 1 to -2 to 2
    return samples


def calculate_area(iterations, samples):
    """
    Using the generated sample points, this function calculates the
    area of the Mandel brot set.
    """
    hits = 0  # number of points inside mandelbrot set area

    for elem in samples:
        hits += mandelbrot(elem[0], elem[1], iterations)

    return (hits / len(samples)) * 16


def iteration_area_difference_i(iterations, samples):
    area_i = np.zeros(iterations + 1)
    # area_M = calculate_area(iterations, samples)
    for i in range(iterations + 1):
        area_i[i] = calculate_area(i, samples)
    diff_area = np.abs(area_i - area_i[-1])

    print(area_i[-20:])
    return diff_area


if __name__ == "__main__":
    iterations = 100
    n = 100000

    x = np.arange(iterations + 1)
    samples = random_sampling(n)
    results = iteration_area_difference_i(iterations, samples)

    fig, ax = plt.subplots()

    ax.plot(x[20:], results[20:])
    # ax.set_yscale("log")
    # ax.axhline(results[-1], color="red")
    # plt.ylim(0)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Difference")
    plt.show()
    plt.savefig("fig.png")

    print(random_sampling(10))
