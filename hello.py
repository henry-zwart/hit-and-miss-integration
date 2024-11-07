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
# from tqdm import trange


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


def calculate_proportion(iterations, samples):
    """
    Using the generated sample points, this function calculates the
    area of the Mandel brot set.
    """
    hits_list = []  # number of points inside mandelbrot set area

    for elem in samples:
        hits_list.append(mandelbrot(elem[0], elem[1], iterations))

    hits_proportion_list = np.array(hits_list, dtype=float).cumsum() / np.arange(
        1, len(samples) + 1
    )
    return hits_proportion_list


def calculate_area(iterations, samples):
    area = np.zeros((iterations + 1, len(samples)))
    # area_M = calculate_area(iterations, samples)
    for i in range(iterations + 1):
        area[i] = calculate_proportion(i, samples) * 16

    return area


def balance_i_s(iterations, n):
    """
    Calculating error for many different amount of sample points, (mean taken over 5).
    Calculating error for many different amount of iterations, (mean taken over 5).
    """
    all_sample_convergence = []
    all_iteration_convergence = []
    for _ in range(1):
        print("hey")
        samples = random_sampling(n)
        results = calculate_area(iterations, samples)

        all_iteration_convergence.append(np.abs(results - results[-1]))
        all_sample_convergence.append(
            np.abs(results[:, :] - np.expand_dims(results[:, -1], 1))
        )

    return (
        np.mean(all_iteration_convergence, axis=0),
        np.mean(all_sample_convergence, axis=0),
    )


def plot_balance_i_s(iterations, n):
    """
    Plot calculated errors.
    """
    iteration_convergence, sample_convergence = balance_i_s(iterations, n)
    x = np.arange(iterations + 1)
    y = np.arange(n)

    fig, ax = plt.subplots(2)
    ax[0].plot(x[20:], iteration_convergence[20:, -1])
    ax[1].plot(y[10000:], sample_convergence[-1, 10000:])
    plt.show()
    plt.savefig("fig.png")


def plot_area_balanced(iterations, n):
    """
    Finds the number of sample points and number of iterations for certain error values
    """
    list_iterations = []
    list_samples = []

    iteration_convergence, sample_convergence = balance_i_s(iterations, n)
    iteration_convergence, sample_convergence = (
        iteration_convergence[20:, -1],
        sample_convergence[-1, 10000:],
    )

    for i, elem in enumerate(iteration_convergence):
        if elem < 0.05 and len(list_iterations) == 0:
            list_iterations.append(i + 1)
        elif elem < 0.04 and len(list_iterations) == 1:
            list_iterations.append(i + 1)
        elif elem < 0.03 and len(list_iterations) == 2:
            list_iterations.append(i + 1)
        elif elem < 0.02 and len(list_iterations) == 3:
            list_iterations.append(i + 1)
        elif elem < 0.01 and len(list_iterations) == 4:
            list_iterations.append(i + 1)

    for i, elem in enumerate(sample_convergence):
        if elem < 0.05 and len(list_samples) == 0:
            list_samples.append(i + 1)
        elif elem < 0.04 and len(list_samples) == 1:
            list_samples.append(i + 1)
        elif elem < 0.03 and len(list_samples) == 2:
            list_samples.append(i + 1)
        elif elem < 0.02 and len(list_samples) == 3:
            list_samples.append(i + 1)
        elif elem < 0.01 and len(list_samples) == 4:
            list_samples.append(i + 1)
    list_area = []
    for i in range(len(list_iterations)):
        samples = random_sampling(list_samples[i])
        list_area.append(calculate_proportion(list_iterations[i], samples)[-1] * 16)
    print(list_area)
    plt.plot([0.05, 0.04, 0.03, 0.02, 0.01], list_area)
    plt.show()


if __name__ == "__main__":
    # plot_balance_i_s(100, 100000)
    plot_area_balanced(100, 100000)
    # balance_i_s(100, 10000)
