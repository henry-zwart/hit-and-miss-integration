"""
Course: Stochastic Simulation
Assignement 1: Hit and Miss Integration
Names: Tika van Bennekum, ..., ...
Student numbers: 13392425, ..., ...

Description:
    ...

next steps:
- rewrite calculate_area to keep track of hits: DONE
- calculate difference for different nr of samples: DONE

- use this information to perform experiments with balanced i and s: IN ACTION

- improve model (perhaps baysian estimation)

"""

import matplotlib.pyplot as plt
import numpy as np

from hit_and_mandelbrot.mandelbrot import estimate_area, estimate_area_per_sample


def calculate_area_2(iterations, n_samples, repeats, sampler):
    area_exp = np.zeros((iterations + 1, n_samples), dtype=np.float64)
    area_ci = np.zeros_like(area_exp)

    for i in range(iterations + 1):
        exp, ci = estimate_area_per_sample(
            n_samples, i, repeats=repeats, sampler=sampler
        )
        area_exp[i] = exp
        area_ci[i] = ci

    return area_exp, area_ci


def balance_i_s(iterations, n):
    """
    Calculating error for many different amount of sample points, (mean taken over 5).
    Calculating error for many different amount of iterations, (mean taken over 5).
    """
    nr_of_runs = 15

    expected_area, area_ci = calculate_area_2(iterations, n, nr_of_runs, sampler="lhs")

    return (
        np.abs(expected_area - expected_area[-1]),
        np.abs(expected_area - expected_area[:, -1][:, None]),
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
    Finds the number of sample points and number of iterations for certain error values.
    Calculates area for these found values.
    Plots result.
    """
    list_iterations = []
    list_samples = []

    # gets error for different values
    iteration_convergence, sample_convergence = balance_i_s(iterations, n)
    # selects right/relevant parts of arrays
    iteration_convergence, sample_convergence = (
        iteration_convergence[20:, -1],
        sample_convergence[-1, 10000:],
    )

    list_errors = np.arange(0.0001, 0.0022, 0.00002)[::-1]
    list_errors = np.arange(0.0001, 0.015, 0.0001)[::-1]

    # Finds the number of iterations for certain error values
    for error in list_errors:
        for i, elem in enumerate(iteration_convergence, start=20):
            if elem < error:
                list_iterations.append(i)
                break

    # Finds the number of sample points for certain error values
    for error in list_errors:
        for i, elem in enumerate(sample_convergence, start=10000):
            if elem < error:
                list_samples.append(i)
                break

    list_area = []

    # calculates area for the corresponding iterations and sample point amount to the specific errors chosen
    for i in range(len(list_iterations)):
        area = estimate_area(list_samples[i], list_iterations[i], repeats=5).mean()
        list_area.append(area)

    # plot
    print("list_area: ", list_area)
    print("list_samples: ", list_samples)
    print("list_iterations: ", list_iterations)
    plt.title("Area of Mandelbrot set, with balanced (s, i) combinations taken")
    plt.ylabel("area of Mandelbrot set")
    plt.xlabel("error")
    print(len(list_area), len(list_errors))
    plt.scatter(list_errors, list_area)
    plt.show()



def adaptive_sampling(iterations, n, x_min, x_max, y_min, y_max, threshold, max_depth, cur_depth):
    """
    This is an adaptation to the random sampling Monte carlo technique.
    The function works recursively.

    Algorithm:
    1. It starts with a (sub)grid
    2. It takes n random sampling points within the subgrid
    3. It calculates the balance of hits and misses
    4. We check if the max depth is reached of our recursion,
        if so -> area subgrid is returned
        if not -> step 5
    5. - If the balance is disproportionate:
            the subgrid is mostly entirely inside or outside the Mandelbrot set area
            -> thus, we return the area of the subgrid
       - If the balance is proportionate:
            the subgrid includes an edge of the Mandelbrot set area (most probably)
            -> thus, we want to zoom in on this subgrid,
               we recursively call this function with 4 subgrids of the current grid
    """
    # makes list of sample points (x_samples[i], y_samples[i]) within current grid space
    x_samples, y_samples = np.random.uniform(x_min, x_max, n), np.random.uniform(y_min, y_max, n)

    # Counts the proportion of sample points that is inside of the Mandelbrot set area
    total_inside_area = 0
    for i in range(n):
        total_inside_area += mandelbrot(x_samples[i], y_samples[i], iterations)
    hits_proportion = total_inside_area / n

    # When max depth is reached the area of the current grid is returned. 
    # This makes sure the algorithm doesn't go on forever.
    if cur_depth == max_depth:
        return (x_max - x_min) * (y_max - y_min) * hits_proportion
    
    # Checks balance hits and misses
    # If the balance is disproportionate (mostly hits or mostly misses):
    #         the subgrid is mostly entirely inside or outside the Mandelbrot set area
    #         -> thus, we return the area of the subgrid
    if hits_proportion < threshold or hits_proportion > 1 - threshold:
        return (x_max - x_min) * (y_max - y_min) * hits_proportion
    
    # recursively calculates area for 4 subgrids
    mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    area = (
        adaptive_sampling(iterations, n, x_min, mid_x, y_min, mid_y, threshold, max_depth, cur_depth + 1) +
        adaptive_sampling(iterations, n, mid_x, x_max, y_min, mid_y, threshold, max_depth, cur_depth + 1) +
        adaptive_sampling(iterations, n, x_min, mid_x, mid_y, y_max, threshold, max_depth, cur_depth + 1) +
        adaptive_sampling(iterations, n, mid_x, x_max, mid_y, y_max, threshold, max_depth, cur_depth + 1)
    )
    
    return area

# def stratisfied_sampling((x_min, x_max, y_min, y_max), , threshold=0.1, max_depth=5, depth=0):
#     return

if __name__ == "__main__":
    iterations = 100
    n = 1000

    # plot_area_balanced(nr_of_iterations, nr_of_sample_points)
    # plot_balance_i_s(100, 100000)
    # balance_i_s(100, 10000)
    # print(random_sampling(10))
    area_estimate = adaptive_sampling(iterations, n, x_min=-2, x_max=2, y_min=-2, y_max=2, threshold=0.1, max_depth=5, cur_depth=0)
    print(f"Estimated area of Mandelbrot set: {area_estimate}")
