import numpy as np
from scipy.stats import qmc

# function defining a mandelbrot
def mandelbrot(x, y, iterations):
    c0 = complex(x, y) # The complex number that is the starting point

    z = 0
    for i in range(1, iterations):
        if abs(z) > 2: # a circle with radius 2, goes around mandelbrot set
            return 0
        z = z * z + c0
    return 1


def random_sampling(n):
    return np.reshape(np.random.uniform(-2, 2, n*2), shape=(n,2))


def hypercube_sampling(n):
    sampler = qmc.LatinHypercube(d=2, strength=1)
    samples = sampler.random(n)
    return samples

def orthognonal_sampling(n):
    sampler = qmc.LatinHypercube(d=2, strength=2)
    samples = sampler.random(n)
    return samples


def calculate_area(iterations, samples):
    hits = 0 # number of points inside mandelbrot set area

    for elem in samples:
        hits += mandelbrot(elem[0], elem[1], iterations)

    return (hits / len(samples)) * 16


print(random_sampling(10))
# result = calculate_area(10000, 500)
# print(result)
