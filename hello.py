"""
    Course: Stochastic Simulation
    Assignement 1: Hit and Miss Integration
    Names: Tika van Bennekum, ..., ...
    Student numbers: 13392425, ..., ...

    Description:
        ...

"""

import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

# function defining a mandelbrot
def mandelbrot(x, y, iterations):
    """
    This function takes a sample point (x, y) and calculates using the nr of 
    iterations wheter it falls within the area of the Mandel brot set or not.
    If it does, 1 is returned and if not 0 is returned.
    """
    c0 = complex(x, y) # The complex number that is the starting point

    z = 0
    for _ in range(1, iterations):
        if abs(z) > 2: # a circle with radius 2, goes around mandelbrot set
            return 0
        z = z * z + c0
    return 1


def random_sampling(n):
    """ Function for the pure random sampling. """
    return np.reshape(np.random.uniform(-2, 2, n*2), shape=(n,2))


def hypercube_sampling(n):
    """ Function for the latin hyercube sampling. """
    sampler = qmc.LatinHypercube(d=2, strength=1)
    samples = sampler.random(n)
    samples = (samples - 0.5)*4 # rescaling from 0 to 1 to -2 to 2
    return samples

def orthognonal_sampling(n):
    """ Function for the orthogonal sampling. """
    sampler = qmc.LatinHypercube(d=2, strength=2)
    samples = sampler.random(n)
    samples = (samples - 0.5)*4 # rescaling from 0 to 1 to -2 to 2
    return samples


def calculate_area(iterations, samples):
    """
    Using the generated sample points, this function calculates the 
    area of the Mandel brot set.
    """
    hits = 0 # number of points inside mandelbrot set area

    for elem in samples:
        hits += mandelbrot(elem[0], elem[1], iterations)

    return (hits / len(samples)) * 16


def iteration_area_difference(iterations,samples):
    area_i = np.zeros(iterations)
    area_M = calculate_area(iterations, samples)
    for i in range(iterations):
	area_i[i] = calculate_area(i,samples)
    diff_area = area_i - area_M
    
    return diff_area    

iterations = 100
samples = random_sampling(1000)
results = iteration_area_difference(iterations,samples)
plt.plot(np.arange(iterations),results)
plt.show() 
	 	
	
	


print(random_sampling(10))
# result = calculate_area(10000, 500)
# print(result)
