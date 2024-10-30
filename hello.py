import numpy as np

# function defining a mandelbrot
def mandelbrot(x, y, iterations):
    c0 = complex(x, y) # The complex number that is the starting point

    z = 0
    for i in range(1, iterations):
        if abs(z) > 2: # a circle with radius 2, goes around mandelbrot set
            return 0
        z = z * z + c0
    return 1

def calculate_area(n, iterations):
    hits = 0 # number of points inside mandelbrot set area

    for _ in range(n):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)

        hits += mandelbrot(x, y, iterations)

    return (hits / n) * 16


result = calculate_area(10000, 500)
print(result)
