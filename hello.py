# function defining a mandelbrot
def mandelbrot(x, y, iterations):
    sample_point = complex(x, y)

    z = 0
    for i in range(1, iterations):
        if abs(z) > 2: # a circle with radius 2, goes around mandelbrot set
            return 0
        z = z * z + sample_point
    return 1

print(mandelbrot(0, 0, 1))
