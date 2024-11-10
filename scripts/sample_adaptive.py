from hit_and_mandelbrot.mandelbrot import (
    adaptive_sampling,
)

if __name__ == "__main__":
    n_samples = 10_000
    iterations = 100
    threshold = 0.1
    max_depth = 5

    area_estimate = adaptive_sampling(
        n_samples,
        iterations,
        x_min=-2,
        x_max=2,
        y_min=-2,
        y_max=2,
        threshold=threshold,
        max_depth=max_depth,
    )
    print(f"Estimated area of Mandelbrot set: {area_estimate}")
