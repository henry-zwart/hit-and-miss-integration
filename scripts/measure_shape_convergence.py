import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot import Sampler, est_area, mean_and_ci
from hit_and_mandelbrot.sampling import sample_complex_uniform


def rel_change(i, samples, cache, z=1.96, ddof=1):
    assert i > 0
    print("\n")
    print("=" * 20)
    print(f"Testing: i = {i}")

    # Check if we already have (i // 2) cached
    if (i // 2) in cache:
        print(f"Reusing cached result for: i//2 = {i // 2}.")
        a1 = cache[i // 2]
    else:
        a1 = est_area(samples=samples, iterations=i // 2)
        cache[i // 2] = a1.copy()

    # We shouldn't have i in cache (since we shouldn't be repeating).
    # But we'll use it if its there :)
    if i in cache:
        print(f"Warning: {i} has already been tested!")
        a2 = cache[i]
    else:
        a2 = est_area(samples=samples, iterations=i)
        cache[i] = a2.copy()
    print()

    expected_area, area_ci = mean_and_ci(a2, z=z, ddof=ddof)
    expected_rc, rc_ci = mean_and_ci(1 - (a2 / a1), z=z, ddof=ddof)
    return (
        np.array([expected_rc - rc_ci, expected_rc + rc_ci]),
        np.array([expected_area - area_ci, expected_area + area_ci]),
    )


def print_rc_ci_msg(ci, threshold):
    expected_val = ci.mean()
    ci_radius = ci[1] - expected_val
    print(f"Relative change: {100*expected_val:.2f}% +- {100*ci_radius:.2f}")
    if ci[1] < threshold:
        print(
            f"Confidence interval below threshold: {100 * expected_val:.2f}% + {100 * ci_radius:.2f}% < {100 * threshold:.2f}%"
        )
    else:
        print(
            f"Confidence interval exceeds threshold: {100 * expected_val:.2f}% + {100 * ci_radius:.2f}% >= {100 * threshold:.2f}%"
        )


def find_pow2_upper_bound(samples, threshold, cache, z=1.96, ddof=1):
    tested_is = []
    rc_cis = []
    area_cis = []

    i = 1
    rc_ci, area_ci = rel_change(2**i, samples, cache, z=z, ddof=ddof)
    tested_is.append(2**i)
    rc_cis.append(rc_ci)
    area_cis.append(area_ci)
    while rc_ci[1] > threshold:
        print_rc_ci_msg(rc_ci, threshold)
        i += 1
        rc_ci, area_ci = rel_change(2**i, samples, cache, z=z, ddof=ddof)
        tested_is.append(2**i)
        rc_cis.append(rc_ci)
        area_cis.append(area_ci)

    print_rc_ci_msg(rc_ci, threshold)

    return tested_is, rc_cis, area_cis


def minimal_convergence_iteration(samples, threshold, z, ddof):
    cache = {}
    tested_is, rc_cis, area_cis = find_pow2_upper_bound(
        samples, threshold, cache, z, ddof
    )

    # Run binary search to find first i which is convergent
    left, right = tested_is[-2:]
    while left <= right:
        mid = (left + right) // 2
        rc_ci, area_ci = rel_change(mid, samples, cache, z=z, ddof=ddof)
        print_rc_ci_msg(rc_ci, threshold)

        tested_is.append(mid)
        rc_cis.append(rc_ci)
        area_cis.append(area_ci)

        if rc_ci[1] < threshold:
            print(f"Lowering upper bound: {right} -> {mid}")
            right = mid - 1
        else:
            print(f"Raising lower bound: {left} -> {mid}")
            left = mid + 1
        if left >= right:
            print("left >= right, quitting.")
            break

    sorted_order = np.argsort(tested_is)
    tested_is = np.array(tested_is)[sorted_order]
    rc_cis = np.array(rc_cis)[sorted_order]
    area_cis = np.array(area_cis)[sorted_order]

    return (
        tested_is,
        rc_cis,
        area_cis,
    )


if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "shape_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    repeats = 3
    n_samples = 281**2
    threshold = 0.1 / 100
    z = 1.96
    ddof = 1
    sampler = Sampler.ORTHO

    print(f"Sampling: {n_samples} from {sampler} sampler, with {repeats} repeats.")
    samples = sample_complex_uniform(
        n_samples=n_samples,
        repeats=repeats,
        method=sampler,
    )

    tested_is, rc_cis, area_cis = minimal_convergence_iteration(
        samples,
        threshold,
        z=z,
        ddof=ddof,
    )

    np.save(RESULTS_ROOT / "iterations.npy", tested_is)
    np.save(RESULTS_ROOT / "relchange_cis.npy", rc_cis)
    np.save(RESULTS_ROOT / "area_cis.npy", area_cis)

    min_convergent_idx = np.argmax(rc_cis[:, 1] < threshold)
    min_convergent_iters = tested_is[min_convergent_idx]
    min_convergent_area = area_cis[min_convergent_idx].mean()

    best_estimate_iters = tested_is[-1]
    best_estimate_area = area_cis[-1].mean()

    metadata = {
        "convergence_threshold": threshold,
        "z": z,
        "ddof": ddof,
        "repeats": repeats,
        "n_samples": n_samples,
        "sampler": sampler,
        "min_convergent_idx": int(min_convergent_idx),
        "min_convergent_iters": int(min_convergent_iters),
        "min_convergent_area": float(min_convergent_area),
        "best_estimate_iters": int(best_estimate_iters),
        "best_estimate_area": float(best_estimate_area),
    }

    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
