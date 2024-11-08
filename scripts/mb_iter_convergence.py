import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube


def sample_lhs(xmin, xmax, n):
    normalised_samples = LatinHypercube(d=1).random(n)
    denormed_samples = (xmax - xmin) * (normalised_samples - 0.5)
    return denormed_samples


def sample_complex_uniform(n_samples, r_min, r_max, i_min, i_max, method="random"):
    match method:
        case "random":
            real_samples = np.random.uniform(r_min, r_max, n_samples)
            imag_samples = np.random.uniform(i_min, i_max, n_samples) * 1.0j
        case "lhs":
            real_samples = sample_lhs(r_min, r_max, n_samples)
            imag_samples = sample_lhs(i_min, i_max, n_samples) * 1.0j
        case _:
            raise ValueError(f"Unknown sampling method: {method}")
    return real_samples + imag_samples


def estimate_area(
    n_samples, iterations, x_min=-2, x_max=2, y_min=-2, y_max=2, repeats=1
):
    # Calculate area of the sample space
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    v = (x_max - x_min) * (y_max - y_min)

    results = np.zeros(repeats, dtype=np.float64)

    for i in range(repeats):
        # Run mandlebrot iterations
        c0 = sample_complex_uniform(n_samples, x_min, x_max, y_min, y_max, method="lhs")
        z = c0.copy()
        for _ in range(iterations):
            still_bounded = np.abs(z) < 2
            z[still_bounded] = np.pow(z[still_bounded], 2) + c0[still_bounded]

        # Return the estimated area
        proportion_bounded = (np.abs(z) < 2).sum() / n_samples
        results[i] = v * proportion_bounded

    return results


def mean_and_ci(arr, z=1.96):
    return (arr.mean(), z * arr.std(ddof=1) / np.sqrt(arr.shape[0]))


def rel_change(i, n_samples, repeats, z=1.96):
    assert i > 0
    a1 = estimate_area(n_samples, i - 1, repeats=repeats)
    a2 = estimate_area(n_samples, i, repeats=repeats)
    expected_area, area_ci = mean_and_ci(a2)
    expected_rc, rc_ci = mean_and_ci((a1 - a2) / a1)
    return expected_rc, rc_ci, expected_area, area_ci


def find_pow2_upper_bound(n_samples, threshold, repeats):
    tested_is = []
    expected_rcs = []
    rc_cis = []
    expected_areas = []
    area_cis = []

    i = 1
    rc_exp, rc_ci, area_exp, area_ci = rel_change(2**i, n_samples, repeats)
    tested_is.append(2**i)
    expected_rcs.append(rc_exp)
    rc_cis.append(rc_ci)
    expected_areas.append(area_exp)
    area_cis.append(area_ci)
    while (rc_exp + rc_ci) > threshold:
        print(f"{2**i}: {rc_exp*100:.2f}% +- {100*rc_ci:.2f}")
        print(
            f"Upper confidence bound exceeds threshold: {100*(rc_exp + rc_ci):.2f}% > {100*threshold}%"
        )
        i += 1
        rc_exp, rc_ci, area_exp, area_ci = rel_change(2**i, n_samples, repeats)
        tested_is.append(2**i)
        expected_rcs.append(rc_exp)
        rc_cis.append(rc_ci)
        expected_areas.append(area_exp)
        area_cis.append(area_ci)

    print(f"{2**i}: {rc_exp*100:.2f}% +- {100*rc_ci:.2f}")
    print(
        f"Threshold outside confidence interval: {100*(rc_exp + rc_ci):.2f}% <= {100*threshold}%"
    )

    return tested_is, expected_rcs, rc_cis, expected_areas, area_cis


def minimal_convergence_iteration(n_samples, threshold, repeats):
    tested_is, expected_rcs, rc_cis, expected_areas, area_cis = find_pow2_upper_bound(
        n_samples, threshold, repeats
    )

    # Run binary search to find first i which is convergent
    left, right = tested_is[-2:]
    while left <= right:
        mid = (left + right) // 2
        print(f"Testing: {mid}")
        rc_exp, rc_ci, area_exp, area_ci = rel_change(mid, n_samples, repeats)
        print(f"{mid}: {rc_exp*100:.2f}% +- {100*rc_ci:.2f}")
        tested_is.append(mid)
        expected_rcs.append(rc_exp)
        rc_cis.append(rc_ci)
        expected_areas.append(area_exp)
        area_cis.append(area_ci)
        if (rc_exp + rc_ci) < threshold:
            print(
                f"Threshold outside confidence interval: {100*(rc_exp + rc_ci):.2f}% <= {100*threshold}%"
            )
            print(f"Lowering upper bound: {right} -> {mid}")
            right = mid - 1
        else:
            print(
                f"Upper confidence bound exceeds threshold: {100*(rc_exp + rc_ci):.2f}% > {100*threshold}%"
            )
            print(f"Raising lower bound: {left} -> {mid}")
            left = mid + 1
        if left >= right:
            print("left >= right, quitting.")
            break

    return (
        np.array(tested_is),
        np.array(expected_rcs),
        np.array(rc_cis),
        np.array(expected_areas),
        np.array(area_cis),
    )


if __name__ == "__main__":
    n_samples = 1000000
    threshold = 0.1 / 100
    repeats = 100
    tested_is, expected_rcs, rc_cis, expected_area, area_cis = (
        minimal_convergence_iteration(n_samples, threshold, repeats)
    )

    sorted_order = np.argsort(tested_is)
    min_convergent_i = tested_is[sorted_order][
        np.argmax(expected_rcs[sorted_order] + rc_cis[sorted_order] < threshold)
    ]
    min_convergent_area = expected_area[sorted_order][
        np.argmax(expected_rcs[sorted_order] + rc_cis[sorted_order] < threshold)
    ]

    fig, axes = plt.subplots(2, sharex=True)

    # Scatterplot of the relative change in estimated area, with confidence intervals
    axes[0].scatter(tested_is, 100 * expected_rcs, s=10, marker="x")
    axes[0].vlines(
        tested_is, 100 * (expected_rcs - rc_cis), 100 * (expected_rcs + rc_cis)
    )

    # Horizontal line at threshold, vertical red line at first i which achieves convergence
    axes[0].axhline(y=100 * threshold, color="grey", linewidth=0.5)
    axes[0].axvline(min_convergent_i, linestyle="dashed", color="red", linewidth=0.5)

    # Make plot pretty
    axes[0].set_xlim(1, None)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative change")
    axes[0].set_title("Percentage relative change from A(i-1) -> A(i)")

    # Scatterplot of the estimated area for each of the tested i's, with confidence intervals
    axes[1].scatter(tested_is, expected_area, s=10)
    axes[1].vlines(tested_is, expected_area - area_cis, expected_area + area_cis)

    # Horizontal red line at minimum convergent area
    axes[1].axhline(min_convergent_area, linestyle="dashed", color="red", linewidth=0.5)
    axes[1].text(
        x=1.01,
        y=min_convergent_area,
        s=f"{min_convergent_area:.4f}",
        va="center",
        ha="left",
        color="red",
        transform=axes[1].get_yaxis_transform(),
    )

    # Make plot pretty
    axes[1].set_ylabel("Estimated area")
    axes[1].set_ylim(0, None)
    axes[1].set_xlabel("Iterations")
    axes[1].set_title("Convergence of A(i)")

    # Prepare and save figure
    fig.tight_layout()
    fig.savefig("iteration_convergence_lhs.png", dpi=500, bbox_inches="tight")
