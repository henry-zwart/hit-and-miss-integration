import matplotlib.pyplot as plt
import numpy as np

from hit_and_mandelbrot import Sampler, estimate_area, mean_and_ci


def rel_change(i, n_samples, repeats, z=1.96, ddof=1):
    assert i > 0
    print(f"Testing: i = {i}")
    a1 = estimate_area(n_samples, i - 1, repeats=repeats, sampler=Sampler.LHS)
    a2 = estimate_area(n_samples, i, repeats=repeats, sampler=Sampler.LHS)
    expected_area, area_ci = mean_and_ci(a2, z=z, ddof=ddof)
    expected_rc, rc_ci = mean_and_ci((a1 - a2) / a1, z=z, ddof=ddof)
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
    print()


def find_pow2_upper_bound(n_samples, threshold, repeats, z=1.96, ddof=1):
    tested_is = []
    rc_cis = []
    area_cis = []

    i = 1
    rc_ci, area_ci = rel_change(2**i, n_samples, repeats, z=z, ddof=ddof)
    tested_is.append(2**i)
    rc_cis.append(rc_ci)
    area_cis.append(area_ci)
    while rc_ci[1] > threshold:
        print_rc_ci_msg(rc_ci, threshold)
        i += 1
        rc_ci, area_ci = rel_change(2**i, n_samples, repeats, z=z, ddof=ddof)
        tested_is.append(2**i)
        rc_cis.append(rc_ci)
        area_cis.append(area_ci)

    print_rc_ci_msg(rc_ci, threshold)

    return tested_is, rc_cis, area_cis


def minimal_convergence_iteration(n_samples, threshold, repeats, z, ddof):
    tested_is, rc_cis, area_cis = find_pow2_upper_bound(
        n_samples, threshold, repeats, z, ddof
    )

    # Run binary search to find first i which is convergent
    left, right = tested_is[-2:]
    while left <= right:
        mid = (left + right) // 2
        rc_ci, area_ci = rel_change(mid, n_samples, repeats, z=z, ddof=ddof)
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
    n_samples = 1000000
    threshold = 0.1 / 100
    repeats = 100
    n_samples = 1000000
    threshold = 0.1 / 100
    z = 1.96
    ddof = 1

    tested_is, rc_cis, area_cis = minimal_convergence_iteration(
        n_samples, threshold, repeats, z=z, ddof=ddof
    )

    min_convergent_idx = np.argmax(rc_cis[:, 1] < threshold)
    min_convergent_i = tested_is[min_convergent_idx]
    min_convergent_area = area_cis[min_convergent_idx].mean()

    fig, axes = plt.subplots(2, sharex=True)

    # Scatterplot of the relative change in estimated area, with confidence intervals
    axes[0].scatter(tested_is, 100 * rc_cis.mean(axis=1), s=10, marker="x")
    axes[0].vlines(tested_is, 100 * rc_cis[:, 0], 100 * rc_cis[:, 1])

    # Horizontal line at threshold, vertical red line at first i which achieves convergence
    axes[0].axhline(y=100 * threshold, color="grey", linewidth=0.5)
    axes[0].axvline(min_convergent_i, linestyle="dashed", color="red", linewidth=0.5)

    # Make plot pretty
    axes[0].set_xlim(1, None)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative change")
    axes[0].set_title("Percentage relative change from A(i-1) -> A(i)")

    # Scatterplot of the estimated area for each of the tested i's, with confidence intervals
    axes[1].scatter(tested_is, area_cis.mean(axis=1), s=10)
    axes[1].vlines(tested_is, area_cis[:, 0], area_cis[:, 1])

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
    fig.savefig(
        "results/figures/iteration_convergence_lhs.png", dpi=500, bbox_inches="tight"
    )
