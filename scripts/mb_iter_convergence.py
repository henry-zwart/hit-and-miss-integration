import matplotlib.pyplot as plt
import numpy as np

from hit_and_mandelbrot import estimate_area, mean_and_ci


def rel_change(i, n_samples, repeats, z=1.96, ddof=1):
    assert i > 0
    a1 = estimate_area(n_samples, i - 1, repeats=repeats)
    a2 = estimate_area(n_samples, i, repeats=repeats)
    expected_area, area_ci = mean_and_ci(a2, z=z, ddof=ddof)
    expected_rc, rc_ci = mean_and_ci((a1 - a2) / a1, z=z, ddof=ddof)
    return expected_rc, rc_ci, expected_area, area_ci


def find_pow2_upper_bound(n_samples, threshold, repeats, z=1.96, ddof=1):
    tested_is = []
    expected_rcs = []
    rc_cis = []
    expected_areas = []
    area_cis = []

    i = 1
    rc_exp, rc_ci, area_exp, area_ci = rel_change(
        2**i, n_samples, repeats, z=z, ddof=ddof
    )
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
        rc_exp, rc_ci, area_exp, area_ci = rel_change(
            2**i, n_samples, repeats, z=z, ddof=ddof
        )
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


def minimal_convergence_iteration(n_samples, threshold, repeats, z, ddof):
    tested_is, expected_rcs, rc_cis, expected_areas, area_cis = find_pow2_upper_bound(
        n_samples, threshold, repeats, z, ddof
    )

    # Run binary search to find first i which is convergent
    left, right = tested_is[-2:]
    while left <= right:
        mid = (left + right) // 2
        print(f"Testing: {mid}")
        rc_exp, rc_ci, area_exp, area_ci = rel_change(
            mid, n_samples, repeats, z=z, ddof=ddof
        )
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

    sorted_order = np.argsort(tested_is)
    tested_is = np.array(tested_is)[sorted_order]
    expected_rcs = np.array(expected_rcs)[sorted_order]
    rc_cis = np.array(rc_cis)[sorted_order]
    expected_areas = np.array(expected_areas)[sorted_order]
    area_cis = np.array(area_cis)[sorted_order]

    return (
        tested_is,
        expected_rcs,
        rc_cis,
        expected_areas,
        area_cis,
    )


if __name__ == "__main__":
    n_samples = 1000000
    threshold = 0.1 / 100
    repeats = 100
    n_samples = 100000
    threshold = 5 / 100
    z = 1.96
    ddof = 1

    tested_is, expected_rcs, rc_cis, expected_area, area_cis = (
        minimal_convergence_iteration(n_samples, threshold, repeats, z=z, ddof=ddof)
    )

    min_convergent_i = tested_is[np.argmax(expected_rcs + rc_cis < threshold)]
    min_convergent_area = expected_area[np.argmax(expected_rcs + rc_cis < threshold)]

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
    fig.savefig(
        "results/figures/iteration_convergence_lhs.png", dpi=500, bbox_inches="tight"
    )
