from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_wdbc_from_zip, standardize_train_test, train_test_split
from src.libsvm_compare import fit_linear_libsvm
from src.metrics import (
    accuracy,
    dual_objective,
    duality_gap,
    max_primal_violation,
    parameter_l2_difference,
    primal_objective,
    support_vector_mask,
)
from src.svm_qp import (
    fit_hard_margin_dual,
    fit_hard_margin_primal,
    fit_soft_margin_dual,
    fit_soft_margin_primal,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_accuracy_plot(path: Path, rows: list[dict[str, object]]) -> None:
    c_values = [float(row["C_assignment"]) for row in rows]
    custom_acc = [float(row["dual_test_accuracy"]) for row in rows]
    libsvm_acc = [float(row["libsvm_test_accuracy"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(c_values, custom_acc, marker="o", label="custom dual")
    plt.plot(c_values, libsvm_acc, marker="s", label="libsvm")
    plt.xscale("log")
    plt.xlabel("Assignment C")
    plt.ylabel("Test accuracy")
    plt.ylim(0.85, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    X, y, _ = load_wdbc_from_zip(ROOT / "data_a1.zip")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    rows: list[dict[str, object]] = []
    for C in [0.1, 1.0, 10.0]:
        primal = fit_soft_margin_primal(X_train, y_train, C=C)
        dual = fit_soft_margin_dual(X_train, y_train, C=C)
        libsvm = fit_linear_libsvm(X_train, y_train, X_test, C_assignment=C)

        primal_train_pred = primal.predict(X_train)
        primal_test_pred = primal.predict(X_test)
        dual_train_pred = dual.predict(X_train)
        dual_test_pred = dual.predict(X_test)

        primal_value = primal_objective(primal.w, primal.zeta, C=C)
        dual_value = dual_objective(X_train, y_train, dual.alpha)

        libsvm_alpha_full = None
        alpha_l2_diff: float | str = "unknown"
        if libsvm.support_indices is not None and libsvm.dual_coefficients is not None:
            libsvm_alpha_full = np.zeros_like(dual.alpha)
            libsvm_alpha_full[libsvm.support_indices] = np.abs(libsvm.dual_coefficients)
            alpha_l2_diff = parameter_l2_difference(dual.alpha, libsvm_alpha_full)

        rows.append(
            {
                "C_assignment": C,
                "C_libsvm": C / len(y_train),
                "primal_train_accuracy": accuracy(y_train, primal_train_pred),
                "primal_test_accuracy": accuracy(y_test, primal_test_pred),
                "dual_train_accuracy": accuracy(y_train, dual_train_pred),
                "dual_test_accuracy": accuracy(y_test, dual_test_pred),
                "libsvm_test_accuracy": accuracy(y_test, libsvm.predictions),
                "primal_dual_w_l2": parameter_l2_difference(primal.w, dual.w),
                "primal_dual_b_abs": abs(primal.b - dual.b),
                "custom_libsvm_w_l2": parameter_l2_difference(dual.w, libsvm.w),
                "custom_libsvm_b_abs": abs(dual.b - libsvm.b),
                "custom_libsvm_alpha_l2": alpha_l2_diff,
                "support_vectors_custom": int(np.sum(support_vector_mask(dual.alpha))),
                "support_vectors_libsvm": int(len(libsvm.support_indices)) if libsvm.support_indices is not None else "unknown",
                "primal_objective": primal_value,
                "dual_objective": dual_value,
                "duality_gap": duality_gap(primal_value, dual_value),
                "max_primal_violation": max_primal_violation(X_train, y_train, primal.w, primal.b, primal.zeta),
                "libsvm_backend": libsvm.backend,
            }
        )

    write_csv(ROOT / "results" / "tables" / "soft_margin_comparison.csv", rows)

    hard_rows: list[dict[str, object]] = []
    try:
        hard_primal = fit_hard_margin_primal(X_train, y_train)
        hard_dual = fit_hard_margin_dual(X_train, y_train)
        hard_rows.append(
            {
                "status": "solved",
                "primal_train_accuracy": accuracy(y_train, hard_primal.predict(X_train)),
                "dual_train_accuracy": accuracy(y_train, hard_dual.predict(X_train)),
                "primal_dual_w_l2": parameter_l2_difference(hard_primal.w, hard_dual.w),
                "primal_dual_b_abs": abs(hard_primal.b - hard_dual.b),
                "max_primal_violation": max_primal_violation(X_train, y_train, hard_primal.w, hard_primal.b),
            }
        )
    except Exception as exc:
        hard_rows.append(
            {
                "status": f"failed: {exc}",
                "primal_train_accuracy": "",
                "dual_train_accuracy": "",
                "primal_dual_w_l2": "",
                "primal_dual_b_abs": "",
                "max_primal_violation": "",
            }
        )
    write_csv(ROOT / "results" / "tables" / "hard_margin_summary.csv", hard_rows)
    write_accuracy_plot(ROOT / "results" / "figures" / "accuracy_vs_c.png", rows)

    print("Wrote results/tables/soft_margin_comparison.csv")
    print("Wrote results/tables/hard_margin_summary.csv")
    print("Wrote results/figures/accuracy_vs_c.png")


if __name__ == "__main__":
    main()
