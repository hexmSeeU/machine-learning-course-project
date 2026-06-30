# P2 SVM Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete P2 SVM deliverable: custom primal/dual linear SVM implementation, libsvm comparison, reproducible experiments, tests, and a Chinese report draft.

**Architecture:** Keep the custom SVM independent from libsvm. Use `cvxpy` only as a QP solver, implement the primal/dual formulations explicitly, and isolate libsvm usage in comparison code. Data loading, metrics, experiments, and report generation stay in separate focused files.

**Tech Stack:** Python 3, NumPy, pandas, cvxpy, scipy, pytest, matplotlib, optional `libsvm-official` or scikit-learn's libsvm-backed `SVC` for comparison.

---

## File Structure

- Create `P2/requirements.txt`: runtime and test dependencies.
- Create `P2/README.md`: how to install, run tests, run experiments, and locate report/results.
- Create `P2/src/__init__.py`: package marker and public exports.
- Create `P2/src/data.py`: load WDBC data from `data_a1.zip`, convert labels, split, standardize.
- Create `P2/src/svm_qp.py`: custom hard/soft margin SVM primal and dual QP solvers.
- Create `P2/src/metrics.py`: objective values, feasibility checks, accuracy, support vector summaries, parameter comparisons.
- Create `P2/src/libsvm_compare.py`: optional libsvm/sklearn comparison helper only.
- Create `P2/scripts/run_experiments.py`: reproducible experiment runner writing CSV/Markdown tables and figures.
- Create `P2/tests/test_data.py`: tests for loading and preprocessing.
- Create `P2/tests/test_svm_qp.py`: tests for primal/dual SVM behavior.
- Create `P2/tests/test_metrics.py`: tests for metrics and duality gap calculations.
- Create `P2/report/report.md`: Chinese report draft covering all required topics.
- Create `P2/results/tables/.gitkeep` and `P2/results/figures/.gitkeep`: output directories.

---

### Task 1: Dependencies And Project Skeleton

**Files:**
- Create: `P2/requirements.txt`
- Create: `P2/README.md`
- Create: `P2/src/__init__.py`
- Create: `P2/results/tables/.gitkeep`
- Create: `P2/results/figures/.gitkeep`

- [ ] **Step 1: Create dependency file**

Create `P2/requirements.txt` with:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
cvxpy>=1.4
pytest>=8.0
matplotlib>=3.7
scikit-learn>=1.3
libsvm-official>=3.32.0
```

- [ ] **Step 2: Create README**

Create `P2/README.md` with setup, testing, experiment, and report instructions.

- [ ] **Step 3: Create package marker and result directories**

Create `P2/src/__init__.py` with:

```python
"""P2 custom SVM implementation package."""
```

Create empty files:

```text
P2/results/tables/.gitkeep
P2/results/figures/.gitkeep
```

- [ ] **Step 4: Verify skeleton**

Run:

```bash
find P2 -maxdepth 3 -type f | sort
```

Expected: the new files are listed together with existing `data_a1.zip` and `requirements.md`.

- [ ] **Step 5: Commit**

```bash
git add P2/requirements.txt P2/README.md P2/src/__init__.py P2/results/tables/.gitkeep P2/results/figures/.gitkeep
git commit -m "Add P2 project skeleton"
```

If git identity is still unset, skip the commit and record that the commit could not be created.

---

### Task 2: Data Loading And Preprocessing

**Files:**
- Create: `P2/src/data.py`
- Create: `P2/tests/test_data.py`

- [ ] **Step 1: Write failing data tests**

Create tests for `load_wdbc_from_zip`, `train_test_split`, and `standardize_train_test`. Assert WDBC shape `(569, 27)`, binary labels `{-1, 1}`, reproducible split, and train-only standardization.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd P2
pytest tests/test_data.py -q
```

Expected: FAIL because `src.data` does not exist.

- [ ] **Step 3: Implement data utilities**

Implement:

```python
def load_wdbc_from_zip(zip_path="data_a1.zip") -> tuple[np.ndarray, np.ndarray, list[str]]
def train_test_split(X, y, test_size=0.25, seed=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def standardize_train_test(X_train, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Details: read `data.csv` from zip using pandas, map `M -> +1` and `B -> -1`, drop `id` and `diagnosis`, return float NumPy arrays. Standardization uses training mean/std and replaces zero std with `1.0`.

- [ ] **Step 4: Run data tests**

Run:

```bash
cd P2
pytest tests/test_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add P2/src/data.py P2/tests/test_data.py
git commit -m "Add P2 data loading utilities"
```

---

### Task 3: Custom SVM QP Solvers

**Files:**
- Create: `P2/src/svm_qp.py`
- Create: `P2/tests/test_svm_qp.py`

- [ ] **Step 1: Write failing SVM tests**

Create tests for a tiny linearly separable dataset and a noisy soft-margin dataset. Check hard-margin primal/dual predictions, soft-margin constraints, alpha bounds `0 <= alpha <= C/N`, and primal/dual decision value agreement.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd P2
pytest tests/test_svm_qp.py -q
```

Expected: FAIL because `src.svm_qp` does not exist.

- [ ] **Step 3: Implement custom QP SVM solvers**

Create `LinearSVMModel` with `w`, `b`, optional `alpha`, optional `zeta`, `decision_function(X) = X @ w - b`, and `predict(X)` returning `+1/-1`.

Implement:

```python
def fit_hard_margin_primal(X, y) -> LinearSVMModel
def fit_soft_margin_primal(X, y, C) -> LinearSVMModel
def fit_hard_margin_dual(X, y) -> LinearSVMModel
def fit_soft_margin_dual(X, y, C) -> LinearSVMModel
```

QP details:

- Hard primal: minimize `0.5 * ||w||^2`, constraints `y * (X @ w - b) >= 1`.
- Soft primal: minimize `0.5 * ||w||^2 + (C/N) * sum(zeta)`, constraints `y * (X @ w - b) >= 1 - zeta`, `zeta >= 0`.
- Hard dual: maximize `sum(alpha) - 0.5 alpha^T Q alpha`, constraints `alpha >= 0`, `y @ alpha == 0`.
- Soft dual: same dual objective, constraints `0 <= alpha <= C/N`, `y @ alpha == 0`.
- Recover `w = sum_i alpha_i y_i x_i` and recover `b` from support vectors using `b = mean(X_sv @ w - y_sv)` under the project's `w^T x - b` convention.

Use cvxpy solvers in order `CLARABEL`, `OSQP`, `SCS`, raising a clear `RuntimeError` if none reaches optimal status.

- [ ] **Step 4: Run SVM tests**

Run:

```bash
cd P2
pytest tests/test_svm_qp.py -q
```

Expected: PASS. If hard-margin tolerances fail, inspect solver status and margins before changing tolerances.

- [ ] **Step 5: Commit**

```bash
git add P2/src/svm_qp.py P2/tests/test_svm_qp.py
git commit -m "Implement custom primal and dual SVM solvers"
```

---

### Task 4: Metrics And Objective Checks

**Files:**
- Create: `P2/src/metrics.py`
- Create: `P2/tests/test_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create tests for accuracy, primal objective with `C/N`, dual objective, maximum primal violation, support-vector mask, and parameter L2 difference.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd P2
pytest tests/test_metrics.py -q
```

Expected: FAIL because `src.metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Implement:

```python
def accuracy(y_true, y_pred) -> float
def primal_objective(w, zeta, C) -> float
def dual_objective(X, y, alpha) -> float
def duality_gap(primal_value, dual_value) -> float
def max_primal_violation(X, y, w, b, zeta=None) -> float
def support_vector_mask(alpha, tol=1e-6) -> np.ndarray
def parameter_l2_difference(a, b) -> float
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
cd P2
pytest tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Run all current tests**

Run:

```bash
cd P2
pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add P2/src/metrics.py P2/tests/test_metrics.py
git commit -m "Add SVM metric utilities"
```

---

### Task 5: libsvm Comparison Helper

**Files:**
- Create: `P2/src/libsvm_compare.py`

- [ ] **Step 1: Implement optional libsvm comparison API**

Implement `LibSVMLinearModel` with `w`, `b`, `predictions`, `backend`, optional support indices and dual coefficients.

Implement:

```python
def fit_linear_libsvm(X_train, y_train, X_eval, C_assignment) -> LibSVMLinearModel
```

Set `C_libsvm = C_assignment / len(y_train)`. Prefer `libsvm-official`; if unavailable, fall back to `sklearn.svm.SVC(kernel="linear")`. Convert intercepts into the project convention `decision = X @ w - b`.

- [ ] **Step 2: Smoke-test import**

Run:

```bash
cd P2
python - <<'PY'
from src.libsvm_compare import fit_linear_libsvm
print(fit_linear_libsvm.__name__)
PY
```

Expected: prints `fit_linear_libsvm`.

- [ ] **Step 3: Commit**

```bash
git add P2/src/libsvm_compare.py
git commit -m "Add libsvm comparison helper"
```

---

### Task 6: Experiment Runner

**Files:**
- Create: `P2/scripts/run_experiments.py`

- [ ] **Step 1: Implement experiment script**

Implement a script that:

1. Loads WDBC data from `data_a1.zip`.
2. Splits train/test with seed `42` and test size `0.25`.
3. Standardizes features using training statistics.
4. Runs soft-margin primal, soft-margin dual, and libsvm for `C in [0.1, 1.0, 10.0]`.
5. Writes `results/tables/soft_margin_comparison.csv` with accuracies, parameter differences, support-vector counts, objectives, duality gap, feasibility violation, and backend.
6. Attempts hard-margin primal/dual and writes `results/tables/hard_margin_summary.csv`.
7. Writes `results/figures/accuracy_vs_c.png`.

- [ ] **Step 2: Run full tests before experiments**

Run:

```bash
cd P2
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run experiments**

Run:

```bash
cd P2
python scripts/run_experiments.py
```

Expected: prints:

```text
Wrote results/tables/soft_margin_comparison.csv
Wrote results/tables/hard_margin_summary.csv
Wrote results/figures/accuracy_vs_c.png
```

- [ ] **Step 4: Inspect result table**

Run:

```bash
cd P2
python - <<'PY'
import pandas as pd
print(pd.read_csv('results/tables/soft_margin_comparison.csv').to_string(index=False))
print(pd.read_csv('results/tables/hard_margin_summary.csv').to_string(index=False))
PY
```

Expected: three rows for soft-margin `C`; finite duality gaps and parameter differences.

- [ ] **Step 5: Commit**

```bash
git add P2/scripts/run_experiments.py P2/results/tables P2/results/figures
git commit -m "Add P2 experiment runner and results"
```

---

### Task 7: Chinese Report Draft

**Files:**
- Create: `P2/report/report.md`

- [ ] **Step 1: Create report draft**

Write `P2/report/report.md` in Chinese with sections:

1. 二分类 SVM 的基本思想。
2. Hard-margin SVM 的原始问题与对偶问题。
3. Soft-margin SVM 与本作业的 `(C/N)` 缩放。
4. 支持向量、KKT 条件与参数恢复。
5. 最大间隔为什么有利于泛化。
6. 泛化误差、泛化界、弱对偶、强对偶、duality gap。
7. 实现方法、数据预处理、libsvm 的 `C` 与偏置符号对应。
8. 实验结果表格和对比结论。

Use the report table columns:

```markdown
| C | custom dual test acc | libsvm test acc | ||w_custom-w_libsvm|| | |b_custom-b_libsvm| | duality gap |
|---:|---:|---:|---:|---:|---:|
```

- [ ] **Step 2: Fill result table from CSV**

Run after Task 6:

```bash
cd P2
python - <<'PY'
from pathlib import Path
import pandas as pd

report = Path('report/report.md')
text = report.read_text()
df = pd.read_csv('results/tables/soft_margin_comparison.csv')
rows = []
for _, row in df.iterrows():
    rows.append(
        f"| {row['C_assignment']:.1f} | {row['dual_test_accuracy']:.4f} | {row['libsvm_test_accuracy']:.4f} | "
        f"{row['custom_libsvm_w_l2']:.4e} | {row['custom_libsvm_b_abs']:.4e} | {row['duality_gap']:.4e} |"
    )
new_table = "\n".join([
    "| C | custom dual test acc | libsvm test acc | ||w_custom-w_libsvm|| | |b_custom-b_libsvm| | duality gap |",
    "|---:|---:|---:|---:|---:|---:|",
    *rows,
])
marker = "| C | custom dual test acc"
start = text.index(marker)
end = text.index("\n\n", start)
report.write_text(text[:start] + new_table + text[end:])
PY
```

Expected: table is filled with experiment numbers.

- [ ] **Step 3: Check report placeholders**

Run:

```bash
cd P2
rg -n "FILL_FROM_RESULTS|TBD|TODO" report/report.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add P2/report/report.md
git commit -m "Draft Chinese SVM report"
```

---

### Task 8: Final Verification And Cleanup

**Files:**
- Modify as needed only if verification reveals concrete failures.

- [ ] **Step 1: Run all tests**

Run:

```bash
cd P2
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run experiments from a clean command**

Run:

```bash
cd P2
python scripts/run_experiments.py
```

Expected: result CSV files and accuracy figure are regenerated successfully.

- [ ] **Step 3: Inspect final outputs**

Run:

```bash
cd P2
find results report -maxdepth 3 -type f | sort
```

Expected includes:

```text
report/report.md
results/figures/accuracy_vs_c.png
results/tables/hard_margin_summary.csv
results/tables/soft_margin_comparison.csv
```

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only intended P2 files are modified/untracked. Existing P1 move changes may still appear from earlier work and must not be reverted.

- [ ] **Step 5: Final commit**

```bash
git add P2
git commit -m "Complete P2 SVM assignment deliverable"
```

If git identity is still unset, leave files uncommitted and report the exact status to the user.
