# P2 Requirements

## Overall

The assignment consists of a report and Python, MATLAB, C, or C++ code on both hard-margin and soft-margin SVM.

In addition, implement the following slightly modified optimization problem:

$$
\begin{aligned}
\min_{w, b, \zeta} \quad
& \frac{1}{2}\|w\|_2^2 + \frac{C}{N}\sum_{i=1}^{n}\zeta_i \\
\text{subject to} \quad
& y_i(w^\top x_i - b) \ge 1 - \zeta_i, \\
& \zeta_i \ge 0, \quad \forall i \in \{1, \ldots, n\}.
\end{aligned}
$$

Test your implementation against [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

## Report

Write a report containing:

- Your understanding of binary SVMs, within 4 pages.
- Experimental comparison of your code and an existing implementation of SVMs, libsvm, within 4 pages.

The full report has a maximum length of 8 pages.

There is no strict report format other than the page limit. The purpose of the report is to show what you have understood about SVMs and what you have done, or at least tried, to make your code correct.

The report should at least cover:

- The primal form and dual form for both hard-margin and soft-margin SVMs.
- The concept of support vectors.
- Why maximum margin is good.
- The concepts of generalisation error and generalisation bounds.
- The concepts of duality gap, weak duality, and strong duality.
- Experimental comparison with libsvm.

## Code

The code requirements are:

- Implement SVMs in Python, MATLAB, C, or C++.
- Python is strongly recommended.
- Solve SVMs in both the primal and the dual.
- Implement the main algorithm of SVMs yourself.
- You may use an off-the-shelf toolbox to solve the resulting Quadratic Programming problem.
- You may not call existing SVM toolboxes or SVM APIs in your own implementation.

## Data

Download `data_a1.zip` from the course shared folder.

For further information, see `dataset description.txt`.

## Self-Check And libsvm Comparison

Self-check the correctness of your code by comparing it with libsvm.

The formulation in libsvm and your implementation may be slightly different, so you need to identify the difference and the correspondence between them.

Compare the parameters of your SVM and libsvm:

- `w`
- `b`
- `alpha`

Once everything is set correctly, the decision boundary determined by the parameters should be the same, up to numerical precision, given the same data and model hyperparameters such as `C`.

You can also check the duality gap.
