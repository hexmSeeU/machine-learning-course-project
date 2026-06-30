# 二分类支持向量机实验报告

**姓名：** 何煦明

**学号：** 22521149

## 1. 二分类 SVM 的基本思想

二分类支持向量机（Support Vector Machine, SVM）解决的是 $y_i\in\{-1,+1\}$ 的监督分类问题。给定训练集 $\{(x_i,y_i)\}_{i=1}^n$，线性 SVM 希望学习一个超平面，把两类样本分到超平面的两侧。本文采用作业中的符号约定，将判别函数写成

$$
f(x)=w^\top x-b,
$$

预测规则为 $\operatorname{sign}(f(x))$。这里 $w$ 决定超平面的方向，$b$ 决定超平面的位置。样本 $x_i$ 的函数间隔为 $y_i(w^\top x_i-b)$；若该值为正，说明样本分类正确，若为负，说明样本被误分类。

SVM 的核心不只是找到一个能分类训练样本的平面，而是在所有可行平面中选择“间隔”最大的平面。在线性可分情况下，若约束归一化为 $y_i(w^\top x_i-b)\ge1$，离超平面最近样本的几何间隔与 $1/\|w\|_2$ 成正比。因此最大化间隔等价于最小化 $\|w\|_2^2$。这也是 SVM 原始问题中出现 $\frac12\|w\|_2^2$ 的原因。

## 2. Hard-margin SVM 的原始问题与对偶问题

当训练数据线性可分时，可以要求所有样本都被正确分类并且至少达到单位函数间隔。hard-margin SVM 的原始问题为

$$
\begin{aligned}
\min_{w,b}\quad & \frac12\|w\|_2^2\\
\text{s.t.}\quad & y_i(w^\top x_i-b)\ge 1,\quad i=1,\ldots,n.
\end{aligned}
$$

这个问题是凸二次规划：目标函数是凸二次函数，约束是线性不等式。为了得到对偶问题，引入拉格朗日乘子 $\alpha_i\ge0$，拉格朗日函数为

$$
L(w,b,\alpha)=\frac12\|w\|_2^2-\sum_i\alpha_i[y_i(w^\top x_i-b)-1].
$$

对 $w$ 和 $b$ 求驻点可得

$$
w=\sum_i\alpha_i y_i x_i,\qquad \sum_i\alpha_i y_i=0.
$$

代回拉格朗日函数后，hard-margin 的对偶问题为

$$
\begin{aligned}
\max_\alpha\quad & \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^\top x_j\\
\text{s.t.}\quad & \alpha_i\ge0,\quad \sum_i\alpha_i y_i=0.
\end{aligned}
$$

对偶形式的重要意义是：最优 $w$ 可以由训练样本的线性组合恢复，而且只有 $\alpha_i>0$ 的样本会真正影响分类边界。这些样本就是支持向量。

## 3. Soft-margin SVM 与本作业的缩放

实际数据可能不可完全线性可分，或者虽然可分但为了分类所有训练样本需要非常小的间隔。soft-margin SVM 引入松弛变量 $\zeta_i\ge0$，允许样本违反 margin 约束。本作业要求实现的原始问题为

$$
\begin{aligned}
\min_{w,b,\zeta}\quad & \frac12\|w\|_2^2+\frac{C}{N}\sum_i\zeta_i\\
\text{s.t.}\quad & y_i(w^\top x_i-b)\ge1-\zeta_i,\\
& \zeta_i\ge0.
\end{aligned}
$$

其中 $C$ 控制间隔大小和训练误差之间的权衡。$C$ 越大，违反约束的代价越高，模型更倾向于减少训练错误；$C$ 越小，模型更重视较大的 margin，允许更多样本落入 margin 内。本作业目标函数中惩罚项是 $\frac{C}{N}\sum_i\zeta_i$，因此它相当于对平均 slack 进行惩罚，而不是直接对 slack 总和惩罚。

通过同样的拉格朗日推导，可以得到 soft-margin 的对偶问题：

$$
\begin{aligned}
\max_\alpha\quad & \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^\top x_j\\
\text{s.t.}\quad & 0\le\alpha_i\le\frac{C}{N},\quad \sum_i\alpha_i y_i=0.
\end{aligned}
$$

与 hard-margin 相比，soft-margin 对偶问题多了上界 $\alpha_i\le C/N$。这个上界正是由 slack 变量的非负性和惩罚系数推出的。libsvm 的标准 soft-margin 形式使用 $C_{libsvm}\sum_i\xi_i$，所以为了与本作业形式一致，实验中设置

$$
C_{libsvm}=\frac{C}{N}.
$$

这是本实验中最重要的参数对应关系之一。如果直接把作业中的 $C$ 传给 libsvm，两个模型的优化问题并不相同，比较 $w,b,\alpha$ 就没有意义。

## 4. 支持向量与 KKT 条件

支持向量是对分类边界有直接影响的训练样本。在对偶解中，通常用 $\alpha_i>0$ 判断样本是否为支持向量。由

$$
w=\sum_i\alpha_i y_i x_i
$$

可知，$\alpha_i=0$ 的样本不会出现在 $w$ 的表达式中，因此它们对最终超平面没有直接贡献。

KKT 条件可以帮助理解不同样本的作用。对于 soft-margin SVM：若 $\alpha_i=0$，该样本通常在 margin 外并且不会影响边界；若 $0<\alpha_i<C/N$，样本位于 margin 边界上，满足 $y_i(w^\top x_i-b)=1$；若 $\alpha_i=C/N$，样本可能在 margin 内，也可能被误分类。恢复偏置 $b$ 时，最理想的是使用满足 $0<\alpha_i<C/N$ 的自由支持向量，因为它们严格位于 margin 边界上。若没有自由支持向量，$b$ 的数值可能在一个区间内都能给出同样的最优目标和预测结果，这也是数值比较时不能只看逐元素完全相等的原因。

## 5. 最大间隔、泛化误差与泛化界

泛化误差指模型在未见数据上的期望错误率。训练误差低并不一定代表泛化误差低，因为过于复杂的模型可能只记住训练集中的偶然模式。SVM 通过最大化 margin 来控制模型复杂度：在同样能分类训练样本的超平面中，较大的 margin 代表分类边界距离样本更远，对小扰动更稳定。

从统计学习理论角度看，泛化界通常把经验误差、样本数量、置信度和假设空间复杂度联系起来。对线性间隔分类器，边界复杂度与 $\|w\|_2$ 有关，而 margin 与 $1/\|w\|_2$ 有关。因此最小化 $\|w\|_2^2$ 可以被理解为控制模型容量。soft-margin SVM 又进一步通过 slack 变量在“训练集约束违反”和“margin 大小”之间做折中。这个折中是实际应用中非常重要的，因为真实数据往往包含噪声或异常点。

## 6. 弱对偶、强对偶与 duality gap

弱对偶表示：任意对偶可行解的目标值都是原始最优值的下界。也就是说，对于最小化的 primal 和最大化的 dual，总有 $d\le p$。强对偶表示在最优处 $p^*=d^*$。SVM 的 primal 是凸二次规划，约束为仿射约束；在满足 Slater 条件等常见条件时，强对偶成立。

实现中可以计算

$$
\text{duality gap}=p^*-d^*
$$

作为自检指标。若 gap 接近 0，说明 primal 和 dual 的求解结果一致；若 gap 很大，可能说明公式写错、参数缩放不一致、约束方向错误，或 QP solver 没有真正收敛。在本实验中，soft-margin 的 duality gap 均在 $10^{-9}$ 量级以内，因此 primal 与 dual 的数值结果是匹配的。

## 7. 实现方法

本文使用 Python 实现线性 hard-margin 和 soft-margin SVM。自实现部分显式构造 primal 和 dual 的二次规划问题，并使用 cvxpy 作为通用 QP 求解器；实现没有调用任何已有 SVM 训练 API。libsvm 只用于实验对比。

数据集为 WDBC，共 569 个样本、27 个实值特征，标签为恶性 M 和良性 B。本文将 M 映射为 $+1$，B 映射为 $-1$。实验固定随机种子划分训练集和测试集，测试集比例为 25%。特征标准化只使用训练集均值和标准差，然后把同一变换应用到测试集，避免测试集信息泄露。

实现中统一采用 $w^\top x-b$ 的判别函数。sklearn/libsvm 返回的线性模型通常写为 $w^\top x+\text{intercept}$，因此比较时需要令 $b=-\text{intercept}$。此外，libsvm 使用的惩罚系数设置为 $C/N$，以匹配作业中的目标函数。实验比较包括训练/测试准确率、$w$ 的 $L_2$ 差异、$b$ 的绝对差异、$\alpha$ 的 $L_2$ 差异、支持向量数量、primal objective、dual objective 和 duality gap。

## 8. 实验结果与 libsvm 比较

soft-margin 实验结果如下。表中的 $C$ 是作业目标函数中的 $C$，libsvm 实际使用 $C_{libsvm}=C/N$。

| $C$ | custom dual test acc | libsvm test acc | w L2 diff | b abs diff | alpha L2 diff | SV count | duality gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.8169 | 0.8169 | 1.0107e-04 | 5.0037e-04 | 1.8759e-05 | 307 | 1.3837e-10 |
| 1.0 | 0.9507 | 0.9507 | 1.1466e-04 | 1.0522e-04 | 2.8214e-05 | 162 | 1.2573e-10 |
| 10.0 | 0.9648 | 0.9648 | 7.7245e-04 | 4.2768e-04 | 1.3666e-03 | 81 | 3.0180e-09 |

hard-margin 在标准化后的训练集上也成功求解，训练准确率为 1.0000，primal/dual 的 $\|w_p-w_d\|_2$ 为 1.1873e-06，最大约束违反为 1.5596e-12。这说明在当前训练划分和标准化后，数据可以被线性超平面完全分开，且 primal 与 dual 的 hard-margin 解基本一致。

从 soft-margin 结果看，自实现 dual SVM 与 libsvm 的测试准确率在三个 $C$ 上完全一致。$C=0.1$ 时，模型允许较多 margin 违反，支持向量数量较多，测试准确率为 0.8169；当 $C$ 增大到 1.0 和 10.0 时，训练约束违反受到更强惩罚，测试准确率提高到 0.9507 和 0.9648，同时支持向量数量从 307 降到 81。这符合 soft-margin SVM 的直觉：较大的 $C$ 会使模型更努力地拟合训练数据，并减少处在 margin 内的样本。

参数比较也支持实现正确性。使用官方 libsvm 对比时，自实现 dual SVM 与 libsvm 的 $w$ 差异约为 $10^{-4}$ 到 $10^{-3}$，$b$ 差异约为 $10^{-4}$，$\alpha$ 的 $L_2$ 差异约为 $10^{-5}$ 到 $10^{-3}$。这说明在正确处理 $C$ 缩放和偏置符号后，两者得到的线性决策边界几乎相同。duality gap 接近 0，进一步说明 primal 和 dual 的目标值一致。由于 QP solver 与 libsvm 的优化算法、停止准则和浮点误差不同，参数不可能逐位完全相同；但从准确率、决策边界参数和 duality gap 来看，自实现 SVM 与 libsvm 的结果已经在数值精度内一致。
