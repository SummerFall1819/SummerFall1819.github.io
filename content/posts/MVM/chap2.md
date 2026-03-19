---
title: "奇异值分解"
date: 2026-03-19T16:41:54+08:00
weight: 0
categories: [高等代数]
tags: [矩阵, SVD, 奇异值分解]
series: [现代数值方法]
draft: false
featured: false
params:
  mathjax: true
---


方阵 $\boldsymbol A \in \mathbb R^{M \times M}$ 可以视为一个线性变换。
在一个 $M-1$ 的单位超球面 $\mathcal X := \{\boldsymbol x \in \mathbb R^M \mid \lVert \boldsymbol x \rVert_2 = 1\} = \mathcal S^{M-1}$ 的原空间上，映射后的空间 $\mathcal Y = \{\boldsymbol {Ax} \mid \boldsymbol x \in \mathcal X\}$ 总是一个超椭球体。
类似于超球体可以由正交基定义，一个超椭球体可以通过 $\boldsymbol u_1,\dots,\boldsymbol u_M \in \mathbb R^M$ 这组正交基向量和对应方向上的缩放标量 $\sigma_1, \dots, \sigma_M$ 表示。
如果记 $\boldsymbol v_1, \dots, \boldsymbol v_M \in \mathbb R^M$ 是单纯形 $\mathcal S^{M - 1}$ 上的一组正交基向量，$\boldsymbol \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_M)$ ,那么以下式子成立。

$$
\boldsymbol {AV} = \boldsymbol {U\Sigma} \implies \boldsymbol A = \boldsymbol {U\Sigma V}^T
$$

这就是奇异值分解(SVD)的公式。其中：
上述的解读说明，每个方阵的线性变换等价于三个变换的叠加：

- 一次旋转或翻转 $\boldsymbol V^T$
- 一次在各个维度上的放缩 $\boldsymbol \Sigma.$
- 再次进行旋转或翻转 $\boldsymbol U$.

<video controls width="100%">
  <source src="/videos/SVDDecomposition.mp4" type="video/mp4">
</video>


对于 $\boldsymbol A \in \mathbb R^{M \times N}$ 的一般矩阵来说：

- $M > N$, 则 $\boldsymbol \Sigma$ 负责将原超球体旋转嵌入到高位的椭球面。
- $M < N$, 则 $\boldsymbol \Sigma$ 负责将拉伸后的超球体投影到低维的椭球面。

对于任意矩阵 $\boldsymbol A \in \mathbb R^{M \times N}$. 都存在一个**奇异值分解**：

$$
\boldsymbol A = \boldsymbol {U\Sigma V}^T
$$

其中

- $\boldsymbol U \in \mathbb R^{M \times M} = [\boldsymbol u_1, \dots, \boldsymbol u_M]$ 称为左奇异向量矩阵， $\boldsymbol u_1, \dots, \boldsymbol u_M \in \mathbb R^M$ 称为左奇异向量。
- $\Sigma \in \mathbb R^{M \times N} = \operatorname{diag(\sigma_1, \dots, \sigma_{\min(M,N)})}$ 称为奇异值矩阵，对角线元素 $\sigma_1 \geq \dots \geq \sigma_{\min(M,N)}$ 称为奇异值。
- $\boldsymbol V \in \mathbb R^{N \times N} = [\boldsymbol v_1, \dots, \boldsymbol v_N]$ 称为右奇异向量矩阵，$\boldsymbol v_1, \dots, \boldsymbol v_N \in \mathbb R^N$ 称为右奇异向量。

根据前文的直观描述，这个事实应该是显然的。下面将通过一个更形式化的证明说明这一点。

存在性证明思路：反向证明存在 $\boldsymbol U^T \boldsymbol {AV}$ 得到的矩阵是一个对角矩阵，先证明存在一对向量 $\boldsymbol u, \boldsymbol v$ 使得存在正交基将 $\boldsymbol A$ 变换为分块对角形式，隔离出第一个奇异值。子矩阵则通过归纳法可以证明完毕。

{{< collapse summary="数学证明" openByDefault=false >}}

记 $\sigma_1 = \lVert \boldsymbol A \rVert_2$.则存在一组单位向量 $\boldsymbol v_1, \boldsymbol u_1$ 满足：

$$
\boldsymbol {Av}_1 = \sigma_1 \boldsymbol u_1
$$

让矩阵 $\boldsymbol U_1$ 和 $\boldsymbol V_1$ 分别是以 $\boldsymbol u_1$, $\boldsymbol v_1$ 扩展而成的的标准正交基矩阵。

$$
\boldsymbol S = \boldsymbol U_1^T \boldsymbol A \boldsymbol V_1
$$

$$
\boldsymbol S \boldsymbol e_1 = \boldsymbol U_1^T \boldsymbol A \boldsymbol V_1 \boldsymbol e_1 = \boldsymbol U_1^T \boldsymbol A \boldsymbol v_1 = \boldsymbol U_1^T (\sigma_1 \boldsymbol u_1) = \sigma_1 \boldsymbol e_1
$$

其中 $\boldsymbol e_1 = [1, 0, \dots, 0]$

从而

$$
\boldsymbol S = \boldsymbol U_1^T \boldsymbol A \boldsymbol V_1 = \begin{bmatrix} \sigma_1 & \boldsymbol w^T \\ \boldsymbol 0 & \boldsymbol B \end{bmatrix}\\
$$

同时，由于 $\boldsymbol U_1, \boldsymbol V_1$ 都是正交矩阵，从而 $\lVert \boldsymbol S \rVert_2 = \sigma_1$

根据

$$
\lVert \boldsymbol S \rVert_2 \left \lVert \begin{bmatrix} \sigma_1 \\ \boldsymbol w \end{bmatrix} \right \rVert_2 \geq \left \lVert \boldsymbol S \begin{bmatrix} \sigma_1 \\ \boldsymbol w \end{bmatrix} \right \rVert_2 \geq \sigma_1^2 + \boldsymbol w^T \boldsymbol w = (\sigma^2 + \boldsymbol w^T \boldsymbol w) \left\lVert \begin{bmatrix} \sigma_1 \\ \boldsymbol w \end{bmatrix} \right\rVert_2
$$

这要求  $\boldsymbol w$ 也是零向量。从而

$$
\boldsymbol S = \boldsymbol U_1^T \boldsymbol A \boldsymbol V_1 = \begin{bmatrix} \sigma_1 & \boldsymbol 0^T  \\ \boldsymbol 0 & \boldsymbol B \end{bmatrix}\\
$$

仅需使用归纳法，就可以完成证明。

{{< /collapse >}}



低秩近似定理（Eckart-Young-Mirsky 定理）：对于任意秩为 $k$ 的矩阵 $\boldsymbol B$, 以下不等式成立：

$$
\lVert \boldsymbol A - \boldsymbol A_k \rVert_F \leq \lVert \boldsymbol A - \boldsymbol B \rVert_F
$$

其中 $\boldsymbol A_k$ 是 $\boldsymbol A$ 的截断 SVD, 也被称为 $\boldsymbol A$ 的最佳 $k$ 秩近似。

$$
\boldsymbol A_k = \Sigma_{i = 1}^k \sigma_i \boldsymbol u_i \boldsymbol v_i^T = \boldsymbol U_k \boldsymbol \Sigma_k \boldsymbol V_k^T
$$

SVD 分解有一些性质。

- 非零奇异值的个数等于矩阵 $\boldsymbol A$ 的秩 r。
- $\boldsymbol v_1, \dots, \boldsymbol v_r$ 张成 $\boldsymbol A$ 的行空间。
- $\boldsymbol v_{r + 1} ,\dots, \boldsymbol v_N$ 张成 $\boldsymbol A$ 的零空间。
- $\boldsymbol u_1, \dots, \boldsymbol u_r$ 张成 $\boldsymbol A$ 的列空间。
- $\boldsymbol u_{r + 1}, \dots, \boldsymbol u_M$ 张成 $\boldsymbol A$ 的左零空间。
- $\sigma_1 = \max_{\lVert \boldsymbol w \rVert_2 = 1} \lVert \boldsymbol {Aw} \rVert_2 = \lVert \boldsymbol A \rVert_2$
- $\sigma_N = \min_{\lVert \boldsymbol w \rVert_2 = 1} \lVert \boldsymbol {Aw} \rVert_2$

## Weyl’s Theorem.