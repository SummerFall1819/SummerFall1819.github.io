---
title: "基础知识"
date: 2026-06-12T15:10:01+08:00
weight: 2
categories: [高等代数]
tags: [矩阵, 线性变换, 符号定义]
series: [现代数值方法]
draft: true
featured: false
params:
  mathjax: true
---

本章会迅速回顾一下一些必要的知识。

## 基础概念与术语

多个标量组合在一起，就形成了**向量(Vector)**，如下所示。
$$
\boldsymbol a := [a_1, a_2, \dots, a_n]
$$

表示 $\boldsymbol a$ 是由 $n$ 个标量 $\{a_i\}_{i = 1}^n$ 组成的向量。

而由多个向量组合在一起，就形成了**矩阵(Matrix)**，如下是一个 $m \times n$ 的矩阵，即这个矩阵有 $m$ 行 $n$ 列。
$$
\boldsymbol A = \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn}
\end{bmatrix}
$$

上述写法也会被简写为 $\boldsymbol A = [a_{ij}]_{m \times n} \in \mathbb R^{m \times n}$.
特殊地，当 $m = 1$ 时，矩阵退化为一个 **行向量(Row Vector)**. $n = 1$ 时，矩阵退化为一个 **列向量(Column Vector)**. 当 $m = n$ 时，矩阵被称为 **方阵(Square Matrix)**.

书写上，通常简写向量为 $\boldsymbol a \in \mathbb R^{n \times 1} \implies \boldsymbol a \in \mathbb R^n$. 如无特殊说明，所有的符号表示的向量均为列向量。

跟随此记法，$\boldsymbol A \in \mathbb R^{m \times n} = [\boldsymbol a_1, \boldsymbol a_2, \dots, \boldsymbol a_n]$ 表示一个由 $n$ 个列向量 $\boldsymbol \{a_i\}_{i = 1}^n$ 组成的矩阵。

{{< callout type="info" title="说明" >}}
我们以下的符号虽然都是基于实空间 $\mathbb R$ 上的，但是除非另有说明，相应的操作可以拓展到复空间 $\mathbb C$ 上。
{{< /callout >}}

---


**向量空间(Vector Space)** 可以被简单的描述成一个集合，这个集合内的元素在缩放和相加后仍然属于这个集合。**线性空间(Linear Space)** 定义的加减和缩放在下一节定义。

称一组向量 **张成** 一个空间，如果这个空间内的任意向量都可以表示为这组向量的线性组合。

一个矩阵的 **值域空间(Range Space)** 描述了这个矩阵可以变换到的所有向量空间，也可以被描述为 **列空间**，因为这个空间是由矩阵的所有列向量张成的。

与之相对的， **零空间(Null Space)** 是指满足方程 $\boldsymbol {Ax} = \boldsymbol 0$ 的所有向量 $\boldsymbol x$ 组成的空间。

一个矩阵的行空间是由矩阵的所有行向量张成的。


一个矩阵的 **秩(Rank)** 表示矩阵中线性无关行或列的最大数量。

矩阵的逆。






## 向量运算

相同大小的两个向量可以进行加，减，以及乘法运算。
记 $\boldsymbol a, \boldsymbol b \in \mathbb R^n$.

$$
\boldsymbol a + \boldsymbol b = [a_1 + b_1, a_2 + b_2, \dots, a_n + b_n]^T
$$

$$
\boldsymbol a - \boldsymbol b = [a_1 - b_1, a_2 - b_2, \dots, a_n - b_n]^T
$$

乘法分为内积和外积。

内积是一个行向量和一个列向量相乘，最后的结果为一个标量。

$$
c = \boldsymbol a^T \boldsymbol b = \sum_{i = 1}^n a_ib_i
$$

外积是一个列向量和一个行向量相乘，最后的结果为一个矩阵。

$$
\boldsymbol C = \boldsymbol a \boldsymbol b^T = \begin{bmatrix}
a_1b_1 & a_1b_2 & \dots & a_1b_n \\
a_2b_1 & a_2b_2 & \dots & a_2b_n \\
\vdots & \vdots & \ddots & \vdots \\
a_nb_1 & a_nb_2 & \dots & a_nb_n
\end{bmatrix}
$$

可以将向量相乘视为矩阵相乘的特殊形式。

标量与向量相乘将会得到一个新的向量。
$$
k \boldsymbol a = [ka_1, ka_2, \dots, ka_n]^T
$$

## 矩阵运算

矩阵的运算需要满足特定的条件。

当两个矩阵大小相同时，可以进行加减运算。记 $\boldsymbol A, \boldsymbol B \in \mathbb R^{m \times n}$.

$$
\boldsymbol A \pm \boldsymbol B = \begin{bmatrix}
a_{11} \pm b_{11} & a_{12} \pm b_{12} & \dots & a_{1n} \pm b_{1n} \\
a_{21} \pm b_{21} & a_{22} \pm b_{22} & \dots & a_{2n} \pm b_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} \pm b_{n1} & a_{n2} \pm b_{n2} & \dots & a_{nn} \pm b_{nn}
\end{bmatrix}
$$

矩阵的乘法需要满足：前一个矩阵的列数与后一个矩阵的行数相同。

记 $\boldsymbol A \in \mathbb R^{m \times n}, \boldsymbol B \in \mathbb R^{n \times k}$, 那么 $\boldsymbol C = \boldsymbol {AB} \in \mathbb R^{m \times k}$. 其中
$$
c_{ij} = \sum_{l = 1}^n a_{il}b_{lj} = \boldsymbol a_i^T \boldsymbol b_j
$$


类似地，一个向量也可以和一个矩阵相乘，最后的结果是一个新的向量。记 $\boldsymbol A \in \mathbb R^{m \times n}, \boldsymbol x \in \mathbb R^n$, 那么 $\boldsymbol b \in \mathbb R^m$.

$$
\boldsymbol b = \boldsymbol {Ax} \tag{1.1}
$$

## 线性变换与线性组合

对于公式 (1.1). 我们可以做两种不同的解读。

我们可以将矩阵 $\boldsymbol A$ 视为一个 $\mathbb R^n \mapsto \mathbb R^m$ 的**线性变换**。

我们也可以依据公式

$$
\boldsymbol b = \sum_{i = 1}^m x_i \boldsymbol a_i
$$
把公式 (1.1) 视为矩阵列向量 $\boldsymbol a$ 的一个**线性组合**。




