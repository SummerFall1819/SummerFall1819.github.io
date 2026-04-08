---
title: "投影矩阵"
date: 2026-03-20T15:50:14+08:00
weight: 0
categories: [高等代数]
tags: [矩阵, 投影矩阵]
series: [现代数值方法]
draft: true
featured: false
params:
  mathjax: true
---

我们称 $\boldsymbol P^2 = \boldsymbol P$ 的矩阵为幂等矩阵。

- $\forall \boldsymbol v \in \operatorname{Range}(\boldsymbol P), \boldsymbol{Pv} = \boldsymbol v$.
- $\forall \boldsymbol v \in \mathbb R^N, \boldsymbol{Pv} - \boldsymbol v \in \operatorname{Null}(\boldsymbol P)$.

对于幂等矩阵 $\boldsymbol P$, 矩阵 $\boldsymbol I - \boldsymbol P$ 也是幂等矩阵。称 $\boldsymbol I - \boldsymbol P$ 与 $\boldsymbol P$ 互补。

- $\operatorname{Range}(\boldsymbol I - \boldsymbol P) = \operatorname{Null}(\boldsymbol P)$
- $\operatorname{Range}(\boldsymbol P) = \operatorname{Null}(\boldsymbol I - \boldsymbol P)$.
- $\operatorname{Null}(\boldsymbol P) \cap \operatorname{Range}(\boldsymbol P) = \{\boldsymbol 0\}$

证明：
对于 $\forall \boldsymbol v \in \operatorname{Range}(\boldsymbol P)$:
$$
\begin{aligned}
&(\boldsymbol I - \boldsymbol P)\boldsymbol v \\
= & \boldsymbol v - \boldsymbol{Pv}\\
= & \boldsymbol v - \boldsymbol v\\
= & \boldsymbol 0
\end{aligned}
$$

对于 $\forall \boldsymbol v \in \operatorname{Null}(\boldsymbol P)$:
$$
\begin{aligned}
&(\boldsymbol I - \boldsymbol P) \boldsymbol v \\
= & \boldsymbol v - \boldsymbol{Pv}\\
= & \boldsymbol v - \boldsymbol 0\\
= & \boldsymbol v
\end{aligned}
$$

如果存在 $\boldsymbol v \in \operatorname{Null}(\boldsymbol P) \cap \operatorname{Range}(\boldsymbol P)$, 则
$$
\boldsymbol {Pv} = \boldsymbol v, \boldsymbol {Pv} = \boldsymbol 0 \implies \boldsymbol v = \boldsymbol 0
$$

由于 $\boldsymbol v = \boldsymbol{Pv} + (\boldsymbol I - \boldsymbol P)\boldsymbol v$, 所以 $\mathbb R^N = \operatorname{Range}(\boldsymbol P) \oplus \operatorname{Null}(\boldsymbol P)$. 这里 $\oplus$ 表示子空间的直和。

给定两个子空间 $\mathcal S_1, \mathcal S_2$.

$$\mathcal S_1 \oplus \mathcal S_2 = \mathbb R^N \iff \mathcal S_1 \cap \mathcal S_2 = \{\boldsymbol 0\} \quad \text {and} \quad \mathcal S_1 + \mathcal S_2 = \mathbb R^N$$

那么称这两个子空间是正交的，也可记为 $\mathcal S_1 = \mathcal S_2^{\perp}$.

对于给定的正交子空间 $\mathcal S_1, \mathcal S_2$, 满足 $\operatorname{Range}(\boldsymbol P) = \mathcal S_1, \operatorname{Null}(\boldsymbol P) = \mathcal S_2$ 的幂等矩阵 $\boldsymbol P$ **存在** 且 **唯一**。

证明：

$$\forall \boldsymbol v \in \mathbb R^N, \exists \boldsymbol v_1 \in \mathcal S_1, \boldsymbol v_2 \in \mathcal S_2 \quad \text  {s.t.} \quad \boldsymbol v = \boldsymbol v_1 + \boldsymbol v_2$$

从而

$$\boldsymbol P \boldsymbol v = \boldsymbol v_1, (\boldsymbol I - P) \boldsymbol v = \boldsymbol v_2$$

如果存在另一组表示 $\boldsymbol v = \boldsymbol u_1 + \boldsymbol u_2$, 则

$$\boldsymbol v = \boldsymbol u_1 + \boldsymbol u_2 = (\boldsymbol v_1 + \boldsymbol \delta) + (\boldsymbol v_2 - \boldsymbol \delta) \implies \boldsymbol \delta \in \mathcal S_1 \cap \mathcal S_2 = \{\boldsymbol 0\}$$

对于任意 $\boldsymbol v$, $\boldsymbol v_1, \boldsymbol v_2$ 都是唯一的，从而 $\boldsymbol P$ 是唯一的。

---

对矩阵 $\boldsymbol P$, 考虑特征值分解 $\boldsymbol P = \boldsymbol Q \boldsymbol \Lambda \boldsymbol Q^{-1}$, 则

$$
\boldsymbol P^2 = \boldsymbol Q \boldsymbol \Lambda \boldsymbol Q^{-1} \boldsymbol Q \boldsymbol \Lambda \boldsymbol Q^{-1} = \boldsymbol Q \boldsymbol \Lambda^2 \boldsymbol Q^{-1} \implies \boldsymbol \Lambda^2 = \boldsymbol \Lambda
$$

从而 $\lambda_i \in \{0,1\}, \forall i$.

> 一个满秩的幂等矩阵必然是单位阵。

对于 $\mathbb R^N$ 下的一个子空间 $\mathcal S$, 可以通过下述方法构造对应的投影矩阵 $\boldsymbol P$, 记为 $\boldsymbol P_{\mathcal S}$
- 找到 $\mathcal S$ 的一个基 $\boldsymbol Q = [\boldsymbol q_1, \boldsymbol q_2, \cdots, \boldsymbol q_k]$.
- $ \boldsymbol P_{\mathcal S} = \boldsymbol Q \boldsymbol Q^T$.

---


对于低维投影，存在正交投影和倾斜投影两种方式。以二维投射到一维为例：

[视频]


一个矩阵 $\boldsymbol P$ 是一个正交投影器，当且仅当：
- $\boldsymbol P$ 是幂等的。
- $\boldsymbol P$ 是厄米特矩阵（实数领域内为实对称矩阵）

等价于存在一个子空间 $\mathcal S$, 使得 $\boldsymbol P = \boldsymbol P_\mathcal S$.

