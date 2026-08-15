---
title: "归一化流"
date: 2026-04-09T15:25:00+08:00
weight: 5
categories: [生成式模型, 机器学习]
tags: [归一化流, Normalizing Flow, NICE, Real-NVP, Neural ODE]
series: [生成式模型]
draft: true
featured: false
params:
  mathjax: true
ShowToc: true
TocOpen: false
---

归一化流（normalizing flow, NF）是一类非常具有代表性的显式生成模型。它的基本出发点是：既然简单分布容易采样、也容易计算密度，那么能否通过一系列可逆变换，把一个简单分布逐步变成复杂数据分布？如果这件事能够做到，那么模型既能像自回归模型那样精确计算似然，又能像 VAE 那样高效采样。

这一想法看起来很直接，但真正困难的地方在于：变换既要足够灵活，能够把简单分布扭成复杂分布；又要足够可计算，使我们仍然能高效地求出变换前后密度之间的关系。归一化流的全部设计，本质上都围绕着这两个目标展开。

{{< callout type="info" title="核心思想" >}}
归一化流用一个可逆映射 $f_{\theta}$ 把简单先验分布 $p_0(\boldsymbol{z})$ 变成复杂数据分布 $p_{\theta}(\boldsymbol{x})$。关键公式是 change of variables：密度值会随着局部体积伸缩而改变，而这个伸缩因子正由 Jacobian 行列式给出。
{{< /callout >}}

## 为什么要引入归一化流

在前几类生成模型中，我们已经见过不同路线的优缺点。

- 自回归模型可以精确计算似然，但采样通常必须逐步进行，因此速度较慢。
- VAE 采样很快，也能得到潜变量表示，但训练时优化的是 ELBO 而不是精确对数似然。
- GAN 则更关注样本质量，通常并不直接给出显式密度。

Normalizing flow 想解决的问题，可以概括为课件中的一句话：can we get the best of both worlds? 也就是说，我们是否能同时得到

- 易于采样的生成过程；
- 可精确计算的对数似然；
- 可逆的隐变量表示。

若能把复杂分布表示成简单分布经过可逆变换后的结果，那么这三点便会自然统一起来。于是，flow model 的核心不再是构造一个复杂条件分布，也不再是引入难解的后验，而是设计一个足够好的可逆映射。

## 随机变量变换与 Change of Variables

### 密度不是概率

要理解 normalizing flow，首先必须分清“密度”和“概率”。

对连续随机变量而言，某一点的概率通常为零。真正有意义的是小区域上的概率质量。例如，在一维情况下，
$$
\mathbb P(x \in [a, a + \Delta a]) \approx p_x(a)\Delta a.
$$
因此，密度 $p_x(x)$ 描述的是“单位长度上的概率质量”，而不是点本身的概率。当变量经过变换后，区间长度会发生拉伸或压缩，所以密度值也必须随之调整。

课件里特别强调了一点：density is not probability。一个变换若把某个小区间压得更短，那么同样的概率质量被挤到更小的区域里，密度就会变大；反之，若把区间拉长，密度就会变小。多维情形下，区间长度对应的便是局部面积或体积。

### 一维 change of variables

设一维随机变量 $z$ 的密度为 $p_z(z)$，并令
$$
x = f(z),
$$
其中 $f$ 是可逆且单调的。若记其逆函数为
$$
z = f^{-1}(x),
$$
则有
$$
p_x(x) = p_z\left(f^{-1}(x)\right)\left|\frac{d f^{-1}(x)}{dx}\right|.
$$
这个公式说明：变换后的密度，等于原密度乘上局部长度缩放因子。

如果改用正向导数来写，也可以表示为
$$
p_x(x) = p_z(z)\left|\frac{dz}{dx}\right| = p_z(z)\left|\frac{1}{f'(z)}\right|, \qquad x = f(z).
$$
因此，若在某一点附近 $|f'(z)|$ 很大，说明该变换把局部区间拉长了，于是目标密度会相应减小。

{{< collapse summary="一维 change of variables 的推导" >}}
设 $x = f(z)$，且 $f$ 单调可逆。对足够小的区间 $[x, x + dx]$，其原像近似为 $[z, z + dz]$，并满足
$$
dx \approx \left|f'(z)\right|dz.
$$
由于变换前后对应区间上的概率质量相同，因此
$$
p_x(x)dx \approx p_z(z)dz.
$$
代入 $dz = \left|\frac{dz}{dx}\right|dx$ 得到
$$
p_x(x) = p_z(z)\left|\frac{dz}{dx}\right| = p_z\left(f^{-1}(x)\right)\left|\frac{d f^{-1}(x)}{dx}\right|.
$$
{{< /collapse >}}

### 多维情形与 Jacobian 行列式

真正用于 flow 的是多维版本。设
$$
\boldsymbol{x} = f(\boldsymbol{z}), \qquad \boldsymbol{x}, \boldsymbol{z} \in \mathbb R^d,
$$
其中 $f$ 可逆且可微。记 Jacobian 矩阵为
$$
\boldsymbol{J}_f(\boldsymbol{z}) = \frac{\partial f(\boldsymbol{z})}{\partial \boldsymbol{z}^T}.
$$
则 change of variables 公式写为
$$
p_x(\boldsymbol{x}) = p_z\left(f^{-1}(\boldsymbol{x})\right)\left|\det \frac{\partial f^{-1}(\boldsymbol{x})}{\partial \boldsymbol{x}^T}\right|.
$$
等价地，如果直接用正向映射写成关于 $\boldsymbol{z}$ 的形式，则有
$$
p_x(\boldsymbol{x}) = p_z(\boldsymbol{z})\left|\det \boldsymbol{J}_f(\boldsymbol{z})\right|^{-1}, \qquad \boldsymbol{x} = f(\boldsymbol{z}).
$$

这里的 determinant 具有清楚的几何意义。在线性变换 $\boldsymbol{x} = \boldsymbol{A}\boldsymbol{z}$ 中，矩阵 $\boldsymbol{A}$ 会把一个小立方体变成一个平行多面体，而其体积缩放倍数正是
$$
\left|\det \boldsymbol{A}\right|.
$$
一般的非线性变换在局部上可以由 Jacobian 近似，因此 Jacobian 行列式就刻画了局部体积缩放。也正因为如此，密度变换公式里会出现 determinant。

{{< callout type="tip" title="几何直觉" >}}
Jacobian 行列式的绝对值越大，说明局部体积被拉得越大，因此密度会被除以更大的数；Jacobian 行列式越小，说明体积被压缩得越厉害，因此密度会相应升高。
{{< /callout >}}

## 归一化流的表示方式

### 从简单先验到复杂数据分布

flow model 的标准写法是先从一个简单基分布（base distribution）采样：
$$
\boldsymbol{z}_0 \sim p_0(\boldsymbol{z}_0),
$$
通常 $p_0$ 取标准高斯
$$
p_0(\boldsymbol{z}_0) = \mathcal N(\boldsymbol{z}_0 \mid \boldsymbol{0}, \boldsymbol{I}).
$$
然后依次施加若干可逆变换：
$$
\boldsymbol{z}_k = f_k(\boldsymbol{z}_{k - 1}), \qquad k = 1,\dots,K,
$$
并令最终输出为
$$
\boldsymbol{x} = \boldsymbol{z}_K.
$$
于是整体映射写成
$$
\boldsymbol{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\boldsymbol{z}_0).
$$

这就是 “a flow of transformations” 的含义。单个变换可能只能做较简单的扭曲，但多个变换复合起来后，便能够逐步把简单分布推送到复杂分布。

### 与 VAE 的一个关键区别

如果从潜变量模型视角看，flow 也可以写成
$$
p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) = p_0(\boldsymbol{z})p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z}).
$$
但与 VAE 不同的是，这里的条件分布不是一个有噪声的高斯或伯努利，而是一个退化分布：
$$
p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z}) = \delta\left(\boldsymbol{x} - f_{\theta}(\boldsymbol{z})\right).
$$
也就是说，一旦给定 $\boldsymbol{z}$，$\boldsymbol{x}$ 就被唯一确定；反过来，一旦映射可逆，给定 $\boldsymbol{x}$ 也能唯一确定 $\boldsymbol{z}$。课件中把这一点概括为：given $\boldsymbol{z}$, $\boldsymbol{x}$ is unique and vice versa。

这意味着 flow 不是通过“随机 decoder”来表达复杂分布，而是通过“确定性可逆变换”来表达复杂分布。

### 似然、采样与表示

由于映射可逆，flow 的三个核心任务都很直接：

- 采样：先从 $p_0(\boldsymbol{z}_0)$ 采样，再做正向变换得到 $\boldsymbol{x}$；
- 似然计算：把观测样本 $\boldsymbol{x}$ 逆向映射回 $\boldsymbol{z}_0$，再应用 change of variables；
- 表示学习：把样本映射到 latent space 中的 $\boldsymbol{z}_0$。

因此，normalizing flow 确实在某种意义上兼具“显式密度”和“高效生成”这两项优点。

## 归一化流的学习与推断

### 链式 change of variables

对由多层可逆变换组成的 flow，密度公式可以逐层展开。若
$$
\boldsymbol{z}_k = f_k(\boldsymbol{z}_{k - 1}), \qquad \boldsymbol{x} = \boldsymbol{z}_K,
$$
则
$$
p_{\theta}(\boldsymbol{x}) = p_0(\boldsymbol{z}_0)\prod_{k = 1}^{K}\left|\det \frac{\partial f_k(\boldsymbol{z}_{k - 1})}{\partial \boldsymbol{z}_{k - 1}^T}\right|^{-1}.
$$
取对数后得到
$$
\log p_{\theta}(\boldsymbol{x}) = \log p_0(\boldsymbol{z}_0) - \sum_{k = 1}^{K}\log \left|\det \frac{\partial f_k(\boldsymbol{z}_{k - 1})}{\partial \boldsymbol{z}_{k - 1}^T}\right|.
$$

这个公式是整个 normalizing flow 的训练基础。只要我们能高效计算每一层 Jacobian 的 log-determinant，就能精确计算对数似然并做最大似然估计。

### 最大似然训练

设训练集为 $\mathcal D = \{\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(N)}\}$，则 flow 的标准学习目标就是
$$
\hat{\theta} = \arg \max_{\theta}\frac{1}{N}\sum_{n = 1}^{N}\log p_{\theta}\left(\boldsymbol{x}^{(n)}\right).
$$
这与自回归模型一样，是精确最大似然；区别只在于，flow 的对数似然来自可逆变换的 Jacobian，而自回归模型来自链式分解。

从推断角度看，对于任意给定样本 $\boldsymbol{x}$，只需逐层应用逆变换
$$
\boldsymbol{z}_{k - 1} = f_k^{-1}(\boldsymbol{z}_k),
$$
便可得到其 latent representation $\boldsymbol{z}_0$，同时累加 log-determinant 项完成密度评估。

## 设计难点：表达能力与可计算性的权衡

课件在中间部分反复强调一个 trade-off：flexibility vs tractability。因为对 flow 来说，一个变换想要真正可用，至少要满足下面几件事：

- 正向映射要容易计算，这样才便于采样；
- 逆向映射要容易计算，这样才便于求密度和做表示提取；
- Jacobian 行列式要容易计算，否则训练代价过高；
- 映射又必须足够灵活，否则只能学到很弱的分布。

一个极端反例是恒等映射
$$
f(\boldsymbol{z}) = \boldsymbol{z}.
$$
它当然完全可逆、计算也最容易，但表达能力几乎为零，因为它根本不能把简单高斯分布变成复杂数据分布。这说明“可计算”本身并不足够，真正的难点是如何在可计算前提下保持足够强的表达能力。

### 三角 Jacobian 的重要性

为了让 determinant 易于计算，一个非常重要的设计原则是让 Jacobian 尽量具有三角结构。若
$$
\boldsymbol{J} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1d} \\
0 & a_{22} & \cdots & a_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{dd}
\end{bmatrix},
$$
则其行列式满足
$$
\det(\boldsymbol{J}) = \prod_{i = 1}^{d} a_{ii}.
$$
于是
$$
\log \left|\det(\boldsymbol{J})\right| = \sum_{i = 1}^{d}\log |a_{ii}|.
$$
这比对一般稠密矩阵直接求 determinant 便宜得多。因此，很多 flow 层的设计本质上都在想办法把 Jacobian 做成三角矩阵或近似三角矩阵。

{{< callout type="warning" title="Flow 设计的核心约束" >}}
如果一个变换非常灵活，但其逆映射或 Jacobian determinant 难以计算，那么它在 normalizing flow 中就几乎无法直接用于精确最大似然训练。
{{< /callout >}}

## NICE：加性耦合层

NICE（Non-linear Independent Components Estimation）是一类很经典的 flow 设计。它的核心是 coupling layer，也就是把变量分成两部分，只让一部分去变换另一部分。

设输入向量拆成
$$
\boldsymbol{x} =
\begin{bmatrix}
\boldsymbol{x}_a \\
\boldsymbol{x}_b
\end{bmatrix}.
$$
NICE 的加性耦合层定义为
$$
\boldsymbol{y}_a = \boldsymbol{x}_a, \qquad \boldsymbol{y}_b = \boldsymbol{x}_b + m_{\theta}(\boldsymbol{x}_a),
$$
其中 $m_{\theta}$ 可以是一个复杂的神经网络。

这一层的优点非常明显。首先，它的逆变换容易写出：
$$
\boldsymbol{x}_a = \boldsymbol{y}_a, \qquad \boldsymbol{x}_b = \boldsymbol{y}_b - m_{\theta}(\boldsymbol{y}_a).
$$
其次，它的 Jacobian 是块三角矩阵，因此 determinant 很容易算。更具体地，
$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}^T} =
\begin{bmatrix}
\boldsymbol{I} & \boldsymbol{0} \\
\frac{\partial m_{\theta}(\boldsymbol{x}_a)}{\partial \boldsymbol{x}_a^T} & \boldsymbol{I}
\end{bmatrix},
$$
于是
$$
\det \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}^T} = 1.
$$

这说明 additive coupling layer 是 volume preserving 的：它可以扭曲分布的形状，但不会改变局部体积。因此，NICE 可以通过堆叠许多层来重新排列、弯折、拉开分布结构，但单靠这种层本身并不能改变整体密度尺度。

### 为什么还需要 rescaling layer

如果所有层的 determinant 都恒为 $1$，那么整个 flow 的 determinant 也始终恒为 $1$。这会显著限制模型能力，因为它无法根据输入位置自适应地拉伸或压缩体积。

因此 NICE 还会额外引入简单的 rescaling layer，例如
$$
\boldsymbol{y} = \exp(\boldsymbol{s}) \odot \boldsymbol{x},
$$
其中 $\boldsymbol{s}$ 是可学习参数。此时 Jacobian 为对角矩阵，故
$$
\log \left|\det \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}^T}\right| = \sum_{i = 1}^{d} s_i.
$$
这样模型便能够整体改变体积尺度。

不过，这种缩放仍然相对简单，因为它主要提供的是全局或逐维的常数缩放，而不是依赖输入位置的自适应拉伸。这也引出了下一步改进：Real-NVP。

## Real-NVP：仿射耦合层

Real-NVP（Real-valued Non-Volume Preserving transformation）可以看作 NICE 的非体积保持扩展。它把加性耦合改成了仿射耦合：
$$
\boldsymbol{y}_a = \boldsymbol{x}_a,
$$
$$
\boldsymbol{y}_b = \boldsymbol{x}_b \odot \exp\left(\boldsymbol{s}_{\theta}(\boldsymbol{x}_a)\right) + \boldsymbol{t}_{\theta}(\boldsymbol{x}_a),
$$
其中 $\boldsymbol{s}_{\theta}$ 和 $\boldsymbol{t}_{\theta}$ 分别输出缩放与平移参数。

其逆变换同样容易计算：
$$
\boldsymbol{x}_a = \boldsymbol{y}_a,
$$
$$
\boldsymbol{x}_b = \left(\boldsymbol{y}_b - \boldsymbol{t}_{\theta}(\boldsymbol{y}_a)\right) \odot \exp\left(-\boldsymbol{s}_{\theta}(\boldsymbol{y}_a)\right).
$$

更重要的是，Jacobian 仍然是三角结构，因此 determinant 仍然容易求：
$$
\log \left|\det \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}^T}\right| = \sum_j s_j(\boldsymbol{x}_a).
$$
与 NICE 不同，这里 determinant 不再是常数 $1$，而是输入的函数。课件中专门指出：the det Jacobian is a function of input instead of constant。也正因为如此，Real-NVP 能够在不同位置上做不同程度的拉伸与压缩，表达能力显著增强。

### 为什么 coupling layer 有效

耦合层看上去似乎有些“保守”，因为一半变量保持不动，另一半变量才被变换。但只要在层与层之间交替改变分组方式、加入 permutation 或 masking，不同维度的信息就会逐步传播到全局。多层复合后，整个分布仍然可以被大幅扭曲。

课件还提到一个理论结果：仅用 affine coupling layers 就足以逼近任意分布，至少在理论上具有普适逼近能力。不过，这个结论并不意味着实际训练一定容易，因为参数化方式可能仍然 ill-posed，优化过程中也可能出现数值不稳定、尺度爆炸等问题。

## 归一化流的表达能力与局限

从表达角度看，flow 非常优雅：它把复杂分布表示成简单分布经过连续可逆扭曲后的结果。单层 flow 也许只能完成比较粗糙的变换，但多层叠加后可以逐步把多峰分布“拉直”、把复杂几何结构“展开”。课件中的单层与多层示例，正是为了说明复合变换会显著增强表达能力。

不过，flow 的可逆性同时也是一种约束。由于映射要求一一对应，因此：

- 潜变量维度与数据维度通常必须一致；
- 若数据真实分布集中在低维流形上，光滑可逆映射未必容易精确表示；
- 为了保持 Jacobian 可计算，很多层结构必须牺牲一部分灵活性。

因此，normalizing flow 虽然兼具精确似然与快速采样，但它并不是“没有代价的最优方案”。它的优势来自可逆性，而限制也正来自可逆性。

## Neural ODE 与连续归一化流

课件最后用 Neural ODE 把 flow 推向了连续深度视角。直观地说，如果把 ResNet 看成
$$
\boldsymbol{h}_{k + 1} = \boldsymbol{h}_k + g_{\theta_k}(\boldsymbol{h}_k),
$$
那么当层数趋于无穷、单层步长趋于无穷小，就会得到一个常微分方程：
$$
\frac{d\boldsymbol{h}(t)}{dt} = g_{\theta}\left(\boldsymbol{h}(t), t\right).
$$
这里的 $g_{\theta}$ 由神经网络参数化，于是模型变成“由神经网络定义的动力系统”。

如果我们把这个动力系统用在概率变换上，就得到 continuous normalizing flow 的基本想法。此时状态随时间连续演化，输出由 ODE solver 给出，而逆变换则由反向积分得到。由于 ODE 解在适当条件下是可逆的，这条路线自然也能用于构造可逆生成模型。

在连续情形下，离散层的 Jacobian determinant 会对应一个瞬时变化率。若记 $\boldsymbol{z}(t)$ 的密度为 $p_t(\boldsymbol{z}(t))$，则满足著名的 instantaneous change of variables 公式：
$$
\frac{d}{dt}\log p_t\left(\boldsymbol{z}(t)\right) = -\operatorname{tr}\left(\frac{\partial g_{\theta}\left(\boldsymbol{z}(t), t\right)}{\partial \boldsymbol{z}(t)^T}\right).
$$
这说明在连续时间下，log-density 的变化由向量场 Jacobian 的 trace 决定。这样，Jacobian determinant 的计算问题被转化为沿时间积分 trace 的问题。

Neural ODE 并不只是 flow 的技术细节补充，而是在更一般的意义上把“层叠可逆变换”推广到了“连续可逆动力系统”。这也是后来连续归一化流、概率 ODE 等方法的重要基础。

## 小结

归一化流的主线很清楚：先从一个简单先验分布出发，再通过一串可逆变换把它推送到复杂数据分布。change of variables 公式提供了密度变换的严格依据；Jacobian 行列式给出了局部体积变化；最大似然训练则建立在可精确计算 log-determinant 的前提上。

从结构设计看，NICE 用加性耦合层把“可逆”和“易算 determinant”结合起来，但它本身保持体积不变；Real-NVP 进一步通过仿射耦合层引入输入相关的缩放，使模型更有表达能力；Neural ODE 则把 flow 推到了连续深度形式。它们共同说明了一件事：normalizing flow 的关键不只是“做可逆映射”，而是在表达能力、逆映射成本与 Jacobian 计算之间找到平衡。

因此，normalizing flow 可以被看作显式生成模型中的另一条核心路线。它没有像 VAE 那样依赖变分下界，也没有像自回归模型那样必须顺序生成，而是把问题转化为一个几何问题：如何通过一系列局部可控的可逆变换，把一个简单分布真正塑造成复杂数据分布。


一张画 change of variables 的局部体积拉伸示意，一张画 coupling layer
  的变量拆分与逆变换流程，会很适合这篇笔记。