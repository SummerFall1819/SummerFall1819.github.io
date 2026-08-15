---
title: "生成对抗网络"
date: 2026-04-09T09:58:51+08:00
weight: 6
categories: [生成式模型, 机器学习]
tags: [GAN]
series: [生成式模型]
draft: true
featured: false
params:
  mathjax: true
ShowToc: true
TocOpen: false
---

## 生成对抗网络

生成对抗网络（Generative Adversarial Network, GAN）是一类典型的隐式生成模型。它并不要求我们显式写出数据分布的概率密度函数，而是通过“生成器”与“判别器”之间的对抗训练，迫使模型生成与真实数据难以区分的样本。与自回归模型、变分自编码器、归一化流等显式密度模型相比，GAN 更直接地把注意力放在样本质量与分布匹配上，而不是显式似然值本身。

从建模角度看，GAN 试图回答的问题是：如果我们只要求模型能够产生与真实样本“无法区分”的结果，是否就能绕开高维密度估计中的许多困难，并得到质量更高的生成样本。围绕这一问题，GAN 引出了三个彼此紧密相连的主题：

- 为什么最大似然并不总是样本质量的理想指标；
- 如何只利用样本而不显式利用密度来训练生成模型；
- 为什么这种对抗式训练在理论上优雅、但在优化上又异常困难。

## 从最大似然到样本质量

在许多经典生成模型中，学习目标都可以写成最大似然估计。设训练样本为 $\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(n)} \sim p_{\text{data}}(\boldsymbol{x})$，则参数学习通常写为
$$
\hat{\theta}=\arg\max_{\theta}\sum_{i=1}^{n}\log p_{\theta}(\boldsymbol{x}^{(i)}).
$$
这一目标的优点非常明确。若模型容量充分、模型族中确实包含真实分布，则最大似然估计具有良好的统计效率；同时，较高的似然也意味着更好的无损压缩性能。因此，自回归模型、VAE 以及归一化流等模型，通常都把最大化或近似最大化似然作为训练的核心。

然而，样本质量与测试似然并不总是一致的。理想情况下，最优生成模型应当同时给出最高测试对数似然和最好的样本质量；但一旦模型并不完美，这两者便可能明显分离。

一种典型反例是“高似然、差样本”。设模型分布为
$$
p_{\theta}(\boldsymbol{x})=0.01\,p_{\text{data}}(\boldsymbol{x})+0.99\,p_{\text{noise}}(\boldsymbol{x}),
$$
其中 $p_{\text{noise}}$ 表示某种与数据毫不相干的噪声分布。这个模型生成的样本中，$99\%$ 都只是噪声，因此感知质量极差；但对来自真实分布的样本 $\boldsymbol{x}$，有
$$
\log p_{\theta}(\boldsymbol{x})=\log\bigl(0.01\,p_{\text{data}}(\boldsymbol{x})+0.99\,p_{\text{noise}}(\boldsymbol{x})\bigr)\ge\log\bigl(0.01\,p_{\text{data}}(\boldsymbol{x})\bigr)=\log p_{\text{data}}(\boldsymbol{x})-\log 100.
$$
于是
$$
\mathbb E_{p_{\text{data}}}\bigl[\log p_{\theta}(\boldsymbol{x})\bigr]\ge\mathbb E_{p_{\text{data}}}\bigl[\log p_{\text{data}}(\boldsymbol{x})\bigr]-\log 100.
$$
另一方面，由 KL 散度的非负性可知
$$
\mathbb E_{p_{\text{data}}}\bigl[\log p_{\text{data}}(\boldsymbol{x})\bigr]\ge\mathbb E_{p_{\text{data}}}\bigl[\log p_{\theta}(\boldsymbol{x})\bigr].
$$
这说明，尽管模型的采样质量极差，它在测试似然上却未必会显著吃亏。特别是在高维情形下，$\log 100$ 这一常数量级常常并不大。

另一种反例是“好样本、低似然”。如果模型只是机械记忆训练集，那么它生成的样本可能看起来非常逼真，甚至与训练样本几乎无法区分；但对于测试集中的新样本，它可能分配极低甚至为零的概率质量。此时，模型的感知样本质量很高，但测试对数似然却很差。

这些现象表明，似然值更适合作为密度估计和压缩任务的指标，而不一定适合作为样本感知质量的直接指标。若目标转向“生成高质量样本”，则有必要考虑一种不直接依赖显式似然函数的学习思路，即 likelihood-free learning。

## 隐式生成模型

GAN 所采用的是隐式生成模型。它从一个简单先验分布中采样隐变量
$$
z \sim p(z),
$$
其中 $p(z)$ 通常取标准高斯分布或均匀分布，然后通过一个确定性映射 $G_{\theta}$ 生成样本
$$
\boldsymbol{x} = G_{\theta}(z).
$$
这里的 $G_{\theta}$ 一般是一个深度神经网络，因此整个模型可以看作“随机输入 $z$”与“确定性生成器 $G_{\theta}$”的组合。

与 VAE 或 Flow 类模型相比，GAN 的关键区别不在于也使用了“隐变量 $\to$ 样本”的生成机制，而在于它通常并不要求显式计算诱导分布 $p_G(\boldsymbol{x})$ 的密度值。我们只要求：

- 能够方便地从 $p(z)$ 采样；
- 能够方便地通过 $G_{\theta}$ 生成样本；
- 不一定能够显式写出 $p_G(\boldsymbol{x})$ 的解析表达式。

因此，GAN 的困难不在采样，而在学习：给定真实样本 $\{\boldsymbol{x} \sim p_{\text{data}}\}$ 和模型样本 $\{\boldsymbol{x} \sim p_G\}$，如何调整 $\theta$，使得 $p_G$ 尽可能接近 $p_{\text{data}}$。

## 两样本检验

由于无法方便地直接比较密度，一个自然想法是：只比较样本。设
$$
S_1=\{\boldsymbol{x} \sim P\},\qquad S_2=\{\boldsymbol{x} \sim Q\},
$$
两样本检验（two-sample test）研究的问题就是：如何仅凭来自 $P$ 与 $Q$ 的有限样本，判断二者是否相同。

标准的统计假设写为
$$
H_0: P=Q,\qquad H_1: P\ne Q.
$$
检验通常通过某个统计量 $T(S_1,S_2)$ 来完成。若 $T$ 大于阈值 $\alpha$，则拒绝原假设 $H_0$；否则认为观测结果与 $H_0$ 相容。这里最重要的一点在于：统计量只依赖样本，而不依赖 $P$ 与 $Q$ 的显式密度表达式。

在低维场景中，可以用样本均值、样本方差等手工构造统计量。例如：

- 线性统计量主要比较一阶矩；
- 二次统计量可以比较二阶矩；
- 更复杂的统计量则对应更高阶结构。

但在图像等高维数据上，仅比较少数低阶矩远远不够。两组样本可能均值和方差都接近，却在语义结构上完全不同。因此，GAN 的核心思想之一就是：不预先手工指定统计量，而是直接学习一个能够区分两组样本的函数。

## 判别器作为可学习的两样本检验器

设真实数据分布为 $p_{\text{data}}$，模型分布为 $p_G$。GAN 引入一个判别器 $D_{\phi}(\boldsymbol{x})$，它接收样本 $\boldsymbol{x}$ 并输出一个介于 $0$ 与 $1$ 之间的数，用来表示“该样本来自真实数据”的概率。

对于固定的生成器 $G$，判别器的训练目标是做二分类：

- 对真实样本 $\boldsymbol{x} \sim p_{\text{data}}$，希望 $D(\boldsymbol{x})$ 尽可能接近 $1$；
- 对生成样本 $\boldsymbol{x} \sim p_G$，希望 $D(\boldsymbol{x})$ 尽可能接近 $0$。

对应的目标函数为
$$
\max_D V(G,D)=\mathbb E_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})]+\mathbb E_{\boldsymbol{x} \sim p_G}[\log(1-D(\boldsymbol{x}))].
$$
这本质上就是交叉熵下的二分类问题。判别器越强，说明它越能区分“真实样本”和“生成样本”，也就说明当前 $p_G$ 与 $p_{\text{data}}$ 的差异越明显。

更进一步，若判别器函数类足够丰富，则它可以看作一种高维空间中的可学习检验统计量。与传统两样本检验相比，GAN 并不是事先固定“比较哪些统计特征”，而是让神经网络自动学习最有助于区分两组样本的特征。

## 生成对抗网络的目标函数

生成器的任务恰好与判别器相反。它希望生成样本 $G(z)$ 能够骗过判别器，使 $D(G(z))$ 尽可能接近 $1$。于是，标准 GAN 的训练目标写为
$$
\min_G \max_D V(G,D)=\mathbb E_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})]+\mathbb E_{z \sim p(z)}[\log(1-D(G(z)))].
$$

这一定义将生成模型训练转化为一个极小极大博弈：

- 判别器试图区分真假样本；
- 生成器试图让假样本看起来足够真实；
- 二者相互推动，理想情况下最终达到平衡。

这一结构给出了 GAN 的名称来源：生成器与判别器彼此对抗（adversarial），而模型的分布匹配正是通过这种对抗机制实现的。

## 最优判别器与 Jensen-Shannon 散度

GAN 最优雅的理论结果之一，是在固定生成器的条件下可以显式求出最优判别器。对每个样本点 $\boldsymbol{x}$，目标函数中与 $D(\boldsymbol{x})$ 相关的部分为
$$
p_{\text{data}}(\boldsymbol{x})\log D(\boldsymbol{x})+p_G(\boldsymbol{x})\log(1-D(\boldsymbol{x})).
$$
更具体地，把
$$
a=p_{\text{data}}(\boldsymbol{x}),\qquad b=p_G(\boldsymbol{x}),\qquad d=D(\boldsymbol{x})
$$
代入后，可把点态目标写成
$$
f(d)=a\log d+b\log(1-d),\qquad d\in(0,1).
$$
对 $d$ 求导，有
$$
f'(d)=\frac{a}{d}-\frac{b}{1-d}.
$$
令一阶导数为零，可得
$$
\frac{a}{d}=\frac{b}{1-d}\quad\Longrightarrow\quad a(1-d)=bd\quad\Longrightarrow\quad d=\frac{a}{a+b}.
$$
再看二阶导数，
$$
f''(d)=-\frac{a}{d^2}-\frac{b}{(1-d)^2}<0,
$$
因此该驻点对应极大值。于是最优判别器为
$$
D_G^{\*}(\boldsymbol{x})=\frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x})+p_G(\boldsymbol{x})}.
$$

将其代回目标函数，有
$$
V(G,D_G^{\*})=\mathbb E_{\boldsymbol{x} \sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x})+p_G(\boldsymbol{x})}\right]+\mathbb E_{\boldsymbol{x} \sim p_G}\left[\log\frac{p_G(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x})+p_G(\boldsymbol{x})}\right].
$$
令
$$
m(\boldsymbol{x})=\frac{p_{\text{data}}(\boldsymbol{x})+p_G(\boldsymbol{x})}{2},
$$
则上式可改写为
$$
V(G,D_G^{\*})=D_{\mathrm{KL}}(p_{\text{data}}\|m)+D_{\mathrm{KL}}(p_G\|m)-\log 4.
$$
因此
$$
V(G,D_G^{\*})=2D_{\mathrm{JS}}(p_{\text{data}},p_G)-\log 4,
$$
其中
$$
D_{\mathrm{JS}}(p,q)=\frac{1}{2}D_{\mathrm{KL}}\left(p \,\middle\|\, \frac{p+q}{2}\right)+\frac{1}{2}D_{\mathrm{KL}}\left(q \,\middle\|\, \frac{p+q}{2}\right)
$$
称为 Jensen-Shannon 散度。

于是，标准 GAN 可以理解为：生成器通过最小化一个由判别器诱导出的 Jensen-Shannon 散度，来逼近真实分布。

Jensen-Shannon 散度具有几个重要性质：

- $D_{\mathrm{JS}}(p,q)\ge 0$；
- 当且仅当 $p=q$ 时，$D_{\mathrm{JS}}(p,q)=0$；
- $D_{\mathrm{JS}}(p,q)=D_{\mathrm{JS}}(q,p)$；
- $\sqrt{D_{\mathrm{JS}}(p,q)}$ 满足三角不等式，因此可诱导一个距离。

因此，GAN 的全局最优解对应于
$$
p_G=p_{\text{data}},
$$
此时
$$
V(G^{\*},D^{\*})=-\log 4.
$$

## GAN 的训练算法

在实践中，GAN 通过交替优化（alternating optimization）来训练。设一个 minibatch 的大小为 $m$，则每一步通常执行以下过程：

1. 从训练集采样 $m$ 个真实样本 $\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(m)}$；
2. 从先验分布采样 $m$ 个噪声向量 $z^{(1)},\dots,z^{(m)}$；
3. 固定生成器参数 $\theta$，更新判别器参数 $\phi$，做随机梯度上升：
$$
\nabla_{\phi}V(G_{\theta},D_{\phi})=\frac{1}{m}\nabla_{\phi}\sum_{i=1}^{m}\bigl[\log D_{\phi}(\boldsymbol{x}^{(i)})+\log(1-D_{\phi}(G_{\theta}(z^{(i)})))\bigr].
$$
4. 固定判别器参数 $\phi$，更新生成器参数 $\theta$，做随机梯度下降：
$$
\nabla_{\theta}V(G_{\theta},D_{\phi})=\frac{1}{m}\nabla_{\theta}\sum_{i=1}^{m}\log(1-D_{\phi}(G_{\theta}(z^{(i)}))).
$$

这一算法形式上非常简单，但真正的困难在于：判别器和生成器不是分别优化独立目标，而是在同一个博弈系统中交替更新。因此，单个目标函数的最优性并不自动保证整个训练过程稳定。

## 理论最优与实践优化之间的差距

从理论上看，如果每次更新时判别器都能达到最优，并且生成器是在函数空间中精确更新，那么生成器应当逐步逼近真实分布。但这些假设在深度学习中通常并不成立：

- 判别器在每一步都只能近似最优；
- 生成器更新是在参数空间而非函数空间进行；
- 判别器与生成器都只用有限容量的神经网络表示；
- 实际训练依赖的是随机梯度法，而不是精确优化。

因此，GAN 训练中常见的现象不是单调收敛，而是振荡、发散甚至周期性循环。与最大似然训练不同，GAN 的损失曲线也缺乏明确、稳健的停止准则。即使判别器损失和生成器损失都看起来“稳定”，也不意味着生成样本已经覆盖了真实分布的全部模式。

## Mode Collapse

GAN 中最著名的训练问题是 mode collapse。所谓 mode collapse，是指生成器只学会了真实分布中的一个或少数几个模式，从而反复生成高度相似的样本，而没有覆盖全部数据变化。

例如，若真实分布是多个高斯峰的混合分布，那么一个理想生成器应当覆盖所有峰；但在实际训练中，生成器可能只在其中一个峰附近集中采样，随后又跳到另一个峰附近，于是出现“在不同模式之间来回震荡”的现象。这种行为会带来两个直接后果：

- 生成样本看起来局部上可能是清晰的；
- 但整体样本缺乏多样性，不能代表真实分布的全貌。

从几何上看，Jensen-Shannon 散度的优化景观在某些场景下存在较差的局部极值。与 KL 散度相比，JSD 往往更偏向“模式选择”而非“模式覆盖”。这种差异可以从任务目标上理解：

- 在密度估计或压缩任务中，希望所有真实样本都被分配概率质量，因此更强调覆盖全部模式；
- 在样本生成任务中，跨越不同模式进行平均，反而容易落在数据流形之外，从而产生模糊而不真实的样本；
- 因而只抓住单个模式，有时会得到更“好看”的样本，但却损失了全局多样性。

这也是为什么“高样本质量”与“高似然覆盖”并不总是相容的。

## Mode-Seeking 与 Mode-Covering

为了更精确地描述这种差异，可以引入 mode-seeking 与 mode-covering 两个术语。若我们最小化前向 KL 散度
$$
D_{\mathrm{KL}}(p_{\text{data}}\|p_G)=\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\log\frac{p_{\text{data}}(\boldsymbol{x})}{p_G(\boldsymbol{x})}\right],
$$
则期望是对真实分布取的。只要某个真实模式上 $p_{\text{data}}(\boldsymbol{x})$ 很大而 $p_G(\boldsymbol{x})$ 很小，损失就会迅速增大。因此，前向 KL 倾向于要求模型给所有真实模式都分配足够概率质量，这种行为常被称为 mode-covering。

与之相对，若最小化反向 KL 散度
$$
D_{\mathrm{KL}}(p_G\|p_{\text{data}})=\mathbb E_{\boldsymbol{x}\sim p_G}\left[\log\frac{p_G(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x})}\right],
$$
则期望是对模型分布取的。模型更关心自己实际会采样到的区域：如果它把概率质量放在真实分布很低甚至为零的区域，代价会很大；但若它完全忽略某些真实模式，只要自己从不采到这些位置，惩罚反而未必明显。因此，反向 KL 更容易表现出 mode-seeking 行为，即宁可集中在少数高密度模式附近，也不愿意在模式之间“铺开”。

GAN 在理想化推导下对应的是 Jensen-Shannon 散度，而不是简单的前向 KL 或反向 KL；但在有限容量、有限步数和非凸优化的实际训练中，它经常表现出更接近 mode-seeking 的倾向。直观上说，生成器更愿意先抓住几个容易骗过判别器的尖锐模式，而不是一开始就覆盖整个数据分布。这也解释了为什么 GAN 往往能生成较清晰的样本，却更容易出现 mode collapse。

## 从 JSD 到更一般的 $f$-divergence

既然 JSD 可能难以优化，一个自然问题是：能否用其他散度替换它。更一般地，给定两个分布 $p$ 与 $q$，$f$-divergence 定义为
$$
D_f(p,q)=\mathbb E_{\boldsymbol{x} \sim q}\left[f\!\left(\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right)\right],
$$
其中 $f$ 是满足 $f(1)=0$ 的凸、下半连续函数。

由 Jensen 不等式可知
$$
D_f(p,q)\ge 0.
$$
很多熟悉的散度都可以写成 $f$-divergence 的特例，例如：

- KL 散度；
- reverse KL；
- Jensen-Shannon 散度；
- total variation；
- Pearson $\chi^2$ 散度；
- Hellinger 距离；
- $\alpha$-divergence。

这表明，GAN 目标并不是孤立的，而是更大一类分布差异度量的具体实例。理论上，我们完全可以构造基于其他 $f$-divergence 的对抗训练框架，例如 f-GAN。

不过，改变散度并不自动解决训练问题。实际困难往往不仅来自“目标值是什么”，更来自“优化动力学如何演化”。即便某个目标在静态意义上更合理，交替 SGD 在动态上仍可能不收敛。

## 从目标函数到优化动力学

GAN 的训练本质上是一个向量场上的迭代过程。与单目标最优化不同，极小极大问题的梯度场通常既包含“朝向最优点”的分量，也包含“绕着最优点旋转”的分量。因此，随机梯度交替更新未必会真正收敛到平衡点。

一个经典的玩具例子是 Dirac GAN。设真实分布与模型分布分别为两个 Dirac 分布：
$$
p_{\text{data}}(x)=\delta(x-c),\qquad p_G(x)=\delta(x-\theta),
$$
并取线性判别器
$$
D(x)=\phi x.
$$
则目标可写为
$$
\min_{\theta}\max_{\phi}V(\theta,\phi)=(c-\theta)\phi.
$$
当 $c=0$ 时，最优点在 $(\theta,\phi)=(0,0)$，但交替梯度法在该点附近往往表现为旋转而非收敛。这个例子说明：即使理论最优解非常简单，训练轨迹仍可能因为梯度场的旋转结构而长期震荡。

因此，GAN 的困难并不只是“目标定义得对不对”，而是“这个极小极大系统是否容易被数值方法稳定求解”。

## Mescheder 等人关于收敛性的分析

Mescheder、Geiger 与 Nowozin 在 2018 年的论文《Which Training Methods for GANs do actually Converge?》中，对这一问题给出了更系统的解释。该文指出，先前一些局部收敛性分析往往假设真实分布与生成分布是绝对连续的，但这一假设对图像建模并不自然，因为真实图像通常集中在高维空间中的低维流形附近，生成器学到的分布也常常具有类似结构。

在这种流形情形下，GAN 的不稳定性不是单纯的数值偶然，而是目标动力学本身的结构性问题。即使在平衡点附近，判别器也可能保留沿数据流形法向方向的非零梯度。这些梯度与真正有用的“回到数据流形”的方向并不一致，却会继续驱动生成器更新，从而让训练轨迹出现旋转、震荡甚至循环。

该文的一个重要结论是：并非所有常见 GAN 训练方法都真的局部收敛。以 Dirac-GAN 为例，当每次生成器更新之间只进行有限次判别器更新时，未正则化的 GAN、WGAN 以及 WGAN-GP 都不一定会收敛到平衡点。相反，instance noise 与 zero-centered gradient penalties 则可以恢复局部收敛。

从这个角度看，GAN 训练的稳定化并不只是“让判别器更弱”或“把损失函数换掉”，而是要主动改变平衡点附近的向量场结构，使系统减少围绕平衡点打转的趋势，更接近一个具有收缩性的优化过程。

## 判别器正则化与稳定化

既然问题出在动力学上，一类自然思路便是对判别器加入适当正则化，以抑制这些无效的法向梯度。Mescheder 等人特别强调了两种有效策略：instance noise 与 zero-centered gradient penalties。

instance noise 的思想是在真实样本与生成样本上都加入小的高斯噪声，使原本集中在低维流形上的分布被适度“抹厚”，从而改善两者支撑集的重叠情况。这样一来，判别器学习到的差异信号会更加平滑，对生成器的反馈也更稳定。

另一类更常用的方法是零中心梯度惩罚。Mescheder 等人强调了两种简化形式：
$$
R_1(\phi)=\frac{\gamma}{2}\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\|\nabla_{\boldsymbol{x}}D_{\phi}(\boldsymbol{x})\|_2^2\right],
$$
以及
$$
R_2(\theta,\phi)=\frac{\gamma}{2}\mathbb E_{\boldsymbol{x}\sim p_G}\left[\|\nabla_{\boldsymbol{x}}D_{\phi}(\boldsymbol{x})\|_2^2\right].
$$
其中 $R_1$ 只在真实样本上惩罚判别器梯度，$R_2$ 只在生成样本上惩罚判别器梯度。它们的共同作用，是鼓励判别器在平衡点附近对数据流形保持更平坦的形状，尤其避免产生无意义的法向梯度。

从动力学角度看，这类零中心正则化等于主动削弱了博弈系统中的旋转分量，使平衡点附近的向量场更接近一个普通的收缩型优化问题。Mescheder 等人的分析说明，在合适的正则化强度与光滑性假设下，这些简化梯度惩罚能够在更一般的流形情形下恢复局部收敛。这也解释了为什么后来的很多稳定化技巧，实质上都在想办法让判别器在数据流形附近“不要过于敏感”。

## DCGAN：早期稳定训练的架构经验

实践中，GAN 的成功很大程度上依赖架构设计。Deep Convolutional GAN（DCGAN）给出了一组影响深远的经验规则。

首先，在生成器中不再使用传统的池化操作，而采用转置卷积逐步上采样；在判别器中则使用带步长卷积进行下采样。这样可以让网络以端到端可学习的方式控制分辨率变化。

其次，非线性与归一化的选择也十分关键：

- 生成器隐藏层常使用 ReLU；
- 判别器常使用 Leaky ReLU；
- 生成器输出层常使用 `tanh`；
- 判别器输出层常使用 `sigmoid`；
- 批归一化通常有助于稳定训练；
- 但生成器输出端与判别器输入端往往不施加批归一化。

优化上，DCGAN 常使用 Adam，并采用较小学习率与较小动量。经验上，类似
$$
\text{LR}\approx 2\times 10^{-4},\qquad\beta_1 \approx 0.5
$$
这样的设置往往更稳定。

DCGAN 的意义不仅在于能生成比早期 GAN 更清晰的图像，还在于它展示了潜在空间中的语义结构。许多属性可以表现为近似线性的向量方向，例如
$$
\text{smiling woman}-\text{neutral woman}+\text{neutral man}\approx\text{smiling man}.
$$
这说明生成器学到的不只是像素级插值，而是某种具有语义规律的表示空间。

## Progressive Growing GAN

随着分辨率提升，GAN 训练会变得更加困难。Progressive Growing of GANs 的核心思想是：从低分辨率开始训练，再逐步加入新层，把模型平滑地扩展到更高分辨率。

这种做法的优势在于：

- 训练初期只需要学习整体轮廓与粗结构，优化更稳定；
- 分辨率逐步增加时，模型可以在已有粗结构基础上补充高频细节；
- 生成器与判别器的容量是随训练过程渐进增加的，而不是一开始就面对最大难度任务。

这一策略显著改善了高分辨率图像生成的稳定性，并使 GAN 首次在高质量人脸和复杂场景图像上取得了非常有影响力的结果。

## StyleGAN：分层控制图像风格

StyleGAN 在 Progressive GAN 的基础上进一步重新设计了生成器结构。它不再直接把潜变量 $z$ 输入合成网络，而是先通过一个映射网络
$$
f: z \mapsto w
$$
把 $z$ 映射到中间潜空间 $\mathcal W$，再由合成网络逐层接收由 $w$ 控制的 style 信号。

典型的 StyleGAN 生成器具有以下结构特征：

- 从一个可学习的常数张量开始，而不是直接把 $z$ reshape 成特征图；
- 每一层卷积都通过 style 参数调制；
- 使用 AdaIN 等机制控制不同通道的统计特征；
- 在不同分辨率层级注入独立噪声，以控制随机细节。

这种设计使潜变量的作用更加分层：

- 粗粒度层更容易控制姿态、脸型、整体布局等全局属性；
- 中等尺度层更容易控制五官、发型等局部结构；
- 细粒度层更容易控制肤质、发丝、雀斑等高频纹理。

因此，StyleGAN 不仅生成质量更高，也表现出更强的可解释性与可控性。style mixing 的实验进一步表明，不同层的风格控制确实对应不同尺度的视觉因素。

## 一些实用经验

GAN 的发展历史表明，单纯更换目标函数往往不如改进架构和正则化来得有效。许多实际有效的经验包括：

- 判别器正则化通常有助于稳定训练；
- 架构设计常常比理论上更换散度更重要；
- 谱归一化（spectral normalization）通常是有效的稳定化手段；
- 条件生成中的 projection discriminator 往往表现较好。

这说明 GAN 的问题具有明显的“工程化”特征：理论目标固然重要，但可训练性常常由网络结构、正则化和优化细节共同决定。

## 生成模型的评价

评价生成模型本身就是一个困难问题。对显式密度模型，如 VAE、PixelCNN、Flow 等，可以直接考虑测试似然；但对 GAN 而言，似然往往不可得，因此评价更多依赖样本质量、多样性、下游任务表现以及人工主观判断。

从原则上说，一个好的生成模型至少应满足两个方面：

- 单个样本应当清晰、自然、语义明确；
- 整体样本集合应当具有足够多样性，而不是重复同一模式。

这也是 Inception Score 与 FID 试图刻画的两个维度。

## Inception Score

Inception Score 依赖一个预训练的 Inception-v3 分类器。对生成样本 $\boldsymbol{x}$，分类器输出条件分布 $p(y\mid \boldsymbol{x})$，其中 $y$ 是 ImageNet 的类别标签。其直观解释是：

- 若样本 $\boldsymbol{x}$ 足够清晰，则 $p(y\mid \boldsymbol{x})$ 应当尖锐，即条件熵较低；
- 若模型整体具有多样性，则边缘分布 $p(y)$ 应当分散，即边缘熵较高。

这里的边缘标签分布定义为
$$
p(y)=\int_{\boldsymbol{x}} p(y\mid \boldsymbol{x})p_G(\boldsymbol{x})\,d\boldsymbol{x}.
$$
于是 Inception Score 定义为
$$
\mathrm{IS}=\exp\left(\mathbb E_{\boldsymbol{x} \sim p_G}\bigl[D_{\mathrm{KL}}(p(y\mid \boldsymbol{x})\|p(y))\bigr]\right).
$$
利用熵的恒等式，上式也可写为
$$
\mathrm{IS}=\exp\bigl(H(y)-H(y\mid \boldsymbol{x})\bigr).
$$

为了理解这一公式，可以回顾熵的基本含义。熵衡量的是不确定性。以二元分布为例，若事件发生概率为 $p$，则其熵为
$$
H(p)=-p\log_2 p-(1-p)\log_2(1-p).
$$
当 $p=0.5$ 时熵最大；当 $p$ 接近 $0$ 或 $1$ 时熵较小。Inception Score 正是在利用这一思想：希望单个样本的类别预测不确定性低，而整个样本集合的类别覆盖不确定性高。

不过，Inception Score 有明显局限。它并不直接参考真实数据分布，而只分析生成样本本身。因此，理论上只要模型在 1000 个 ImageNet 类别上各生成一个非常清晰的样本，就可能得到极高的分数，即使这些样本根本不符合目标数据集的真实分布。

## Fréchet Inception Distance

为弥补 Inception Score 不比较真实数据的缺陷，Fréchet Inception Distance（FID）把真实样本与生成样本都映射到 Inception-v3 的特征空间，再比较两组特征分布。

具体地，将图像送入 Inception-v3 的 pool3 层，得到 2048 维特征表示。分别记真实数据特征的均值与协方差为 $(m,C)$，生成数据特征的均值与协方差为 $(m_w,C_w)$。FID 定义为两个高斯近似分布之间的 Fréchet 距离：
$$
\mathrm{FID}=\|m-m_w\|_2^2+\mathrm{Tr}\!\left(C+C_w-2(CC_w)^{1/2}\right).
$$

FID 有几个重要特点：

- 它显式比较真实样本与生成样本；
- 它同时考虑均值差异与协方差差异；
- 它通常比 Inception Score 更符合人的直觉；
- 对模糊、遮挡、噪声、结构破坏等失真往往更敏感。

因此，在现代 GAN 研究中，FID 往往是更常用也更可信的评价指标。

## 小结

生成对抗网络提供了一种与最大似然截然不同的生成模型学习思路。它不要求显式写出概率密度，而是通过判别器构造一个可学习的两样本检验，并在此基础上驱动生成器逼近真实分布。

从理论上看，标准 GAN 在最优判别器下等价于最小化 Jensen-Shannon 散度，这使其具有简洁而优雅的分布匹配解释；但从实践上看，GAN 的核心难点在于极小极大博弈的优化动力学，mode collapse、振荡与不稳定训练几乎贯穿整个发展过程。

GAN 的发展历程也说明，成功往往不来自单一因素，而来自目标函数、优化方法、判别器正则化以及网络架构的共同改进。DCGAN、Progressive GAN 与 StyleGAN 分别体现了稳定训练、高分辨率生成与可控表示学习三个关键方向。最后，在评价上，单个样本是否逼真只是问题的一部分，如何同时衡量清晰度、多样性以及与真实分布的一致性，同样是理解 GAN 所不可回避的问题。
