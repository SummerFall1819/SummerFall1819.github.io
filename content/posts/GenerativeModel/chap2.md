---
title: "自回归模型"
date: 2026-04-09T14:40:24+08:00
weight: 3
categories: [生成式模型, 机器学习]
tags: [自回归模型, NADE, PixelCNN, Transformer]
series: [生成式模型]
draft: false
featured: false
params:
  mathjax: true
ShowToc: true
TocOpen: false
---

自回归模型（autoregressive model, AR）是最直接的一类显式生成模型。它并不试图一次性写出整个高维联合分布，而是先给随机变量规定一个顺序，再把联合分布拆成一串条件分布。这样做的结果是：模型既可以精确计算样本概率，又可以按顺序采样生成新样本。因此，自回归模型长期是密度估计、语言模型与图像生成中的核心方法。

{{< callout type="info" title="核心思想" >}}
任意联合分布都可以由链式法则写成条件分布的乘积。自回归模型真正要解决的问题，不是这个分解是否存在，而是如何用参数共享的模型高效地近似每一个条件分布。
{{< /callout >}}




## 自回归模型的基本思想

设观测样本记为 $\boldsymbol{x} = (x_1,\dots,x_n)^T$。一旦给定变量顺序，就有链式法则
$$
p(\boldsymbol{x}) = \prod_{i = 1}^{n} p(x_i \mid x_{< i}),
$$
其中 $x_{< i} = (x_1,\dots,x_{i-1})$ 表示第 $i$ 个变量之前的全部历史。链式法则始终成立，但需要的参数数量是指数级别的。基于这个样本顺序，有些条件概率可能会比较复杂，我们假定这些这些条件概率可以被共享参数的函数族进行参数化的表达（如逻辑斯蒂回归等），这样的模型就被称为自回归模型。



从训练角度看，自回归分解还有一个非常重要的性质：
$$
-\log p_{\theta}(\boldsymbol{x}) = \sum_{i = 1}^{n} -\log p_{\theta}(x_i \mid x_{< i}).
$$
也就是说，最大化联合分布的对数似然，等价于把每一个位置上的条件预测都做好。于是自回归建模可以理解为一种顺序预测问题：给定真实历史 $x_{< i}$，预测当前位置 $x_i$ 的条件分布。

{{< callout type="warning" title="注意" >}}
模型必须依赖一个顺序，不同顺序会影响学习难度与表示效果；同时，生成过程几乎总是顺序的，因此**采样速度**常常成为自回归模型的主要瓶颈。
{{< /callout >}}

## 经典自回归模型

### FVSBN：最基本的伯努利自回归模型

FVSBN(Fully Visible Sigmoid Belief Network) 是最简单的一种自回归模型，它假设每个像素 $x_i \in \{0,1\}$ 的条件分布都是伯努利分布，并用 **logistic regression** 来参数化该条件分布：

在二值图像建模中，最简单的自回归模型是 FVSBN（Fully Visible Sigmoid Belief Network）。它假设每个像素 $x_i \in \{0,1\}$ 的条件分布都是伯努利分布，并用 logistic regression 来参数化该条件分布：
$$
\hat{x}_i = p_{\theta}(x_i = 1 \mid x_{< i}; \boldsymbol{\alpha}^{(i)}) = \sigma\left(\alpha_0^{(i)} + \sum_{j = 1}^{i - 1}\alpha_j^{(i)}x_j\right).
$$
整个联合分布写成
$$
p_{\theta}(\boldsymbol{x}) = \prod_{i = 1}^{n} p_{\theta}(x_i \mid x_{< i}).
$$

这个模型的参数数量为 $\sum_{i = 1}^n i = \dfrac{n^2}{2}$. 全部来自于 $\boldsymbol {\alpha}^{(i)}$.

{{< figure src="/images/generativemodel/chap2/FVSBN.png" caption="FVBSN 示意" align="center" >}}

这个模型有两个优点。第一，它完全符合自回归分解，因此概率计算和采样都很直接。第二，每个条件分布的训练都只是一个二分类问题，因此优化并不复杂。

但它的表达能力也非常有限。因为每个位置只做一次线性回归，模型只能捕捉相对简单的条件关系。当图像结构变得复杂时，FVSBN 往往不足以表达强非线性依赖。换言之，它告诉我们自回归思路是可行的，但也暴露了“逐位置线性预测”过于简单的问题。

### NADE：用隐藏层提升表达能力

NADE (Neural Autoregressive Density Estimation) 把 logistic regression 升级为单隐藏层神经网络，以解决单层逻辑回归表达能力不够的问题。

对于 $x_i$, 先根据所有的前缀 $\boldsymbol x_{<i}$ 计算一个隐藏表示：
$$
\boldsymbol h_i = \sigma(\boldsymbol A^{(i)} \boldsymbol x_{<i} +  \boldsymbol c^{(i)})
$$

再针对隐藏表示得到新的 $x_i$:
$$
\hat x_i =p (x_i \mid \boldsymbol x_{<i}; \boldsymbol A^{(i)}, \boldsymbol c^{(i)}, \boldsymbol \alpha^{(i)}, b_i) = \sigma\left(\left(\boldsymbol{\alpha}^{(i)}\right)^T \boldsymbol h_i + b_i\right)
$$

{{< figure src="/images/generativemodel/chap2/NADE.png" caption="NADE 示意" align="center" >}}

在参数不共享的情况下，如果 $\boldsymbol h_i \in \mathbb R^d$, 则所有的参数总量为：
$$
\sum_{i = 1}^{n} \left[d(i - 1) + d + d + 1\right] = \sum_{i = 1}^{n} \left[d(i + 1) + 1\right] = \frac{dn(n + 3)}{2} + n.
$$
其中，$d(i - 1)$ 来自 $\boldsymbol A^{(i)} \in \mathbb R^{d \times (i - 1)}$，两个 $d$ 分别来自隐藏层偏置 $\boldsymbol c^{(i)} \in \mathbb R^d$ 和输出权重 $\boldsymbol \alpha^{(i)} \in \mathbb R^d$，最后的 $1$ 来自输出偏置 $b_i$。

可以通过共享参数的方式缩减参数数量。令 $\boldsymbol A^{(i)} := \boldsymbol W_{:, <i}$，并令隐藏层偏置 $\boldsymbol c^{(i)} := \boldsymbol c$ 共享。此时 $\boldsymbol W \in \mathbb R^{d \times n}$，$\boldsymbol c \in \mathbb R^d$，所有输出权重可以写成 $\boldsymbol V \in \mathbb R^{n \times d}$，输出偏置写成 $\boldsymbol b \in \mathbb R^n$，参数总量为：
$$
dn + d + nd + n = 2nd + d + n.
$$
若只计算实际会被前缀使用到的输入权重列，即 $\boldsymbol W_{:,1},\dots,\boldsymbol W_{:,n - 1}$，则输入侧参数是 $d(n - 1) + d = dn$，总量也可写成 $2nd + n$。实际实现中通常保留 $\boldsymbol W \in \mathbb R^{d \times n}$ 的矩阵形状，因此常用计数是 $2nd + d + n$。参数规模从“对每个位置都单独建一个网络”显著降了下来。


NADE 不再直接用原始前缀做一次线性预测，而是先构造一个隐藏表示 $\boldsymbol{h}_i$ 来概括历史信息。这样，每个条件分布都可以是非线性的，表达能力明显增强。


更进一步，NADE 还能递推地更新隐藏层预激活量。令
$$
\boldsymbol{a}_1 = \boldsymbol{c}, \qquad \boldsymbol{a}_{i + 1} = \boldsymbol{a}_i + \boldsymbol{W}_{:,i}x_i, \qquad \boldsymbol{h}_i = \sigma(\boldsymbol{a}_i),
$$
则从 $i$ 到 $i+1$ 时只需把第 $i$ 个输入对应的那一列权重加进去，而不必重新计算整次矩阵乘法。于是，NADE 既提升了表达能力，也改善了参数与计算效率。

#### 从 NADE 到一般离散变量与连续变量

二值变量只是最简单的情况。若 $x_i$ 不是二元的，而是 $K$ 类离散变量，那么可以把伯努利条件分布改成 categorical 分布：
$$
p_{\theta}(x_i \mid x_{< i}) = \operatorname{Cat}(p_i^1,\dots,p_i^K),
$$
其中
$$
(p_i^1,\dots,p_i^K)^T = \operatorname{softmax}(\boldsymbol{V}_i \boldsymbol{h}_i + \boldsymbol{b}_i).
$$
这里的 softmax 把一个实向量变成合法概率向量，因此自然适用于多分类情形。像像素灰度值、离散词表中的词元，都是这种建模方式的直接对象。

若观测变量是连续的，则需要进一步为每个条件分布选择合适的连续概率族。例如，可以令
$$
p_{\theta}(x_i \mid x_{< i}) = \mathcal N\bigl(\mu_{\theta}(x_{< i}), \Sigma_{\theta}(x_{< i})\bigr),
$$
或者使用 mixture of logistics、Gaussian mixture 等更灵活的条件分布。RNADE（Real-valued NADE）就是把 NADE 的思想推广到连续变量的代表模型。

### MADE：更深的前馈自回归网络

NADE 使用的是单隐藏层结构。若希望进一步提升表达能力，一个自然问题是：能否使用更深的 MLP，同时仍然保持自回归性质。MADE（Masked Autoencoder for Distribution Estimation）给出的做法是，对前馈网络的权重施加掩码，使得第 $i$ 个输出只依赖于 $x_{< i}$，而不会“偷看”当前位置及其未来位置。

因此，MADE 的重要性不在于它换了一个新的概率分解，而在于它说明：深层前馈网络同样可以被改造成合法的自回归模型。这样一来，自回归建模便不再局限于浅层结构，而可以直接继承现代深度网络的表达能力。

## 序列、图像与语言中的自回归结构

### RNN：用递归状态压缩长历史

对于长度可变的序列，直接把全部历史 $x_{< t}$ 展开后输入模型并不方便，因为历史会随着时间不断变长。RNN 的思路是维护一个递归更新的隐藏状态 $\boldsymbol{h}_t$ 来概括过去信息：
$$
\boldsymbol{h}_{t + 1} = f_{\theta}(\boldsymbol{h}_t, x_t), \qquad p_{\theta}(x_{t + 1} \mid x_{1:t}) = g_{\theta}(\boldsymbol{h}_{t + 1}).
$$
其中 $\boldsymbol{h}_t$ 可以理解为“到时刻 $t$ 为止历史信息的摘要”。这样，模型参数规模不再随着序列长度增长，而是固定不变。

RNN 自回归模型的优点很清楚。它能够自然处理任意长度序列，而且只要隐藏状态容量足够，理论上可以表示非常复杂的历史依赖。但它的缺点同样明显：对数似然的计算和样本生成都需要严格按时间顺序展开，因此训练较慢；同时，递归结构还会带来梯度消失与梯度爆炸问题。

### PixelRNN 与 PixelCNN：把图像也看成序列

图像虽然是二维对象，但同样可以按照 raster scan 顺序被重写成一维序列。于是，图像建模仍然可以使用自回归分解。对彩色图像而言，一个像素还包含三个通道，因此常见写法是
$$
p_{\theta}(\boldsymbol{x}) = \prod_{t = 1}^{n} p_{\theta}(x_t^r \mid x_{< t}) p_{\theta}(x_t^g \mid x_{< t}, x_t^r) p_{\theta}(x_t^b \mid x_{< t}, x_t^r, x_t^g).
$$
这意味着，模型不仅要遵守像素的顺序，还要在同一像素内部再遵守颜色通道的顺序。

PixelRNN 使用二维方向上的递归结构来建模图像中的长程依赖，本质上仍然是在做“根据已知上下文预测下一个像素”的工作。它的优点是表达能力强，但缺点也继承了 RNN 的顺序性。

PixelCNN 则改用卷积架构。为了不破坏自回归性质，卷积核不能看到当前位置及其未来位置，因此必须采用 masked convolution。掩码卷积使模型在训练时能够并行处理整张图像，但如果结构设计不当，会出现 blind spot，也就是某些理论上应该被利用的历史区域没有真正进入感受野。后续很多改进工作，正是在解决这种由掩码结构带来的信息盲区。

### 语言模型、Transformer 与 GPT

语言模型是自回归思想最自然的应用场景之一。对词元序列 $w_{1:T}$，其联合分布总可以写成
$$
p_{\theta}(w_{1:T}) = \prod_{t = 1}^{T} p_{\theta}(w_t \mid w_{< t}).
$$
于是，语言建模的核心任务就变成了 next-token prediction。

早期神经语言模型通常只看一个固定窗口的上下文，用嵌入加 MLP 去预测下一个词。这样做已经比传统 $n$-gram 更有表达力，但固定窗口会限制模型的上下文范围。

Transformer 的出现改变了这一点。通过 causal mask，自注意力层可以在训练阶段并行处理整个序列，同时又保证位置 $t$ 的输出只能依赖 $w_{< t}$。因此，Transformer 既保留了自回归建模的概率意义，又显著提升了长程依赖建模能力和硬件并行效率。

GPT 则可以看成“大规模 Transformer 自回归语言模型”的典型代表。它的训练目标并没有改变，本质上仍然是最大化
$$
\sum_{t = 1}^{T} \log p_{\theta}(w_t \mid w_{< t}),
$$
但模型规模、数据规模与训练技巧的提升，使这种简单的自回归预训练在语言理解与生成上取得了极强的效果。类似地，ImageGPT、VideoPoet 等工作也是把图像或视频先离散化为 token 序列，再沿用同样的自回归建模原则。

## 自回归模型的学习

### 最大似然估计

设训练集为 $\mathcal D = \{\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(m)}\}$。自回归模型的标准学习原则是最大似然估计：
$$
\hat{\theta} = \arg \max_{\theta} \frac{1}{m}\sum_{j = 1}^{m} \log p_{\theta}(\boldsymbol{x}^{(j)}).
$$
利用自回归分解，这个目标立即变成
$$
\hat{\theta} = \arg \max_{\theta} \frac{1}{m}\sum_{j = 1}^{m}\sum_{i = 1}^{n} \log p_{\theta}(x_i^{(j)} \mid x_{< i}^{(j)}).
$$

这个公式非常关键，因为它说明训练一个自回归模型，本质上就是在训练大量“给定真实前缀，预测下一个变量”的局部任务。对于文本，这就是 teacher forcing；对于图像，这就是已知前面像素时预测当前位置的条件分布。联合概率学习因此被拆解成了很多条件监督信号。

### 一个最简单的例子：有偏硬币

若只有一个伯努利变量 $x \in \{0,1\}$，并假设样本 $x^{(1)},\dots,x^{(m)}$ 相互独立且服从参数为 $p$ 的分布，则似然函数为
$$
L(p) = \prod_{j = 1}^{m} p^{x^{(j)}}(1-p)^{1-x^{(j)}}.
$$
取对数后得到
$$
\ell(p) = \sum_{j = 1}^{m} \left[x^{(j)}\log p + (1-x^{(j)})\log(1-p)\right].
$$
令导数为零，可以得到
$$
\hat{p} = \frac{1}{m}\sum_{j = 1}^{m} x^{(j)}.
$$
也就是说，最大似然估计恰好就是样本频率。这个例子说明，在简单模型中，MLE 有时可以得到闭式解；但在神经网络参数化的自回归模型中，通常已经没有解析解，只能依靠数值优化。

### 从 KL 散度理解最大似然

从分布逼近的角度看，自回归模型学习也可以写成“让模型分布尽量接近真实分布”。一种自然的差异度量是前向 KL 散度：
$$
D\bigl(p_{\text{data}} \parallel p_{\theta}\bigr) = \mathbb E_{p_{\text{data}}}\left[\log \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\theta}(\boldsymbol{x})}\right].
$$

将其展开可得
$$
D\bigl(p_{\text{data}} \parallel p_{\theta}\bigr) = \mathbb E_{p_{\text{data}}}[\log p_{\text{data}}(\boldsymbol{x})] - \mathbb E_{p_{\text{data}}}[\log p_{\theta}(\boldsymbol{x})].
$$
其中第一项与参数 $\theta$ 无关，因此
$$
\arg \min_{\theta} D\bigl(p_{\text{data}} \parallel p_{\theta}\bigr) = \arg \max_{\theta} \mathbb E_{p_{\text{data}}}[\log p_{\theta}(\boldsymbol{x})].
$$

{{< callout type="tip" title="最大似然的含义" >}}
最小化前向 KL 等价于最大化数据的期望对数似然。因此，自回归模型的最大似然训练不仅是一个方便的工程目标，也是在直接逼近真实数据分布。
{{< /callout >}}

这个观点还有一个很重要的解释：前向 KL 可以看成用模型分布 $p_{\theta}$ 代替真实分布 $p_{\text{data}}$ 时产生的额外压缩损失。因此，自回归模型常常既是生成模型，也是很强的密度估计与压缩模型。

### 梯度下降、随机梯度下降与并行化

对神经网络形式的自回归模型，目标函数通常是非凸的，因此实践中依赖梯度下降或其变体。若记
$$
\ell(\theta) = \sum_{j = 1}^{m}\sum_{i = 1}^{n} \log p_{\theta}(x_i^{(j)} \mid x_{< i}^{(j)}),
$$
则基本更新形式为
$$
\theta \leftarrow \theta + \eta \nabla_{\theta}\ell(\theta).
$$
在数据量很大时，通常使用随机梯度下降或 mini-batch 方法，以小批量样本来近似整体梯度：
$$
\nabla_{\theta}\ell(\theta) \approx \frac{m}{|\mathcal B|}\sum_{\boldsymbol{x} \in \mathcal B}\sum_{i = 1}^{n} \nabla_{\theta}\log p_{\theta}(x_i \mid x_{< i}).
$$

不过，是否能够高效并行，强烈依赖于具体架构。若模型采用 RNN，则隐藏状态本身就是递归定义的，因此即使在训练阶段，对同一个样本的条件概率计算也需要按时间顺序展开。相比之下，NADE、MADE、PixelCNN 这类前馈掩码模型虽然仍然表示同一个自回归分解，但训练时往往可以在一次前向传播中同时得到多个位置的条件分布，因此更适合现代并行硬件。

{{< callout type="warning" title="自回归模型的主要瓶颈" >}}
许多自回归模型在训练时可以部分并行，但在采样时仍然必须依次生成 $x_1,x_2,\dots,x_n$。因此，生成速度通常明显慢于非自回归模型。
{{< /callout >}}

## 小结

自回归模型的核心非常统一：先用链式法则把联合分布拆成条件分布，再用神经网络去刻画这些条件分布。FVSBN 展示了最基本的逻辑回归式自回归建模；NADE、RNADE 与 MADE 展示了如何通过隐藏层、参数共享和掩码结构提升表达能力；RNN、PixelRNN、PixelCNN、Transformer 与 GPT 则说明，同一个概率原则可以落到完全不同的网络架构上。

从学习角度看，自回归模型通常采用最大似然训练，而最大似然又等价于最小化 $D(p_{\text{data}} \parallel p_{\theta})$。这使它们成为最标准、最稳定的一类显式密度模型。它们的主要优点是概率可计算、训练目标明确、生成质量通常较强；主要缺点则是依赖变量顺序、缺少显式隐变量表示，而且采样过程往往较慢。

因此，自回归模型既是理解现代生成模型的重要起点，也是理解大语言模型的一条最直接路径。无论对象是像素、字符、词元还是离散化后的视频片段，其背后的数学骨架始终是同一个式子：
$$
p_{\theta}(\boldsymbol{x}) = \prod_{i = 1}^{n} p_{\theta}(x_i \mid x_{< i}).
$$
