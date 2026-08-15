---
title: "变分自编码器"
date: 2026-04-09T15:26:00+08:00
weight: 4
categories: [生成式模型, 机器学习]
tags: [变分自编码器, VAE, 变分推断, ELBO, VQ-VAE]
series: [生成式模型]
draft: true
featured: false
params:
  mathjax: true
ShowToc: true
TocOpen: false
---

变分自编码器（Variational Auto-Encoder, VAE）是深度生成模型中的一条非常重要的路线。它一方面保留了概率模型的视角，试图显式描述数据如何由潜变量生成；另一方面又借助神经网络，使模型能够处理高维、复杂的数据分布。因此，VAE 常常被看作连接“概率图模型”“变分推断”“深度学习”这三类思想的桥梁。

与自回归模型相比，VAE 并不把联合分布拆成一长串条件分布，而是假设观测样本 $\boldsymbol{x}$ 是由某个低维或较低维的潜变量 $\boldsymbol{z}$ 生成的。这样做的直觉是：高维数据往往并不是在整个观测空间中任意分布，而是受一些更本质、更紧凑的潜在因素控制。对于图像，这些因素可能是姿态、形状、光照；对于语音，它们可能是音素、说话风格与内容结构。

{{< callout type="info" title="核心思想" >}}
VAE 的目标不是直接求出难解的后验分布 $p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})$，而是引入一个可学习的近似后验 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$，再通过最大化证据下界（evidence lower bound, ELBO）来同时训练生成模型与推断模型。
{{< /callout >}}

## 从高斯混合模型到隐变量模型

要理解 VAE，先看一个更经典的潜变量模型：高斯混合模型（Gaussian Mixture Model, GMM）。若只用单个高斯分布去拟合数据，
$$
p(\boldsymbol{x}) = \mathcal N(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$
模型只能表达单峰（unimodal）结构。当真实数据呈现多个簇时，单高斯往往明显不足。

GMM 的做法是引入一个离散潜变量 $z \in \{1,\dots,K\}$，表示样本来自第几个高斯成分。于是模型写成
$$
p_{\theta}(\boldsymbol{x}) = \sum_{k = 1}^{K}\pi_k \mathcal N(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
$$
其中 $\pi_k \ge 0$ 且 $\sum_{k = 1}^{K}\pi_k = 1$。如果进一步把 $z$ 看作隐变量，则联合分布可以写为
$$
p_{\theta}(\boldsymbol{x}, z = k) = p(z = k)p_{\theta}(\boldsymbol{x} \mid z = k) = \pi_k \mathcal N(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).
$$
边缘分布则由对潜变量求和得到：
$$
p_{\theta}(\boldsymbol{x}) = \sum_{k = 1}^{K} p_{\theta}(\boldsymbol{x}, z = k).
$$

这一写法已经体现了 latent variable model 的一般模式：先写联合分布 $p_{\theta}(\boldsymbol{x}, \boldsymbol{z})$，再通过对潜变量边缘化得到观测分布 $p_{\theta}(\boldsymbol{x})$。引入潜变量的意义并不只是让公式变复杂，而是在不直接枚举所有数据模式的情况下，提升模型的表达能力。

在 GMM 中，给定样本 $\boldsymbol{x}^{(n)}$ 后，潜变量后验分布为
$$
p_{\theta}(z = k \mid \boldsymbol{x}^{(n)}) = \frac{\pi_k \mathcal N(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{i = 1}^{K}\pi_i \mathcal N(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)}.
$$
这正是 EM 算法中 E-step 所计算的 responsibility。换句话说，E-step 的本质就是在当前参数下，对隐变量进行推断。

### 为什么这一步很重要

GMM 给出的真正启发并不是“混合几个高斯”，而是下面这件事：一旦模型写成
$$
p_{\theta}(\boldsymbol{x}) = \int p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z}
$$
或其离散版本
$$
p_{\theta}(\boldsymbol{x}) = \sum_{\boldsymbol{z}} p_{\theta}(\boldsymbol{x}, \boldsymbol{z}),
$$
那么学习问题通常会自动分裂为两个部分：

- 怎样对潜变量做推断；
- 怎样在推断结果的基础上更新生成模型参数。

对于 GMM，这两步都还能得到相对漂亮的闭式解，于是有了 EM。可一旦联合分布由深度网络参数化，这种解析解通常就不存在了，这也正是 VAE 要解决的核心困难。

## 变分推断与 ELBO

### 深度隐变量模型为什么难

现在考虑更一般的连续潜变量模型。设 $\boldsymbol{x} \in \mathbb R^d$，$\boldsymbol{z} \in \mathbb R^m$，并假设
$$
p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) = p(\boldsymbol{z})p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z}).
$$
其中 $p(\boldsymbol{z})$ 是先验分布，通常取标准高斯
$$
p(\boldsymbol{z}) = \mathcal N(\boldsymbol{z} \mid \boldsymbol{0}, \boldsymbol{I}).
$$
若条件分布 $p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$ 由神经网络输出参数，那么观测边缘分布
$$
p_{\theta}(\boldsymbol{x}) = \int p(\boldsymbol{z})p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z}) d\boldsymbol{z}
$$
一般就很难精确计算。进一步地，真正理想的后验分布
$$
p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x}) = \frac{p(\boldsymbol{z})p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})}{p_{\theta}(\boldsymbol{x})}
$$
也随之变得难以求解，因为分母正是上面的积分。

因此，困难并不在于“模型没有定义”，而在于：

- 似然 $\log p_{\theta}(\boldsymbol{x})$ 难以直接优化；
- 后验 $p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})$ 难以直接推断。

### 用近似后验代替真实后验

变分推断（variational inference, VI）的做法是人为引入一个容易处理的分布族
$$
q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}),
$$
用它去近似真实后验 $p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})$。这里 $\phi$ 是变分分布的参数。在深度学习语境下，$\phi$ 通常由一个神经网络给出。

我们希望 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 与 $p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})$ 足够接近。一个自然的度量是 KL 散度：
$$
D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})\right).
$$
但真实后验本身就不可解，因此不能直接最小化这个量。VAE 的关键转折点在于：虽然这个 KL 不能直接算，但它可以导出一个可优化的下界。

### ELBO 的形式

对任意满足支持集条件的 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$，都有
$$
\log p_{\theta}(\boldsymbol{x}) = \mathcal L(\theta,\phi;\boldsymbol{x}) + D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})\right),
$$
其中
$$
\mathcal L(\theta,\phi;\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) - \log q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})\right].
$$
由于 KL 散度非负，立刻得到
$$
\log p_{\theta}(\boldsymbol{x}) \ge \mathcal L(\theta,\phi;\boldsymbol{x}).
$$
这就是 evidence lower bound，也就是 ELBO。

更进一步，把联合分布拆开后可得
$$
\mathcal L(\theta,\phi;\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})\right] - D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p(\boldsymbol{z})\right).
$$
这一形式尤其重要，因为它直接揭示了 VAE 的两个部分：

- 第一项是 reconstruction term，要求潜变量能够保留足够的信息，以便 decoder 重建样本；
- 第二项是 regularization term，要求编码后的后验不要偏离先验太远，从而让隐空间保持规则。

{{< callout type="tip" title="ELBO 的直观解释" >}}
最大化 ELBO，等价于一边提升样本在模型下的解释能力，一边让近似后验尽量接近真实后验。若 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) = p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})$，那么下界就被“顶满”，此时 ELBO 与对数似然相等。
{{< /callout >}}

{{< collapse summary="ELBO 恒等式的推导" >}}
从 KL 散度出发，
$$
D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})\right) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log \frac{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}{p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})}\right].
$$
代入 Bayes 公式
$$
p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x}) = \frac{p_{\theta}(\boldsymbol{x}, \boldsymbol{z})}{p_{\theta}(\boldsymbol{x})},
$$
得到
$$
D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})\right) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) - \log p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) + \log p_{\theta}(\boldsymbol{x})\right].
$$
由于 $\log p_{\theta}(\boldsymbol{x})$ 与 $\boldsymbol{z}$ 无关，可移出期望，于是
$$
\log p_{\theta}(\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) - \log q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})\right] + D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p_{\theta}(\boldsymbol{z} \mid \boldsymbol{x})\right).
$$
这正是上面的恒等式。
{{< /collapse >}}

## 变分自编码器的结构

### Decoder：从潜变量生成观测

VAE 的生成方向可以写成：
$$
\boldsymbol{z} \sim p(\boldsymbol{z}) = \mathcal N(\boldsymbol{0}, \boldsymbol{I}), \qquad \boldsymbol{x} \sim p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z}).
$$
其中 $p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$ 由解码器（decoder）网络参数化。给定 $\boldsymbol{z}$ 后，网络输出观测分布的参数，例如：

- 若数据是二值化图像，可以令 $p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$ 为逐维伯努利分布；
- 若数据是连续变量，可以令 $p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$ 为高斯分布，其均值与方差由网络输出。

因此，VAE 并不是“直接输出一张图”，而是先输出一个条件概率分布，再从该分布中解释或生成样本。这一点使它与普通自编码器在概念上有根本区别。

### Encoder：用摊还推断近似后验

如果对每一个样本 $\boldsymbol{x}$ 都单独优化一个 $q(\boldsymbol{z} \mid \boldsymbol{x})$，那么推断本身会变成一张巨大查找表。课件中把这一点概括得很直接：每个样本都对应一行，行里存的就是该样本的变分参数。这样的做法虽然在理论上可行，但在训练集大时极不经济，也无法自然泛化到新样本。

于是，VAE 采用摊还变分推断（amortized variational inference）：不再为每个样本单独维护一组变分参数，而是学习一个统一的函数近似器
$$
q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}).
$$
这个函数就是编码器（encoder）。最常见的选择是对角高斯：
$$
q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) = \mathcal N\left(\boldsymbol{z} \mid \boldsymbol{\mu}_{\phi}(\boldsymbol{x}), \operatorname{diag}\left(\boldsymbol{\sigma}_{\phi}^{2}(\boldsymbol{x})\right)\right).
$$
也就是说，encoder 输入样本 $\boldsymbol{x}$，输出均值向量 $\boldsymbol{\mu}_{\phi}(\boldsymbol{x})$ 与方差向量 $\boldsymbol{\sigma}_{\phi}^{2}(\boldsymbol{x})$，它们共同决定近似后验。

### 为什么它叫 Autoencoder

普通自编码器（autoencoder）通常包含一个确定性编码器与一个确定性解码器，其目标是把输入压缩到低维表示，再从该表示重建输入。VAE 与它相似的地方在于也存在“编码”和“解码”两个方向，但差别同样明显：

- 编码器输出的不是一个确定点，而是一个分布；
- 解码器做的不是简单回归，而是定义 $p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})$；
- 训练目标不是单纯重建误差，而是重建项加上 KL 正则项。

因此，VAE 可以理解为“带有随机编码和概率约束的自编码器”。课件里用一句很形象的话概括这一点：它既做 reconstruction with noise，也做 regularization to Gaussian。

## VAE 的训练

### 训练目标

设训练集为 $\mathcal D = \{\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(N)}\}$，VAE 的标准目标是最大化平均 ELBO：
$$
\max_{\theta,\phi} \frac{1}{N}\sum_{n = 1}^{N}\mathcal L(\theta,\phi;\boldsymbol{x}^{(n)}).
$$
把单样本目标展开后得到
$$
\mathcal L(\theta,\phi;\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\theta}(\boldsymbol{x} \mid \boldsymbol{z})\right] - D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p(\boldsymbol{z})\right).
$$
其中 $\theta$ 对应 decoder，$\phi$ 对应 encoder。于是训练过程就是同时学习生成模型与推断模型。

### 对 decoder 参数求梯度

若把 ELBO 写成
$$
\mathcal L(\theta,\phi;\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_{\theta}(\boldsymbol{x}, \boldsymbol{z}) - \log q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})\right],
$$
则对 $\theta$ 求导相对直接，因为采样分布 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 本身不依赖于 $\theta$。因此可以交换梯度与期望：
$$
\nabla_{\theta}\mathcal L(\theta,\phi;\boldsymbol{x}) = \mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[\nabla_{\theta}\log p_{\theta}(\boldsymbol{x}, \boldsymbol{z})\right].
$$
若积分无法解析计算，就用 Monte Carlo 近似即可。

### 对 encoder 参数求梯度为什么更难

真正麻烦的是 $\phi$。原因在于，期望是对 $q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})$ 取的，而这个分布本身也依赖于 $\phi$。因此不能像上面那样简单地把梯度推进期望内部。

一种一般性的做法是 score function estimator，也就是 REINFORCE 或 policy gradient 思想。它基于恒等式
$$
\nabla_{\phi}\mathbb E_{q_{\phi}(\boldsymbol{z})}\left[f(\boldsymbol{z})\right] = \mathbb E_{q_{\phi}(\boldsymbol{z})}\left[f(\boldsymbol{z})\nabla_{\phi}\log q_{\phi}(\boldsymbol{z})\right].
$$
这说明即使采样过程本身不可导，也可以得到无偏梯度估计。不过它往往方差较大，因此在连续 VAE 中通常不是首选。

{{< collapse summary="REINFORCE 恒等式的由来" >}}
从期望的积分形式出发，
$$
\nabla_{\phi}\mathbb E_{q_{\phi}(\boldsymbol{z})}\left[f(\boldsymbol{z})\right] = \nabla_{\phi}\int q_{\phi}(\boldsymbol{z})f(\boldsymbol{z}) d\boldsymbol{z}.
$$
把梯度移入积分后，
$$
= \int \nabla_{\phi}q_{\phi}(\boldsymbol{z})f(\boldsymbol{z}) d\boldsymbol{z}.
$$
再利用
$$
\nabla_{\phi}q_{\phi}(\boldsymbol{z}) = q_{\phi}(\boldsymbol{z})\nabla_{\phi}\log q_{\phi}(\boldsymbol{z}),
$$
即可得到
$$
\nabla_{\phi}\mathbb E_{q_{\phi}(\boldsymbol{z})}\left[f(\boldsymbol{z})\right] = \int q_{\phi}(\boldsymbol{z})f(\boldsymbol{z})\nabla_{\phi}\log q_{\phi}(\boldsymbol{z}) d\boldsymbol{z} = \mathbb E_{q_{\phi}(\boldsymbol{z})}\left[f(\boldsymbol{z})\nabla_{\phi}\log q_{\phi}(\boldsymbol{z})\right].
$$
{{< /collapse >}}

### 重参数化技巧

VAE 在连续潜变量场景中的关键工程技巧，是重参数化（reparameterization trick）。若
$$
q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) = \mathcal N\left(\boldsymbol{z} \mid \boldsymbol{\mu}_{\phi}(\boldsymbol{x}), \operatorname{diag}\left(\boldsymbol{\sigma}_{\phi}^{2}(\boldsymbol{x})\right)\right),
$$
则可以把采样过程改写为
$$
\boldsymbol{\epsilon} \sim \mathcal N(\boldsymbol{0}, \boldsymbol{I}), \qquad \boldsymbol{z} = \boldsymbol{\mu}_{\phi}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\phi}(\boldsymbol{x}) \odot \boldsymbol{\epsilon}.
$$
此时随机性被转移到了与参数无关的噪声 $\boldsymbol{\epsilon}$ 上，而 $\boldsymbol{z}$ 则变成了关于 $\phi$ 的可导函数。于是
$$
\mathbb E_{q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x})}\left[f(\boldsymbol{z})\right] = \mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\boldsymbol{0}, \boldsymbol{I})}\left[f\left(\boldsymbol{\mu}_{\phi}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\phi}(\boldsymbol{x}) \odot \boldsymbol{\epsilon}\right)\right],
$$
从而可以直接通过反向传播同时更新 encoder 与 decoder。

在实践中，单样本 Monte Carlo 近似已经常常够用：
$$
\mathcal L(\theta,\phi;\boldsymbol{x}) \approx \log p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}\right) - D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p(\boldsymbol{z})\right),
$$
其中
$$
\boldsymbol{z} = \boldsymbol{\mu}_{\phi}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\phi}(\boldsymbol{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal N(\boldsymbol{0}, \boldsymbol{I}).
$$

### 对角高斯下的 KL 闭式解

若先验取标准高斯，encoder 采用对角高斯后验，则 KL 项可以写成闭式：
$$
D_{\mathrm{KL}}\left(q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) \parallel p(\boldsymbol{z})\right) = \frac{1}{2}\sum_{j = 1}^{m}\left(\mu_j(\boldsymbol{x})^2 + \sigma_j(\boldsymbol{x})^2 - \log \sigma_j(\boldsymbol{x})^2 - 1\right).
$$
这使得训练时真正需要采样估计的，通常只剩下 reconstruction term，因此优化会稳定得多。

{{< callout type="warning" title="VAE 的训练代价与收益" >}}
VAE 的似然训练是近似的，因为我们最大化的是 ELBO 而不是精确的 $\log p_{\theta}(\boldsymbol{x})$。但换来的好处是：采样只需先采 $\boldsymbol{z}$ 再过一次 decoder，因此通常比自回归模型快得多。
{{< /callout >}}

## 如何理解 VAE 学到的表示

训练完成后，encoder 会把每个样本映射到一个高斯分布而不是单一点，因此隐空间不是普通的“压缩坐标”，而是一种带不确定性的表示。相近样本在隐空间中往往对应相近的均值和较为连续的区域，这使得插值、条件控制和语义操作成为可能。

由于先验常被设成标准高斯，理想情况下，很多样本编码后的后验应该分布在一个比较规则的隐空间里。这样我们就可以直接采样
$$
\boldsymbol{z} \sim \mathcal N(\boldsymbol{0}, \boldsymbol{I}),
$$
再通过 decoder 生成新样本。这也是课件中强调的一个优点：VAE 的 latent space 更接近高斯，因此可以直接从先验采样。

不过，这里有一个需要区分的点。训练目标约束的是每个样本对应的 posterior
$$
q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}),
$$
而不是整体聚合后的分布。真正与生成质量直接相关的，其实是 aggregated posterior：
$$
q(\boldsymbol{z}) = \int p_{\text{data}}(\boldsymbol{x})q_{\phi}(\boldsymbol{z} \mid \boldsymbol{x}) d\boldsymbol{x}.
$$
即使每个条件后验看起来都被 KL 项拉向高斯，聚合后的 $q(\boldsymbol{z})$ 仍然可能与标准高斯之间存在 hole 或 mismatch。此时若直接从先验采样，就可能落在 decoder 不熟悉的区域，导致生成质量下降。

## VAE 与自回归模型的比较

课件最后把 VAE 与 autoregressive model 做了一个很清楚的对照。两者都属于显式生成模型，但侧重点不同。

- VAE 的学习目标是 ELBO，也就是近似最大似然；自回归模型通常直接做精确最大似然。
- VAE 采样时只需经过一次 decoder，因此生成较快；自回归模型通常必须逐步生成，因此采样较慢。
- VAE 的似然值通常需要近似评估；自回归模型的似然则可以精确而快速地计算。
- VAE 天然带有 encoder，因此可以较方便地做表示学习；标准自回归模型一般没有一个对称的“编码器”。

因此，若任务更强调表示学习、快速采样和隐变量操控，VAE 往往是很自然的选择；若任务更强调精确似然与高保真逐步建模，自回归模型通常更有优势。

## VQ-VAE：离散潜变量的延伸

标准 VAE 使用连续高斯潜变量，但很多模态中的潜在结构天然更接近离散表示，例如语言中的 token、语音中的音素片段、图像中的离散视觉单元。VQ-VAE（Vector Quantized VAE）就是沿着这个方向发展的代表模型。

它的核心思想是：encoder 先输出一个连续向量，再把它量化到码本（codebook）中最近的离散嵌入向量上。这样，潜变量不再服从连续高斯，而是来自有限个离散原型。重建部分仍然保留 autoencoder 结构，但训练梯度需要借助 straight-through estimator 近似传递。

VQ-VAE 的另一个关键点，是它往往不再直接依赖标准高斯先验。课件中提到一种常见做法：先训练好 encoder 与 decoder，再使用 encoder 提取离散特征，并在这些特征上训练一个额外的先验模型，例如 PixelCNN。推断时，先由 PixelCNN 生成离散 latent，再由 decoder 还原为数据。这样做的原因正是为了绕开隐空间中的 hole，使生成过程遵循更真实的潜变量分布。

这一思路后来被许多强模型采用。像离散 token 图像生成、跨模态生成以及早期的 DALL-E 路线，都可以看成是在“VAE 学表示，再对 latent 建模”这一框架上的延伸。

## 小结

VAE 的主线可以概括为：先用潜变量模型刻画“数据如何生成”，再用变分推断解决“后验难算”的问题，最后借助神经网络把生成器与推断器统一为可训练的 encoder-decoder 结构。GMM 与 EM 提供了理解 latent variable model 的起点；ELBO 给出了可优化的学习目标；amortized variational inference 让推断从“每个样本单独优化”变成“学习一个统一的编码器”；reparameterization trick 则让连续潜变量上的梯度计算真正落地。

因此，VAE 的意义并不只是一个具体模型，而是一整套方法论：当显式似然难以直接优化、真实后验难以精确求解时，可以通过构造可学习的近似后验，将生成与推断放在同一个目标函数里联合训练。这也是后续大量深度隐变量模型的共同出发点。


后续如果你愿意，可以再补两张图：一张画 VAE 的 encoder-decoder 结构，一张画 ELBO 的两项分解，会更利于阅读。