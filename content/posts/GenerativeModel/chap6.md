---
title: "能量模型"
date: 2026-04-23T10:01:27+08:00
weight: 7
categories: [生成式模型, 机器学习]
tags: ["Energy-Based Model", EBM, RBM, "Score Matching", "Contrastive Divergence"]
series: [生成式模型]
draft: true
featured: false
params:
  mathjax: true
ShowToc: true
TocOpen: false
---

能量模型（energy-based model, EBM）是生成式建模里一条很有代表性的路线。它既不像自回归模型与归一化流那样，必须把概率分布写成一个容易归一化、容易计算似然的形式；也不像 GAN 那样，干脆放弃显式概率，只要求模型能够生成样本。EBM 保留了概率建模的语言，但把建模重点放在“如何给样本打分”上，而不是“如何把密度显式写成一个容易计算的闭式表达式”上。

这种思路的吸引力在于灵活。只要我们能设计一个足够强的函数来刻画“什么样的样本更合理”，理论上就可以把它放进模型里。然而代价也非常直接：一旦归一化常数难以计算，采样、似然评估与最大似然训练都会立刻变得困难。因此，理解 EBM 的关键，并不只是记住它的定义，而是看清它所处的那组张力：表达能力（flexibility）很强，但可计算性（tractability）很差。

{{< callout type="info" title="核心思想" >}}
能量模型先学习一个打分函数 $f_{\theta}(\boldsymbol{x})$，再把它归一化成概率分布：
$$
p_{\theta}(\boldsymbol{x})=\frac{\exp\left(f_{\theta}(\boldsymbol{x})\right)}{Z(\theta)}=\frac{\exp\left(-E_{\theta}(\boldsymbol{x})\right)}{Z(\theta)}.
$$
这里 $E_{\theta}(\boldsymbol{x})=-f_{\theta}(\boldsymbol{x})$ 称为能量（energy），$Z(\theta)$ 称为配分函数（partition function）。低能量对应高概率，高能量对应低概率。
{{< /callout >}}

## 从归一化困难到能量模型

### 概率参数化的基本矛盾

要把一个函数当作概率分布，它至少要满足两个条件：

- 非负性（non-negativity）：$p(\boldsymbol{x}) \ge 0$；
- 归一化（normalization）：$\int p(\boldsymbol{x})d\boldsymbol{x}=1$，离散情形则改成求和。

这两个条件里，真正困难的通常不是非负性，而是归一化。若我们先任意写下一个非负函数 $g_{\theta}(\boldsymbol{x})$，那么总可以形式上定义
$$
p_{\theta}(\boldsymbol{x})=\frac{g_{\theta}(\boldsymbol{x})}{\int g_{\theta}(\boldsymbol{x})d\boldsymbol{x}}.
$$
问题在于，分母中的高维积分通常并没有解析解。于是，“构造一个非负函数”这件事本身并不难，真正难的是知道这个函数的总质量有多大。

在经典概率建模里，一个常见做法是故意把 $g_{\theta}$ 选成那些积分可以解析计算的函数。例如高斯分布、指数分布以及更一般的指数族（exponential family），都属于这一路线。自回归模型与带潜变量的很多模型，也可以理解为在可归一化结构上做组合，从而把复杂分布拆成若干个可计算的局部对象。

EBM 的出发点恰恰是：如果我们不再强迫 $g_{\theta}$ 的积分具有闭式表达式，而是把归一化常数单独拿出来处理，那么模型的表达空间就会变得大得多。

### EBM 的定义与能量解释

最常见的 EBM 写法是
$$
p_{\theta}(\boldsymbol{x})=\frac{\exp\left(f_{\theta}(\boldsymbol{x})\right)}{Z(\theta)}, \qquad Z(\theta)=\int \exp\left(f_{\theta}(\boldsymbol{x})\right)d\boldsymbol{x}.
$$
其中 $f_{\theta}(\boldsymbol{x})$ 可以看作未归一化对数密度（unnormalized log-density），而
$$
E_{\theta}(\boldsymbol{x})=-f_{\theta}(\boldsymbol{x})
$$
则是与之等价的能量函数。于是也常把模型写成
$$
p_{\theta}(\boldsymbol{x})=\frac{\exp\left(-E_{\theta}(\boldsymbol{x})\right)}{Z(\theta)}.
$$

为什么这里几乎总是使用指数形式，而不是平方、绝对值之类别的非负化方式？主要有三点原因。

第一，概率在数值上经常跨越很多个数量级，而对数概率是更自然的工作尺度。若用 $f_{\theta}(\boldsymbol{x})$ 直接表示“未归一化对数概率”，那么指数映射就是最自然的还原方式。

第二，很多熟悉的概率分布本来就能写成指数形式，因此 EBM 与指数族在形式上是兼容的。

第三，统计物理中 Boltzmann 分布的形式正是 $\exp(-E)$，所以“能量”这一术语也就自然延续了下来。

更重要的是，EBM 的定义只要求我们能计算 $f_{\theta}(\boldsymbol{x})$ 或 $E_{\theta}(\boldsymbol{x})$，而不要求 $Z(\theta)$ 容易求出。这正是它的灵活性来源。

### 灵活性与可计算性的张力

若把前面的式子和前几章的模型对比，会看到 EBM 的优缺点都很鲜明。

- 优点是极强的表达能力。原则上，几乎任何可微神经网络都可以拿来当 $f_{\theta}$。
- 缺点是配分函数 $Z(\theta)$ 往往没有闭式形式，而且在高维下数值计算会遭遇维数灾难（curse of dimensionality）。
- 因而直接评估似然、直接做最大似然训练、以及直接从模型采样，都不会像自回归模型或 normalizing flow 那样直接。

{{< callout type="warning" title="EBM 的主要瓶颈" >}}
EBM 的困难不在于“写不出概率分布”，而在于虽然概率分布形式上存在，但其中的配分函数 $Z(\theta)$ 往往不可解。只要这一点不解决，似然、采样与训练就都要额外绕路。
{{< /callout >}}

## 为什么不知道配分函数也能工作

### 相对比较通常不需要 $Z(\theta)$

虽然单独计算 $p_{\theta}(\boldsymbol{x})$ 很困难，但比较两个样本的相对大小却是容易的。对任意两个点 $\boldsymbol{x}$ 与 $\boldsymbol{x}'$，有
$$
\frac{p_{\theta}(\boldsymbol{x})}{p_{\theta}(\boldsymbol{x}')}=\exp\left(f_{\theta}(\boldsymbol{x})-f_{\theta}(\boldsymbol{x}')\right).
$$
因为配分函数在比值里完全抵消了，所以我们可以直接判断“哪个样本更合理”。这也是 EBM 在很多判别或结构化任务中很有用的原因。

例如，在条件预测里常把模型写成
$$
p_{\theta}(\boldsymbol{y}\mid \boldsymbol{x})=\frac{\exp\left(f_{\theta}(\boldsymbol{x},\boldsymbol{y})\right)}{\sum_{\boldsymbol{y}'}\exp\left(f_{\theta}(\boldsymbol{x},\boldsymbol{y}')\right)}.
$$
若我们的目标只是求最大后验预测，那么
$$
\arg\max_{\boldsymbol{y}}p_{\theta}(\boldsymbol{y}\mid \boldsymbol{x})=\arg\max_{\boldsymbol{y}}f_{\theta}(\boldsymbol{x},\boldsymbol{y}),
$$
归一化常数并不影响最优解。于是，目标识别（object recognition）、序列标注（sequence labeling）、图像恢复（image restoration）等问题，都可以从 EBM 的角度来理解。

### Ising 模型与 Markov Random Field

课件中用图像恢复说明了这一点。设观测到的是带噪图像 $\boldsymbol{x}$，而真实图像记为 $\boldsymbol{y}$。我们希望在给定 $\boldsymbol{x}$ 的情况下恢复 $\boldsymbol{y}$。一种典型建模方式是使用马尔可夫随机场（Markov random field, MRF）：
$$
p(\boldsymbol{y},\boldsymbol{x})=\frac{1}{Z}\exp\left(\sum_{i}\psi_i(x_i,y_i)+\sum_{(i,j)\in E}\psi_{ij}(y_i,y_j)\right).
$$

这里的两个势函数（potential）有不同作用。

- $\psi_i(x_i,y_i)$ 是单点项（unary term），刻画第 $i$ 个观测像素与其真实像素之间的一致性；
- $\psi_{ij}(y_i,y_j)$ 是邻接项（pairwise term），刻画相邻像素倾向于取相近值的先验平滑性。

因为观测图像 $\boldsymbol{x}$ 已经固定，所以恢复时只需最大化后验分布
$$
\arg\max_{\boldsymbol{y}}p(\boldsymbol{y}\mid \boldsymbol{x})=\arg\max_{\boldsymbol{y}}p(\boldsymbol{y},\boldsymbol{x}),
$$
等价地就是寻找最低能量的配置。这正体现了 EBM 的一个核心优势：即使全局归一化常数很难算，只要我们关心的是相对优劣或最优配置，模型仍然能直接使用。

课件中的下图把这个恢复问题画得很直观：观测像素与真实像素之间通过单点项相连，而真实像素之间再通过邻接项形成一个 MRF。

{{< figure src="/images/generativemodel/chap6/ising-mrf.png" caption="课件中的 Ising / MRF 图像恢复示意。" align="center" >}}

### Product of Experts

另一个很有启发性的例子是专家乘积（product of experts, PoE）。假设我们已经有若干个打分器或概率模型 $q_1(\boldsymbol{x}),\dots,q_K(\boldsymbol{x})$，那么可以把它们组合成
$$
p(\boldsymbol{x})=\frac{1}{Z}\prod_{k=1}^{K}q_k(\boldsymbol{x}).
$$
若每个 $q_k$ 都代表一个“专家”的意见，那么总模型只会对同时满足所有专家约束的样本赋予高概率。写到对数域上就是
$$
f(\boldsymbol{x})=\sum_{k=1}^{K}\log q_k(\boldsymbol{x}),
$$
因此 PoE 本质上就是把多个能量或分数相加。

PoE 与混合模型（mixture model）的差异非常重要。

- 混合模型更像“OR”：只要某一个分量认为样本合理，它就可能获得较高概率；
- PoE 更像“AND”：只要有一个专家强烈反对，样本的总体概率就会被明显压低。

这使得 PoE 特别适合做约束组合。课件也提到，diffusion model 中的 guidance，以及某些偏好对齐（preference alignment）过程，都可以从 PoE 的角度得到很直观的理解：多个评分器共同塑造了最终分布。

### RBM 与深玻尔兹曼机

受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）是最经典的含潜变量 EBM。它把可见变量 $\boldsymbol{x}\in\{0,1\}^n$ 与隐变量 $\boldsymbol{z}\in\{0,1\}^m$ 建模为
$$
p_{\theta}(\boldsymbol{x},\boldsymbol{z})=\frac{1}{Z(\theta)}\exp\left(\boldsymbol{x}^{T}\boldsymbol{W}\boldsymbol{z}+\boldsymbol{b}^{T}\boldsymbol{x}+\boldsymbol{c}^{T}\boldsymbol{z}\right),
$$
其中 $\boldsymbol{W}\in\mathbb R^{n\times m}$，$\boldsymbol{b}\in\mathbb R^n$，$\boldsymbol{c}\in\mathbb R^m$。

RBM 之所以叫 restricted，是因为它只允许可见层与隐层之间连接，而不允许可见层内部和隐层内部连接。正因如此，它的条件分布会分解为独立项：
$$
p_{\theta}(z_j=1\mid \boldsymbol{x})=\sigma\left(\boldsymbol{W}_{:,j}^{T}\boldsymbol{x}+c_j\right), \qquad p_{\theta}(x_i=1\mid \boldsymbol{z})=\sigma\left(\boldsymbol{W}_{i,:}\boldsymbol{z}+b_i\right).
$$
这意味着在给定一层时，另一层可以并行采样，因此 Gibbs sampling 变得可行。

如果只看结构，RBM 可以理解为“可见层和隐层之间的二部图”，这也是它能够高效做 Gibbs sampling 的原因。

{{< figure src="/images/generativemodel/chap6/rbm-structure.png" caption="课件中的 RBM 结构图：层间连接存在，层内连接被去掉。" align="center" >}}

从建模角度看，RBM 说明了一件重要的事：纯粹的 $f_{\theta}(\boldsymbol{x})$ 形式虽然灵活，但没有显式的特征层次；一旦加入隐变量，EBM 也可以学习内部表示。进一步把多个 RBM 叠起来，就得到深玻尔兹曼机（Deep Boltzmann Machine, DBM）以及早期深度信念网络（Deep Belief Network, DBN）。它们在深度学习早期曾经非常重要，因为当时的深网络往往需要用这种逐层预训练（pretraining）方式才能稳定训练。

更一般地，Yann LeCun 提出的 world model 也常被理解为一种非概率的 EBM：它不一定显式定义归一化概率，而是在潜空间中定义“正确状态低代价、错误状态高代价”的能量结构，再通过自监督学习让模型学会这个能量地形。

## EBM 的最大似然学习

### 对数似然与正相、负相

对一个训练样本 $\boldsymbol{x}$，EBM 的对数似然写成
$$
\log p_{\theta}(\boldsymbol{x})=f_{\theta}(\boldsymbol{x})-\log Z(\theta).
$$
因此最大似然训练的梯度是
$$
\nabla_{\theta}\log p_{\theta}(\boldsymbol{x})=\nabla_{\theta}f_{\theta}(\boldsymbol{x})-\mathbb E_{\tilde{\boldsymbol{x}}\sim p_{\theta}}\left[\nabla_{\theta}f_{\theta}(\tilde{\boldsymbol{x}})\right].
$$

这个公式极其重要，因为它把训练拆成了两部分：

- 第一项是正相（positive phase），提高训练样本的分数；
- 第二项是负相（negative phase），降低模型自己典型样本的分数。

于是，最大似然训练并不是简单地把数据点“拉高”就结束了。若只提高 $f_{\theta}(\boldsymbol{x}_{\text{train}})$，模型可能会把所有点一起抬高，导致归一化常数也同步增大，训练数据的相对概率并不会真的改善。负相的作用就是把那些“不该高分”的区域压下去，从而真正塑造出一个以数据为低能量谷底的能量地形。

{{< collapse summary="对数似然梯度的推导" >}}
记
$$
Z(\theta)=\int \exp\left(f_{\theta}(\boldsymbol{x})\right)d\boldsymbol{x}.
$$
则
$$
\log p_{\theta}(\boldsymbol{x})=f_{\theta}(\boldsymbol{x})-\log Z(\theta).
$$
对参数求导，
$$
\nabla_{\theta}\log p_{\theta}(\boldsymbol{x})=\nabla_{\theta}f_{\theta}(\boldsymbol{x})-\nabla_{\theta}\log Z(\theta).
$$
进一步有
$$
\nabla_{\theta}\log Z(\theta)=\frac{1}{Z(\theta)}\nabla_{\theta}Z(\theta)=\frac{1}{Z(\theta)}\int \nabla_{\theta}\exp\left(f_{\theta}(\tilde{\boldsymbol{x}})\right)d\tilde{\boldsymbol{x}}.
$$
因为
$$
\nabla_{\theta}\exp\left(f_{\theta}(\tilde{\boldsymbol{x}})\right)=\exp\left(f_{\theta}(\tilde{\boldsymbol{x}})\right)\nabla_{\theta}f_{\theta}(\tilde{\boldsymbol{x}}),
$$
所以
$$
\nabla_{\theta}\log Z(\theta)=\frac{1}{Z(\theta)}\int \exp\left(f_{\theta}(\tilde{\boldsymbol{x}})\right)\nabla_{\theta}f_{\theta}(\tilde{\boldsymbol{x}})d\tilde{\boldsymbol{x}}=\int p_{\theta}(\tilde{\boldsymbol{x}})\nabla_{\theta}f_{\theta}(\tilde{\boldsymbol{x}})d\tilde{\boldsymbol{x}}.
$$
于是得到
$$
\nabla_{\theta}\log p_{\theta}(\boldsymbol{x})=\nabla_{\theta}f_{\theta}(\boldsymbol{x})-\mathbb E_{\tilde{\boldsymbol{x}}\sim p_{\theta}}\left[\nabla_{\theta}f_{\theta}(\tilde{\boldsymbol{x}})\right].
$$
{{< /collapse >}}

### Contrastive Divergence

从上面的公式看，真正困难的是负相，因为它要求我们从当前模型分布 $p_{\theta}$ 中采样。若能精确完成这一步，就可以做真正的最大似然；但在高维下，这往往太贵。

对比散度（Contrastive Divergence, CD）给出的想法是：既然精确采样难，那就用短链 MCMC 近似负相。其最常见的 CD-$k$ 步骤是

1. 令 $\boldsymbol{x}^{(0)}=\boldsymbol{x}_{\text{data}}$；
2. 从 $\boldsymbol{x}^{(0)}$ 出发运行 $k$ 步 MCMC，得到 $\boldsymbol{x}^{(k)}$；
3. 用

   $$
   \nabla_{\theta}\log p_{\theta}(\boldsymbol{x}_{\text{data}})\approx \nabla_{\theta}f_{\theta}(\boldsymbol{x}_{\text{data}})-\nabla_{\theta}f_{\theta}\left(\boldsymbol{x}^{(k)}\right)
   $$

   近似真实梯度。

直观地说，CD 并不试图在每次训练都得到真正的模型平衡样本，而是只要求“把数据点附近的能量地形稍微往正确方向推一小步”。这会引入偏差，但代价远低于精确 MCMC，因此在 RBM 等模型里非常有效。

### 从 MCMC 到 Langevin dynamics

即便似然难算，EBM 仍然可以采样，因为接受率只需要概率比值，不需要绝对归一化常数。最简单的思路是 Metropolis-Hastings：给当前样本 $\boldsymbol{x}_t$ 添加随机扰动得到候选点 $\boldsymbol{x}'$，再按
$$
a\left(\boldsymbol{x}_t\to \boldsymbol{x}'\right)=\min\left\{1,\exp\left(f_{\theta}(\boldsymbol{x}')-f_{\theta}(\boldsymbol{x}_t)\right)\right\}
$$
决定是否接受。由于 $Z(\theta)$ 抵消了，这一步是可计算的。

但在高维连续空间里，单纯随机游走往往太慢，因此现代 EBM 更常用 Langevin dynamics。记
$$
s_{\theta}(\boldsymbol{x})=\nabla_{\boldsymbol{x}}\log p_{\theta}(\boldsymbol{x}),
$$
则离散 Langevin 更新可写成
$$
\boldsymbol{x}_{t+1}=\boldsymbol{x}_t+\frac{\epsilon}{2}s_{\theta}(\boldsymbol{x}_t)+\sqrt{\epsilon}\,\boldsymbol{\xi}_t, \qquad \boldsymbol{\xi}_t\sim\mathcal N(\boldsymbol{0},\boldsymbol{I})
$$
其中 $\epsilon$ 是步长。有些教材会把噪声项写成 $\sqrt{2\epsilon}$，这只是离散化约定不同，本质思想完全一样。

这个更新式包含两种力量：

- 梯度项把样本推向更高密度、也就是更低能量的区域；
- 高斯噪声让链有机会跳出局部模式，从而探索整个分布。

因此，Langevin dynamics 可以理解为“带噪声的密度上升”。课件里的图也展示了这一点：轨迹会沿着 score field 向多个高密度模式移动，而不是直接确定性地掉进某一个局部极值。

{{< figure src="/images/generativemodel/chap6/langevin-dynamics.png" caption="课件中的 Langevin dynamics 示意：样本轨迹在 score field 的引导下向高密度区域移动。" align="center" >}}

## Score 视角与无采样训练

### Score function 为什么不含 $Z(\theta)$

对 EBM 而言，一个非常关键的观察是：当我们对输入 $\boldsymbol{x}$ 求梯度时，配分函数是常数。于是
$$
s_{\theta}(\boldsymbol{x})=\nabla_{\boldsymbol{x}}\log p_{\theta}(\boldsymbol{x})=\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x})-\nabla_{\boldsymbol{x}}\log Z(\theta)=\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x}).
$$

这说明 score function 只依赖于未归一化分数 $f_{\theta}$，与配分函数无关。于是我们得到了一条新的训练思路：如果直接学习概率太难，那能否直接学习 score？

{{< callout type="tip" title="一个极重要的观察" >}}
EBM 的 score function
$$
s_{\theta}(\boldsymbol{x})=\nabla_{\boldsymbol{x}}\log p_{\theta}(\boldsymbol{x})
$$
不含 $Z(\theta)$。这使得很多原本依赖归一化常数的训练难题，能够被改写成只涉及 $\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x})$ 的形式。
{{< /callout >}}

### Score matching

若真实数据分布为 $p_{\text{data}}(\boldsymbol{x})$，那么一个自然想法是让模型的 score 尽量逼近真实分布的 score。Hyvarinen 提出的 score matching 就是最典型的做法。它最初定义为最小化 Fisher divergence：
$$
D_F\left(p_{\text{data}},p_{\theta}\right)=\frac{1}{2}\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\left\|\nabla_{\boldsymbol{x}}\log p_{\text{data}}(\boldsymbol{x})-\nabla_{\boldsymbol{x}}\log p_{\theta}(\boldsymbol{x})\right\|_2^2\right]
$$

把 EBM 的 score 代入后，有
$$
D_F\left(p_{\text{data}},p_{\theta}\right)=\frac{1}{2}\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\left\|\nabla_{\boldsymbol{x}}\log p_{\text{data}}(\boldsymbol{x})-\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x})\right\|_2^2\right]
$$
表面上看，这还是有问题，因为 $\nabla_{\boldsymbol{x}}\log p_{\text{data}}(\boldsymbol{x})$ 并不知道。但 score matching 的精妙之处就在于：通过分部积分（integration by parts），可以把这项完全消掉。

最终得到的目标是
$$
J_{\text{SM}}(\theta)=\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\frac{1}{2}\left\|\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x})\right\|_2^2+\operatorname{tr}\left(\nabla_{\boldsymbol{x}}^2 f_{\theta}(\boldsymbol{x})\right)\right]+\text{const.}
$$
其中 $\nabla_{\boldsymbol{x}}^2 f_{\theta}(\boldsymbol{x})$ 是 Hessian 矩阵，$\operatorname{tr}$ 表示迹。这个目标只需要数据样本和模型分数函数本身，不需要从 EBM 采样。

{{< collapse summary="score matching 如何用分部积分消去未知的数据分布导数" >}}
先看一维情形。记
$$
s_{\theta}(x)=\partial_x \log p_{\theta}(x)=\partial_x f_{\theta}(x).
$$
Fisher divergence 的一维目标为
$$
J(\theta)=\frac{1}{2}\int p_{\text{data}}(x)\left(s_{\theta}(x)-\partial_x\log p_{\text{data}}(x)\right)^2dx.
$$
展开后得到
$$
J(\theta)=\frac{1}{2}\int p_{\text{data}}(x)s_{\theta}(x)^2dx-\int p_{\text{data}}(x)s_{\theta}(x)\partial_x\log p_{\text{data}}(x)dx+\text{const.}
$$
注意
$$
p_{\text{data}}(x)\partial_x\log p_{\text{data}}(x)=\partial_x p_{\text{data}}(x),
$$
于是中间项变成
$$
\int s_{\theta}(x)\partial_x p_{\text{data}}(x)dx.
$$
对它做分部积分，并假设边界项 $p_{\text{data}}(x)s_{\theta}(x)\to 0$，则有
$$
\int s_{\theta}(x)\partial_x p_{\text{data}}(x)dx=-\int p_{\text{data}}(x)\partial_x s_{\theta}(x)dx.
$$
代回去得到
$$
J(\theta)=\int p_{\text{data}}(x)\left(\frac{1}{2}s_{\theta}(x)^2+\partial_x s_{\theta}(x)\right)dx+\text{const.}
$$
再利用 $s_{\theta}(x)=\partial_x f_{\theta}(x)$，便有
$$
J(\theta)=\int p_{\text{data}}(x)\left(\frac{1}{2}\left(\partial_x f_{\theta}(x)\right)^2+\partial_x^2 f_{\theta}(x)\right)dx+\text{const.}
$$

多维情形只是把一维导数推广成梯度与 Hessian，最终结果为
$$
J_{\text{SM}}(\theta)=\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\frac{1}{2}\left\|\nabla_{\boldsymbol{x}}f_{\theta}(\boldsymbol{x})\right\|_2^2+\operatorname{tr}\left(\nabla_{\boldsymbol{x}}^2 f_{\theta}(\boldsymbol{x})\right)\right]+\text{const.}
$$
{{< /collapse >}}

### Sliced score matching 与现代 score-based 模型

score matching 的优点很清楚：训练时不必在每一步都从 EBM 中采样，也不必计算配分函数。但它也有自己的代价，即目标函数里出现了 Hessian 的迹。在高维数据上，这一项计算很重，因此直接 score matching 并不总是实用。

为此，后续工作提出了 sliced score matching、denoising score matching 等变体。它们的共同思路是：继续利用“score 与 $Z(\theta)$ 无关”这一关键事实，但把原本昂贵的二阶项换成更容易估计的形式。现代 score-based generative model 与 diffusion model 的很多技术，本质上都延续了这条路线：不直接学习归一化密度，而是学习其梯度场，再通过 Langevin dynamics 或更一般的随机微分方程采样。

课件还给出了一个现代 EBM 的采样示意：从随机噪声出发，经由 Langevin sampling 逐步靠近数据流形，最终得到具有可辨结构的人脸样本。

{{< figure src="/images/generativemodel/chap6/modern-ebm-samples.png" caption="现代 EBM 的采样结果示意：从噪声初始化，经 Langevin sampling 得到清晰样本。" align="center" >}}

因此，从今天回头看，EBM 并不是一条已经过时的路线。相反，现代 diffusion 模型里的很多核心思想，恰恰与 EBM 的 score 观点高度一致。

### 对抗式训练

除了 score matching，课件最后还介绍了另一类无采样训练思路，即对抗式训练（adversarial training for EBM）。它的基本想法是引入一个容易采样的变分分布 $q_{\phi}(\boldsymbol{x})$，对对数似然做上界估计。若把 Shannon 熵记为
$$
H\left(q_{\phi}\right)=-\mathbb E_{\boldsymbol{x}\sim q_{\phi}}\left[\log q_{\phi}(\boldsymbol{x})\right],
$$
则有
$$
\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[\log p_{\theta}(\boldsymbol{x})\right]\le \mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[f_{\theta}(\boldsymbol{x})\right]-\mathbb E_{\boldsymbol{x}\sim q_{\phi}}\left[f_{\theta}(\boldsymbol{x})\right]-H\left(q_{\phi}\right),
$$
其中最后一项鼓励 $q_{\phi}$ 保持足够的分散性，而不是塌缩成少数几个点。

于是训练可以写成一个极大极小问题：
$$
\max_{\theta}\min_{\phi}\mathbb E_{\boldsymbol{x}\sim p_{\text{data}}}\left[f_{\theta}(\boldsymbol{x})\right]-\mathbb E_{\boldsymbol{x}\sim q_{\phi}}\left[f_{\theta}(\boldsymbol{x})\right]-H\left(q_{\phi}\right).
$$

这个式子很有直观意义。

- EBM 希望把真实数据的分数拉高；
- 同时把由 $q_{\phi}$ 生成的负样本分数压低；
- $q_{\phi}$ 则要不断制造“难负样本”，并保持一定熵，避免退化成几个点。

若把 $q_{\phi}$ 看成一个生成器，那么这与 GAN 的对抗结构就有明显相似性。不同之处在于，此时判别器不再输出一个二分类概率，而是直接输出 energy 或 score。

顺便一提，课件最后的总结还提到噪声对比估计（noise contrastive estimation, NCE）。它也是经典的无采样训练方法：把学习未归一化密度的问题转化为“数据样本与已知噪声样本的二分类”，并同时估计配分函数。这说明，一旦接受“未归一化概率”这个视角，训练方法其实并不只剩下最大似然这一条路。

## 本章小结

能量模型的核心并不复杂：用一个灵活的函数为样本打分，再通过配分函数把它归一化为概率分布。真正值得反复体会的是它的结构性 trade-off。

- 在表达上，EBM 很自由，因为 $f_{\theta}(\boldsymbol{x})$ 几乎可以由任意强大的神经网络参数化。
- 在计算上，EBM 很困难，因为配分函数通常不可解，导致采样与似然评估都不直接。
- 在应用上，EBM 又并非完全失去可用性，因为相对概率比较、MAP 推断以及很多结构化任务并不显式需要 $Z(\theta)$。
- 在学习上，最大似然对应“拉高数据、压低模型样本”的正负相结构，Contrastive Divergence 用短链 MCMC 近似这一过程。
- 在更现代的视角下，score function 消去了配分函数，于是 score matching、sliced score matching、对抗式训练乃至 diffusion 的很多思想，都可以放回 EBM 框架中理解。

从生成模型这条主线看，EBM 处在一个很特别的位置：它不像 autoregressive model 那样计算友好，也不像 GAN 那样完全放弃显式概率；它以“保留概率形式但牺牲归一化可解性”为代价，换来了极强的建模自由度。正因为这个位置特殊，后续很多看似不同的方法，都会在某种意义上重新回到 EBM 的思想上来。
