## 多项式回归

### 一元线性回归

假设现在有一组变量 $\boldsymbol x = \{x_i\}_{i = 1}^N, \boldsymbol y = \{y_i\}_{i = 1}^N$. 而我们希望研究这两个变量之间的相互关系。在得到成对的数据 $\{(x_i,y_i)\}_{i = 1}^N$ 后，我们可以通过拟合一条直线来描述它们之间的关系。假设这条直线为 $y = kx + b$.

判断一条直线拟合的是否合理，应当看 $(x_i, kx_i + b)$ 和 $(x_i,y_i)$ 之间的距离。这两个点越接近，说明这条直线在 $x_i$ 附近拟合得越好，因此，可以用所有预测点和实际点的距离的和来判断这条直线的拟合程度。距离是非负的，一般采用平方来表示。

按照这个思路，可以用下面的指标可以用来判断直线的拟合程度：
$$
\begin{equation}
Q = \sum_{i = 1}^N \left(y_i - kx_i - b\right)^2
\end{equation}
$$
其中第 $i$ 项是第 $i$ 个原始点和第 $i$ 个预测点的距离平方。

这个函数是以 $k,b$ 为自变量的函数，观察可以看到，它是 $k,b$ 的一个二元函数。


作一个简单的展开，则有
$$
\begin{aligned}
    Q &= \sum_{i = 1}^N \left(y_i - kx_i - b\right)^2\\
    &= \sum_{i = 1}^N [(y_i - b) - kx_i] ^2 \\
    &= \sum_{i = 1}^N [(y_i - b)^2 + k^2 x_i^2 - 2kx_i(y_i - b)] \\
    &= \sum_{i = 1}^N x_i^2 \times k^2 - 2\sum_{i = 1}^N x_i(y_i - b) \times k + \sum_{i = 1}^N (y_i - b)^2 \\
\end{aligned}
$$
在 $k = -\dfrac{\sum_{i = 1}^N x_i(y_i - b)}{\sum_{i = 1}^N x_i^2}$ 处，这个函数可以取到最小值。

同理：
$$
\begin{aligned}
    Q &= \sum_{i = 1}^N (y_i - kx_i - b)^2\\
    &= \sum_{i = 1}^N [(y_i - kx_i) - b]^2 \\
    &= \sum_{i = 1}^N [(y_i - kx_i)^2 + b^2 - 2(y_i - kx_i)b] \\
    &=  N b^2 - 2\sum_{i = 1}^N (y - kx_i) \times b + \sum_{i = 1}^N (y_i - kx_i)^2\\
\end{aligned}
$$
在 $b = \dfrac{\sum_{i = 1}^N (y_i - kx_i)}{N}$ 处，这个函数可以取到最小值。

分别带入，则可以获得 $k$ 和 $b$ 的值
$$
\begin{equation}
\begin{cases}
k = \dfrac{N \sum_{i = 1}^N x_iy_i - (\sum_{i = 1}^N x_i) (\sum_{i = 1}^N y_i)}{N\sum_{i = 1}^N x_i^2 - (\sum_{i = 1}^N x_i)^2} \\
b = \dfrac{\sum_{i = 1}^N y_1}{N} - k \dfrac{\sum_{i = 1}^N x_i}{N}
\end{cases}
\end{equation}
$$

这是在初等数学中，根据二次函数的性质即可推导的内容。

一个更通用的做法是，根据二次函数的凸性，令其一阶导数为 $0$ 求解。

$$
\begin{cases}
\dfrac{\partial Q}{\partial k} = \sum_{i = 1}^N 2x_i(y_i - kx_i - b) = 0 \\[5mm]
\dfrac{\partial Q}{\partial b} = -\sum_{i = 1}^N 2(y_i - kx_i - b) = 0 \\
\end{cases}
$$

上式同样也可以计算得到之前的答案。

一元线性回归是最简单的一种回归，主要用于拟合 $(x_i,y_i)$ 之间的直线关系。更多时候， $x_i$ 和 $y_i$ 之间的关系并不是简单的一次函数关系。这就引出了多项式回归。



### 多项式回归

#### 扩展：从一元到多项式

形式上，称具有以下形式的式子为一个多项式。

$$
p(x) = \sum_{i = 0}^N a_i x^i
$$

其中，$a_i$ 是系数，$x$ 是自变量，$N$ 是多项式的次数。

在这里，多项式函数有用的性质为 **凸性** 和 **线性性**。

类似的，给定一组变量 $\boldsymbol x = \{x_i\}_{i = 1}^N, \boldsymbol y = \{y_i\}_{i = 1}^N$ 和成对的数据 $\{(x_i,y_i)\}_{i = 1}^N$ 后，我们可以通过一个多项式来描述它们之间的关系。假设这个多项式为 $p(x) = \sum_{i = 0}^M a_i x^i$. 注意当 $M = 1$ 时，就是上述的一元线性回归的例子。

可以采取同样的方法来定义拟合的损失函数：

$$
Q = \sum_{i = 1}^N \left(y_i - p(x_i)\right)^2
$$

通过求一阶导并令其为 $0$ 的方式获得这个函数的最小值。
$$
\dfrac{\partial Q}{\partial a_k} = \sum_{i = 1}^N 2x_i^k(y_i - p(x_i)) = 0 \quad(k = 0,1,\dots, M)
$$

可以得到如下的 $N + 1$ 个等式作为方程组
$$
\begin{equation}
\begin{cases}
\sum_{j = 0}^M (\sum_{i = 1}^N 2x_i) a_j = \sum_{i = 1}^N 2y_i \\
\sum_{j = 0}^M (\sum_{i = 1}^N 2x_i^2) a_j = \sum_{i = 1}^N 2x_iy_i \\
\cdots\\
\sum_{j = 0}^M (\sum_{i = 1}^N 2x_i^{k + 1}) a_j = \sum_{i = 1}^N 2x_i^ky_i \\
\cdots\\
\sum_{j = 0}^M (\sum_{i = 1}^N 2x_i^{N + 2}) a_j = \sum_{i = 1}^N 2x_i^{N + 1}y_i \\
\end{cases}
\end{equation}
$$

对这个方程组的求解依赖于线性代数的相关结论。

#### 更紧凑的表示

线性代数多以矩阵和向量表示数据。

记
- $\boldsymbol x = [x_1, x_2, \dots, x_N]^\top \in \mathbb R^N $ 表示输入变量，
- $\boldsymbol y = [y_1, y_2, \dots, y_N]^\top \in \mathbb R^N $ 表示输出变量，
- $\boldsymbol w = [w_0, w_1, \dots, w_M]^\top \in \mathbb R^M $ 表示多项式函数 $p(x)$ 的系数，即 $p(x) = \sum_{i = 0}^M w_ix^i$.
- $p(\boldsymbol x) = [p(x_1),p(x_2),\dots, p(x_N)] ^\top \in \mathbb R^N$ 表示 $\boldsymbol x$ 经过 $p(x)$ 函数映射得到的向量表示。

则映射
$$
\boldsymbol w \mapsto p(\boldsymbol x)
$$

可以表示为矩阵和向量的乘法。

$$
\begin{aligned}
p(\boldsymbol x) &= \boldsymbol X \boldsymbol w\\
&= 
\begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^{M} \\
1 & x_2 & x_2^2 & \cdots & x_2^{M} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_N & x_N^2 & \cdots & x_N^{M} \\
\end{bmatrix}
\begin{bmatrix}
a_0 \\
a_1 \\
\vdots\\
a_M
\end{bmatrix}
\end{aligned}
$$

特殊地，矩阵 $\boldsymbol X \in \mathbb R^{N \times M}$ 称为 **范德蒙德矩阵** (Vandermonde matrix).

对应的，损失函数表示为

$$
L(\boldsymbol w; \boldsymbol X, \boldsymbol y) = \lVert \boldsymbol y - \boldsymbol {Xw} \rVert _p^p
$$

其中 $\lVert \cdot \rVert_p$ 表示向量的 $p$ 范数。
$p = 2$ 的情形下，前面的讨论说明了这是一个凸函数，因此只需要
$$
\begin{equation}
\dfrac{\partial L(\boldsymbol w; \boldsymbol X, \boldsymbol y)}{\partial \boldsymbol w} = 0 \implies 2 \boldsymbol X ^ \top (\boldsymbol {Xw} - \boldsymbol y) = 0
\end{equation}
$$
即可获得最小值，其存在闭式解：

$$
\boldsymbol w^* = (\boldsymbol X^\top \boldsymbol X)^{-1} \boldsymbol X^\top \boldsymbol y
$$

这便是对原数据的最佳 $N + 1$ 阶多项式拟合，该方法也被称为 **最小二乘法**。

#### 幕后：物理意义与数学原理

暂略。


