---
title: CS224W-Lecture 7-GNN (2)
date: 2021-11-11 23:52:38
tags:
mathjax: true
---

## TL; DR

本文是斯坦福大学 [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/index.html) 第七讲的内容。

<!-- more -->

该讲首先从一个宏观的角度介绍了 GNN 学习的整体流程，分别是单层 GNN 网络的 Message 与 Aggregation、多层 GNN 网络的连接 (Layer connectivity)、图增强 (Graph augmentation) 与下游任务的学习 (Learning objective)。

其次分析了单层 GNN 的 Message 与 Aggregation，并且介绍了三种最常见的 GNN 模型：GCN、GraphSAGE 与 GAT。

然后介绍多层 GNN 网络的连接方式，但是 GNN 的层数增多会导致 over-smoothing 的问题，所以还介绍了避免该问题的方法。

对于图增强和下游任务的学习这两部分将在第八讲中进行介绍。

## 1. A General Perspective on Graph Neural Networks

<img src="/blogs/images/image-20211108223219368.png" alt="图"  />

上图是一个由六个结点六条边构成的简单图。其中结点 A 是我们的目标结点，我们希望通过 GNN 的学习到 A 结点的 embedding 表示，并且将学习到的结果应用到下游的机器学习任务中，例如分类、回归等任务。

下图展示了图机器学习的流程。首先面向单层 GNN 网络，两个最重要的操作是 (1) Message 与 (2) Aggregation。多层 GNN 网络堆叠在一起，构成一个大的 GNN 模型 (Layer connectivity)。由于结点特征丢失或者图的结构问题，我们可能需要图增强 (Graph augmentation) 操作。最后，我们会将 GNN 学习到的结点的 embedding 特征交给下游机器学习任务进行学习 (Learning objective)。

<img src="/blogs/images/image-20211108222913077.png" alt="A General Perspective on Graph Neural Networks" style="zoom: 50%;" />



## 2. GNN Layer

<img src="/blogs/images/image-20211108223614427.png" alt="image-20211108223614427" style="zoom: 80%;" />

**单层 GNN 网络在计算过程中主要分为两个步骤**：

- **Message**：网络的输入结点产生一条 message 并传递给它的邻居结点。下式中 $\mathbf{h}_{u}^{(l-1)}$ 表示结点 $u$ 在第 $l-1$ 层的中间结果；$\operatorname{MSG}^{(l)}$ 表示第 $l$ 层自定义的消息函数，该函数可能是常见的线性变换；$\mathbf{m}_{u}^{(l)}$ 表示该结点产生的消息。

$$
\mathbf{m}_{u}^{(l)}=\operatorname{MSG}^{(l)}\left(\mathbf{h}_{u}^{(l-1)}\right)
$$

- **Aggregation**：网络的输出结点收集它的邻居结点产生的 message 并进行聚合计算。下式中 $N(v)$ 表示结点 $v$ 的所有邻居结点的集合；$\mathrm{AGG}^{(l)}$ 是自定义的聚合函数，例如 $\mathrm{max}(\cdot), \mathrm{min}(\cdot), \mathrm{sum}(\cdot)$ 等；$\mathbf{h}_{v}^{(l)}$ 表示结点 $v$ 聚集了它的邻居结点后得到中间结果。

$$
\mathbf{h}_{v}^{(l)}=\mathrm{AGG}^{(l)}\left(\left\{\mathbf{m}_{u}^{(l)}, u \in N(v)\right\}\right)
$$

在上面 $\mathbf{m}_{u}^{(l)}$ 和 $\mathbf{h}_{v}^{(l)}$ 的定义中，我们没有考虑结点 $v$ 第 $l-1$ 的 中间结果 $\mathbf{h}_{v}^{(l-1)}$ 对于第 $l$ 层的中间结果 $\mathbf{h}_{v}^{(l)}$ 的影响，即结点 $v$ 自身的信息会在聚合的过程中的丢失。为了解决这个问题，我们对上面的消息函数和聚合函数进行修正。**修正后的结果消息函数和聚合函数的定义如下**。
$$
\begin{cases}

\mathbf{m}_{u}^{(l)}  = &\operatorname{MSG}^{(l)}\left(\mathbf{h}_{u}^{(l-1)}\right), u \in\{N(v) \cup v\} \\

\mathbf{h}_{v}^{(l)}  = & \mathrm{AGG}^{(l)}\left(\left\{\mathbf{m}_{u}^{(l)}, u \in N(v)\right\}, \mathbf{m}_{v}^{(l)}\right)

\end{cases}
$$

基于以上对于单层 GNN 网络的理解，下面介绍三种最常见图神经网络模型：GCN，GraphSAGE 与 GAT。

### 2.1 CASE 1: GCN

$$
\mathbf{h}_{v}^{(l)} =\sigma\left(\mathbf{W}^{(l)} \sum_{u \in N(v)} \frac{\mathbf{h}_{u}^{(l-1)}}{|N(v)|}\right) = \sigma\left(\sum_{u \in N(v)} \mathbf{W}^{(l)} \frac{\mathbf{h}_{u}^{(l-1)}}{|N(v)|}\right)
$$
GCN (Graph Convolutional Networks)，即图卷积神经网络的公式如上。其中，$\mathbf{m}_{u}^{(l)} =  \mathbf{W}^{(l)} \frac{\mathbf{h}_{u}^{(l-1)}}{|N(v)|}$  可以视为 GCN 的消息函数，当结点 $u$ 向它的邻居结点 $v$ 发送消息时，会除以结点 $v$ 的度进行正则化（原论文的正则化方式稍有不同）。而聚合函数 $\mathbf{h}_{v}^{(l)}=\sigma\left(\operatorname{Sum}\left(\left\{\mathbf{m}_{u}^{(l)}, u \in N(v)\right\}\right)\right)$ 则会对所有的邻居结点产生的消息进行加和，然后通过某激活函数 $\sigma(\cdot)$ 进行非线性变换。

### 2.2 CASE 2: GraphSAGE 

$$
\mathbf{h}_{v}^{(l)}=\sigma\left(\mathbf{W}^{(l)} \cdot \operatorname{CONCAT}\left(\mathbf{h}_{v}^{(l-1)}, \operatorname{AGG}\left(\left\{\mathbf{h}_{u}^{(l-1)}, \forall u \in N(v)\right\}\right)\right)\right)
$$
上式中，为了表示结点 $v$ 在第 $l$ 层的中间表示 $\mathbf{h}_{v}^{(l)}$，首先计算结点 $v$ 的所有邻居结点的消息  $\mathbf{h}_{u}^{(l-1)}$ （个人理解此处应该表示为 $\mathbf{m}_{u}^{l}$），然后进行消息的聚合。聚合过程分为两个阶段：

- 邻居结点消息的聚合 $\operatorname{AGG}\left(\left\{\mathbf{h}_{u}^{(l-1)}, \forall u \in N(v)\right\}\right)$
- 为了避免结点 $v$ 自身信息在聚合过程中的丢失，我们通过 $\mathrm{CONCAT}(\cdot)$ 操作将结点 $v$ 上一层的输出 $\mathbf{h}_{v}^{(l-1)}$ 和邻居结点聚合后的结果进行拼接。

与 GCN 相比，GraphSAGE 中聚合消息函数有更多形式的表达。论文中提供的消息函数有以下三种：

- Mean：对邻居结点的上一层输出进行求和然后取平均。
  $$
  \operatorname{AGG} = \sum_{u \in N(v)} \frac{\mathbf{h}_{u}^{(l-1)}}{|N(v)|}
  $$

- Pool：对邻居结点的上一层输出进行线性变换，然后进行最大池化或者平均池化操作。其中，线性变化可被视为 message 操作，池化可被视为聚合操作。
  $$
  \mathrm{AGG}=\operatorname{Mean}\left(\left\{\operatorname{MLP}\left(\mathbf{h}_{u}^{(l-1)}\right), \forall u \in N(v)\right\}\right)
  $$

- LSTM：将 LSTM 用于随机随机排列的邻居结点。此时将 LSTM 进行聚合操作。
  $$
  \mathrm{AGG}=\operatorname{LSTM}\left(\left[\mathbf{h}_{u}^{(l-1)}, \forall u \in \pi(N(v))\right]\right)
  $$

此外，GraphSAGE 在实现中为每一层的中间结果增加了 $\ell_{2}$ 正则化的操作。添加正则化的目的是使得结点 $v$ 在进行聚合时，它的所有邻居结点 $N(v)$ 都有相同的 $\ell_{2}$ 范数；否则，结点 $v$ 的两个邻居结点 $u_1, u_2$ 的上一层的表示 $\mathbf{h}_{u_1}^{(l-1)}, \mathbf{h}_{u_2}^{(l-1)}$的数据范围不同。在一些情况下，增加正则化的操作可以提升图的表现性能。

### 2.3 CASE 3: GAT

GAT (Graph Attention Networks) 的思想是将注意力机制（Attention）引入到聚合操作中。即：
$$
\mathbf{h}_{v}^{(l)}=\sigma\left(\sum_{u \in N(v)} \alpha_{v u} \mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}\right)
$$
$\alpha_{v u}$ 表示结点 $u$ 产生的 message 对于结点 $v$ 的重要性。我们可以认为 GCN 和 GraphSAGE 中, $\alpha_{uv} = \frac{1}{|N(v)|}$。GCN 和 GraphSAGE 中 $\alpha_{uv}$ 的值只和结点 $v$ 的度相关，它的所有的邻居结点被认为具有相同的重要性。但是，基于注意力机制，聚合操作可以给更重要的邻居结点较大的权重。结点的重要性取决于它所在的环境，但是也是可以通过模型的训练进行学习。

下面介绍**引入注意力机制**的方法。

- 计算 attention coefficient：假设 $a(\cdot, \cdot)$ 是某种注意力机制，那么可以依据下式计算 $e_{vu}$， $e_{vu}$ 表示结点 $u$ 产生的 meassage 对于 结点 $v$ 的重要性。
$$
e_{v u}=a\left(\mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}, \mathbf{W}^{(l)} \boldsymbol{h}_{v}^{(l-1)}\right)
$$

- 正则化：使用 softmax 的方法进行正则化，将 $e_{uv}$ 转换成 $\alpha_{uv}$，满足 $\sum_{u \in N(v)} \alpha_{v u}=1$。

$$
\alpha_{v u}=\frac{\exp \left(e_{v u}\right)}{\sum_{k \in N(v)} \exp \left(e_{v k}\right)}
$$

- 根据式（9）计算结点 $v$ 聚合后的结果。如下图所示，通过步骤1、2可以计算结点 A 的邻居结点 B/C/D 的 attention weight $\alpha_{AB}, \alpha_{AD}, \alpha_{AC}$, 然后计算结点 A 的下一层中间表示为$\mathbf{h}_{A}^{(l)}=\sigma\left(\alpha_{A B} \mathbf{W}^{(l)} \mathbf{h}_{B}^{(l-1)}+\alpha_{A C} \mathbf{W}^{(l)} \mathbf{h}_{C}^{(l-1)}+ \alpha_{A C} \mathbf{W}^{(l)} \mathbf{h}_{C}^{(l-1)}\right)$。

<img src="/blogs/images/image-20211109121418611.png" alt="image-20211109121418611" style="zoom:50%;" />



这里展示一种注意力机制 $a(\cdot, \cdot)$ 的选取：对于上一层的中间表示 $\mathbf{h}_{A}^{(l-1)}, \mathbf{h}_{B}^{(l-1)}$，可以先计算它们产生的消息，拼接之后进行线性变换。线性变换的参数可以随着图神经网络的下游任务一起进行训练。

![注意力机制](/blogs/images/image-20211111214023361.png)

Multi-head attention 可以稳定注意力机制的训练过程。具体的实现方式是创建多组注意力机制的参数，每组参数都会计算出一个中间表示结果，最终使用拼接（concatenation）或者求和（summation）的方法将输出结果聚集。

## 3. GNN Layers in Pratice

<img src="/blogs/images/image-20211109124636402.png" alt="A suggested GNN Layer" style="zoom: 50%;" />

GNN 的设计可以将在许多领域被证明有用的现代深度学习的模块纳入考虑之中。推荐的单层 GNN layer 如上图所示：

- Linear：线性变换
- BatchNorm：稳定神经网络训练
- Dropout：防止过拟合
- Activation：非线性变换
- Attention/Gating：为不同的邻居结点分配 attention weight

## 4. Stacking Layers of a GNN

以上部分讨论了单层 GNN layer 的设计，接下来考虑如何连接多层 GNN 层从而构建一个大的 GNN 网络。

首先，最直接的连接方式是顺序连接，如下图所示。其中 $x_v$ 表示结点 $v$ 的初始 embedding 特征，它将作为第一层 GNN Layer 的输入。

<img src="/blogs/images/image-20211111151057809.png" alt="Sequentially Stack GNN Layers" style="zoom: 67%;" />

但是，堆叠较多的 GNN 层的问题是模型会 over-smoothing——所有结点的 embedding 特征趋同。我们可以从感受野的角度理解 over-smoothing 产生的原因。在 GNN 模型中，某一结点感受野可以理解为影响该结点 embedding 结果的“邻居”结点。一个有了 $K$ 个 GNN 层的模型，每个结点都会有与它距离至多为 $K$ 的 $K$-hop 邻居结点集合。当模型的层数增加时，重叠的两个不同的结点相重合的 K-hop 邻居结点集合将迅速扩大。如下图所示，两个黄色结点均是我们的目标结点，但是随着 $K$ 从 1 增长到 3，重合的邻居结点数量也迅速扩散。

![GNN 的感受野](/blogs/images/image-20211111151523692.png)

因此，如果我们为您堆叠较多的 GNN 层，不同的结点可以共享的 $K$-hop 邻居结点的数量增多，导致不同结点的 embedding 特征趋同，也就是 over-smoothing 问题的产生。

为了避免的 over-smoothing 的问题的产生，我们通常设计层数较少的 GNN 模型。那么，如何在有限的 GNN 层数的条件下增强图神经网络的表达能力呢？课程介绍了两种可以尝试的实现方案。

**方案一：增加单 GNN 层的复杂度**

![增加单 GNN 层的复杂度](/blogs/images/image-20211111153656652.png)

第一种方案是增加单 GNN 层的复杂度。如上图所示，在前面的介绍中，message 和 aggregation 函数都是单层线性变换。我们可以使用深度神经网络替换这两种变换，从而增强单 GNN 层的表达能力。

**方案二：增加无消息传递的网络层**

<img src="/blogs/images/image-20211111154046035.png" alt="GNN 中的预处理层与后处理层" style="zoom: 80%;" />

GNN 网络中不是所有的神经网络层都需要像邻居结点传递消息。我们可以在模型中增加预处理层与后处理层。实际中，这些层的增加对于模型的表达能力的提升效果显著。

- 预处理层可以应用在我们需要对结点特征进行编码时，例如图像、文本数据。
- 后处理曾可以应用在我们想利用图神经网络的结点 embedding 进行下游分类、回归等任务。 

**方案二：增加 Skip Connection**

<img src="/blogs/images/image-20211111154850416.png" alt="增加 Skip Connection 的 GNN 模型" style="zoom:80%;" />

增加 Skip Connection 的思想来源于 ResNet 与 DenseNet 等含有 short-cut 的深度神经网络：前一层网络的输入可能会被重新使用，加和或者拼接到后续网络层的输出中。