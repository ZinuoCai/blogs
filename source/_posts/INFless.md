---
title: INFless--A Native Serverless System for Low-Latency, High-Throughput Inference
date: 2022-3-30 12:06:00
tags:
mathjax: true
---

本文由天津大学、同济大学与 58 同城合作完成，发表在 ASPLOS’ 22 上。

<!--more-->

## 1. Overview

**背景**：由于无服务器计算系统资源免配置、自动扩缩容与“Pay-as-you-go”的付费模式，其已经被广泛应用到各种领域，包括将机器学习推理任务部署到无服务器计算系统中。

**问题**：但是，常见的无服务器计算服务，包括 Amazon Lambda，Google Cloud 与 Azure Functions 不能满足推理服务低延迟、高吞吐量的性能需求。主要原因在于他们没有允许用户指定 SLO 需求，同时对于计算密集型的推理任务没有提供加速硬件支持。

**已有方案**：已有的解决方案包括 Mark (ATC‘ 19)、BATCH (SC’ 20) ，它们为了满足吞吐量的需求都采用了批处理的方案——将多个任务集成到一次执行。这种 On-Top-of-Platform (OTP) 的设计在已有的无服务器计算框架的基础上增加新的批处理层，尽管可以提高系统的吞吐量，但是低延时的性能需求没有妥善的解决。而且，这种设计也不利于对底层资源进行合理的调度。

**设计**：因此，研究者提出了一个基于无服务器计算的原生的机器学习推理框架，该框架可以满足低延时、高吞吐量、低开销的性能需求。该方案的主要设计包括：

- 内置批处理解决方案与异构硬件支持；
- Combined Operator Profiling 方法找到资源配置方案；
- Long-Short Term Histogram (LSTH) 策略降低冷启动时间与减少资源浪费。

**结果**：最终的实验表明在满足低延时的系统需求的基础上，可以提高2到5倍的系统吞吐量。

## 2. Motivation

![观测模型集](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20220329232009631.png)

将无服务器计算应用到已经成为了当前研究的趋势。研究者从上面表格中的11个模型入手，在 AWS Lambda 平台上展开测试，得到以下六条经验性的结论：

- 由于缺少加速硬件的支持，商用的无服务器计算平台在部署计算密集型的任务时往往会有高时延问题；
- 使用批处理的方法优化系统吞吐量时，对于一些小模型不能提供低时延的服务；
- 商用的无服务器计算平台 CPU-memory 按比例分配的资源调度策略对于计算密集型的任务导致内存的资源浪费；
- 已有的 “one-to-one mapping” 资源分配的策略不能根据用户负载的变化自动选择更有的资源调度方案；
- OTP 的设计方案缺少批处理参数、函数示例调度与资源分配的协同设计，因此仅带来系统吞吐量的优化。
- （针对 Combined Operator Profiling 设计方案）机器学习的模型推理任务通常由一些通用的模型算子的组成，而且少部分算子占据了执行的大部分时间，因此可以使用 profiling 的方案估计模型的执行时间。

## 3. Design and Implementation

这一小节主要介绍 INFless 的系统设计与实现。

- 3.1 介绍系统的整体框架；
- 3.2-3.5 是系统的设计细节：
  - 3.2 介绍 INFless 内置的批处理方案；
  - 3.3 介绍 INFless 通过对机器学习模型算子进行 profiling 预估任务执行时间；
  - 3.4 介绍 INFless 基于线性规划的资源调度方案，研究者使用了贪心算法简化最优解的求解；
  - 3.5 介绍 INFless 基于 Long-Short Term Histogram 的冷启动处理方案；
- 3.6 介绍了 INFless 基于 OpenFaaS 的开发与实现。

### 3.1. System Architecture

![系统架构图](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20220329224552859.png)

上图是整体的系统框架图。机器学习推理服务的提供者在部署模型时 (1)，对于机器学习模型有向无环图的特定结构 (2)，系统会根据已经 profiling 的算子集对模型的执行时间进行预测 (3)。推理服务的使用者进行推理请求时 (4)，根据用户请求的 arrival 特征，系统会生成对应的请求处理与资源调度方案 (5)，系统中的 auto-scaling engine 会根据任务负载的变化自动优化资源配置，从而在保证 SLO 需求的情况下降低资源消耗。同时针对无服务器计算平台独有的冷启动问题，INFless 使用了基于 LSTH 的方法在冷启动与资源消耗之间达到平衡 (7)。

### 3.2. Built-in, Non-Uniform Batching

假设任务请求的速率在 $[r_{low}, r_{up}]$ 区间，该实例处理的批处理大小为 $b$，任务的执行时间为 $t_{exec}$，用户指定的最高时延为 $t_{slo}$，则：

$$
r_{up} = \frac{1}{t_{exec}} * b, r_{low} = \frac{1}{t_{slo}-t_{exec}}*b
$$

为了保证$r_{low} \le r_{up}$，任务执行时间与用户指定的SLO需求应该满足$t_{exec} \le t_{slo} / 2$ 的约束。例如，如果一个任务的 SLO 需求是不超过 200 毫秒，任务的执行时间是 50 毫秒，批处理大小为 4，那么工作负载的速率应该是每秒 28~80 次请求之间。

假设已经存在 $n$ 个实例处理该计算任务，则 $R_{\max }=\sum_{i \in[1, \ldots, n]} r_{u p}^{i}$，而且 $R_{\min }=\sum_{i \in[1, \ldots, n]} r_{\text {low }}^{i}$，那么根据实际的任务请求速率 RPS (request per second) ，系统会做出以下的性能调整：

- 当 $R > R_{max}$，实际的任务请求速率超过当前系统的最大处理能力时，auto-scaling engine 将增加实例来满足超过部分的处理需求；
- 当 $\alpha R_{min} + (1-\alpha)R_{max} \le R \le R_{max}$ ，在这种情况下，系统的负载比较均衡，每个实例处理的请求数量在 $r_{i}=r_{u p}^{i}-\frac{R_{\max }-R}{R_{\max }-R_{\min }} \times\left(r_{u p}^{i}-r_{l o w}^{i}\right)$。在系统实现中，为了提高系统的吞吐量，$\alpha$ 的值设置为0.8。
- 当 $R \le \alpha R_{min} +(1-\alpha) R_{max}$，这种情况下，auto-scaling engine 将减少多余的实例来降低资源需求。

### 3.3. Combined Operator Profiling

对于一个算子 $o_i$，对其进行 profiling 时收集其对应的五元组，$\left\langle p_{i}, b_{i}, c_{i}, g_{i}, t_{i}\right\rangle$。其中，每个元素对应的意义如下：

- $p_i$ 算子的输入规模；
- $b_i$ 批处理的尺寸，只考虑 $b_{i} \in\left\{2^{0}, 2^{1}, \ldots, 2^{\max }\right\}$的情况；
- $c_i$ CPU相关参数，包括核数、内存带宽、内存大小与缓存；
- $g_i$ GPU相关参数，包括GPU显存，SM数量与PCI-e带宽；
- $t_i$ 任务执行时间。

研究者收集了超过 100 个算子的 profling 结果并组成了一个数据库。该数据库可以用于估计在某种资源配置的条件下，整体模型的执行时间。由于机器学习推理任务往往是一个有向无环图的执行流程，顺序执行的部分执行时间为各算子的执行时间之和，带有分支部分的计算时间为所有路径执行时间的最大值。下图展示了对于ResNet-50，MobileNet 与 LSTM-2365 这三种模型在不同的配置条件下的模型执行时间与预估执行时间的误差，结果显示最终的误差率在10%以内。因此，基于此方法研究者在系统实际运行时将预测时间扩大 10% 以保证不违背用户的SLO需求。

![Combined Operator Profiling 结果分析](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20220329211711153.png)

### 3.4. Scheduling

研究者将系统调度问题形式化为了一个线性规划问题。假设集群中可用的服务器的数量为 $m$，至多启动 $n$ 个实例执行计算任务。对于实例 $i$ ，需要决定它的资源配置包括 $b_i, c_i，g_i$ 以及一个二元变量 $x_{ij} \in \{0, 1\}$（实例 $i$ 是否调度到服务器$j$上）。同时对于每个服务器 $j$ 设置二元变量 $y_j\in \{0, 1\}$ 判断当前服务器是否可用。
$$
\begin{aligned}
\text { minimize }: & \sum_{j}^{m}\left(\beta C_{j}+G_{j}\right) y_{j}     &(2)\\
t_{\text {wait }}^{i}+t_{\text {exec }}^{i} \leq t_{\text {slo }}^{i}, & \forall i \in[1, \ldots, n] &(3)\\
t_{\text {exec }}^{i} \leq t_{\text {wait }}^{i}, & \forall i \in[1, . ., n] &(4)\\
\sum_{i}^{n} c_{i} x_{i j} \leq C_{j} y_{j}, & \forall j \in[1, . ., m] &(5)\\
\sum_{i}^{n} g_{i} x_{i j} \leq G_{j} y_{j}, & \forall j \in[1, . ., m] &(6)\\
\alpha R_{\max }^{k}+(1-\alpha) R_{\min }^{k} \leq R_{k} \leq R_{\max }^{k}, & \forall k \in I &(7)\\
x_{i j} \in\{0,1\}, & y_{j} \in\{0,1\} &(8)\\
b_{i}, c_{i} \leqslant Z_{+}, & g_{i} \in Z &(9)
\end{aligned}
$$
调度策略的优化目标为 (2) ,其中 $C_j$ 与 $G_j$ 分别代表了服务器 $j$ 上使用的 CPU 与 GPU 的资源使用情况，由于 CPU 与 GPU 的计算效能不同，因此使用参数 $\beta$ 进行平衡；(3) 和 (4) 保证用户的 SLO 需求得以满足，(5) 和 (6) 保证 CPU 与 GPU 资源不会超载，(7) 保证当前系统负载均衡。由于该线性规划问题是一个 NP 难的问题，因此作者设计了贪心算法进行最优解的求取，详细算法参考原论文。

### 3.5. Managing Cold Starts with LSTH

冷启动时间是无服务器框架中常见的时延来源。解决该问题的 SOTA 方法是微软在 Serverless in the Wild (ATC' 20) 提出的 HHP 方法，通过过去一段时间的任务请求情况绘制柱状图，推导出两个参数 pre-warming window 与 keep-alive window。研究者发现将该方法融入到已有的框架中会产生较多的资源浪费，因此他们提出了 LSTH 方法平衡冷启动次数与资源使用效率之间的关系。

![58 同城的资源请求模式与 LSTH 示意图](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20220329215633800.png)

研究者对 58 同城的机器学习任务请求进行分析，其任务请求呈现两个明显的特征，如上图中的 (a) 子图所示。特征一从长周期来看，任务请求呈现周期性的变化趋势。特征二从短周期来看，任务中存在 burst 的情况（突然上升或者下降）。基于 HHP 的方法，特征一会采用较低的 pre-warming window，尽管可以降低冷启动时间，但是造成较高的系统资源浪费；特征二会导致柱状图不具有代表性，使得冷启动的次数增多。

基于以上的观察，研究者提出使用 LSTH 的方法进行 pre-warming window 与 keep-alive window 的参数选取。与 PPH 方法相比，作者基于短期的任务请求模式与长期的任务请求模式绘制了两幅柱状图，如上图中的子图(b) 所示。通过柱状图可以推导出长周期模式下的 pre-warming window 与 keep-alive window 参数 $L_{\text{pre-warm}}$ 和 $L_{\text{keep-alive}}$, 短周期下的参数 $S_{\text{pre-warm}}$ 和 $S_{\text{keep-alive}}$。最终对于某一特定任务的两个滑动窗口的值将是两种模式下的trade-off。实际实现中，$\gamma = 0.5$。
$$
\text { pre-warm }=\gamma L_{\text {pre-warm }}+(1-\gamma) S_{\text {pre-warm }}
$$
$$
\text { keep-alive }=\gamma L_{\text{keep-alive}}+(1-\gamma) S_{\text {kee-palive }}
$$

### 3.6. Implementation

**INFless** 基于开源的无服务器计算框架 OpenFaaS 实现，框架部署在 Kubernetes 集群中。一些代码实现细节如下：

- 接近 9300行代码主要修改了 **faas-netes**，**faas-cli** 与 **faas-Gateway** 组件，接近 5000 行代进行系统测试、模拟与工作负载生成。
- 弹性收缩引擎集成在 **faas-netes** 中，替代了原来的调度模块。
- 使用 Nvidia-docker 为函数提供GPU硬件支持，为了提高GPU的使用效率，使用MPS方法允许多个任务共享GPU执行。使用linux cgroups来限制CPU的资源使用情况。
-  在应用层，修改了 **faas-cli** 允许用户指定 SLO 的需求；在系统层 (1) 增加了 profiling 的数据存储，(2) 增加了基于 combined operator profiling 的模型执行时间预测模块，(3) 修改了原系统的h函数触发机制和 (4) 实现了调度算法和基于 LSTH 的冷启动优化算法。

## 4. Conclusion

本文面向机器学习推理任务的无服务器计算框架进行设计，最终实现了支持异构硬件设备支持的无服务器计算框架。框架的主要设计包括在本文的第三部分进行了详细介绍，包括**自适应的批处理**方案、**任务执行时间估计**方案、**资源调度**方案与冷启动处理方案。总体来说，本文对于问题的定义明确，解决方案简洁同时 workable，值得学习！