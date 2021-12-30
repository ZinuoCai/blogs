---
title: AntMan--Dynamic Scaling on GPU Clusters for Deep Learning
date: 2021-12-30 09:49:38
tags:
---

## Overview


AntMan 是发表在 OSDI'20 Machine Learning Session 的论文，主要解决**深度学习的 GPU 集群资源使用率低的问题**。一作是**肖文聪**（Wencong Xiao，<https://wencongxiao.github.io/>），北航与微软亚洲研究院联培博士，现任职于阿里巴巴 PAI group。代表作包括 AntMan (OSDI'20) 和 Gandiva (OSDI' 18)。

<!--more-->

之前读过这篇文章，初读时由于基础知识的缺乏，所以没有再读时体会深刻。现在重读觉得这篇文章一气呵成，尤其是图表的设计自成体系，13 张图和 5 张表勾勒出文章的骨架。本文就主要从这 18 张图表介绍 AntMan 的设计，体会顶会论文是如何炼成的。Let's go!

## 1. Background

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228161343891.png)

**问题描述**：集群中 GPU 使用率较低。**图1**展示了在实际生产集群中 GPU 资源的使用情况：仅有 20% 的机器，GPU 的显存使用率超过 50%；仅有 10% 的机器，GPU 的计算单元使用率超过 80%。

**原因阐释**：低使用率通常源于两个方面。一是为了避免任务的抢占，集群往往给用户的任务分配独占的 GPU 显卡，但是任务无法充分使用被分配的显存资源或者计算资源；二是对于需要多张显卡执行机器学习任务的模型，他们需要花费较长的时间等待 GPU 的分配。而且等待的过程中已经分配的 GPU 既不会执行任务，也不会释放给其他用户使用。**图2**展示了当用户需要使用多块 GPU 时，GPU 的平均空载时间。

**现有方案及不足**：可行的解决方案是允许多个任务使用同一块 GPU 执行任务。但是，packing 一方面会引起计算资源竞争，导致该 GPU 上的任务完成效率同时下降。另一方面则是显存资源竞争，显存资源竞争的后果则是引起 `RuntimeError: CUDA out of memory` 的问题。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228232930713.png)

**模型训练的资源使用特点**：模型训练的特征 GPU 资源的低使用率与较长的空闲等待时间之外，任务需要的资源也动态变化。**图3**和**图4**分别显示了 DeepFM 模型与 ESPnet 模型在模型训练时 GPU 的显存使用与计算单元的使用率动态变化的特点。图4 似乎更有意思。我们知道，模型结构决定了模型的计算特征：计算密集型或者 IO 密集型。**在单个模型的内部， ESPnet 出现了计算密集型与 IO 密集型的分化**： 1200～1400s 是计算密集型（SM util 较高）， 1400～1600s 是 IO 密集型（Memory util 较高）。

除了归纳模型训练的资源使用特点，AntMan 还结合了以下实践得出的机会点进行系统的设计：

- 模型使用的显存比较小。**图5**对一周机器学习的训练任务进行统计，**图5(a)** 显示90%的机器学习训练任务，模型本身占用的显存都小于300MB。
- 模型训练过程中 mini-batch 的训练时间较短。**图5(b)** 显示每个 mini-batch 的训练时间普遍小于1500毫秒。
- 训练过程周期性变化。我们可以利用周期性变化的特点，分析在某种资源配置下模型训练的性能，从而对 resource-guarantee 与 opportunistic 的任务提供更合理的资源配置。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228193409487.png)

## 2. Design

这一部分主要讨论 AntMan 如何设计应对 GPU 共享的挑战。首先介绍的是对深度学习框架的扩展，从而支持显存和计算单元的动态分配。然后讨论 AntMan 中在 master node 的 Global Scheduler 与在slave node的Local Coordinator 相互协作，共同调度 GPU 资源设计，支持对深度学习框架的扩展。最后介绍上述的调度方案在阿里云平台的实践。

### 2.1 Dynamic Scheduling

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228194121975.png)

首先介绍显存资源的动态分配。

**图6**对比了现有的深度学习框架与 AntMan 的显存分配设计。**图6(a, b, c)** 展示深度学习框架在在申请显存后，当变量被销毁时，申请的显存不会释放，而是缓存在深度学习框架自己维护的显存池中。例如，这里 <https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp> 介绍了 PyTorch 是如何将显存据为己有，自主分配。**图6(d, e, f)** 则是 AntMan 的显存机制。当显存资源有限时，AntMan 会降低 opportunistic 任务的显存分配，因此部分的计算张量只能拷贝到 CPU 设备的内存中；当显存资源有盈余时，内存上的张量又会重新拷贝到 GPU 显存中。**图7**则是介绍了上述的动态分配显存资源的机制如何和机器学习训练任务结合。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228201835645.png)

其次介绍计算单元的动态分配。在现代操作系统中，`cgroup` 的设计保证了进程在使用 CPU 计算时的性能隔离，但是 GPU 却没有保证多个进程性能隔离的机制。因此，动态分配计算单元的目的则是保证 opportunistic 的任务不会影响到 resource-guarantee 任务的完成时间 (JCT, job completion time)。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228202203689.png)

**图8** 展示了动态分配计算资源的方案。当我们在使用 GPU 执行任务时，实际上是使用 GPU 的流处理器 (SM, stream multiprocessor) 执行自定义的 kernel 函数。图上蓝色的方格表示 Job A 的 kernel 函数执行流，Job A 是 resource-guarantee 的任务；绿色方格表示 Job B的 kernel 函数执行流，它是 opportunistic 类型的任务。**图8(a)** 中是 Job A 独占时的 GPU 使用情况，我们发现 GPU 并未被充分使用。在 **图8(b)** 如果我们将 Job B 不加限制地扔到 GPU 上执行时，尽管 GPU 的 cycle 被完全占据了，但是 Job A 的 JCT 大受影响。

AntMan 的计算资源分配如 **图8(c)** 所示。具体的实现方案是在深度学习框架中引入 `GpuOpManager`模块。它的作用是控制 GPU 的 kernel 函数发送到 GPU 设备频率。`GpuOpManager`会在 GPU 设备空闲的时候将 opportunistic 任务的 GPU 算子发送到 GPU 上执行，从而不会影响 resource-guarantee 任务的执行效率。

### 2.2 Collaborative Scheduler

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228210634012.png)

**图9**展示了分别位于 master node 与 slave node 的 Global Scheduler 与 Local Coordinator 相互协作，调度 GPU 资源的过程。整体来说，这一部分的设计中规中矩，master 与 local 相互协作调度的设计受到 Kubernetes 的影响痕迹明显，没有 Dynamic Scheduling 部分的设计巧妙。

Global Scheduler 维护全局的信息，包括硬件设备的信息 (GPU Computing  Utilization 与 GPU Memory Usage) 和深度学习任务的执行信息 (mini-batch 的执行时间，最高、最低与平均显存使用信息以及 CPU 端的内存信息)，这些信息都是 slave node 反馈给 master node 的。Global Scheduler 利用这些信息更有效的指导深度学习任务的安置与资源分配。

### 2.3 Scheduling Policy

Design 的最后一部分介绍了资源动态分配与主从节点协作调度在实际生产集群中的实践。

设计的**总目标**是通过引入 opportunistic 类型的任务使用空闲的 GPU cycle，从而提高 GPU 集群的资源使用率。

我们重点关注 Local Coordinator 的设计。Local Coordinator 的设计融入了动态分配显存与计算资源考量，主要考虑了以下三个问题：

- 如何保证 resource-guarantee 任务的性能？
- 如何处理 resource-guarantee 任务的需求激增情况？
- 如何使用贪心的方法最大化 opportunistic 任务的表现？

对于第一个问题，AntMan 的做法是首先保证 resource-guarantee 的任务稳定执行；然后为 opportunistic 的任务分配 GPU 显存和计算单元；最后观察 resource-guarantee 的性能是否受到了影响。判断性能是否受到影响的方法则是观察 mini-batch 的执行时间进行判断。

对于第二个问题，如果是 memory 突变的情况，首先使用 host memory 作为临时存储，减少 opportunistic 任务的显存使用；接着增加 resource-guarantee 任务的显存分配。同样的方法适用于计算单元需求的激增。

最后一个问题的场景是多 GPU 任务在等待 GPU 资源时，已经分配到的 GPU 空闲的情况。对于这个问题，AntMan 的解决方案是使用贪心算法，最大化 GPU 显存的使用效率。**图10**展示了使用贪心算法的现实依据——不同的模型对 GPU 显存的限制变化在性能上的感知差距很大，例如降低 SR 模型 90% 的显存分配仅带来 25% 的性能损失，而降低 ResNet 模型 10% 的显存就引起 60% 的性能损失。因此，AntMan 在 opportunistic 任务竞争的场景下优先将 GPU 分配给能够带来性能提升的任务（ResNet 的优先级高于 SR，“会哭得孩子有奶吃”？）。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228212107240.png)

## 3. Evaluation

实验结果分成三个部分。一是在 benchmark 上分析 AntMan 提出的动态资源调度的效果；二是在公开数据集上进行测试；三是在阿里云上带有 5000 块异构的 GPU 集群上的效果测试。

### 3.1 Benchmark

**实验一：Dynamic GPU memory scaling.**

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228222730165.png)

**表1**展示了两个不同的任务，其中 Job A 是 opportunistic 类型的任务，Job B 是 resource guarantee 类型的任务。在第 26 分钟，两个任务会同时执行在显存为 32 GB 的 GPU 上。**表2**提供了不同的解决方案以及对应的结果，其中抢占或者 Pack 会导致其中某一个任务崩溃，而 FIFO 与 UMem 方法不能保证 resource guarantee 任务的效率。AntMan 的动态显存分配满足要求。而且，我们可以发现，Job A 的性能下降也是非常低的。

**实验二：Efficient memory shrinkage and growth.**

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228223550574.png)

**图11**展示了 AntMan 在进行显存的分配与压缩需要的额外开销。其中,**图11(b)** 分别列举了 VGG16，InceptionV3 与 GoogleNet 在显存变化 17GB，16GB和4GB时的开销时间，以此证明显存动态分配的开销微乎其微。

**实验三：Dynamic GPU computation unit scaling.**

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228223607063.png)

为了展示 AntMan 动态的计算资源分配不会影响 resource-guarantee 任务的性能，作者将 AntMan 与 Packing 模式相对比。实验的模型是 ResNet50（opportunistic，计算密集型）与ESPNet（resource-guarantee，非计算密集型）。**图12(a)** 展示直接进行 Packing 会导致 ESPNet 模型的 SM 使用率受到 ResNet50 的约束，而 AntMan 会自适应的调节 ResNet50 的 SM 使用率，确保 ESPNet 模型正常执行。实验结果表明，Packing 模式，ESPNet 的执行时间为 105.2 分钟，严重受到影响；而 AntMan 的执行时间保持在 20 分钟左右，**niubility**！

### 3.2 Trace Experiment

这部分的实验数据来自微软在 Gandiva 中使用的数据集，模型的分布如**表3**。实验对比了 YARN-CS 与 Gandiva，结果在**图13**。实验结果显示 AntMan 可以达到更短的平均任务完成时间 (JCT) 与最大任务完成时间 (makespan)。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228223631589.png)

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228223649095.png)

### 3.3 Cluster Experiment

最后一部分实验在案例云的集群中进行测试，集群拥有超过 5000 块 GPU。**表4**展示了部署 AntMan 前（2019年12月）后（2020年4月）任务在分配 GPU 时的等待时间。（表格数据是否存在问题，为什么 2020年4月Average 时间比 90% 和 95% 分位的时间长？）**表5**对10000个训练任务的 mini-batch 训练时间进行分析，发现99%的任务 mini-batch 时间误差在 10 毫秒内，认为是没有收到影响。

![](https://raw.githubusercontent.com/ZinuoCai/Image-Bed/image-bed/2021/image-20211228223701862.png)

## 4. Conclusion

没啥好总结的，谈一谈我重读这篇文章发现的一些问题与研究点，希望可以抛砖引玉。

- **研究如何更好地模型组合共享资源**：什么样的模型结构任务适合去共享 GPU 资源呢？计算密集型与 IO 密集型能否进行更好地组合，以此能够充分利用 GPU 为我们提供的计算单元与显存这两种资源？

- **研究减少显存对模型性能的影响**： Scheduling Policy 中多个 opportunistic 任务的显存分配中提到了不同的模型对显存分配降低时的性能损失不同，AntMan 使用的方法是 profile 不同模型在不同显存分配下的性能表现，然后根据测量的结果进行分配？是否用更系统或者理论的方法，通过分析模型的结构来预测性能的变化？
- **研究共享任务的计算单元分配问题**：还是上述场景，任务的执行时间和分配的计算单元的速率有很大的关系，AntMan 却没有明确说明这些任务计算单元是如何分配的。有没有分配计算单元的方法，胜过没有任何限制的抢占得到的效率？Dynamic Scheduling 中的计算单元的动态分配值得参考。

