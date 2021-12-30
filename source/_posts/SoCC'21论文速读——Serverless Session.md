---
title: SoCC' 21 论文速读——Serverless Session
date: 2021-12-09 21:20:00
tags:
---

## TL; DR

SoCC 是云计算领域的顶会。2021 年 SoCC 共计录用 46 篇论文，其中无服务器计算有两个 Session，共收录 7 篇论文。<!--more-->本文对其中的 4 篇论文进行简要的介绍，其中 VSP[1] 从服务的使用者角度考虑问题，尝试打破云计算提供商的技术壁垒；Kraken[2] 解决有向无环图应用的资源配置问题；Faa$T [3] 考虑 serverless 框架中的存储问题，通过设计本地的缓存代替吞吐量受限制的 remote storage；ServerMore[4] 提出将 serverless 的工作负载和 serverful 的虚拟机工作负载混合执行，进一步提高整体的资源利用率。

完整的论文列表链接 https://dblp.uni-trier.de/db/conf/cloud/socc2021.html。

## 1. VSP

On Merits and Viability of Multi-Cloud Serverless

针对市场上的多家云服务厂商提供的无服务器计算，本文设计了一个虚拟的无服务器计算服务的提供者 (VSP, virtual serverless provider)。

VSP 提出的目的是比较不同的云计算厂商的无服务器计算费用，从而避免开发者局限于一种云计算厂商的困境。同时，VSP 可以促进云计算提供商之间的“内卷”，鼓励他们提供更便宜、高性能的计算服务。

VSP 的提出有以下的理论基础：(1) 无服务器计算函数通常是轻量级的，因此可以“无痛”地在多家云计算提供商之间部署和迁移服务；(2) 存在较多开源的无服务器计算框架；(3) 已有的工作研究无服务器计算中函数性能的可预测性，有助于优化服务厂商的选择。

**评价**：本文提出的 VSP 从使用无服务器计算的开发者角度考虑问题，通过比较不同厂商提供的服务性能与价格进行选择。VSP 的提出，可以打破无服务器计算的开山之作 *Cloud Programming Simplified: A Berkeley View on Serverless Computing* 中提出的各个厂商之间的服务生态壁垒，给予开发者更多的云服务选择空间。

## 2. Kraken

Kraken : Adaptive Container Provisioning for Deploying Dynamic DAGs in Serverless Platforms

微服务带动了以云服务为基础的应用的流行，这些微服务常常以有向无环图 (DAG, directed acyclic graph) 的形式被调度执行。由于这些应用大多数是面向用户服务的，因此有着明确的 SLO 的需求。无服务器计算的短执行时间、细粒度资源分配与良好的扩展性等特点可以满足有着 SLO 需求的微服务应用的需求，但是由于 serverless 对有向无环图的执行顺序无法感知，因此导致了资源的过度分配等问题。

基于以上的考虑，我们提出了 Kraken。它是一个面向无服务器计算框架的 workflow-aware 的资源管理框架，它可以最大限度地减少为应用程序 DAG 配置的容器数量，同时确保符合 SLO 的需求。我们将 Kraken 部署在开源的无服务器计算框架 OpenFaaS 上，在一个多节点的 Kubernetes 集群中进行实验。实验结果表明，Kraren 可以减少容器的数量达到 76%，将系统资源利用率提高了 4 倍，在集群范围内节约 48% 的能源。 

## 3. Faa$T

Faa$T: A Transparent Auto-Scaling Cache for Serverless Applications

由于无服务器计算中函数 stateless 的特点，函数需要依赖于外部存储 (remote storage) 保存必要的状态。已有的工作在解决该问题时面临以下三个问题：(1) 忽略了函数变化的特点 (adaptivity)；(2) 无法依据数据读写模式进行扩缩 (scalability)；(3) 对用户不透明，违背了 serverless 的设计初衷 (transparent)。

我们通过对 Microsoft Azure 的无服务器计算服务进行分析，拟合函数的数据读写模式。并且以此为基础，在 Azure 上实现了 Faa$T。实验表明 Faa$T 可以提升 serverless 函数的性能平均达到 57%，最高可以达到 92%。并且，与已有的通过外部存储作为 caching 系统的方法相比，Faa$T可以降低绝大多数的开销。

## 4. ServerMore

ServerMore: Opportunistic Execution of Serverless Functions in the Cloud

因为 serverless 函数的执行特点是短执行时间与低资源需求，因此 ServerMore 尝试将 serverless 函数与 serverful 的虚拟机调度在同一台物理机上执行任务 (co-locating)。ServerMore 的整体目标是保证 serverful 工作负载的性能损失在可接受的范围内，同时执行 serverless  函数，从而提高系统的资源利用率。

ServerMore 在实现 co-locating 的过程中，主要解决以下的挑战： 从**性能**角度，一要保证将 serverless 函数调度到运行着 serverful 的虚拟机的物理机上时，不会对虚拟机上的任务有过多的性能影响；二是 co-locating 一定会引起 CPU, memory bandwidth 与 cache 的资源竞争，因此还需要协调资源间的共享；三是考虑性能的可预测性，对于虚拟机任务 ServerMore 实现了一套性能预测的机制。从系统**自适应性**的角度，从co-locating 一方面要考虑 serverful 的虚拟机任务的 arrival 模式，另一方面要考虑 serverless 函数的资源需求与执行时间的多样性。

## 总结

无服务器计算是几年来云计算的热点，除了云计算顶会 SoCC 以外，其他系统顶会也广泛涉及，例如 OSDI' 21 中的 *Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads* 将无服务器计算将无服务器计算与图神经网络结合。

本文不涉及以下三篇论文，有兴趣的同学可以自己研究：

1. *Tell me when you are sleepy and what may wake you up!* （与 serverless 的关联度不如其他论文紧密）

2. *Speedo: Fast dispatch and orchestration of serverless workflows*（论文的写作实在是拗口，“不忍卒读”）

3. *Atoll: A Scalable Low-Latency Serverless Platform*

