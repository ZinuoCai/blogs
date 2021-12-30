---
title: SoCC' 21 论文速读——ML Session
date: 2021-11-13 23:44:15
tags:
---

## TL; DR

SoCC 是云计算领域的顶会。2021 年 SoCC 共计录用 46 篇论文，其中机器学习有两个 Session，名称为 **Efficient, Robust ML Training and Inference**，共收录 8 篇论文。<!--more-->本文对其中的 7 篇论文进行简要的介绍，个人认为 *Clamor: Extending Functional Cluster Computing Frameworks with Fine-Grained Remote Memory Access* 的贡献点与机器学习的关联度不大，暂且略过。


在这 7 篇论文中，Chronus[1]、SEER[2] 面向机器学习训练任务，Morphling[3]、Lorien[4] 面向部署任务，他们都在探讨在不同的目标下，如何优化资源的利用以及模型的性能。Scrooge[5] 和 Llama[6] 讨论了如何在云计算的场景下部署视频应用；值得一提的是 Llama[6] 将无服务器计算应用到视频应用中，个人认为它更适合在 Serverless 的 Session。最后一篇 SIREN[7] 讨论了联邦学习中的攻击与防御问题，是我们实验室研二学长的工作。完整的论文列表链接 <https://dblp.uni-trier.de/db/conf/cloud/socc2021.html>。

## 1. Chronus

Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs

现代的 GPU 集群以分布式训练的方式支持深度学习任务 (DLT, deep learning task) 的训练。高效的任务调度是优化模型性能、提高系统资源利用率以及保证不同用户的公平性的关键因素。不同的训练任务有不同的目标 (SLO, severice-level objective 或者 best-effort) 以及资源需求。然而，目前对于如何能够高效的满足不同训练任务的需求还没有广泛的研究。

我们提出了 Chronus，它是一个端到端的调度系统，确保 SLO job 在 deadline 之前完成任务，对于 best-effort job 则优化模型性能。Chronus 基于深度学习任务的以下特征进行设计：

- 利用训练任务执行时间的可预测性，对深度学习任务进行分析，并且估计任务在动态的资源条件下的执行时间。
- 利用训练任务执行抢占的特点，设计 lease-based 的调度方案选择训练任务。
- 考虑训练任务的位置敏感性 (placement sensitivity)，设计新的本地搜索策略分配资源。

在真实数据集上的模拟实验表明：对于  SLO job，Chronus 可以降低 miss rate 14.7 倍；对于 best-effort job，Chronus  可以减少任务完成时间 19.9 倍。我们还将 Chronus 部署在了一个拥有 120 块 GPU 的Kubernetes 集群中验证它的可行性。

## 2. SEER

Elastic Hyperparameter Tuning on the Cloud

超参数选择是训练与部署机器学习模型重要的一环。之前超参数选择的工作假设系统资源有限，如何在较短的时间提高模型的准确率。在云计算的场景下，有限资源限制的问题得到缓解，但优化模型的准确率的同时，考虑时间和金钱预算非常必要。相关工作 (HyperSched) 研究了对于弹性的系统资源，如何降低超参数调节的金钱开销，但他们没有考虑保证模型准确率的问题。

我们工作的目标是在指定时间和金钱预算的条件下，研究如何调节超参数提高模型准确率。我们提出 SEER (Sequential Elimination with Elastic Resources)。SEER 算法初始阶段测试不同的超参数数值，并且保证可能的最佳参数能够在时间限制的 deadline 之前能得到充足的训练。与有限资源的训练方案不同，SEER 能够利用系统资源弹性分配的特点，避免 sublinear scaling 的副作用。而且，SEER 可以被容易地集成到已有的系统中，而且对于工作负载的类型不敏感。

## 3. Morphling

Morphling: Fast, Near-Optimal Auto-Configuration for Cloud-Native Model Serving

机器学习推理系统已经广泛地部署在了云计算的场景中。高效的推理方案需要优化硬件参数 (GPU Type, GPU Memory, Timeshare) 与运行时配置参数 (batch size)。但是，现有的自动配置参数的方案，例如 Bayesian optimization 与 white-box prediction 在面向高维的参数搜索空间时是低效的，带来的问题则是需要对硬件参数与运行时参数进行过多的采样。

我们提出了 Morphling，它使用 meta-learning 的方法对云原生的模型部署服务进行参数的搜索。Morphling 首先训练一个元模型来收集模型在不同的参数配置下的表现；然后，对于新的模型推理部署服务，Morphing 对参数配置空间进行少量采样，结合元模型可以迅速找到最佳的配置方案。我们在 Kubernetes 中实现了 Morphing 并且使用流行的 CV 与 NLP 模型进行测试。实施结果表明，Morphing 可以降低搜索开销 3 到 22 倍，在 720 种可能的配置空间中，仅需要对其中 30 种进行采样就可以找到最佳的配置方案。

## 4. Lorien

Lorien: Efficient Deep Learning Workloads Delivery

面对快速变化的深度学习算子与不断涌现的硬件平台，现代深度学习系统使用“编译”的思想来自动产生部署在不同硬件平台的代码，并且寻找代码高效执行的策略。为了保证这些自动产生的代码的高效性，我们会通过一些 auto-tuning 的框架来找到合适的算子的执行策略，但 auto-tuning 往往是个耗时的过程。

因此我们提出了 Lorien，系统地 tune 深度学习的算子，组织它们的执行策略。Lorien 首先用 Gluon 的 CV 模型库中 29 个常用的深度学习模型，组成了上千种算子层级的 tuning 任务，并且在 X86 CPU、ARM CPU 与 NVIDIA GPU 上 tune，构建了一个 tuning 的数据库。然后，为了保证能够为没有出现过的执行任务同样在秒或者分钟的时间级别内生成合适的算子调度方案，Lorien 集成了 AutoML 方法，使用已经收集到的数据训练了一个 cost model。

我们的实验结果表明 AutoML 方法生成的调度方案，不需要再重新 fine-tune 或者在硬件设备上进行实际的测量。与已有的 auto-tuning 框架相比，Lorien 可以节省至少 10 倍的时间就能够找到最优的调度方案。 

## 5. Scrooge

Scrooge: A Cost-Effective Deep Learning Inference System

深度学习的发展促进了使用云计算资源实时处理音视频等多媒体流的应用。这些应用必须满足吞吐量与时延的目标，而且应当应对不同的动态性的考量，而且还要尽可能低的金钱开销。

我们提出了 Scrooge 来提供 media-application-as-a-service。Scrooge 的实现策略包括 (1) 将计算任务打包并在 GPU-equipped 的云上虚拟机执行；(2) 使用 optimization formulation 找到最低廉的虚拟机使用方案；(3) 快速应对输入视频流的复杂性；

我们的实验表明相对于 SOTA 的解决方案，Scrooge 可以减少预算 16-32%。在动态工作负载的条件下，Scrooge 可以满足 98% 任务的时延要求。

## 6. Llama

Llama: A Heterogeneous & Serverless Framework for Auto-Tuning Video Analytics Pipelines

视频数据的增长导致视频处理应用的兴起。视频处理操作包括视频预处理、元数据提取、VQA (video question answering)等。这些操作组成的 DAG (directed acyclic graph，有向无环图) 形成了视频处理的 pipeline。通过调节采样率、批处理大小以及硬件种类可以优化视频处理的时延与系统资源的使用率。但是，找到高效的超参数往往是 challenging 的，主要由于 (1) 搜索空间较大；(2) 最佳的参数与用户自定义的时延与金钱预算等参数相关；(3) 不同的视频输入内容会产生不一样的中间结果，因此影响 DAG 图上的分支选择。当前的视频处理系统需要用户手动配置超参数并且选择硬件类型。

我们提出了 LLAMA，一个异构的 (heterogeneous，LLAMA 考量了 CPU 和 GPU 两种硬件资源)、基于无服务器计算的视频处理框架。指定视频处理的总时延目标，LLANA 通过 (1) 为 DAG 中的每一个子操作计算时延目标； (2) 动态执行 cost-based 优化器选择合适的硬件满足时延目标。LLAMA 在拥有 CPU 与 GPU 计算资源的无服务器计算框架中进行实验，实验结果表明，LLAMA 可以降低平均时延 7.8 倍，减少金钱开销 16 倍。 

## 7. SIREN

Siren: Byzantine-robust Federated Learning via Proactive Alarming

联邦学习的诞生主要是为了解决分布式机器学习中的数据隐私问题。但是，分布式的训练方式使得联邦学习易受不同种类的攻击。当前应对这些攻击的防御主要是利用对于模型权重的分析来鉴别恶意的客户端。这种方案面临很多的限制，例如需要提前知道恶意客户端的数量。但是在实际的应用场景中，这种假设是不现实的。

我们的工作提出了 SIREN，一个通过主动报警机制来应对拜占庭攻击的联邦学习系统。与已有的防御方案相比，SIREN 可以在保证全局模型性能优化的同时，防御更高比例的恶意客户端的攻击。通过在 IID 与 non-IID 的数据集上进行不同方式的攻击，我们的实验证明了 SIREN 防御方案的有效性。

