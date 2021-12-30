---
title: 机器学习推理任务 GPU 显存使用分析
date: 2021-09-01 08:34:32
tags:
mathjax: true
---
## 1. 总体目标

使用 GPU 加速机器学习任务时，程序运行时常见的错误 GPU 的显存空间不足：`RuntimeError: CUDA Error: out of memory`。熟练的炼丹师往往会通过减小 batch size，降低程序对 GPU 显存的需求，使程序正常执行。但是，为什么减小 batch size 可以降低显存需求，是否能够提前预测程序需要的显存大小呢？本文以机器学习推理任务为例，通过简单的实际案例，对如上的问题进行回答。

<!-- more -->

## 2. 理论分析

### 2.1 谁占用了显存？

当前对机器学习通用的定义是 **机器学习 = 数据 + 模型 + 算法**。在训练任务中，我们通过在训练集中优化我们的算法，从而获得在验证集或者测试集上表现优异的模型。而在推理任务中，我们使用生成的模型在测试数据上进行前向计算，获得预测的结果。

**在推理任务中，占用 GPU 显存的主要包括三个部分：模型权重、输入输出以及中间结果。**

- 模型权重：深度学习模型往往是结构相似的 layer 堆砌而成，例如卷积层、池化层、全连接层、激活函数层。一些 layer 是有参数的，例如卷积层的参数是一个高维的卷积核，全连接层的参数是一个二维矩阵。也有一些 layer 没有参数，例如激活函数层、池化层等。
- 中间结果：前向计算中，前一层的输出对应后一层的输入，连接相邻两层的中间结果也需要 GPU 显存来保存。
- 输入输出：相较于模型权重和中间结果，输入输出占用的 GPU 显存比较小。

这一结论在 OSDI'20 的论文 [*Serving DNNs like Clockwork: Performance Predictability from the Bottom Up*](https://www.usenix.org/conference/osdi20/presentation/gujarati) 中也得到了验证。该论文在设计机器学习推理系统的显存管理模块时，将显存占用分为了 Workspace/IOCache/PageCache。这三部分分别对应于中间结果、输入输出和模型权重的显存占用。

### 2.2 占用了多少显存？

模型权重、中间结果和输入输出往往都是结构化的数据，在 PyTorch 和 MXNet 等深度学习框架中都是以高维数组的形式保存。因此，我们可以通过数学计算的方式直接求解显存使用情况。

我们以自定义的一个模型为例，该模型有三层卷积层和三层全连接层组成。卷积层的 channel/kernel size/padding/stride 等参数均在代码中体现。模型的输入为 3×224×224,我们设置的 batch size 为 32。我们逐步分析这些层的模型权重和输入输出占用的显存。

```Python
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd


def my_model_conv():
    my_model = nn.HybridSequential()
    my_model.add(
        nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1)),
        nn.Conv2D(channels=256, kernel_size=7, padding=(1, 1), strides=(7, 7)),
        nn.Conv2D(channels=512, kernel_size=7, padding=(1, 1), strides=(7, 7)),
        nn.Dense(4096, activation="relu"),
        nn.Dense(4096, activation="relu"),
        nn.Dense(1000)
    )
    return my_model


if __name__ == '__main__':
    my_model = my_model_conv()
    my_model.initialize()
    my_model.hybridize()
    X = nd.zeros((32, 3, 224, 224))
    y = my_model.forward(X)
    my_model.export("my_model_conv")
```

首先是最简单的输入输出的显存占用。模型的输入为 (32, 3, 224, 224)，参数的数量为 $num = 32 * 3 * 224 * 224$，每个浮点数需要 32 bit，也就是 4 bytes 的存储空间，因此输入的显存占用为 $num*4 \text{ bytes} = num*4 / 1024 / 1024 \text{ MiB} = 18.375 \text{ MiB}$。模型的输出为 (32, 1000)，显存空间为$0.12 \text{ MiB}$。

对于卷积层，输入的维度为$(N, C, H, W)$, 卷积核的 padding 为 $P$, stride 为 $S$, kernel size 为 $K$, 输出通道数为 $C\prime$。 那么它输出的 $H\prime$  为 $\frac{H + 2*P- K}{S} + 1$，输出的 $W\prime$  为 $\frac{W + 2*P- K}{S} + 1$，输出的维度为$(N, C\prime, H\prime, W\prime)$，卷积核的参数数量为 $C*C\prime*K^2$。因此对于第一个卷积层，它的输出通道数为 64, padding/stride 为 1，kernel size 为 3。所以

$$
H\prime = \frac{H + 2*P- K}{S} + 1 = (224 + 2 - 3) \div 1 + 1 = 224
$$

$$
W\prime = \frac{W + 2*P- K}{S} + 1 = (224 + 2 - 3) \div 1 + 1 = 224
$$

第一个卷积层的输出维度则为 (32, 64, 224, 224)。这对应的显存大小为数量为 $num = 32 * 64 * 224 * 224$，所有的输出占用的空间为 $num*4 \text{ bytes} = num*4 / 1024 / 1024 \text{ MiB} = 392 \text{ MiB}$。第一个卷积层的参数量为 $params = 3 * 64 * 3 *3$，占用的显存空间为$6.75 \text{ KiB}$。

而全连接的计算逻辑相对而言简单很多，在此不再赘述，可以参考以下链接进一步深入了解如何计算显存使用情况。

- [经典神经网络参数的计算【不定期更新】](https://zhuanlan.zhihu.com/p/49842046)
- [科普帖：深度学习中GPU和显存分析](https://zhuanlan.zhihu.com/p/31558973)

我们将自定义的模型汇总，得到结果如下：

![自定义模型显存使用汇总](/blogs/images/image-20210831191029151.png)

最终，我们得到理论上的计算结果：

- 模型权重的显存占用为 $0.01 + 3.06 + 24.5 + 128 + 64 + 15.63 = 235.2 \text{ MiB}$。
- 中间结果的显存占用为 $392 + 32 + 1 + 0.5 + 0.5 + 0.12 = 426.1 \text{ MiB}$。
- 输入输出的显存占用为 $18.38 + 0.12 = 18.5 \text{ MiB}$。

## 3. 代码分析

### 3.1 代码解读

上面模型定义的代码将导出两个文件 `my_model_conv-0000.params` 和 `my_model_conv-symbol.json`。前者是模型的权重，后者定义了模型的结果。我们继续使用 MXNet 导入我们定义好的模型，进行推理任务。

首先导入需要的 Python Package，并且自定义了一个 MemoryLogger。MemoryLogger 在初始化的时候会打印已经使用的显存、剩余显存和当前 GPU 的总的显存。每次调用 `MemoryLogger.log()` 函数时，会打印已经使用的显存、剩余显存和与上次相比又增加的显存。

```python
import logging

import GPUtil
import mxnet as mx
import mxnet.ndarray as nd

logging.getLogger().setLevel(logging.INFO)


class MemoryLogger:
    def __init__(self, device_id: int):
        gpu = GPUtil.getGPUs()[device_id]
        logging.info(f'Initializing: Used memory = {gpu.memoryUsed}MiB, Free memory = {gpu.memoryFree}MiB, '
                     f'Total memory = {gpu.memoryTotal}MiB')

        self.device_id = device_id
        self.currentMemoryUsed = gpu.memoryUsed
        self.allocatedMemory = gpu.memoryUsed
        self.context = mx.gpu(device_id)

    def log(self, mark: str):
        # synchronize
        nd.waitall()
        gpu = GPUtil.getGPUs()[self.device_id]
        freeMem = gpu.memoryFree
        usedMem = gpu.memoryUsed
        logging.info(f'{mark}: Allocate memory = {usedMem - self.allocatedMemory: .1f}MiB, '
                     f'Free memory = {freeMem: .1f}MiB, Increased memory = {usedMem - self.currentMemoryUsed: .1f}')
        self.currentMemoryUsed = usedMem
```

在主函数中，第一步是定义一些全局变量，更重要的是初始化 CUDA 上下文。我们初始化上下文的方法是在 GPU 上定义一个维度为 (1,) 的变量 `dummy_input`。这个变量在后续的代码中不会使用。

```python
gpu_ctx = mx.gpu()
input_shape = (32, 3, 224, 224)
memory_logger = MemoryLogger(0)

# step 1: initialize CUDA context
dummy_input = nd.zeros(shape=(1,), ctx=gpu_ctx)

memory_logger.log("Step 1 (initialize CUDA context)")
```

第二步则是定义模型的输入，模型的输入是一个 (32, 3, 224, 224) 的高维矩阵。

```python
# step 2: define input
from collections import namedtuple

inputs = nd.zeros(shape=input_shape, ctx=gpu_ctx)
inputs = inputs.as_in_context(gpu_ctx)
Batch = namedtuple('Batch', ['data'])
print(f'define input, theoretical memory used = {32 * 3 * 224 * 224 * 4 / 1024 / 1024: .3f} MiB')

memory_logger.log("Step 2 (define input)")
```

第三步则是初始化模型结构并且为模型分配空间。MXNet 可以通过 `mx.symbol.load` 方法读取上述的模型结构 `my_model_conv-symbol.json`文件来初始化模型结构，然后调用 `mx.mod.Module.bind` 方法为模型分配空间。注意，`mx.mod.Module.bind` 方法在申请显存是也会为输入数据申请显存，因此，该步打印的显存增加的数量比实际模型的占用空间多出一个模型输入需要的空间。最后通过`mx.mod.Module.set_params`方法将 CPU 端的模型参数拷贝到 GPU 的显存中。

```python
# step 3: load model
model_name = 'my_model_conv'
m = mx.mod.Module(symbol=mx.symbol.load(f"models/{model_name}-symbol.json"),
                  context=gpu_ctx,
                  data_names=["data"], label_names=[])
m.bind(for_training=False, data_shapes=[("data", input_shape)])

arg_params, aux_params = mx.model.load_params(f"models/{model_name}", 0)
m.set_params(arg_params, aux_params)

memory_logger.log("Step 3 (load model)")
```

最后一步是进行前向计算。

```python
m.forward(Batch([inputs]), is_train=False)
memory_logger.log("Step 4 (forward computation)")
```

### 3.2 结果分析

执行上述代码，我们得到了以下的输出：

![执行结果](/blogs/images/image-20210831194341135.png)

我们主要分析结果中的 **Increased memory** 的结果。

- Step 1 的上下文初始化阶段。MXNet 需要使用 641 MiB 的显存空间进行 CUDA 的上下文加载，实验显示 PyTorch 也需要类似的操作进行 GPU 上下文初始化。
- Step 2 在定义输入时，理论上 (32, 3, 224, 224) 的张量需要的空间为 18.375 MiB 的显存，实际上分配了 20 MiB 的显存。这主要和 MXNet 的显存管理机制相关。为了避免频繁调用 `cudaMalloc` 函数申请显存，MXNet 会维护一个 memory pool。因此，我们看到当需要 18.375 MiB 的显存时，申请的空间会略高，多余的空间有 MXNet 底层的显存管理机制进行维护。
- Step 3 在为模型权重分配显存，增加的显存量为 258 MiB，而理论上的显存应该为模型本身的显存 235.2 MiB 与 重复定义的模型输入的显存 18.375 MiB 之和  254 MiB。考虑到显存碎片问题，多出的 4 MiB 可在接受范围。
- Step 4 进行前向计算出的中间结果的显存与理论计算值相同。

## 4. 总结

本文分析了在机器学习推理任务中 GPU 的显存使用情况。通过结果可以发现，降低 batch size 来避免 out of memory 的问题主要是减少了输入输出和中间结果需要的显存量，而不会对模型权重需要的显存带来帮助。我们着重分析了推理阶段的显存使用情况，训练阶段由于反向传播的存在，GPU 显存的使用情况会更加复杂。