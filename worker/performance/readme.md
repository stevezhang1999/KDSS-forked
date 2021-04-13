# worker层执行性能提升方法一览

performance文件夹下的所有例子，因为要做实验获取各种数据，所以并不严格遵循以下的性能提升方法。

## 1.使用流式计算

使用流式计算可以使得三个过程加以时序上的重叠，它们分别是：从内存传输数据到显存，逐层进行计算，从显存传输数据到内存。单请求（batch_size = 1）下性能可能并不明显，当批量请求（batch_size > 1）时，性能会有一定的提升。

查阅```computer_worker.hpp```中的```ComputeWithStream```和```ComputeWithStreamWithoutExecDeviceMemory```获取更多信息。

## 2.进行上下文重用

模型的执行伴随着TensorRT上下文的构建，而TensorRT上下文是根据引擎构建且可以重用的。构建TensorRT上下文非常地耗费时间（在RTX 2080 Ti测试环境下Resnet-50的上下文构建最大可以到9~12ms，占执行总过程的80%左右的时间），也是性能瓶颈之一。

查阅```compute_worker.hpp```中的```ComputeWithoutExecDeviceMemory```和```ComputeWithStreamWithoutExecDeviceMemory```获取更多信息。

在TensorRT 7中，上下文的执行显存不允许二次设置。TensorRT 6中这是允许的。因此，为了统一表现，在TensorRT 6和TensorRT 7中，均不允许复用```Compute```和```ComputeWithStream```消费过的IExecuteContext指针。

如果使用自定义ctx调用```Compute```及```ComputeWithStream```，请不要复用该ctx（也不用且不需要调用destroy销毁它）。

## 3.批量计算

批量计算（batch_size > 1）可以让GPU的吞吐量得到提升。目前KDSS从ONNX构建引擎时，限制batch_size = 1。后续会针对TensorRT 7修改API，增加batch_size参数，使用户可以自定义模型的batch_size。

从TensorRT引擎文件反序列化得到的引擎不支持设置batch_size参数。