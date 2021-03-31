# KDSS - KarkLi's DNN Serve System
## 1.简介

KDSS（KarkLi's DNN Serve System）是一种类似于TensorFlow Serve System和PyTorch Serve System的统一部署系统，它可以完成模型在生产环境上的服务器的快速部署。通过简单的HTTP请求即可以对外提供模型的计算服务。

该系统有几大模块，自顶向下的顺序为：

（可用性：❌）```gateway``` —— 用于接收Client发送的请求，并进行参数合法性校验。同时也返回计算完成的结果。

（可用性：❌）```controller``` —— 用于接收由gateway预处理过的请求，并将其放入到调度队列中等待调度。当某个任务就绪时，controller便会唤起computation_worker进行计算。整个系统可以有多个transfer_worker，但只会有一个computation_worker。

（可用性：部分✅）```worker``` —— 用于提供对模型的操作，包括加载ONNX模型/反序列化已缓存的TensorRT引擎（如果选择TensorRT作为worker的底层实现），载入模型到显存中/从显存中卸载模型，传递输入/输出，执行某个模型等操作。

其中加载/反序列化模型文件、引擎，载入/卸载模型，传递输入/输出数据的worker我们称之为```transfer_worker```，根据用途的不同可以进一步细分```transfer_worker```，但在宏观上不作区分。而执行模型的worker称为```computation_worker```，整个系统只会有一个。

（可用性：✅）```kgmalloc``` —— 一个独立于该系统的GPU显存管理模块，```worker```层可基于其提供的原语进行显存管理。kgmalloc的地址：[kgmalloc](https://git.code.tencent.com/karkli/kgmalloc)。