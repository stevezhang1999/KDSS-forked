#pragma once

#include "base.hpp"
#include "trt_allocator.hpp"
#include "common.hpp"
#include <exception>
#include <NvInfer.h>

using namespace nvinfer1;

class ComputationWorker final : public IWorker
{
public:
    ComputationWorker();
    virtual ~ComputationWorker() {}

    // LoadModel 加载模型到显存中
    //
    // ComputationWorker 并不负责这些工作，会直接throw
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，为ONNX文件
    // \param file_path 模型文件的路径
    // \param type 输入流代表的实际类型
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    virtual int LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type);

    // UnLoadModel 从显存中卸载模型
    //
    // ComputationWorker 并不负责这些工作，会直接throw
    // \param model_name 模型名称，该名称是在LoadModel时指定的。
    // \returns 如果卸载不成功或模型不存在，将会返回-1，并在logger中输出错误信息
    virtual int UnloadModel(std::string model_name);

    // TransferInput 将输入从内存转移到显存，并申请对应的显存
    //
    // ComputationWorker 并不负责这些工作，会直接throw
    // \param model_name 该输入对应的模型名称
    // \param input_data 输入的字节流载荷
    // \param input_ptr 该输入对应的显存地址数组指针
    // \param allocator 转移到显存时需要用的分配器
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferInput(std::string model_name, const std::vector<std::vector<char>> input_data, void **(&input_ptr), nvinfer1::IGpuAllocator *allocator);

    // TransferOutput 将输出从显存转移到内存，并释放对应的显存
    //
    // ComputationWorker 并不负责这些工作，会直接throw
    // \param model_name 该输出对应的模型名称
    // \param output_ptr 该输出对应的显存地址数组指针
    // \param output_data 输出的字节流载荷
    // \param allocator 分配时用的allocator，必须是Compute使用的allocator
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferOutput(std::string model_name, void **output_ptr, std::vector<std::vector<char>> &output_data, nvinfer1::IGpuAllocator *allocator);

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    virtual std::string GetModelName(int index) const;

    // Compute 开始根据模型执行计算。
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int Compute(std::string model_name, void **input, void **(&output));

    // Compute 开始根据模型执行计算。
    // 此过程会根据模型引擎创建上下文，使用allocator分配输入/输出显存及运行时显存，
    // 并在退出时将其全部销毁。
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \param allocator 分配显存使用的allocator
    // \param ctx 执行需要用的上下文。如果ctx为nullptr，则从引擎表获取引擎构建上下文。当ctx不为nullptr时，model_name无效。
    // \param eInfo 使用给定上下文执行时需要自带的EngineInfo信息，当ctx为nullptr时，该值将被忽略。
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int Compute(std::string model_name, void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo);

    // ComputeWithoutExecDeviceMemory 根据上下文进行模型计算。
    // 此函数不会给上下文分配执行显存，如果传入的ctx没有预先分配显存，则会由TensorRT抛出错误。
    // 不建议使用此函数进行计算，除非确定ctx已分配好显存且该显存会被正确回收。
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \param allocator 分配输入/输出显存使用的allocator
    // \param ctx 执行需要用的上下文。如果ctx由CreateContextWithoutDeviceMemory构建且没有预先分配显存，则会导致未定义行为。
    // \param eInfo 使用给定上下文执行时需要自带的EngineInfo信息。
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int ComputeWithoutExecDeviceMemory(void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo);

    // ComputeWithStream 使用CUDA stream+global_allocator进行overlapped异步计算
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int ComputeWithStream(std::string model_name, void **input, void **(&output));

    // ComputeWithStream 使用CUDA stream+自定义分配器进行overlapped异步计算
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \param allocator 用于分配显存的分配器
    // \param ctx 执行需要用的上下文。如果ctx为nullptr，则从引擎表获取引擎构建上下文。当ctx不为nullptr时，model_name无效。
    // \param eInfo 使用给定上下文执行时需要自带的EngineInfo信息，当ctx为nullptr时，该值将被忽略。
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int ComputeWithStream(std::string model_name, void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo);

    // ComputeWithStreamWithoutExecDeviceMemory 根据上下文进行流式模型计算。
    // 此函数不会给上下文分配执行显存，如果传入的ctx没有预先分配显存，则会由TensorRT抛出错误。
    // 不建议使用此函数进行计算，除非确定ctx已分配好显存且该显存会被正确回收。
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \param allocator 分配输入/输出显存使用的allocator
    // \param ctx 执行需要用的上下文。如果ctx由CreateContextWithoutDeviceMemory构建且没有预先分配显存，则会导致未定义行为。
    // \param eInfo 使用给定上下文执行时需要自带的EngineInfo信息。
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int ComputeWithStreamWithoutExecDeviceMemory(void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo);
};

// end of computation_worker.hpp