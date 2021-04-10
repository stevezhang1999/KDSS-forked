#include "computation_worker.hpp"
#include "transfer_worker.hpp" // for alignment
#include "utils.h"
#include "common.hpp"
#include "common/logger.h" // On TensorRT/sample
#include "common/common.h" // On TensorRT/samples
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

using std::endl;

extern std::shared_ptr<nvinfer1::IGpuAllocator> global_allocator;
extern uint64_t alignment;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

ComputationWorker::ComputationWorker()
{
    // do nothing
}

std::string ComputationWorker::GetModelName(int index) const
{
    mt_rw_mu.rlock();
    auto iter = model_table.find(index);
    mt_rw_mu.runlock();
    if (iter == model_table.end())
        return "";
    return iter->second;
}

int ComputationWorker::LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type)
{
    throw "LoadModel method not supported.";
    return -1;
}

int ComputationWorker::UnloadModel(std::string model_name)
{
    throw "UnloadModel method not supported.";
    return -1;
}

int ComputationWorker::TransferInput(std::string model_name, const std::vector<std::vector<char>> input_data, void **(&input_ptr), nvinfer1::IGpuAllocator *allocator)
{
    throw "TransferInput method not supported.";
    return -1;
}

int ComputationWorker::TransferOutput(std::string model_name, void **output_ptr, std::vector<std::vector<char>> &output_data, nvinfer1::IGpuAllocator *allocator)
{
    throw "TransferOutput method not supported.";
    return -1;
}

int ComputationWorker::Compute(std::string model_name, void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx = nullptr, EngineInfo *eInfo = nullptr)
{
    static int FirstComputeExecute = 0;
    if (FirstComputeExecute == 0)
    {
        FirstComputeExecute = 1;
        int device;
        int result;
        check_cuda_success(cudaGetDevice(&device), result);
        if (result != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get current executing device, compute aborted."
                      << endl;
            return -1;
        }
        gLogInfo << "Compute on device " << device << ".\n";
    }
    EngineInfo ef;
    nvinfer1::ICudaEngine *engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    if (ctx == nullptr)
    {
        // 从table中取得已注册的引擎
        int executed = GetModel(model_name, &ef);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get model" << model_name << endl;
            return -1;
        }

        engine = ef.engine.get();
        if (!engine)
        {
            gLogError << __CXX_PREFIX << "engine not vaild."
                      << endl;
            return -1;
        }

        context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
        if (!context)
        {
            gLogError << __CXX_PREFIX << "engine start failed, context error."
                      << endl;
            return -1;
        }
    }
    else
    {
        context = nullptr;
        if (eInfo == nullptr)
        {
            gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..."
                      << endl;
            return -1;
        }
        ef = *eInfo;
        engine = ef.engine.get();
    }

    // 为Engine分配执行显存
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    if (!execution_memory)
    {
        gLogError << __CXX_PREFIX << "Can not allocate execution memory for model " << model_name << " execution."
                  << endl;
        return -1;
    }
    if (ctx == nullptr)
        context->setDeviceMemory(execution_memory);
    else
        ctx->setDeviceMemory(execution_memory);

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers是从外面来的，无需管理其显存
    
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[input_i_index] = input[i];
    }

    // 处理device端output的映射
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[output_i_index] = output[i];
    }

    // 执行模型
    bool status;
    if (ctx == nullptr)
    {
        status = context->execute(1, buffers.get());
    }
    else
    {
        status = ctx->execute(1, buffers.get());
    }

    // 释放执行显存
    allocator->free(execution_memory);

    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }

    return 0;
}

int ComputationWorker::Compute(std::string model_name, void **input, void **(&output))
{
    return this->Compute(model_name, input, output, global_allocator.get());
}

int ComputationWorker::ComputeWithStream(std::string model_name, void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx = nullptr, EngineInfo *eInfo = nullptr)
{
    static int FirstStreamComputeExecute = 0;
    if (FirstStreamComputeExecute == 0)
    {
        FirstStreamComputeExecute = 1;
        int device;
        int result;
        check_cuda_success(cudaGetDevice(&device), result);
        if (result != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get current executing device, compute aborted."
                      << endl;
            return -1;
        }
        gLogInfo << "Compute with CUDA stream on device " << device << endl;
    }
    EngineInfo ef;
    nvinfer1::ICudaEngine *engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    if (ctx == nullptr)
    {
        // 从table中取得已注册的引擎
        et_rw_mu.rlock();
        auto iter = engine_table.find(model_name);
        et_rw_mu.runlock();
        if (iter == engine_table.end())
        {
            gLogError << __CXX_PREFIX << "engine not vaild."
                      << endl;
            return -1;
        }
        ef = iter->second;
        engine = ef.engine.get();
        if (!engine)
        {
            gLogError << __CXX_PREFIX << "engine not vaild."
                      << endl;
            return -1;
        }

        context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
        if (!context)
        {
            gLogError << __CXX_PREFIX << "engine start failed, context error."
                      << endl;
            return -1;
        }
    }
    else
    {
        context = nullptr;
        if (eInfo == nullptr)
        {
            gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..."
                      << endl;
            return -1;
        }
        ef = *eInfo;
        engine = ef.engine.get();
    }

    // 为Engine分配执行显存
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    if (ctx == nullptr)
        context->setDeviceMemory(execution_memory);
    else
        ctx->setDeviceMemory(execution_memory);

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers是从外面来的，无需管理其显存
    
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // 创建CUDA stream
    cudaStream_t stream;
    int res = 0;
    check_cuda_success(cudaStreamCreate(&stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not create cuda stream."
                  << endl;
        return -1;
    }

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[input_i_index] = input[i];
    }

    // 处理device端output的映射
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[output_i_index] = output[i];
    }

    // 将模型计算任务加入到CUDA流
    bool status;
    if (ctx == nullptr)
        status = context->enqueue(1, buffers.get(), stream, nullptr);
    else
        status = ctx->enqueue(1, buffers.get(), stream, nullptr);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }

    check_cuda_success(cudaStreamSynchronize(stream), res);

    // 释放执行显存
    allocator->free(execution_memory);

    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not synchrnonize cuda stream."
                  << endl;
        return -1;
    }

    // 销毁CUDA stream
    check_cuda_success(cudaStreamDestroy(stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not destroy cuda stream."
                  << endl;
        return -1;
    }
    return 0;
}

int ComputationWorker::ComputeWithStream(std::string model_name, void **input, void **(&output))
{
    return this->ComputeWithStream(model_name, input, output, global_allocator.get());
}

int ComputationWorker::ComputeWithoutExecDeviceMemory(void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo)
{
    static int FirstComputeWithoutExecDeviceMemory = 0;
    if (FirstComputeWithoutExecDeviceMemory == 0)
    {
        FirstComputeWithoutExecDeviceMemory = 1;
        int device;
        int result;
        check_cuda_success(cudaGetDevice(&device), result);
        if (result != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get current executing device, compute aborted."
                      << endl;
            return -1;
        }
        gLogInfo << "Compute on device " << device << ".\n";
    }
    EngineInfo ef;
    nvinfer1::ICudaEngine *engine;
    if (eInfo == nullptr)
    {
        gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..."
                  << endl;
        return -1;
    }
    ef = *eInfo;
    engine = ef.engine.get();

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers是从外面来的，无需管理其显存
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[input_i_index] = input[i];
    }

    // 处理device端output的映射
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[output_i_index] = output[i];
    }

    // 执行模型
    bool status;
    status = ctx->execute(1, buffers.get());
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }

    return 0;
}

int ComputationWorker::ComputeWithStreamWithoutExecDeviceMemory(void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx, EngineInfo *eInfo)
{
    static int FirstComputeWithStreamWithoutExecDeviceMemory = 0;
    if (FirstComputeWithStreamWithoutExecDeviceMemory == 0)
    {
        FirstComputeWithStreamWithoutExecDeviceMemory = 1;
        int device;
        int result;
        check_cuda_success(cudaGetDevice(&device), result);
        if (result != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get current executing device, compute aborted."
                      << endl;
            return -1;
        }
        gLogInfo << "Compute with CUDA stream on device " << device << endl;
    }
    EngineInfo ef;
    nvinfer1::ICudaEngine *engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    if (eInfo == nullptr)
    {
        gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..."
                  << endl;
        return -1;
    }
    ef = *eInfo;
    engine = ef.engine.get();

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers是从外面来的，无需管理其显存
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // 创建CUDA stream
    cudaStream_t stream;
    int res = 0;
    check_cuda_success(cudaStreamCreate(&stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not create cuda stream."
                  << endl;
        return -1;
    }

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[input_i_index] = input[i];
    }

    // 处理device端output的映射
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // 将已确定的显存地址分配过去
        buffers.get()[output_i_index] = output[i];
    }

    // 将模型计算任务加入到CUDA流
    bool status;
    status = ctx->enqueue(1, buffers.get(), stream, nullptr);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }

    check_cuda_success(cudaStreamSynchronize(stream), res);

    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not synchrnonize cuda stream."
                  << endl;
        return -1;
    }

    // 销毁CUDA stream
    check_cuda_success(cudaStreamDestroy(stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not destroy cuda stream."
                  << endl;
        return -1;
    }
    return 0;
}
// end of computation_worker.cpp