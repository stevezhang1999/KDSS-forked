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

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

using std::endl;

extern std::shared_ptr<nvinfer1::IGpuAllocator> global_allocator;
extern uint64_t alignment;

class cudaStreamDelegate
{
public:
    cudaStreamDelegate()
    {
        int result = 0;
        check_cuda_success(cudaStreamCreate(&stream), result);
        if (result != 0)
            std::terminate();
    }
    ~cudaStreamDelegate()
    {
        int result = 0;
        check_cuda_success(cudaStreamDestroy(stream), result);
        if (result != 0)
            std::terminate();
    }
    cudaStream_t get() const { return stream; }

private:
    cudaStream_t stream = nullptr;
};



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
    size_t free, total;
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
        gLogInfo << "Compute on device " << device << "." << endl;
    }
    EngineInfo ef;
    nvinfer1::ICudaEngine *engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    if (ctx == nullptr)
    {
        // ???table???????????????????????????
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
        // gLogWarning << "Call Compute with own context has been depreated. To reuse context, use ComputeWithoutExecDeviceMemory instead." << endl;
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

    // ???Engine??????????????????
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, before allocate: " << "free " << free << " total " << total << endl;
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, after allocate: " << "free " << free << " total " << total << endl;
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
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, after set: " << "free " << free << " total " << total << endl;

    // ????????????ef??????????????????InputName???OutputName?????????????????????
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers??????????????????????????????????????????

    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // ??????host???input???device???input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // ?????????????????????????????????
        int input_i_index = ef.InputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[input_i_index] = input[i];
#ifdef __DEBUG
        gLogInfo << "Input " << i << "'s device address:" << buffers.get()[input_i_index] << endl;
#endif
    }

    // ??????device???output?????????
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // ?????????????????????????????????
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[output_i_index] = output[i];
#ifdef __DEBUG
        gLogInfo << "Output " << i << "'s device address:" << buffers.get()[output_i_index] << endl;
#endif
    }
    // ????????????
    bool status;
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, before execute: " << "free " << free << " total " << total << endl;
    if (ctx == nullptr)
    {
#if NV_TENSORRT_MAJOR < 7
        status = context->execute(1, buffers.get());
#else
        status = context->executeV2(buffers.get());
#endif
    }
    else
    {
#if NV_TENSORRT_MAJOR < 7
        status = ctx->execute(1, buffers.get());
#else
        status = ctx->executeV2(buffers.get());
#endif
    }

    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, after execute: " << "free " << free << " total " << total << endl;

    // ??????????????????
    allocator->free(execution_memory);

    return 0;
}

int ComputationWorker::Compute(std::string model_name, void **input, void **(&output))
{
    return this->Compute(model_name, input, output, global_allocator.get());
}

int ComputationWorker::ComputeWithStream(std::string model_name, void **input, void **(&output), nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx = nullptr, EngineInfo *eInfo = nullptr)
{
    size_t free, total;
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
        // ???table???????????????????????????
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
        // gLogWarning << "Call ComputeWithStream with own context has been depreated. To reuse context, use ComputeWithStreamWithoutExecDeviceMemory instead." << endl;
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

    static cudaStreamDelegate stream;
    int res = 0;

    // ???Engine??????????????????
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, before allocate, stream: " << "free " << free << " total " << total << endl;
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    cudaMemGetInfo(&free, &total);
    gLogInfo << "In worker, after allocate, stream: " << "free " << free << " total " << total << endl;
    if (ctx == nullptr)
        context->setDeviceMemory(execution_memory);
    else
        ctx->setDeviceMemory(execution_memory);

    // ????????????ef??????????????????InputName???OutputName?????????????????????
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers??????????????????????????????????????????
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // ??????host???input???device???input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // ?????????????????????????????????
        int input_i_index = ef.InputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[input_i_index] = input[i];
#ifdef __DEBUG
        gLogInfo << "Input " << i << "'s device address:" << buffers.get()[input_i_index] << endl;
#endif
    }

    // ??????device???output?????????
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // ?????????????????????????????????
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[output_i_index] = output[i];
#ifdef __DEBUG
        gLogInfo << "Output " << i << "'s device address:" << buffers.get()[output_i_index] << endl;
#endif
    }

    // ??????????????????????????????CUDA???
    bool status;
    if (ctx == nullptr)
    {
#if NV_TENSORRT_MAJOR < 7
        status = context->enqueue(1, buffers.get(), stream.get(), nullptr);
#else
        status = context->enqueueV2(buffers.get(), stream.get(), nullptr);
#endif
    }
    else
    {
#if NV_TENSORRT_MAJOR < 7
        status = ctx->enqueue(1, buffers.get(), stream.get(), nullptr);
#else
        status = ctx->enqueueV2(buffers.get(), stream.get(), nullptr);
#endif
    }

    check_cuda_success(cudaStreamSynchronize(stream.get()), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not synchrnonize cuda stream."
                  << endl;
        return -1;
    }

    // ??????????????????
    allocator->free(execution_memory);
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
        gLogInfo << "Compute on device " << device << "." << endl;
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

    // ????????????ef??????????????????InputName???OutputName?????????????????????
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers??????????????????????????????????????????
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    // ??????host???input???device???input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // ?????????????????????????????????
        int input_i_index = ef.InputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[input_i_index] = input[i];
#ifdef __DEBUG
        gLogInfo << "Input " << i << "'s device address:" << buffers.get()[input_i_index] << endl;
#endif
    }

    // ??????device???output?????????
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // ?????????????????????????????????
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[output_i_index] = output[i];
#ifdef __DEBUG
        gLogInfo << "Output " << i << "'s device address:" << buffers.get()[output_i_index] << endl;
#endif
    }

    // ????????????
    bool status;
#if NV_TENSORRT_MAJOR < 7
    status = ctx->execute(1, buffers.get());
#else
    status = ctx->executeV2(buffers.get());
#endif
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

    static cudaStreamDelegate stream_without_execmem;

    // ????????????ef??????????????????InputName???OutputName?????????????????????
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // buffers??????????????????????????????????????????
    CPUMemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed."
                  << endl;
        return -1;
    }

    int res = 0;

    // ??????host???input???device???input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // ?????????????????????????????????
        int input_i_index = ef.InputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[input_i_index] = input[i];
#ifdef __DEBUG
        gLogInfo << "Input " << i << "'s device address:" << buffers.get()[input_i_index] << endl;
#endif
    }

    // ??????device???output?????????
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // ?????????????????????????????????
        int output_i_index = ef.OutputNetworkIndex.at(i);
        // ???????????????????????????????????????
        buffers.get()[output_i_index] = output[i];
#ifdef __DEBUG
        gLogInfo << "Output " << i << "'s device address:" << buffers.get()[output_i_index] << endl;
#endif
    }

    // ??????????????????????????????CUDA???
    bool status;
    // status = ctx->enqueue(1, buffers.get(), stream, nullptr);
#if NV_TENSORRT_MAJOR < 7
    status = ctx->enqueue(1, buffers.get(), stream_without_execmem.get(), nullptr);
#else
    status = ctx->enqueueV2(buffers.get(), stream_without_execmem.get(), nullptr);
#endif
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!"
                  << endl;
        return -1;
    }

    check_cuda_success(cudaStreamSynchronize(stream_without_execmem.get()), res);

    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not synchrnonize cuda stream."
                  << endl;
        return -1;
    }

    return 0;
}

void *ContextSetDeviceMemory(nvinfer1::IExecutionContext *ctx, nvinfer1::IGpuAllocator *allocator)
{
    if (ctx == nullptr || allocator == nullptr)
    {
        gLogError << __CXX_PREFIX << "Get nullptr." << endl;
        return nullptr;
    }
    // ???Engine??????????????????
    uint64_t execute_memory_size = ctx->getEngine().getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    if (!execution_memory)
    {
        gLogError << __CXX_PREFIX << "Can not allocate execution memory for context " << ctx->getName() << " execution."
                  << endl;
        return nullptr;
    }
    ctx->setDeviceMemory(execution_memory);
    return execution_memory;
}
// end of computation_worker.cpp