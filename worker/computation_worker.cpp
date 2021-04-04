#include "computation_worker.hpp"
#include "common.hpp"
#include "common/logger.h" // On TensorRT/sample
#include "common/common.h" // On TensorRT/samples
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

using std::endl;

extern std::shared_ptr<nvinfer1::IGpuAllocator> kg_allocator;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

void *WrapInput(void *host_memory, uint64_t size);
void *UnwrapOutput(void *device_memory);

std::string ComputationWorker::GetModelName(int index) const
{
    mt_rw_mu.rlock();
    auto iter = model_table.find(index);
    mt_rw_mu.runlock();
    if (iter == model_table.end())
        return "";
    return iter->second;
}

void *ComputationWorker::Compute(std::string model_name, void *input)
{
    // 从table中取得已注册的引擎
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }

    IHostMemory *h_memory = engine->serialize();
    std::string serialize_str;
    serialize_str.resize(h_memory->size());
    memcpy((void *)serialize_str.data(), h_memory->data(), h_memory->size());

    // 判断是否和ef里面储存的序列化结果相等？
    if (serialize_str == iter->second.engine_serialize)
    {
        gLogInfo << "Engine matched!" << endl;
    }
    else
    {
        gLogError << "Engine not matched!" << endl;
        // 执行反序列化
        engine = createInferRuntime(gLogger)->deserializeCudaEngine(iter->second.engine_serialize.data(), iter->second.engine_serialize.size());
    }

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        gLogError << "engine start failed, context error." << endl;
        return nullptr;
    }
    int input_index = engine->getBindingIndex(iter->second.InputName.c_str());
    int output_index = engine->getBindingIndex(iter->second.OutputName.c_str());

    void *d_input = WrapInput(input, iter->second.InputSize);
    // void *d_input;
    // cudaMalloc(&d_input, 28 * 28 * sizeof(float));
    // cudaMemcpy(d_input, input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);

    void *buffers[2];
    buffers[input_index] = d_input;
    void *d_output = kg_allocator->allocate(iter->second.OutputSize, 0, 0);
    if (!d_output)
    {
        gLogError << "allocate output memory failed." << endl;
        return nullptr;
    }
    buffers[output_index] = d_output;
    bool status = context->execute(1, buffers);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
        return nullptr;
    }
    void *h_output = UnwrapOutput(d_output);
    // float *output = new float[10];
    // cudaError_t err = cudaMemcpy(output, buffers[output_index], sizeof(float) * 10, cudaMemcpyDeviceToHost);
    return h_output;
}

// GetModelInputSize 获取指定模型的输入大小
uint64_t ComputationWorker::GetModelInputSize(std::string model_name) const
{
    uint64_t res = 1;
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return 0;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return 0;
    }
    int input_index = engine->getBindingIndex(iter->second.InputName.c_str());
    auto inputDims = engine->getBindingDimensions(input_index);
    for (int i = 0; i < inputDims.nbDims; i++)
    {
        res *= inputDims.d[i];
    }
    return res;
}

// GetModelOutputSize 获取指定模型的输出大小
uint64_t ComputationWorker::GetModelOutputSize(std::string model_name) const
{
    uint64_t res = 1;
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return 0;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return 0;
    }
    int output_index = engine->getBindingIndex(iter->second.OutputName.c_str());
    auto outputDims = engine->getBindingDimensions(output_index);
    for (int i = 0; i < outputDims.nbDims; i++)
    {
        res *= outputDims.d[i];
    }
    return res;
}

// GetModelInputDim 获取指定模型的输入总大小
const int *ComputationWorker::GetModelInputDim(std::string model_name) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }
    int input_index = engine->getBindingIndex(iter->second.InputName.c_str());
    return engine->getBindingDimensions(input_index).d;
}

// GetModelOutputDim 获取指定模型的输出总大小
const int *ComputationWorker::GetModelOutputDim(std::string model_name) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return nullptr;
    }
    int output_index = engine->getBindingIndex(iter->second.OutputName.c_str());
    return engine->getBindingDimensions(output_index).d;
}

void *WrapInput(void *host_memory, uint64_t size)
{
    void *res = kg_allocator->allocate(size, 0, 0);
    int result = 0;
    check_cuda_success(cudaMemcpy(res, host_memory, size, cudaMemcpyHostToDevice), result);
    if (result != 0)
    {
        gLogError << "Wrap output failed." << endl;
        return nullptr;
    }
    return res;
}

void *UnwrapOutput(void *device_memory)
{
    uint64_t size = dynamic_cast<KGAllocator *>(kg_allocator.get())->GetDeviceMemorySize(device_memory);
    // int *h_ptr = static_cast<int *>((void *)new char[size]);
    float *h_ptr = new float[size / sizeof(float)];
    memset(h_ptr, 0, size);
    int result = 0;
    check_cuda_success(cudaMemcpy(h_ptr, device_memory, size, cudaMemcpyDeviceToHost), result);
    if (result != 0)
        gLogError << "Unwrap output failed." << endl;
    kg_allocator->free(device_memory);
    return h_ptr;
}

EngineInfo *ComputationWorker::GetModel(std::string model_name) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    if (iter == engine_table.end())
    {
        et_rw_mu.runlock();
        return nullptr;
    }
    et_rw_mu.runlock();
    return new EngineInfo(iter->second);
}
// end of computation_worker.cpp