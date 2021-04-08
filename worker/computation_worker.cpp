#include "computation_worker.hpp"
#include "utils.h"
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

void *WrapInput(void *host_memory, uint64_t size, IGpuAllocator *allocator = kg_allocator.get());
void *UnwrapOutput(void *device_memory, size_t size, IGpuAllocator *allocator = kg_allocator.get());

void *WrapInputAsync(void *host_memory, uint64_t size, IGpuAllocator *allocator, cudaStream_t stream);
void *UnwrapOutputAsync(void *device_memory, size_t size, IGpuAllocator *allocator, cudaStream_t stream);

struct MemoryDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            for (int i = 0; i < current_length; i++)
            {
                if (allocator)
                    allocator->free(obj[i]);
            }
            delete[] obj;
            obj = nullptr;
        }
    }
    // Current_length of obj
    size_t current_length = 0;

    // Current allocator
    IGpuAllocator *allocator = nullptr;
};

template <typename T>
using MemoryUniquePtr = std::unique_ptr<T, MemoryDeleter>;

ComputationWorker::ComputationWorker()
{
    cudaDeviceProp prop;
    int device;
    int result;
    check_cuda_success(cudaGetDevice(&device), result);
    if (result)
    {
        cerr << __CXX_PREFIX << "CUDA error." << endl;
        throw "";
    }
    check_cuda_success(cudaGetDeviceProperties(&prop, device), result);
    if (result)
    {
        cerr << __CXX_PREFIX << "CUDA error." << endl;
        throw "";
    }
    // See also: https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously
    alignment = prop.textureAlignment;
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

int ComputationWorker::Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output, nvinfer1::IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx = nullptr, EngineInfo *eInfo = nullptr)
{
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
            gLogError << __CXX_PREFIX << "engine not vaild." << endl;
            return -1;
        }
        ef = iter->second;
        engine = ef.engine.get();
        if (!engine)
        {
            gLogError << __CXX_PREFIX << "engine not vaild." << endl;
            return -1;
        }

        context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
        if (!context)
        {
            gLogError << __CXX_PREFIX << "engine start failed, context error." << endl;
            return -1;
        }
    }
    else
    {
        context = SampleUniquePtr<nvinfer1::IExecutionContext>(ctx);
        if (eInfo == nullptr)
        {
            gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..." << endl;
            return -1;
        }
        memcpy(&ef, eInfo, sizeof(EngineInfo));
    }

    // 为Engine分配执行显存
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    context->setDeviceMemory(execution_memory);

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // fix: input没有被free掉，对buffers使用智能指针
    MemoryUniquePtr<void *> buffers(new void *[(input_num + output_num)]);
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed." << endl;
        return -1;
    }

    buffers.get_deleter().current_length = input_num + output_num;
    buffers.get_deleter().allocator = allocator;

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        if (input_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            // buffers = (void **)realloc(buffers, sizeof(void *) * input_i_index);
            auto temp_buffers = buffers.release();
            void **new_buffers = new void *[input_i_index + 1];
            memmove(new_buffers, temp_buffers, sizeof(void *) * (input_num + output_num));
            buffers.reset(temp_buffers);
            delete[] temp_buffers;
        }
        uint64_t input_i_size;
        int res = this->GetModelInputSize(model_name, ef.InputName.at(i), &input_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get input size of : " << ef.InputName.at(i);
            return -1;
        }
        buffers.get()[input_i_index] = WrapInput(input[i].data(), input_i_size);
    }

    // 申请device端output空间
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        if (output_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            // buffers = (void **)realloc(buffers, sizeof(void *) * output_i_index);
            auto temp_buffers = buffers.release();
            temp_buffers = (void **)realloc(temp_buffers, sizeof(void *) * output_i_index + 1);
            buffers.reset(temp_buffers);
        }
        uint64_t output_i_size;
        int res = this->GetModelOutputSize(model_name, ef.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << ef.OutputName.at(i);
            return -1;
        }
        buffers.get()[output_i_index] = allocator->allocate(output_i_size, 0, 0);
    }

    // 执行模型
    bool status;
    status = context->execute(1, buffers.get());

    // 释放执行显存
    allocator->free(execution_memory);

    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
        return -1;
    }

    // 对output逐个unwrap
    void **h_output = (void **)malloc(sizeof(void *) * output_num);
    if (!h_output)
    {
        gLogError << __CXX_PREFIX << "Output allocation failed." << endl;
        return -1;
    }

    // 处理device端output到host端output
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = engine->getBindingIndex(ef.OutputName.at(i).c_str());

        uint64_t output_i_size;
        int res = this->GetModelOutputSize(model_name, ef.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << ef.OutputName.at(i);
            return -1;
        }
        h_output[i] = UnwrapOutput(buffers.get()[output_i_index], output_i_size);
        if (!h_output[i])
        {
            // 释放所有h_output[0,i-1]的内存
            for (int j = 0; j < i; j++)
            {
                free(h_output[j]);
            }
            free(h_output);
            h_output = nullptr;
            output.clear();
            return -1;
        }
        // 对h_output进行逐字节写入到result
        std::vector<char> temp;
        for (int j = 0; j < output_i_size; j++)
        {
            temp.push_back(*((char *)h_output[i] + j));
        }
        output.push_back(temp);
    }
    // free(buffers);
    // 释放h_output
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        free(h_output[i]);
    }
    free(h_output);
    return 0;
    // void *h_output = UnwrapOutput(d_output);
    // return h_output;
}

int ComputationWorker::Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output)
{
    // 使用默认分配器kg_allocator
    int device;
    int result;
    check_cuda_success(cudaGetDevice(&device), result);
    if (result != 0)
    {
        gLogError << "Can not get current executing device, compute aborted." << endl;
        return -1;
    }
    gLogInfo << "Compute on device " << device << endl;
    return this->Compute(model_name, input, output, kg_allocator.get());
}

int ComputationWorker::ComputeWithStream(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output, IGpuAllocator *allocator, nvinfer1::IExecutionContext *ctx = nullptr, EngineInfo *eInfo = nullptr)
{
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
            gLogError << __CXX_PREFIX << "engine not vaild." << endl;
            return -1;
        }
        ef = iter->second;
        engine = ef.engine.get();
        if (!engine)
        {
            gLogError << __CXX_PREFIX << "engine not vaild." << endl;
            return -1;
        }

        context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContextWithoutDeviceMemory());
        if (!context)
        {
            gLogError << __CXX_PREFIX << "engine start failed, context error." << endl;
            return -1;
        }
    }
    else
    {
        context = SampleUniquePtr<nvinfer1::IExecutionContext>(ctx);
        if (eInfo == nullptr)
        {
            gLogError << __CXX_PREFIX << "Compute with given context but without engine info, exiting..." << endl;
            return -1;
        }
        memcpy(&ef, eInfo, sizeof(EngineInfo));
    }

    // 为Engine分配执行显存
    uint64_t execute_memory_size = engine->getDeviceMemorySize();
    void *execution_memory = allocator->allocate(execute_memory_size, alignment, 0);
    context->setDeviceMemory(execution_memory);

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();
    void **buffers = (void **)malloc(sizeof(void *) * (input_num + output_num));
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed." << endl;
        return -1;
    }
    memset(buffers, 0, sizeof(void *) * (input_num + output_num));

    // 创建CUDA stream
    cudaStream_t stream;
    int res = 0;
    check_cuda_success(cudaStreamCreate(&stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not create cuda stream." << endl;
        return -1;
    }

    // 处理host端input到device端input
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        if (input_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            buffers = (void **)realloc(buffers, sizeof(void *) * input_i_index);
        }
        uint64_t input_i_size;
        int res = this->GetModelInputSize(model_name, ef.InputName.at(i), &input_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get input size of : " << ef.InputName.at(i);
            return -1;
        }
        buffers[input_i_index] = WrapInputAsync(input[i].data(), input_i_size, allocator, stream);
    }

    // 申请device端output空间
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = ef.OutputNetworkIndex.at(i);
        if (output_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            buffers = (void **)realloc(buffers, sizeof(void *) * output_i_index);
        }
        uint64_t output_i_size;
        int res = this->GetModelOutputSize(model_name, ef.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << ef.OutputName.at(i);
            return -1;
        }
        buffers[output_i_index] = allocator->allocate(output_i_size, 0, 0);
    }

    // 将模型计算任务加入到CUDA流
    bool status;
    status = context->enqueue(1, buffers, stream, nullptr);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
        return -1;
    }

    void **h_output = (void **)malloc(sizeof(void *) * output_num);
    if (!h_output)
    {
        gLogError << __CXX_PREFIX << "Output allocation failed." << endl;
        return -1;
    }

    // 处理device端output到host端output
    uint64_t output_i_size;
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = engine->getBindingIndex(ef.OutputName.at(i).c_str());
        int res = this->GetModelOutputSize(model_name, ef.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << ef.OutputName.at(i);
            return -1;
        }

        // 对每个output的unwrap加入到CUDA流中
        h_output[i] = UnwrapOutputAsync(buffers[output_i_index], output_i_size, allocator, stream);
    }

    check_cuda_success(cudaStreamSynchronize(stream), res);

    // 释放执行显存
    allocator->free(execution_memory);

    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not synchrnonize cuda stream." << endl;
        return -1;
    }

    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        if (!h_output[i])
        {
            // 释放所有h_output[0,i-1]的内存
            for (int j = 0; j < i; j++)
            {
                free(h_output[j]);
            }
            free(h_output);
            h_output = nullptr;
            output.clear();
            return -1;
        }
        // 对h_output进行逐字节写入到result
        std::vector<char> temp;
        for (int j = 0; j < output_i_size; j++)
        {
            temp.push_back(*((char *)h_output[i] + j));
        }
        output.push_back(temp);
    }

    // 释放h_output
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        free(h_output[i]);
    }
    free(h_output);

    // 销毁CUDA stream
    check_cuda_success(cudaStreamDestroy(stream), res);
    if (res != 0)
    {
        gLogError << __CXX_PREFIX << "Can not destroy cuda stream." << endl;
        return -1;
    }
    return 0;
}

int ComputationWorker::ComputeWithStream(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output)
{
    int device;
    int result;
    check_cuda_success(cudaGetDevice(&device), result);
    if (result != 0)
    {
        gLogError << "Can not get current executing device, compute aborted." << endl;
        return -1;
    }
    gLogInfo << "Compute on device " << device << endl;
    return this->ComputeWithStream(model_name, input, output, kg_allocator.get());
}

// GetModelInputSize 获取指定模型的输入总大小
int ComputationWorker::GetModelInputSize(std::string model_name, int index, uint64_t *result) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << __CXX_PREFIX << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;
    for (int i = 0; i < ef.InputDim.at(index).nbDims; i++)
    {
        if (ef.InputDim.at(index).d[i] == 0)
            continue;
        *result *= ef.InputDim.at(index).d[i];
    }
    // 还需要乘系数因子
    int factor = 1;
    switch (ef.InputType.at(index))
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        factor = sizeof(int8_t);
        break;
    }
    *result *= factor;

    return 0;
}

int ComputationWorker::GetModelInputSize(std::string model_name, std::string input_name, uint64_t *result) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << __CXX_PREFIX << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;

    // 先找到该输出的名称对应的索引
    auto input_iter = std::find(ef.InputName.begin(), ef.InputName.end(), input_name);
    if (input_iter == ef.InputName.end())
        return -1;
    int index = std::distance(ef.InputName.begin(), input_iter);

    *result = 1;
    for (int i = 0; i < ef.InputDim.at(index).nbDims; i++)
    {
        if (ef.InputDim.at(index).d[i] == 0)
            continue;
        *result *= ef.InputDim.at(index).d[i];
    }
    // 还需要乘系数因子
    int factor = 1;
    switch (ef.InputType.at(index))
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        factor = sizeof(int8_t);
        break;
    }
    *result *= factor;

    return 0;
}

// GetModelOutputSize 获取指定模型的输出总大小
int ComputationWorker::GetModelOutputSize(std::string model_name, int index, uint64_t *result) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << __CXX_PREFIX << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;
    for (int i = 0; i < ef.OutputDim.at(index).nbDims; i++)
    {
        if (ef.OutputDim.at(index).d[i] == 0)
            continue;
        *result *= ef.OutputDim.at(index).d[i];
    }
    // 还需要乘系数因子
    int factor = 1;
    switch (ef.OutputType.at(index))
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        factor = sizeof(int8_t);
        break;
    }
    *result *= factor;

    return 0;
}

// GetModelOutputSize 获取指定模型的输出总大小
int ComputationWorker::GetModelOutputSize(std::string model_name, std::string output_name, uint64_t *result) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << __CXX_PREFIX << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;

    // 先找到该输出的名称对应的索引
    auto output_iter = std::find(ef.OutputName.begin(), ef.OutputName.end(), output_name);
    if (output_iter == ef.OutputName.end())
        return -1;
    int index = std::distance(ef.OutputName.begin(), output_iter);

    *result = 1;
    for (int i = 0; i < ef.OutputDim.at(index).nbDims; i++)
    {
        if (ef.OutputDim.at(index).d[i] == 0)
            continue;
        *result *= ef.OutputDim.at(index).d[i];
    }
    // 还需要乘系数因子
    int factor = 1;
    switch (ef.OutputType.at(index))
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        factor = sizeof(int8_t);
        break;
    }
    *result *= factor;
    return 0;
}

void *WrapInput(void *host_memory, uint64_t size, IGpuAllocator *allocator)
{
    void *res = allocator->allocate(size, 0, 0);
    int result = 0;
    check_cuda_success(cudaMemcpy(res, host_memory, size, cudaMemcpyHostToDevice), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Wrap output failed." << endl;
        return nullptr;
    }
    return res;
}

void *UnwrapOutput(void *device_memory, size_t size, IGpuAllocator *allocator)
{
    // int *h_ptr = static_cast<int *>((void *)new char[size]);
    void *h_ptr = malloc(size);
    if (!h_ptr)
    {
        gLogError << __CXX_PREFIX << "Can not malloc h_ptr" << endl;
        return nullptr;
    }
    memset(h_ptr, 0, size);
    int result = 0;
    check_cuda_success(cudaMemcpy(h_ptr, device_memory, size, cudaMemcpyDeviceToHost), result);
    if (result != 0)
        gLogError << __CXX_PREFIX << "Unwrap output failed." << endl;
    return h_ptr;
}

void *WrapInputAsync(void *host_memory, uint64_t size, IGpuAllocator *allocator, cudaStream_t stream)
{
    void *res = allocator->allocate(size, 0, 0);
    int result = 0;
    check_cuda_success(cudaMemcpyAsync(res, host_memory, size, cudaMemcpyHostToDevice, stream), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Wrap output failed." << endl;
        return nullptr;
    }
    return res;
}

void *UnwrapOutputAsync(void *device_memory, size_t size, IGpuAllocator *allocator, cudaStream_t stream)
{
    // int *h_ptr = static_cast<int *>((void *)new char[size]);
    void *h_ptr = malloc(size);
    if (!h_ptr)
    {
        gLogError << __CXX_PREFIX << "Can not malloc h_ptr" << endl;
        return nullptr;
    }
    memset(h_ptr, 0, size);
    int result = 0;
    check_cuda_success(cudaMemcpyAsync(h_ptr, device_memory, size, cudaMemcpyDeviceToHost, stream), result);
    if (result != 0)
        gLogError << __CXX_PREFIX << "Unwrap output failed." << endl;
    return h_ptr;
}

int ComputationWorker::GetModel(std::string model_name, EngineInfo *ef) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    if (iter == engine_table.end())
    {
        et_rw_mu.runlock();
        return -1;
    }
    et_rw_mu.runlock();
    *ef = iter->second;
    return 0;
}
// end of computation_worker.cpp