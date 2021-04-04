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

std::vector<std::vector<char>> ComputationWorker::Compute(std::string model_name, std::vector<std::vector<char>> &input)
{
    std::vector<std::vector<char>> result;
    // 从table中取得已注册的引擎
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return result;
    }
    nvinfer1::ICudaEngine *engine = iter->second.engine.get();
    if (!engine)
    {
        gLogError << "engine not vaild." << endl;
        return result;
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
        return result;
    }
    // int input_index = engine->getBindingIndex(iter->second.InputName[0].c_str());
    // int output_index = engine->getBindingIndex(iter->second.OutputName[0].c_str());

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = iter->second.InputName.size();
    int output_num = iter->second.OutputName.size();
    void **buffers = (void **)malloc(sizeof(void *) * (input_num + output_num));
    if (!buffers)
    {
        gLogError << __CXX_PREFIX << "Buffer for computation alloc failed." << endl;
        return result;
    }
    memset(buffers, 0, sizeof(void *) * (input_num + output_num));

    // 处理host端input到device端input
    for (int i = 0; i < iter->second.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = engine->getBindingIndex(iter->second.InputName.at(i).c_str());
        if (input_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            buffers = (void **)realloc(buffers, sizeof(void *) * input_i_index);
        }
        uint64_t input_i_size;
        int res = this->GetModelInputSize(model_name, iter->second.InputName.at(i), &input_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get input size of : " << iter->second.InputName.at(i);
            return result;
        }
        // 测试用
        // 显示一下输入数据到底是什么
        gLogInfo << "Input " << i << ":" << endl;
        {
            float *temp = (float *)new char[input_i_size];
            memset(temp, 0, input_i_size);
            memcpy((void *)temp, (void *)input[i].data(), input_i_size);
            for (int j = 0; j < 28; j++)
            {
                for (int k = 0; k < 28; k++)
                {
                    cout << temp[28 * j + k] << " ";
                }
                cout << endl;
            }
            free(temp);
        }
        buffers[input_i_index] = WrapInput(input[i].data(), input_i_size);
    }

    // 申请device端output空间
    for (int i = 0; i < iter->second.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = engine->getBindingIndex(iter->second.OutputName.at(i).c_str());
        if (output_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogInfo << "Buffers not long enough, reallocating..." << endl;
            buffers = (void **)realloc(buffers, sizeof(void *) * output_i_index);
        }
        uint64_t output_i_size;
        int res = this->GetModelOutputSize(model_name, iter->second.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << iter->second.OutputName.at(i);
            return result;
        }
        buffers[output_i_index] = kg_allocator->allocate(output_i_size, 0, 0);
    }

    bool status = context->execute(1, buffers);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
        return result;
    }

    // 对output逐个unwrap
    void **h_output = (void **)malloc(sizeof(void *) * output_num);
    if (!h_output)
    {
        gLogError << __CXX_PREFIX << "Output allocation failed." << endl;
        return result;
    }

    // 处理device端output到host端output
    for (int i = 0; i < iter->second.OutputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int output_i_index = engine->getBindingIndex(iter->second.OutputName.at(i).c_str());

        uint64_t output_i_size;
        int res = this->GetModelOutputSize(model_name, iter->second.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << iter->second.OutputName.at(i);
            return result;
        }
        h_output[i] = UnwrapOutput(buffers[output_i_index]);
        if (!h_output[i])
        {
            // 释放所有h_output[0,i-1]的内存
            for (int j = 0; j < i; j++)
            {
                free(h_output[j]);
            }
            free(h_output);
            h_output = nullptr;
            result.clear();
            break;
        }
        // 对h_output进行逐字节写入到result
        std::vector<char> temp;
        for (int j = 0; j < output_i_size; j++)
        {
            temp.push_back(*((char *)h_output[i] + j));
        }
        result.push_back(temp);
    }
    free(buffers);
    // 释放h_output
    for (int i = 0; i < iter->second.OutputName.size(); i++)
    {
        free(h_output[i]);
    }
    free(h_output);
    return result;
    // void *h_output = UnwrapOutput(d_output);
    // return h_output;
}

// GetModelInputSize 获取指定模型的输入总大小
int ComputationWorker::GetModelInputSize(std::string model_name, int index, uint64_t *result) const
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        gLogError << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;
    for (int i = 0; i < ef.InputSize.at(index).nbDims; i++)
    {
        if (ef.InputSize.at(index).d[i] == 0)
            continue;
        *result *= ef.InputSize.at(index).d[i];
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
        gLogError << "engine not vaild." << endl;
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
    for (int i = 0; i < ef.InputSize.at(index).nbDims; i++)
    {
        if (ef.InputSize.at(index).d[i] == 0)
            continue;
        *result *= ef.InputSize.at(index).d[i];
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
        gLogError << "engine not vaild." << endl;
        return -1;
    }

    EngineInfo ef = iter->second;
    *result = 1;
    for (int i = 0; i < ef.OutputSize.at(index).nbDims; i++)
    {
        if (ef.OutputSize.at(index).d[i] == 0)
            continue;
        *result *= ef.OutputSize.at(index).d[i];
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
        gLogError << "engine not vaild." << endl;
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
    for (int i = 0; i < ef.OutputSize.at(index).nbDims; i++)
    {
        if (ef.OutputSize.at(index).d[i] == 0)
            continue;
        *result *= ef.OutputSize.at(index).d[i];
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
        gLogError << "Unwrap output failed." << endl;
    kg_allocator->free(device_memory);
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