#pragma once

#include "base.hpp"
#include "trt_allocator.hpp"
#include "common.hpp"
#include <exception>
#include <NvInfer.h>

class ComputationWorker final : public IWorker
{
public:
    ComputationWorker()
    {
        cudaDeviceProp prop;
        int device;
        int result;
        check_cuda_success(cudaGetDevice(&device), result);
        if (!result)
        {
            cerr << __CXX_PREFIX << "CUDA error." << endl;
            throw "";
        }
        check_cuda_success(cudaGetDeviceProperties(&prop, device), result);
        if (!result)
        {
            cerr << __CXX_PREFIX << "CUDA error." << endl;
            throw "";
        }
        // See also: https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously
        alignment = prop.textureAlignment;
    }
    virtual ~ComputationWorker() {}
    virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type)
    {
        throw "Load method not supported.";
        return 0;
    }

    virtual int Unload(std::string model_name)
    {
        throw "Unload method not supported.";
        return 0;
    }

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    virtual std::string GetModelName(int index) const;

    // Compute 开始根据模型执行计算。
    // 此过程会根据模型引擎创建上下文，使用allocator分配输入/输出显存及运行时显存，
    // 并在退出时将其全部销毁。
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    // \param output 将会被写入输出数据的vector
    // \param allocator 分配内存使用的allocator
    int Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output, nvinfer1::IGpuAllocator *allocator);

    // Compute 根据模型使用kg_allocator执行计算。
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    // \param output 将会被写入输出数据的vector
    virtual int Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output);

    // ComputeWithStream 使用CUDA stream+kg_allocator进行overlapped异步计算
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    // \param output 将会被写入输出数据的vector
    int ComputeWithStream(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output);

    // ComputeWithStream 使用CUDA stream+自定义分配器进行overlapped异步计算
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    // \param output 将会被写入输出数据的vector
    int ComputeWithStream(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output,nvinfer1::IGpuAllocator *allocator);

    // GetModelInputSize 获取指定模型的输入总大小
    int GetModelInputSize(std::string model_name, int index, uint64_t *result) const;

    // GetModelInputSize 获取指定模型的输入总大小
    int GetModelInputSize(std::string model_name, std::string input_name, uint64_t *result) const;

    // GetModelOutputSize 获取指定模型的输出总大小
    int GetModelOutputSize(std::string model_name, int index, uint64_t *result) const;

    // GetModelOutputSize 获取指定模型的输出总大小
    int GetModelOutputSize(std::string model_name, std::string output_name, uint64_t *result) const;

    // GetModelInfo 获取指定模型的信息
    int GetModel(std::string model_name, EngineInfo *ef) const;

private:
    // 获取alignment
    uint64_t alignment;
};

// end of computation_worker.hpp