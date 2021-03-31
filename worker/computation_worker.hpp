#pragma once

#include "base.hpp"
#include "trt_allocator.hpp"
#include <exception>
#include <NvInfer.h>

class ComputationWorker final : public IWorker
{
public:
    ComputationWorker() {}
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

    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 指向host_memory的数据指针
    virtual void *Compute(std::string model_name, void *input);
};

// end of computation_worker.hpp