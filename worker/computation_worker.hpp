#pragma once

#include "base.hpp"
#include "trt_allocator.hpp"
#include <exception>
#include <NvInfer.h>

class ComputationWorker final : public IWorker
{
public:
    int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type)
    {
#pragma message("Computation_worker doesn't not support Load method.")
        throw "Load method not supported.";
        return 0;
    }

    int Unload(std::string model_name)
    {
#pragma message("Computation_worker doesn't not support Unload method.")
        throw "Unload method not supported.";
        return 0;
    }
    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    std::string GetModelName(int index);

    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 指向host_memory的数据指针
    void *Compute(std::string model_name, void *input);
};

// end of computation_worker.hpp