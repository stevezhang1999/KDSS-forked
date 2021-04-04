#pragma once

#include "base.hpp"
#include <exception>
#include <unordered_map>
#include <mutex>

class TransferWorker final : public IWorker
{
public:
    TransferWorker(){};
    virtual ~TransferWorker();
    virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type);
    // virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type, void *test_input);
    virtual int Unload(std::string model_name);
    virtual std::string GetModelName(int index) const;
    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    virtual std::vector<std::vector<char>> Compute(std::string model_name, std::vector<std::vector<char>> &input);
};

// preProcessHostInput 对已经写好数据的input转移到vector中
// 需要对每个input分别操作
// \param input_vec 需要传入Compute做计算的数组
// \param input 需要写入的input
// \param size 该input包含元素的个数
// \param type 该input的类型（float传入kFLOAT，uint8_t传入kINT8，uint32_t传入kINT32）
int preProcessHostInput(std::vector<std::vector<char>> &input_vec, void *input, uint64_t num, nvinfer1::DataType type);

// preProcessHostInput 对已经写好数据的output_vec提取数据到output中
// 需要对每个output分别操作
// \param output_vec 从Compute计算得到的output数组
// \param index 需要取得的第index个output
// \param output 指向output的指针
// \param size 该output包含元素的个数
// \param type 该output的类型（float传入kFLOAT，uint8_t传入kINT8，uint32_t传入kINT32）
int preProcessHostOutput(const std::vector<std::vector<char>>& output_vec, int index, void **output, uint64_t num, nvinfer1::DataType type);

// end of transfer_worker.hpp