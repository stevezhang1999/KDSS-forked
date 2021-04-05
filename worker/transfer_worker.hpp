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

    // LoadWithDefaultAllocator 使用默认allocator构建模型
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，可以为TensorRT引擎或ONNX文件
    // \param file_path 模型文件的路径
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    int LoadWithDefaultAllocator(std::string model_name, std::string model_file, std::string file_path);
    // LoadFromEngineFile 从本机读取由本机该模型序列化后的结果
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，可以为TensorRT引擎或ONNX文件
    // \param file_path 模型文件的路径
    // \param inTensorVec 该引擎对应的模型的输入名称集合
    // \param outTensorVec 该引擎对应的模型的输出名称集合
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    int LoadFromEngineFile(std::string model_name, std::string model_file, std::string file_path, std::vector<std::string> inTensorVec, std::vector<std::string> outTensorVec);
    // virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type, void *test_input);
    virtual int Unload(std::string model_name);
    virtual std::string GetModelName(int index) const;
    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    virtual int Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output);
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
int preProcessHostOutput(const std::vector<std::vector<char>> &output_vec, int index, void **output, uint64_t num, nvinfer1::DataType type);

// end of transfer_worker.hpp