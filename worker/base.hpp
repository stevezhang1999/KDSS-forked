#pragma once

#include <string>
#include <vector>
#include <istream>
#include <NvInfer.h>
#include <unordered_map>
#include <atomic>
#include <memory>
#include "../util/RWMutex/rwmutex.hpp"

enum ModelType
{
    ONNX_FILE,
    TRT_ENGINE
};

typedef struct EngineInfo
{
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::string engine_serialize;
    std::vector<std::string> InputName;  //可能有多个输入
    std::vector<std::string> OutputName; //可能有多个输出
    std::vector<nvinfer1::Dims> InputDim;
    std::vector<nvinfer1::Dims> OutputDim;
    std::vector<nvinfer1::DataType> InputType;
    std::vector<nvinfer1::DataType> OutputType;
    std::vector<uint> InputNetworkIndex;  // 在Network中该input的index
    std::vector<uint> OutputNetworkIndex; // 在Network中该output的index
} EngineInfo;

// model_table 全局唯一索引与模型名称对照表
extern std::unordered_map<int, std::string> model_table;

// max_index 当前最大index
extern atomic<int> max_index;

// mt_rw_mu model_table配套RW锁
extern RWMutex mt_rw_mu;

// engine_table 全局唯一模型名称与引擎对照表
extern std::unordered_map<std::string, EngineInfo> engine_table;

// et_rw_mu engine_table配套RW锁
extern RWMutex et_rw_mu;

class IWorker
{
public:
    IWorker(){};
    virtual ~IWorker(){};
    // Load 加载模型到GPU中
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，可以为TensorRT引擎或ONNX文件
    // \param file_path 模型文件的路径
    // \param type 输入流代表的实际类型
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type) = 0;

    // UnLoad 从GPU中卸载模型
    // \param model_name 模型名称，该名称是在LoadModel时指定的。
    // \returns 如果卸载不成功或模型不存在，将会返回-1，并在logger中输出错误信息
    virtual int Unload(std::string model_name) = 0;

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    virtual std::string GetModelName(int index) const = 0;

    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 载有数据载荷的vector
    // \param output 将会被写入输出数据的vector
    virtual int Compute(std::string model_name, std::vector<std::vector<char>> &input, std::vector<std::vector<char>> &output) = 0;
};

// end of base.hpp