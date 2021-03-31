#pragma once

#include <string>
#include <istream>
#include <unordered_map>
#include <NvInfer.h>
#include "../util/RWMutex/rwmutex.hpp"

enum ModelType
{
    ONNX_FILE,
    TRT_ENGINE
};

typedef struct EngineInfo {
    nvinfer1::ICudaEngine * engine;
    std::string InputName;
    std::string OutputName;
    uint32_t InputSize;
    uint32_t OutputSize;
} EngineInfo;
// model_table 全局唯一索引与模型名称对照表
static std::unordered_map<int, std::string> model_table;

// mt_rw_mu model_table配套RW锁
static RWMutex mt_rw_mu;

// engine_table 全局唯一模型名称与引擎对照表
static std::unordered_map<std::string, EngineInfo> engine_table;

// et_rw_mu engine_table配套RW锁
static RWMutex et_rw_mu;

class IWorker
{
public:
    // Load 加载模型到GPU中
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型输入流，可以为TensorRT引擎或ONNX文件
    // \param type 输入流代表的实际类型
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    virtual int Load(std::string model_name, std::istream model_file, ModelType type) = 0;

    // UnLoad 从GPU中卸载模型
    // \param model_name 模型名称，该名称是在LoadModel时指定的。
    // \returns 如果卸载不成功或模型不存在，将会返回-1，并在logger中输出错误信息
    virtual int Unload(std::string model_name) = 0;

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串，并在logger中输出错误信息
    virtual std::string GetModelName(int index) const = 0;

    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 指向host_memory的数据指针
    virtual void *Compute(std::string model_name, void *input) const = 0;
};

// end of base.hpp