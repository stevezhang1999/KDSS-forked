#pragma once

#include <string>
#include <vector>
#include <istream>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <memory>

#include "NvInfer.h"

#include "../util/RWMutex/rwmutex.hpp"

#if defined(_WIN32) || defined(_MSC_VER)
typedef unsigned int uint;
#endif

enum ModelType
{
    ONNX_FILE,
    TRT_ENGINE
};

typedef struct EngineInfo
{
    EngineInfo();
    EngineInfo(const EngineInfo &ef);

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

    // LoadModel 加载模型到显存中
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，可以为TensorRT引擎或ONNX文件
    // \param file_path 模型文件的路径
    // \param type 输入流代表的实际类型
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    virtual int LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type) = 0;

    // UnLoadModel 从显存中卸载模型
    // \param model_name 模型名称，该名称是在LoadModel时指定的。
    // \returns 如果卸载不成功或模型不存在，将会返回-1，并在logger中输出错误信息
    virtual int UnloadModel(std::string model_name) = 0;

    // TransferInput 将输入从内存转移到显存，并申请对应的显存
    // \param model_name 该输入对应的模型名称
    // \param input_data 输入的字节流载荷
    // \param input_ptr 该输入对应的显存地址数组指针
    // \param allocator 转移到显存时需要用的分配器
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferInput(std::string model_name, const std::vector<std::vector<char>> input_data, void **(&input_ptr), nvinfer1::IGpuAllocator *allocator) = 0;

    // TransferOutput 将输出从显存转移到内存，并释放对应的显存
    // \param model_name 该输出对应的模型名称
    // \param output_ptr 该输出对应的显存地址数组指针
    // \param output_data 输出的字节流载荷
    // \param allocator 分配时用的allocator，必须是Compute使用的allocator
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferOutput(std::string model_name, void **output_ptr, std::vector<std::vector<char>> &output_data, nvinfer1::IGpuAllocator *allocator) = 0;

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    virtual std::string GetModelName(int index) const = 0;

    // Compute 开始根据模型执行计算
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针的引用
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int Compute(std::string model_name, void **input, void **(&output)) = 0;
};

inline int GetModel(std::string model_name, EngineInfo *ef)
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
        return -1;
    *ef = iter->second;
    return 0;
}

// GetModelInputSize 获取指定模型的输入总大小
int GetModelInputSize(std::string model_name, int index, uint64_t *result);

// GetModelInputSize 获取指定模型的输入总大小
int GetModelInputSize(std::string model_name, std::string input_name, uint64_t *result);

// GetModelOutputSize 获取指定模型的输出总大小
int GetModelOutputSize(std::string model_name, int index, uint64_t *result);

// GetModelOutputSize 获取指定模型的输出总大小
int GetModelOutputSize(std::string model_name, std::string output_name, uint64_t *result);

// end of base.hpp