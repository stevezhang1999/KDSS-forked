#include "base.hpp"

// model_table 全局唯一索引与模型名称对照表
std::unordered_map<int, std::string> model_table;

// max_index 当前最大index
atomic<int> max_index(-1);

// mt_rw_mu model_table配套RW锁
RWMutex mt_rw_mu;

// engine_table 全局唯一模型名称与引擎对照表
std::unordered_map<std::string, EngineInfo> engine_table;

// et_rw_mu engine_table配套RW锁
RWMutex et_rw_mu;

int GetModelInputSize(std::string model_name, int index, uint64_t *result)
{
    EngineInfo ef;
    int executed = GetModel(model_name, &ef);
    if (executed != 0)
        return executed;

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

int GetModelInputSize(std::string model_name, std::string input_name, uint64_t *result)
{
    EngineInfo ef;
    int executed = GetModel(model_name, &ef);
    if (executed != 0)
        return executed;

    // 先找到该输出的名称对应的索引
    auto input_iter = std::find(ef.InputName.begin(), ef.InputName.end(), input_name);
    if (input_iter == ef.InputName.end())
        return -1;
    int index = std::distance(ef.InputName.begin(), input_iter);

    return GetModelInputSize(model_name, index, result);
}

int GetModelOutputSize(std::string model_name, int index, uint64_t *result)
{

    EngineInfo ef;
    int executed = GetModel(model_name, &ef);
    if (executed != 0)
        return executed;

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

int GetModelOutputSize(std::string model_name, std::string output_name, uint64_t *result)
{

    EngineInfo ef;
    int executed = GetModel(model_name, &ef);
    if (executed != 0)
        return executed;

    // 先找到该输出的名称对应的索引
    auto output_iter = std::find(ef.OutputName.begin(), ef.OutputName.end(), output_name);
    if (output_iter == ef.OutputName.end())
        return -1;
    int index = std::distance(ef.OutputName.begin(), output_iter);

    return GetModelOutputSize(model_name, index, result);
}

EngineInfo::EngineInfo()
{
    this->engine = nullptr;
}

EngineInfo::EngineInfo(const EngineInfo& ef)
{
    // initalize
    this->engine = nullptr;
    this->engine_serialize = "";
    this->InputName.clear();
    this->OutputName.clear();
    this->InputDim.clear();
    this->OutputDim.clear();
    this->InputType.clear();
    this->OutputType.clear();
    this->InputNetworkIndex.clear();
    this->OutputNetworkIndex.clear();

    // memory copy
    this->engine = ef.engine;
    this->engine_serialize = engine_serialize;
    for (auto e : ef.InputName)
        this->InputName.push_back(e);
    for (auto e : ef.OutputName)
        this->OutputName.push_back(e);
    for (auto e : ef.InputDim)
        this->InputDim.push_back(e);
    for (auto e : ef.OutputDim)
        this->OutputDim.push_back(e);
    for (auto e : ef.InputType)
        this->InputType.push_back(e);
    for (auto e : ef.OutputType)
        this->OutputType.push_back(e);
    for (auto e : ef.InputNetworkIndex)
        this->InputNetworkIndex.push_back(e);
    for (auto e : ef.OutputNetworkIndex)
        this->OutputNetworkIndex.push_back(e);
}