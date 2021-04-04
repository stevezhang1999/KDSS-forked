#include "transfer_worker.hpp"
#include "trt_allocator.hpp"
#include "common.hpp"
#include <NvOnnxParser.h>
#include <NvInferRuntimeCommon.h>
#include "common/logger.h" // On TensorRT/samples
#include "common/common.h" // On TensorRT/samples
#include <memory>
#include <iostream>
using std::cerr;
using std::endl;
using std::string;

extern std::shared_ptr<nvinfer1::IGpuAllocator> kg_allocator;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

TransferWorker::~TransferWorker()
{
    // 获得所有模型的名称，并开始删除它们
    vector<std::string> model_vec;
    mt_rw_mu.rlock();
    for (auto name : model_table)
    {
        model_vec.push_back(name.second);
    }
    mt_rw_mu.runlock();

    // 开始删除
    et_rw_mu.lock();
    for (auto name : model_vec)
    {
        engine_table.erase(name);
    }
    et_rw_mu.unlock();
}

bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                      SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                      SampleUniquePtr<nvonnxparser::IParser> &parser, std::string file_path, std::string model_file)
{
    auto parsed = parser->parseFromFile(
        (file_path + model_file).c_str(), static_cast<int>(ILogger::Severity::kINFO));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(16_MiB);

    return true;
}

int TransferWorker::Load(std::string model_name, std::string model_file, std::string file_path, ModelType type)
{
    // 从网络构建引擎，并保存在engine_table中
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        return -1;
    }
    builder->setGpuAllocator(kg_allocator.get());
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return -1;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return -1;
    }

    switch (type)
    {
    case ONNX_FILE:
    {
        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser)
        {
            return -1;
        }
        auto constructed = constructNetwork(builder, network, config, parser, file_path, model_file);
        if (!constructed)
        {
            return false;
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!mEngine)
        {
            return false;
        }
        break;
    }
    case TRT_ENGINE:
        gLogError << "error! TensorRT deserilaize is not avaliable now." << endl;
        return -1;
        break;
    default:
        break;
    }

    // 我们保存一下当前引擎序列化后的字符串表现形式
    IHostMemory *h_memory = mEngine->serialize();
    std::string serialize_str;
    serialize_str.resize(h_memory->size());
    memcpy((void *)serialize_str.data(), h_memory->data(), h_memory->size());

    // 生成索引
    mt_rw_mu.lock();
    max_index++;
    int current_index = max_index.load();
    model_table.insert(std::pair<int, std::string>(current_index, model_name));
    mt_rw_mu.unlock();

    // 插入到engine_table
    EngineInfo ef;
    ef.engine = mEngine;
    ef.engine_serialize = serialize_str;
    // 保存输入输出信息
    for (int i = 0; i < network->getNbInputs(); i++)
    {
        // 先获得该输入的名字，存入ef
        ef.InputName.push_back(network->getInput(i)->getName());
        // 然后获取对应Tensor的大小
        ef.InputSize.push_back(network->getInput(i)->getDimensions());
        // 获取类型，存入
        ef.InputType.push_back(network->getInput(i)->getType());
    }

    for (int i = 0; i < network->getNbOutputs(); i++)
    {
        // 先获得该输入的名字，存入ef
        ef.OutputName.push_back(network->getOutput(i)->getName());
        // 然后获取对应Tensor的大小
        ef.OutputSize.push_back(network->getOutput(i)->getDimensions());
        // 获取类型，存入
        ef.OutputType.push_back(network->getOutput(i)->getType());
    }
    et_rw_mu.lock();
    engine_table.insert(std::pair<std::string, EngineInfo>(model_name, ef));
    et_rw_mu.unlock();
    return current_index;
}

int TransferWorker::Unload(std::string model_name)
{
    // 从engine_table删除该模型的引擎
    et_rw_mu.lock();
    auto iter = engine_table.find(model_name);
    if (iter == engine_table.end())
    {
        et_rw_mu.unlock();
        return -1;
    }
    engine_table.erase(iter);
    et_rw_mu.unlock();

    // 再从model_table中删除
    mt_rw_mu.lock();
    int index = -1;
    for (auto iter = model_table.begin(); iter != model_table.end(); ++iter)
    {
        if (iter->second == model_name)
        {
            model_table.erase(iter);
            break;
        }
    }
    mt_rw_mu.unlock();
    return index;
}

std::string TransferWorker::GetModelName(int index) const
{
    mt_rw_mu.rlock();
    auto iter = model_table.find(index);
    mt_rw_mu.runlock();
    if (iter == model_table.end())
        return "";
    return iter->second;
}

std::vector<std::vector<char>> TransferWorker::Compute(std::string model_name, std::vector<std::vector<char>> &input)
{
    throw "Compute method not supported.";
    return std::vector<std::vector<char>>();
}

int preProcessHostInput(std::vector<std::vector<char>> &input_vec, void *input, uint64_t num, nvinfer1::DataType type)
{
    // 对h_output进行逐字节写入到test_data
    int factor = 1;
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
    {
        factor = sizeof(float);
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        factor = sizeof(uint32_t);
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        factor = sizeof(uint8_t);
        break;
    }
    }
    std::vector<char> temp;
    for (int j = 0; j < num * factor; j++)
    {
        temp.push_back(*((char *)input + j));
    }
    input_vec.push_back(temp);
    return 0;
}

int preProcessHostOutput(const std::vector<std::vector<char>>& output_vec, int index, void **output, uint64_t num, nvinfer1::DataType type)
{
    if (index >= output_vec.size())
        return -1;
    std::vector<char> data = output_vec.at(index);
    int factor = 1;
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
    {
        factor = sizeof(float);
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        factor = sizeof(uint32_t);
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        factor = sizeof(uint8_t);
        break;
    }
    }
    memcpy(*output, data.data(), num * factor);
    return 0;
