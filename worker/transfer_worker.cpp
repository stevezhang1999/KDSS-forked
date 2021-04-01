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

extern nvinfer1::IGpuAllocator *kg_allocator;

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
int TransferWorker::Load(std::string model_name, std::string model_file, std::string file_path, ModelType type)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return -1;
    }

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
        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
        if (!parser)
        {
            return -1;
        }
        auto parsed = parser->parseFromFile((file_path + model_file).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
        if (!parsed)
        {
            return -1;
        }
        if (!kg_allocator)
        {
            kg_allocator = new KGAllocator();
        }
        builder->setGpuAllocator(kg_allocator);
        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(32 * (1 << 20));
        // config->setFlag(nvinfer1::BuilderFlag::kFP16);
        break;
    }
    case TRT_ENGINE:
        gLogError << "error! TensorRT deserilaize is not avaliable now." << endl;
        return -1;
        break;
    default:
        break;
    }
    // 从网络构建引擎，并保存在engine_table中
    auto engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        return -1;
    }
    // 生成索引
    mt_rw_mu.lock();
    max_index++;
    int current_index = max_index.load();
    model_table.insert(std::pair<int, std::string>(current_index, model_name));
    mt_rw_mu.unlock();

    // 获取输入/输出最大空间
    uint32_t inputMaxSize = 0UL;
    uint32_t outputMaxSize = 0UL;

    auto inputDims = network->getInput(0)->getDimensions();
    auto outputDims = network->getOutput(0)->getDimensions();

    // 在batch_size = 1的情况下，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
    for (int i = 0; i < inputDims.nbDims; i++)
    {
        if (inputDims.d[i] == 0)
            continue;
        if (inputMaxSize == 0)
        {
            inputMaxSize += sizeof(float) * inputDims.d[i];
        }
        else
        {
            inputMaxSize *= sizeof(float) * inputDims.d[i];
        }
    }

    // 在batch_size = 1的情况下，对于image_classfication来说，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
    for (int i = 0; i < outputDims.nbDims; i++)
    {
        if (outputDims.d[i] == 0)
            continue;
        if (outputMaxSize == 0)
        {
            outputMaxSize += sizeof(float) * outputDims.d[i];
        }
        else
        {
            outputMaxSize *= sizeof(float) * outputDims.d[i];
        }
    }

    // 插入到engine_table
    et_rw_mu.lock();
    EngineInfo ef = {
        .engine = engine,
        .InputName = network->getInput(0)->getName(),
        .OutputName = network->getOutput(0)->getName(),
    };
    ef.InputSize = inputMaxSize;
    ef.OutputSize = outputMaxSize;
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
        return -1;
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

void *TransferWorker::Compute(std::string model_name, void *input)
{
    throw "Compute method not supported.";
    return nullptr;
}