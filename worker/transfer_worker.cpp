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
        // auto parsed = parser->parseFromFile((file_path + model_file).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
        // if (!parsed)
        // {
        //     return -1;
        // }
        // builder->setMaxBatchSize(1);
        // config->setMaxWorkspaceSize(32 * (1 << 20));
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

    int input_index = mEngine->getBindingIndex("Input3");
    int output_index = mEngine->getBindingIndex("Plus214_Output_0");

    // 测试引擎
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    void *d_input;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    std::ifstream in("/home/lijiakang/TensorRT-6.0.1.5/samples/sampleOnnxMNIST/hostDataBuffer.txt", ios_base::in);
    for (int i = 0; i < 28 * 28; i++)
    {
        in >> test_data[i];
    }
    in.close();
    cudaMalloc(&d_input, 28 * 28 * sizeof(float));
    cudaMemcpy(d_input, test_data, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
    void *buffers[2];
    buffers[input_index] = d_input;
    // void *d_output = kg_allocator->allocate(iter->second.OutputSize, 0, 0);
    void *d_output;
    cudaMalloc(&d_output, 40);
    // void *output;
    // cudaMalloc(&output, iter->second.OutputSize);
    if (!d_output)
    {
        gLogError << "allocate output memory failed." << endl;
    }
    buffers[output_index] = d_output;
    bool status = context->execute(1, buffers);
    if (!status)
    {
        gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
    }
    // void *output = UnwrapOutput(d_output);
    float output[10];
    cudaError_t err = cudaMemcpy(output, buffers[output_index], sizeof(float) * 10, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    // 生成索引
    mt_rw_mu.lock();
    max_index++;
    int current_index = max_index.load();
    model_table.insert(std::pair<int, std::string>(current_index, model_name));
    mt_rw_mu.unlock();

    // 获取输入/输出最大空间
    uint32_t inputMaxSize = 1UL;
    uint32_t outputMaxSize = 1UL;

    auto inputDims = mEngine->getBindingDimensions(input_index);
    auto outputDims = mEngine->getBindingDimensions(output_index);

    // 在batch_size = 1的情况下，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
    for (int i = 0; i < inputDims.nbDims; i++)
    {
        if (inputDims.d[i] == 0)
            continue;
        inputMaxSize *= inputDims.d[i];
    }

    // 在batch_size = 1的情况下，对于image_classfication来说，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
    for (int i = 0; i < outputDims.nbDims; i++)
    {
        if (outputDims.d[i] == 0)
            continue;
        outputMaxSize *= outputDims.d[i];
    }

    // 插入到engine_table
    et_rw_mu.lock();
    EngineInfo ef = {
        .engine = mEngine,
        .engine_serialize = serialize_str,
        .InputName = "Input3",
        .OutputName = "Plus214_Output_0",
    };
    // 获得输入输出类型，对inputMaxSize和outputMaxSize乘以系数
    int input_factor = 0;
    int output_factor = 0;
    auto inputType = nvinfer1::DataType::kFLOAT;
    switch (inputType)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        input_factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        input_factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        input_factor = sizeof(int8_t);
        break;
    }
    auto outputType = nvinfer1::DataType::kFLOAT;
    switch (outputType)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF:
        output_factor = sizeof(float);
        break;
    case nvinfer1::DataType::kINT32:
        output_factor = sizeof(int32_t);
        break;
    case nvinfer1::DataType::kINT8:
        output_factor = sizeof(int8_t);
        break;
    }
    ef.InputSize = input_factor * inputMaxSize;
    ef.OutputSize = output_factor * outputMaxSize;
    engine_table.insert(std::pair<std::string, EngineInfo>(model_name, ef));
    et_rw_mu.unlock();
    return current_index;
}

// int TransferWorker::Load(std::string model_name, std::string model_file, std::string file_path, ModelType type, void *test_input)
// {
//     // 这个版本的Load会直接反序列化由本机标准代码保存的引擎
//     std::string cached_engine = "";
//     std::ifstream fin("/home/lijiakang/TensorRT-6.0.1.5/data/mnist/mnist.trtengine");
//     while (fin.peek() != EOF)
//     { // 使用fin.peek()防止文件读取时无限循环
//         std::stringstream buffer;
//         buffer << fin.rdbuf();
//         cached_engine.append(buffer.str());
//     }
//     fin.close();

//     auto runtime = createInferRuntime(gLogger);
//     if (!runtime)
//     {
//         return -1;
//     }
//     auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size()), samplesCommon::InferDeleter());
//     if (!engine)
//     {
//         return -1;
//     }

//     // 我们保存一下当前引擎序列化后的字符串表现形式
//     IHostMemory *h_memory = engine->serialize();
//     std::string serialize_str;
//     serialize_str.resize(h_memory->size());
//     memcpy((void *)serialize_str.data(), h_memory->data(), h_memory->size());

//     // 测试引擎
//     auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
//     if (!context)
//     {
//         return false;
//     }
//     int input_index = engine->getBindingIndex("Input3");
//     int output_index = engine->getBindingIndex("Plus214_Output_0");
//     void *d_input;
//     cudaMalloc(&d_input, 28 * 28 * sizeof(float));
//     cudaMemcpy(d_input, test_input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
//     void *buffers[2];
//     buffers[input_index] = d_input;
//     // void *d_output = kg_allocator->allocate(iter->second.OutputSize, 0, 0);
//     void *d_output;
//     cudaMalloc(&d_output, 40);
//     // void *output;
//     // cudaMalloc(&output, iter->second.OutputSize);
//     if (!d_output)
//     {
//         gLogError << "allocate output memory failed." << endl;
//     }
//     buffers[output_index] = d_output;
//     bool status = context->execute(1, buffers);
//     if (!status)
//     {
//         gLogError << __CXX_PREFIX << "Execute model failed!" << endl;
//     }
//     // void *output = UnwrapOutput(d_output);
//     float output[10];
//     cudaError_t err = cudaMemcpy(output, buffers[output_index], sizeof(float) * 10, cudaMemcpyDeviceToHost);
//     cudaFree(d_input);
//     cudaFree(d_output);
//     // 生成索引
//     mt_rw_mu.lock();
//     max_index++;
//     int current_index = max_index.load();
//     model_table.insert(std::pair<int, std::string>(current_index, model_name));
//     mt_rw_mu.unlock();

//     // 获取输入/输出最大空间
//     uint32_t inputMaxSize = 1UL;
//     uint32_t outputMaxSize = 1UL;

//     auto inputDims = engine->getBindingDimensions(input_index);
//     auto outputDims = engine->getBindingDimensions(output_index);

//     // 在batch_size = 1的情况下，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
//     for (int i = 0; i < inputDims.nbDims; i++)
//     {
//         if (inputDims.d[i] == 0)
//             continue;
//         inputMaxSize *= inputDims.d[i];
//     }

//     // 在batch_size = 1的情况下，对于image_classfication来说，通常i不会超过3（CHW），对于灰度图来说，i一般为2（HW）
//     for (int i = 0; i < outputDims.nbDims; i++)
//     {
//         if (outputDims.d[i] == 0)
//             continue;
//         outputMaxSize *= outputDims.d[i];
//     }

//     // 插入到engine_table
//     et_rw_mu.lock();
//     EngineInfo ef = {
//         .engine = engine,
//         .engine_serialize = serialize_str,
//         .InputName = "Input3",
//         .OutputName = "Plus214_Output_0",
//     };
//     // 获得输入输出类型，对inputMaxSize和outputMaxSize乘以系数
//     int input_factor = 0;
//     int output_factor = 0;
//     auto inputType = nvinfer1::DataType::kFLOAT;
//     switch (inputType)
//     {
//     case nvinfer1::DataType::kFLOAT:
//     case nvinfer1::DataType::kHALF:
//         input_factor = sizeof(float);
//         break;
//     case nvinfer1::DataType::kINT32:
//         input_factor = sizeof(int32_t);
//         break;
//     case nvinfer1::DataType::kINT8:
//         input_factor = sizeof(int8_t);
//         break;
//     }
//     auto outputType = nvinfer1::DataType::kFLOAT;
//     switch (outputType)
//     {
//     case nvinfer1::DataType::kFLOAT:
//     case nvinfer1::DataType::kHALF:
//         output_factor = sizeof(float);
//         break;
//     case nvinfer1::DataType::kINT32:
//         output_factor = sizeof(int32_t);
//         break;
//     case nvinfer1::DataType::kINT8:
//         output_factor = sizeof(int8_t);
//         break;
//     }
//     ef.InputSize = input_factor * inputMaxSize;
//     ef.OutputSize = output_factor * outputMaxSize;
//     engine_table.insert(std::pair<std::string, EngineInfo>(model_name, ef));
//     et_rw_mu.unlock();
//     return current_index;
// }

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

void *TransferWorker::Compute(std::string model_name, void *input)
{
    throw "Compute method not supported.";
    return nullptr;
}
