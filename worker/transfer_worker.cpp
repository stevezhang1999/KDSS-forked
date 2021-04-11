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

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

using nvinfer1::IGpuAllocator;

extern std::shared_ptr<IGpuAllocator> global_allocator;

uint64_t alignment = 0; // for alignment

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

std::string getTRTEngine(std::string file_path, std::string model_file);

TransferWorker::TransferWorker(ALLOCATOR_TYPE type)
{
    if (global_allocator == nullptr)
    {
        switch (type)
        {
        case DEFAULT_ALLOCATOR:
            cout << "Using default allocator."
                 << endl;
            global_allocator.reset(new DefaultAllocator());
            break;
        case KGMALLOC_ALLOCATOR:
            cout << "Using kgmalloc allocator."
                 << endl;
            global_allocator.reset(new KGAllocator());
            break;
        case KGMALLOCV2_ALLOCATOR:
            cout << "Using kgmalloc v2 allocator."
                 << endl;
            global_allocator.reset(new KGAllocatorV2());
            break;
        default:
            global_allocator.reset(new DefaultAllocator());
            break;
        }
        if (alignment == 0)
        {
            // 获取当前GPU的对齐字节数
            cudaDeviceProp prop;
            int device;
            int result;
            check_cuda_success(cudaGetDevice(&device), result);
            if (result)
            {
                cerr << __CXX_PREFIX << "CUDA error."
                     << endl;
                throw "";
            }
            check_cuda_success(cudaGetDeviceProperties(&prop, device), result);
            if (result)
            {
                cerr << __CXX_PREFIX << "CUDA error."
                     << endl;
                throw "";
            }
            // See also: https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously
            alignment = prop.textureAlignment;
        }
    }
}

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
    KGAllocator::destroy();
}

bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                      SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                      SampleUniquePtr<nvonnxparser::IParser> &parser, std::string file_path, std::string model_file, uint64_t workspace_size)
{
    auto parsed = parser->parseFromFile(
        (file_path + model_file).c_str(), static_cast<int>(ILogger::Severity::kINFO));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(workspace_size);

    return true;
}

int TransferWorker::LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type, IGpuAllocator *allocator, uint64_t workspace_size = 32_MiB)
{
    switch (type)
    {
    case TRT_ENGINE:
    {
        throw "Call TransferWorker::LoadFromEngineFile() instead of this function.";
        return -1;
    }
    }
    // 先查看engine_table是否已有同名引擎
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    if (iter != engine_table.end())
    {
        gLogError << __CXX_PREFIX << "Inserting model " << model_name << " which is already exist on current system."
                  << endl;
        et_rw_mu.runlock();
        return -1;
    }
    et_rw_mu.runlock();
    // 从网络构建引擎，并保存在engine_table中
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        return -1;
    }
    builder->setGpuAllocator(allocator);
#if NV_TENSORRT_MAJOR >= 7
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
#else
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
#endif
    if (!network)
    {
        return -1;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return -1;
    }
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        return -1;
    }
    auto constructed = constructNetwork(builder, network, config, parser, file_path, model_file, workspace_size);
    if (!constructed)
    {
        return -1;
    }
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return -1;
    }

    // 我们保存一下当前引擎序列化后的字符串表现形式
    auto h_memory = SampleUniquePtr<IHostMemory>(mEngine->serialize());
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
        ef.InputDim.push_back(network->getInput(i)->getDimensions());
        // 获取类型，存入
        ef.InputType.push_back(network->getInput(i)->getType());
        // 获取在network中的索引，存入
        ef.InputNetworkIndex.push_back(mEngine->getBindingIndex(network->getInput(i)->getName()));
    }

    for (int i = 0; i < network->getNbOutputs(); i++)
    {
        // 先获得该输入的名字，存入ef
        ef.OutputName.push_back(network->getOutput(i)->getName());
        // 然后获取对应Tensor的大小
        ef.OutputDim.push_back(network->getOutput(i)->getDimensions());
        // 获取类型，存入
        ef.OutputType.push_back(network->getOutput(i)->getType());
        // 获取在network中的索引，存入
        ef.OutputNetworkIndex.push_back(mEngine->getBindingIndex(network->getOutput(i)->getName()));
    }
    et_rw_mu.lock();
    engine_table.insert(std::pair<std::string, EngineInfo>(model_name, ef));
    et_rw_mu.unlock();
    return current_index;
}

int TransferWorker::LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type)
{
    return this->LoadModel(model_name, model_file, file_path, type, global_allocator.get());
}

int TransferWorker::LoadFromEngineFile(std::string model_name, std::string model_file, std::string file_path, std::vector<std::string> inTensorVec, std::vector<std::string> outTensorVec)
{
    // 先查看engine_table是否已有同名引擎
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    if (iter != engine_table.end())
    {
        gLogError << __CXX_PREFIX << "Inserting model " << model_name << " which is already exist on current system."
                  << endl;
        et_rw_mu.runlock();
        return -1;
    }
    et_rw_mu.runlock();

    IRuntime *runtime = createInferRuntime(gLogger.getTRTLogger());
    std::string serialize_str = getTRTEngine(file_path, model_file);
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(serialize_str.data(), serialize_str.size()), samplesCommon::InferDeleter());
    runtime->destroy();

    if (!mEngine)
        return -1;
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
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        if (std::find(inTensorVec.begin(), inTensorVec.end(), mEngine->getBindingName(i)) != inTensorVec.end())
        {
            // 先获得该输入的名字，存入ef
            ef.InputName.push_back(mEngine->getBindingName(i));
            // 然后获取对应Tensor的大小
            ef.InputDim.push_back(mEngine->getBindingDimensions(i));
            // 获取类型，存入
            ef.InputType.push_back(mEngine->getBindingDataType(i));
            // 获取在network中的索引，存入
            ef.InputNetworkIndex.push_back(i);
        }
        else if (std::find(outTensorVec.begin(), outTensorVec.end(), mEngine->getBindingName(i)) != outTensorVec.end())
        {
            // 先获得该输入的名字，存入ef
            ef.OutputName.push_back(mEngine->getBindingName(i));
            // 然后获取对应Tensor的大小
            ef.OutputDim.push_back(mEngine->getBindingDimensions(i));
            // 获取类型，存入
            ef.OutputType.push_back(mEngine->getBindingDataType(i));
            // 获取在network中的索引，存入
            ef.OutputNetworkIndex.push_back(i);
        }
        else
        {
            gLogFatal << __CXX_PREFIX << "Input or output vector not vaild!"
                      << endl;
            throw "";
        }
    }
    et_rw_mu.lock();
    engine_table.insert(std::pair<std::string, EngineInfo>(model_name, ef));
    et_rw_mu.unlock();
    return current_index;
}

int TransferWorker::UnloadModel(std::string model_name)
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

int TransferWorker::TransferInput(std::string model_name, const std::vector<std::vector<char>> input_data, void **(&input_ptr), nvinfer1::IGpuAllocator *allocator)
{
    // 按照EngineInfo遍历的顺序对输入进行填充
    EngineInfo ef;
    int executed = 0;
    executed = GetModel(model_name, &ef);
    if (executed != 0)
    {
        gLogError << __CXX_PREFIX << "Can not get model " << model_name << endl;
        return executed;
    }

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    for (int i = 0; i < ef.InputName.size(); i++)
    {
        // 找到它们在全局中的索引
        int input_i_index = ef.InputNetworkIndex.at(i);
        if (input_i_index >= (input_num + output_num))
        {
            // This should not happen
            gLogWarning << __CXX_PREFIX << "Buffers not long enough, reallocating..."
                     << endl;
            input_ptr = (void **)realloc(input_ptr, sizeof(void *) * input_i_index);
            if (!input_ptr)
            {
                gLogError << __CXX_PREFIX << "Could not allocate input_ptr memory."
                          << endl;
                return -1;
            }
        }
        uint64_t input_i_size;
        int res = GetModelInputSize(model_name, ef.InputName.at(i), &input_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get input size of : " << ef.InputName.at(i);
            return -1;
        }
        input_ptr[input_i_index] = WrapInput((void *)input_data[i].data(), input_i_size, allocator);
    }
    return 0;
}

int TransferWorker::TransferOutput(std::string model_name, void **output_ptr, std::vector<std::vector<char>> &output_data, nvinfer1::IGpuAllocator *allocator)
{
    // 按照EngineInfo遍历的顺序对输入进行填充
    EngineInfo ef;
    int executed = 0;
    executed = GetModel(model_name, &ef);
    if (executed != 0)
    {
        gLogError << __CXX_PREFIX << "Can not get model " << model_name << endl;
        return executed;
    }

    // 首先遍历ef，取得所有的InputName和OutputName，逐个申请内存
    int input_num = ef.InputName.size();
    int output_num = ef.OutputName.size();

    // 对output逐个unwrap
    void **h_output = new void *[output_num];
    if (!h_output)
    {
        gLogError << __CXX_PREFIX << "Output allocation failed."
                  << endl;
        return -1;
    }

    // 处理device端output到host端output
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        uint64_t output_i_size;
        int res = GetModelOutputSize(model_name, ef.OutputName.at(i), &output_i_size);
        if (res)
        {
            // This should not happen
            gLogError << __CXX_PREFIX << "Can not get output size of : " << ef.OutputName.at(i);
            return -1;
        }
        h_output[i] = UnwrapOutput(output_ptr[i], output_i_size);
        allocator->free(output_ptr[i]);
        if (!h_output[i])
        {
            // 释放所有h_output[0,i-1]的内存
            for (int j = 0; j < i; j++)
            {
                allocator->free(output_ptr[j]);
                delete[](char *) h_output[j];
            }
            delete[] h_output;
            h_output = nullptr;
            output_data.clear();
            return -1;
        }
        // 对h_output进行逐字节写入到result
        std::vector<char> temp;
        for (int j = 0; j < output_i_size; j++)
        {
            temp.push_back(*((char *)h_output[i] + j));
        }
        output_data.push_back(temp);
    }

    // 释放h_output
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        delete[](char *) h_output[i];
    }
    delete[] h_output;
    return 0;
}

int TransferWorker::SaveModel(std::string model_name, std::string model_path, std::string file_name)
{
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
        return -1;
    EngineInfo ef = iter->second;
    std::string serializer = ef.engine_serialize;
    if (serializer.length() == 0)
    {
        // 说明模型没有被序列化，执行序列化
        auto h_memory = SampleUniquePtr<IHostMemory>(ef.engine->serialize());
        serializer.resize(h_memory->size());
        memcpy((void *)serializer.data(), h_memory->data(), h_memory->size());
    }
    if (file_name.length() == 0)
        file_name = model_path + model_name + ".tengine";
    else
        file_name = model_path + file_name;
    std::ofstream fout(file_name);
    if (!fout.is_open())
        return -1;
    fout.write(serializer.data(), serializer.size());
    fout.close();
    return 0;
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

int TransferWorker::Compute(std::string model_name, void **input, void **(&output))
{

    throw "Compute method not supported.";
    return -1;
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

int preProcessHostOutput(const std::vector<std::vector<char>> &output_vec, int index, void **output, uint64_t num, nvinfer1::DataType type)
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
}

std::string getTRTEngine(std::string file_path, std::string model_file)
{
    std::string engine_serializer;
    std::ifstream fin(file_path + model_file);
    if (!fin)
        return "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        engine_serializer.append(buffer.str());
    }
    return engine_serializer;
}

void *WrapInput(void *host_memory, uint64_t size, IGpuAllocator *allocator)
{
    void *res = allocator->allocate(size, 0, 0);
    int result = 0;
    check_cuda_success(cudaMemcpy(res, host_memory, size, cudaMemcpyHostToDevice), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Wrap output failed."
                  << endl;
        return nullptr;
    }
    return res;
}

void *UnwrapOutput(void *device_memory, size_t size)
{
    void *h_ptr = (void *)new char[size];
    if (!h_ptr)
    {
        gLogError << __CXX_PREFIX << "Can not malloc h_ptr"
                  << endl;
        return nullptr;
    }
    memset(h_ptr, 0, size);
    int result = 0;
    check_cuda_success(cudaMemcpy(h_ptr, device_memory, size, cudaMemcpyDeviceToHost), result);
    if (result != 0)
        gLogError << __CXX_PREFIX << "Unwrap output failed."
                  << endl;
    return h_ptr;
}

void *WrapInputAsync(void *host_memory, uint64_t size, IGpuAllocator *allocator, cudaStream_t stream)
{
    void *res = allocator->allocate(size, 0, 0);
    int result = 0;
    check_cuda_success(cudaMemcpyAsync(res, host_memory, size, cudaMemcpyHostToDevice, stream), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Wrap output failed."
                  << endl;
        return nullptr;
    }
    return res;
}

void *UnwrapOutputAsync(void *device_memory, size_t size, cudaStream_t stream)
{
    void *h_ptr = (void *)new char[size];
    if (!h_ptr)
    {
        gLogError << __CXX_PREFIX << "Can not malloc h_ptr"
                  << endl;
        return nullptr;
    }
    memset(h_ptr, 0, size);
    int result = 0;
    check_cuda_success(cudaMemcpyAsync(h_ptr, device_memory, size, cudaMemcpyDeviceToHost, stream), result);
    if (result != 0)
        gLogError << __CXX_PREFIX << "Unwrap output failed."
                  << endl;
    return h_ptr;
}