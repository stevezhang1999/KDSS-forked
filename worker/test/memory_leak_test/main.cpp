#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
// #include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <string>

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

using namespace std;

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        cout << "Usage: ./main <allocator_name> [execution_times]" << endl;
        return -1;
    }
    ALLOCATOR_TYPE type;
    string type_string = argv[1];
    transform(type_string.begin(), type_string.end(), type_string.begin(), ::tolower);
    if (type_string == "default")
        type = DEFAULT_ALLOCATOR;
    else if (type_string == "kgmalloc")
        type = KGMALLOC_ALLOCATOR;
    else
        type = KGMALLOCV2_ALLOCATOR;
    TransferWorker transfer_worker(type);
    ComputationWorker computation_worker;

    int loaded;
    loaded = transfer_worker.LoadFromEngineFile("mnist", "mnist.tengine", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", {"Input3"}, {"Plus214_Output_0"});
    if (loaded == -1)
    {
        gLogFatal << "Loading mnist model into memory failed."
                  << endl;
        return loaded;
    }

    int execution = 0;
    int execution_time = 0;

    gLogInfo << "Running model mnist performance test..." << endl;
    EngineInfo ef;
    int executed = GetModel("mnist", &ef);
    if (executed != 0)
    {
        gLogError << __CXX_PREFIX << "Get model mnist info failed."
                  << endl;
        return -1;
    }

    GPUMemoryUniquePtr<void *> d_input(new void *[ef.InputName.size()]);
    GPUMemoryUniquePtr<void *> d_output(new void *[ef.OutputName.size()]);

    if (!d_input || !d_output)
    {
        gLogError << __CXX_PREFIX << "Allocate host memory for mnist input and output failed."
                  << endl;
        return -1;
    }

    memset(d_input.get(), 0, sizeof(void *) * ef.InputName.size());
    memset(d_output.get(), 0, sizeof(void *) * ef.OutputName.size());

    std::vector<uint8_t> fileData(28 * 28 * sizeof(float));
    // int mNumber = rand() % 10;
    readPGMFile("/home/lijiakang/TensorRT-6.0.1.5/data/mnist/1.pgm", fileData.data(), 28, 28);
    auto input_size = 28 * 28;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    for (int i = 0; i < 28 * 28; i++)
    {
        test_data[i] = 1.0 - float(fileData[i] / 255.0);
    }

    std::vector<std::vector<char>> input_data;
    preProcessHostInput(input_data, test_data, 28 * 28, nvinfer1::DataType::kFLOAT);
    std::vector<std::vector<char>> output_data;

    // 处理host端input
    // 暂时解除智能指针的托管
    auto d_input_ptr = d_input.release();
    executed = transfer_worker.TransferInput("mnist", input_data, d_input_ptr, global_allocator.get());
    // 恢复
    d_input.reset(d_input_ptr);
    if (executed != 0)
    {
        gLogError << __CXX_PREFIX << "Invaild input, exit..."
                  << endl;
        return -1;
    }

    // 申请device端output空间
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        uint64_t size;
        executed = GetModelOutputSize("mnist", i, &size);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Can not get mnist output size"
                      << endl;
            return -1;
        }
        d_output.get()[i] = global_allocator->allocate(size, alignment, 0);
        if (!d_output.get()[i])
        {
            // 释放从0~i-1的所有显存
            for (int j = 0; j < i; j++)
                global_allocator->free(d_output.get()[j]);
            gLogError << __CXX_PREFIX << "allocating for device memory failed."
                      << endl;
            return -1;
        }
    }

    // 创建上下文
    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter>
        ctx1(ef.engine->createExecutionContextWithoutDeviceMemory());
    if (!ctx1)
    {
        gLogFatal << __CXX_PREFIX << "Can not create execution context of ResNet-50" << endl;
        return -1;
    }

    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter> ctx2(ef.engine->createExecutionContextWithoutDeviceMemory());
    if (!ctx2)
    {
        gLogFatal << __CXX_PREFIX << "Can not create execution context of ResNet-50" << endl;
        return -1;
    }

    uint64_t loop_time = 0;
    gLogInfo << "Looping for mnist computation for memory leak..." << endl;
    while (1)
    {
        if (loop_time % 10000 == 0)
            gLogInfo << "Executed " << loop_time << " times" << endl;
        // 暂时解除智能指针的托管
        auto d_output_ptr = d_output.release();
        executed = computation_worker.Compute("mnist", d_input.get(), d_output_ptr, global_allocator.get(), ctx1.get(), &ef);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Compute failed, exit..."
                      << endl;
            return -1;
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);
#if NV_TENSORRT_MAJOR <= 6 // TensorRT 7好像并不支持流式传输重用上下文
        // 暂时解除智能指针的托管
        d_output_ptr = d_output.release();
        executed = computation_worker.ComputeWithStream("mnist", d_input.get(), d_output_ptr, global_allocator.get(), ctx2.get(), &ef);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Compute failed, exit..."
                      << endl;
            return -1;
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);
#endif
        loop_time++;
    }
    return 0;
}