#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
// #include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm> // for transform to lower
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
    if (argc < 3)
    {
        gLogInfo << "Not found execution param, setting 2000 times."
                 << endl;
        execution_time = 2000;
    }
    else
    {
        execution_time = std::atoi(argv[2]);
        if (execution_time == 0)
        {
            gLogError << "Execution times param error, usage: [program] <allocator> <execution-time>"
                      << endl;
            gLogInfo << "setting 2000 times."
                     << endl;
            execution_time = 2000;
        }
    }

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

    std::ofstream fout[2];
    const string file_name[] = {"compute_time.txt",
                                "compute_with_stream_time.txt"};
    for (int i = 0; i < 2; i++)
    {
        auto name = file_name[i];
        switch (type)
        {
        case DEFAULT_ALLOCATOR:
            name = "default_allocator_" + name;
            break;
        case KGMALLOC_ALLOCATOR:
            name = "kgmalloc_allocator_" + name;
            break;
        case KGMALLOCV2_ALLOCATOR:
            name = "kgmallocv2_allocator_" + name;
            break;
        default:
            break;
        }
        fout[i].open(name);
        if (!fout[i].is_open())
        {
            gLogError << __CXX_PREFIX << "error."
                      << endl;
            return -1;
        }
    }

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

    for (int i = 1; i <= execution_time; i++)
    {
        if (i % 100 == 0)
            gLogInfo << "Executed " << i << " times" << endl;
        // 暂时解除智能指针的托管
        auto d_output_ptr = d_output.release();
        _CXX_MEASURE_TIME(executed = computation_worker.Compute("mnist", d_input.get(), d_output_ptr, global_allocator.get(), ctx1.get(), &ef), fout[0]);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Compute failed, exit..."
                      << endl;
            return -1;
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);

        // 暂时解除智能指针的托管
        d_output_ptr = d_output.release();
        _CXX_MEASURE_TIME(executed = computation_worker.ComputeWithStream("mnist", d_input.get(), d_output_ptr, global_allocator.get(), ctx2.get(), &ef), fout[1]);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Compute failed, exit..."
                      << endl;
            return -1;
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);
    }

    printCurrentPool(dynamic_cast<KGAllocatorV2 *>(global_allocator.get()));

    for (int i = 0; i < 2; i++)
        fout[i].close();
    return 0;
}