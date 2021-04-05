#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
// #include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main()
{
    TransferWorker transfer_worker;
    ComputationWorker computation_worker;
    std::vector<uint8_t> fileData(28 * 28 * sizeof(float));
    // int mNumber = rand() % 10;
    readPGMFile(std::string() + "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/" + std::to_string(1) + ".pgm", fileData.data(), 28, 28);
    auto input_size = 28 * 28;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    for (int i = 0; i < 28 * 28; i++)
    {
        test_data[i] = 1.0 - float(fileData[i] / 255.0);
    }
    int loaded;
    loaded = transfer_worker.Load((std::string("mnist")).c_str(), "mnist.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", ONNX_FILE);
    if (loaded == -1)
    {
        gLogFatal << "Loading mnist model into memory failed." << endl;
        return loaded;
    }

    std::vector<std::vector<char>> input;
    preProcessHostInput(input, test_data, 28 * 28, nvinfer1::DataType::kFLOAT);
    std::vector<std::vector<char>> h_output;

    std::ofstream fout_1("compute_time.txt");
    std::ofstream fout_2("compute_with_stream_time.txt");
    if (!fout_1.is_open() || !fout_2.is_open())
    {
        gLogError << __CXX_PREFIX << "error." << endl;
        return -1;
    }

    int execution = 0;
    for (int i = 1; i <= 30000; i++)
    {
        _CXX_MEASURE_TIME(execution = computation_worker.Compute("mnist", input, h_output), fout_1);
        h_output.clear();
        if (execution != 0)
        {
            gLogFatal << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }
        _CXX_MEASURE_TIME(execution = computation_worker.ComputeWithStream("mnist", input, h_output), fout_2);
        if (execution != 0)
        {
            gLogFatal << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }
        h_output.clear();
    }
    fout_1.close();
    fout_2.close();
    return 0;
}