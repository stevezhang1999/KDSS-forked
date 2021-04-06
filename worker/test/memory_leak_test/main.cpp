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
    int mNumber = rand() % 10;
    readPGMFile(std::string() + "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/" + std::to_string(mNumber) + ".pgm", fileData.data(), 28, 28);
    auto input_size = 28 * 28;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    for (int i = 0; i < 28 * 28; i++)
    {
        test_data[i] = 1.0 - float(fileData[i] / 255.0);
    }

    std::vector<std::string> model_input_name = {"Input3"};
    std::vector<std::string> model_output_name = {"Plus214_Output_0"};
    int loaded;
    loaded = transfer_worker.LoadFromEngineFile("mnist", "mnist.trtengine", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", model_input_name, model_output_name);
    if (loaded == -1)
    {
        gLogFatal << "Loading mnist model into memory failed." << endl;
        return loaded;
    }

    std::vector<std::vector<char>> input;
    preProcessHostInput(input, test_data, 28 * 28, nvinfer1::DataType::kFLOAT);
    std::vector<std::vector<char>> h_output;

    int executed;
    uint64_t loop_time = 0;
    gLogInfo << "Looping for mnist computation for memory leak..." << endl;
    while (1)
    {
        executed = computation_worker.Compute("mnist", input, h_output);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Invaild output, exit..." << endl;
            return -1;
        }
        loop_time++;
        if (loop_time % 100 == 0)
        {
            gLogInfo << "Looped " << loop_time << " times." << endl;
        }
    }
    return 0;
}