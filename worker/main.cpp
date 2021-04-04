#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
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
    int loaded = transfer_worker.Load("mnist", "mnist.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", ONNX_FILE);
    std::vector<std::vector<char>> input;
    preProcessHostInput(input, test_data, 28 * 28, nvinfer1::DataType::kFLOAT);
    std::vector<std::vector<char>> h_output;

    // Cold start
    h_output = computation_worker.Compute("mnist", input);
    h_output = computation_worker.ComputeWithStream("mnist", input);

    for (int i = 0; i < 20; i++)
    {
        cout << endl
             << i << " times compute: " << endl;
        MEASURE_TIME(h_output = computation_worker.Compute("mnist", input));
        MEASURE_TIME(h_output = computation_worker.ComputeWithStream("mnist", input));
    }
    return 0;
    if (!h_output.size())
    {
        gLogError << "Invaild output, exit..." << endl;
        return -1;
    }
    // float *output = static_cast<float *>(h_output[0].data());
    float *output = (float *)new char[h_output[0].size()];
    if (output)
    {
        int executed = preProcessHostOutput(h_output, 0, (void **)&output, 10, nvinfer1::DataType::kFLOAT);
        if (executed)
        {
            gLogError << "Process output failed, exit..." << endl;
            return -1;
        }
    }

    {
        uint64_t outputSize = 0;
        int executed = computation_worker.GetModelOutputSize("mnist", 0, &outputSize);
        if (executed)
        {
            gLogError << "Cannot get output size of output[0]" << endl;
            return -1;
        }
        outputSize /= sizeof(float);
        float val{0.0f};
        int idx{0};

        // Calculate Softmax
        float sum{0.0f};
        for (int i = 0; i < outputSize; i++)
        {
            output[i] = exp(output[i]);
            sum += output[i];
        }

        gLogInfo << "Output:" << std::endl;
        for (int i = 0; i < outputSize; i++)
        {
            output[i] /= sum;
            val = std::max(val, output[i]);
            if (val == output[i])
            {
                idx = i;
            }

            gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                     << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
        }
        gLogInfo << std::endl;
    }
    delete output;
    return 0;
}