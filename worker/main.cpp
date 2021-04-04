#include "worker.hpp"
#include "common/common.h"
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
    readPGMFile(std::string() + "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/" + std::to_string(6) + ".pgm", fileData.data(), 28, 28);
    auto input_size = 28 * 28;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    for (int i = 0; i < 28*28; i++)
    {
        test_data[i] = 1.0 - float(fileData[i] / 255.0);
    }
    transfer_worker.Load("mnist", "mnist.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", ONNX_FILE);
    // transfer_worker.Load("mnist_standard", "mnist.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/", ONNX_FILE, test_data);

    void *h_output = computation_worker.Compute("mnist", test_data);
    if (!h_output)
    {
        gLogError << "Invaild output, exit..." << endl;
        return -1;
    }
    float *output = static_cast<float *>(h_output);
    {
        const int outputSize = computation_worker.GetModelOutputDim("mnist")[0];
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