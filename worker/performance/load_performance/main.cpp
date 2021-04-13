#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif
int main(int argc,char **argv)
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
    std::vector<uint8_t> fileData(28 * 28 * sizeof(float));
    // int mNumber = rand() % 10;
    readPGMFile(std::string() + "/home/lijiakang/TensorRT-7.1.3.4/data/mnist/" + std::to_string(1) + ".pgm", fileData.data(), 28, 28);
    auto input_size = 28 * 28;
    float test_data[28 * 28];
    memset(test_data, 0, sizeof(float) * 28 * 28);
    for (int i = 0; i < 28 * 28; i++)
    {
        test_data[i] = 1.0 - float(fileData[i] / 255.0);
    }
    int loaded;
    std::ofstream fout_1("onnx_load_time.txt");
    std::ofstream fout_2("trtengine_load_time.txt");
    if (!fout_1.is_open() || !fout_2.is_open())
    {
        gLogError << __CXX_PREFIX << "error." << endl;
        return -1;
    }
    for (int i = 1; i <= 20; i++)
    {
        _CXX_MEASURE_TIME(loaded = transfer_worker.LoadModel((std::string("mnist") + std::to_string(i)).c_str(), "mnist-7.onnx", "/home/lijiakang/KDSS/model/", ONNX_FILE), fout_1);
        if (loaded == -1)
        {
            gLogFatal << "Loading mnist model into memory failed." << endl;
            return loaded;
        }
        vector<string> in_vec;
        in_vec.push_back("Input3");
        vector<string> out_vec;
        out_vec.push_back("Plus214_Output_0");
        _CXX_MEASURE_TIME(loaded = transfer_worker.LoadFromEngineFile((std::string("mnist_standard") + std::to_string(i)).c_str(), "mnist.tengine", "/home/lijiakang/KDSS/model/", in_vec, out_vec), fout_2);
        if (loaded == -1)
        {
            gLogFatal << "Loading mnist model into memory failed." << endl;
            return loaded;
        }
    }
    fout_1.close();
    fout_2.close();
    return 0;
}