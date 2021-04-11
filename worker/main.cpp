#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
// #include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

#if !defined(_MSC_VER)
#include <getopt.h> // support on Windows MinGW
#endif
typedef struct Args
{
    string model_path;
    string engine_path;
    vector<string> input_path;
    vector<string> engine_input;
    vector<string> engine_output;
} Args;

std::vector<std::string> split(const std::string &str, const std::string &pattern)
{
    //const char* convert to char*
    char *strc = new char[strlen(str.c_str()) + 1];
    strcpy(strc, str.c_str());
    std::vector<std::string> resultVec;
    char *tmpStr = strtok(strc, pattern.c_str());
    while (tmpStr != NULL)
    {
        resultVec.push_back(std::string(tmpStr));
        tmpStr = strtok(NULL, pattern.c_str());
    }

    delete[] strc;

    return resultVec;
};

bool parseArgs(Args &args, int argc, char *argv[])
{
    if (argc == 1)
    {
        cerr << "使用-h参数或--help参数以获得更多信息。"
             << endl;
        return false;
    }

    while (1)
    {
        int arg;
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"model", required_argument, 0, 'm'}, {"engine", required_argument, 0, 'e'}, {"engine_input", required_argument, 0, 'y'}, {"engine_output", required_argument, 0, 'z'}, {"input", required_argument, 0, 'i'}, {nullptr, 0, nullptr, 0}};

        int option_index = 0;
        arg = getopt_long(argc, argv, "hm:e:y:z:i:", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h':
            cout
                << "Usage: ./main [-h or --help] [--model=<path to onnx model>] [--engine=<path to TensorRT engine>] [--input=<path to input file>]"
                << endl;
            cout << left << setw(20) << "--help "
                 << "显示帮助信息。"
                 << endl;
            cout << left << setw(20) << "--model"
                 << "ONNX模型文件的路径"
                 << endl;
            cout << left << setw(20) << "--engine"
                 << "TensorRT引擎的路径，该选项与--model参数冲突。当指定了--engine参数之后，必须指定--engine_input和--engine_output以确定模型的输入输出张量名称。"
                 << endl;
            cout << left << setw(20) << "--input"
                 << "需要用于识别的图像路径（可同时输入多张图片的路径，以逗号分隔）。"
                 << endl;
            cout << left << setw(20) << "--engine_input"
                 << "TensorRT引擎输入张量名称（可同时输入多个输入张量名称，以逗号分隔）。"
                 << endl;
            cout << left << setw(20) << "--engine_output"
                 << "TensorRT引擎输出张量名称（可同时输入多个输出张量名称，以逗号分隔）。"
                 << endl;
            return false;
        case 'm':
            if (optarg && args.engine_path.length() == 0)
            {
                args.model_path = optarg;
            }
            else
            {
                cerr << "未能读取到模型名称，或同时使用了-m和-e参数。"
                     << endl;
                return false;
            }
            break;
        case 'e':
            if (optarg && args.model_path.length() == 0)
            {
                args.engine_path = optarg;
            }
            else
            {
                cerr << "未能读取到模型名称，或同时使用了-m和-e参数。"
                     << endl;
                return false;
            }
            break;
        case 'i':
            if (optarg)
            {
                // 用户的输入保证以逗号分隔就行了
                args.input_path = split(optarg, ",");
            }
            else
            {
                cerr << "-i参数需要至少一张图片的路径。"
                     << endl;
                return false;
            }
            break;
        case 'y':
            if (optarg)
            {
                // 用户的输入保证以逗号分隔就行了
                args.engine_input = split(optarg, ",");
            }
            else
            {
                cerr << "--engine_input 参数需要至少一个输入张量的名称。"
                     << endl;
                return false;
            }
            break;
        case 'z':
            if (optarg)
            {
                // 用户的输入保证以逗号分隔就行了
                args.engine_output = split(optarg, ",");
            }
            else
            {
                cerr << "--engine_output 参数需要至少一个输出张量的名称。"
                     << endl;
                return false;
            }
            break;
        default:
            cerr << "参数错误。"
                 << endl;
            return false;
        }
    }
    return true;
}

extern uint64_t alignment;

int main(int argc, char **argv)
{
    Args args;
    auto parsed = parseArgs(args, argc, argv);
    if (!parsed)
        return -1;
    TransferWorker transfer_worker(KGMALLOCV2_ALLOCATOR);
    ComputationWorker computation_worker;
    std::vector<uint8_t> fileData(28 * 28 * sizeof(float));

    int loaded;
    if (args.model_path.length() > 0)
    {
        string model_name;
        auto pos = args.model_path.find_last_of('/');
        // For Windows
        if (pos == -1)
            auto pos = args.engine_path.find_last_of('\\');
        if (pos == -1)
        {
            cerr << "ONNX文件路径不合法。"
                 << endl;
            return 0;
        }
        pos++;
        while (pos < args.model_path.length() && args.model_path.at(pos) != '.')
        {
            model_name += args.model_path.at(pos);
            pos++;
        }
        loaded = transfer_worker.LoadModel(model_name, args.model_path, "", ONNX_FILE);
    }
    else if (args.engine_path.length() > 0)
    {
        string engine_name;
        auto pos = args.engine_path.find_last_of('/');
        // For Windows
        if (pos == -1)
            auto pos = args.engine_path.find_last_of('\\');
        if (pos == -1)
        {
            cerr << "TensorRT引擎路径不合法。"
                 << endl;
            return 0;
        }
        pos++;
        while (pos < args.engine_path.length() && args.engine_path.at(pos) != '.')
        {
            engine_name += args.engine_path.at(pos);
            pos++;
        }
        loaded = transfer_worker.LoadFromEngineFile(engine_name, args.engine_path, "", args.engine_input, args.engine_output);
    }
    else
    {
        cerr << "请至少选择一种模型进行载入。"
             << endl;
        return -1;
    }

    if (loaded == -1)
    {
        gLogFatal << "Loading mnist model into memory failed."
                  << endl;
        return loaded;
    }

    gLogInfo << "载入文件" << (args.engine_path.length() > 0 ? args.engine_path : args.model_path) << "完毕。"
             << endl;

    EngineInfo ef;
    int executed = GetModel("mnist", &ef);
    if (executed != 0)
    {
        gLogError << __CXX_PREFIX << "Get model mnist info failed."
                  << endl;
        return -1;
    }

    for (int i = 0; i < args.input_path.size(); i++)
    {
        GPUMemoryUniquePtr<void *> d_input(new void *[ef.InputName.size()]);
        d_input.get_deleter().allocator = global_allocator.get();
        d_input.get_deleter().current_length = ef.InputName.size();
        CPUMemoryUniquePtr<void *>d_output(new void *[ef.OutputName.size()]);

        if (!d_input || !d_output)
        {
            gLogError << __CXX_PREFIX << "Allocate host memory for mnist input and output failed."
                      << endl;
            return -1;
        }
        readPGMFile(args.input_path[i], fileData.data(), 28, 28);
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
        int executed = transfer_worker.TransferInput("mnist", input_data, d_input_ptr, global_allocator.get());
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
        // 暂时解除智能指针的托管
        auto d_output_ptr = d_output.release();
        executed = computation_worker.Compute("mnist", d_input.get(), d_output_ptr);
        // 恢复
        d_output.reset(d_output_ptr);

        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Compute failed, exit..."
                      << endl;
            return -1;
        }
        executed = transfer_worker.TransferOutput("mnist", d_output.get(), output_data, global_allocator.get());
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Invaild output, exit..."
                      << endl;
            return -1;
        }
        // float *output = static_cast<float *>(h_output[0].data());
        float *output ((float *)new char[output_data[0].size()]);
        if (!output)
        {
            gLogError << __CXX_PREFIX << "Can not allocate memory for output data." << endl;
            return -1;
        }
        memcpy(output, output_data[0].data(), output_data[0].size());
        {
            uint64_t OutputDim = 0;
            int executed = GetModelOutputSize("mnist", 0, &OutputDim);
            if (executed)
            {
                gLogError << __CXX_PREFIX << "Cannot get output size of output[0]"
                          << endl;
                return -1;
            }
            OutputDim /= sizeof(float);
            float val{0.0f};
            int idx{0};

            // Calculate Softmax
            float sum{0.0f};
            for (int i = 0; i < OutputDim; i++)
            {
                output[i] = exp(output[i]);
                sum += output[i];
            }

            gLogInfo << "Output:" << std::endl;
            for (int i = 0; i < OutputDim; i++)
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
        
    }
    return 0;
}