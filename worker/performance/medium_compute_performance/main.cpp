#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int ProcessBGRImage(cv::Mat img, Dims input_dims, char *(&output));
int readCategory(std::string cate_file, std::string cate_path, std::vector<std::string> &output);
int Softmax(float *(&input), size_t size);

int main(int argc, char **argv)
{
    TransferWorker transfer_worker;
    ComputationWorker computation_worker;

    std::ofstream fout_1("compute_time.txt");
    std::ofstream fout_2("compute_with_stream_time.txt");
    // std::ofstream fout_3("compute_default_time.txt");
    if (!fout_1.is_open() || !fout_2.is_open())
    {
        gLogError << __CXX_PREFIX << "error." << endl;
        return -1;
    }

    int loaded;
    ifstream fin(std::string("/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/") + std::string("resnet_50.tengine"));
    if (!fin)
    {
        loaded = transfer_worker.Load("resnet_50", "resnet50-v1-7.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/", ONNX_FILE);
        if (loaded == -1)
        {
            gLogFatal << "Loading resnet_50 model into memory failed." << endl;
            return loaded;
        }
        gLogInfo << "Loading resnet_50 model into memory successfully." << endl;
        int saved = transfer_worker.SaveModel("resnet_50", "/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/");
        if (saved != 0)
        {
            gLogFatal << "Saving resnet_50 model into disk failed." << endl;
            return saved;
        }
        gLogInfo << "Saving resnet_50 model into disk successfully." << endl;
    }
    else
    {
        fin.close();
        loaded = transfer_worker.LoadFromEngineFile("resnet_50", "resnet_50.tengine", "/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/", vector<string>{"data"}, vector<string>{"resnetv17_dense0_fwd"});
        if (loaded == -1)
        {
            gLogFatal << "Loading resnet_50 model into memory failed." << endl;
            return loaded;
        }
    }

    EngineInfo ef;
    int got = computation_worker.GetModel("resnet_50", &ef);
    if (got != 0)
    {
        gLogFatal << "Loading resnet_50 model from memory failed." << endl;
        return got;
    }
    gLogInfo << "ResNet-50 input:" << endl;
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        uint64_t size;
        computation_worker.GetModelInputSize("resnet_50", ef.InputName.at(i), &size);
        gLogInfo
            << ef.InputName.at(i) << " Dims: " << ef.InputDim.at(i) << " Size: " << size
            << " bytes" << endl;
    }
    gLogInfo << "ResNet-50 output:" << endl;
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        uint64_t size;
        computation_worker.GetModelOutputSize("resnet_50", ef.OutputName.at(i), &size);
        gLogInfo << ef.OutputName.at(i) << " Dims: " << ef.OutputDim.at(i) << " Size: " << size
                 << " bytes" << endl;
    }

    uint64_t input_size;
    if (computation_worker.GetModelInputSize("resnet_50", 0, &input_size) != 0)
    {
        gLogError << "Can not get model input size of 0." << endl;
        return -1;
    }

    int execution = 0;
    int execution_time = 0;
    if (argc < 2)
    {
        gLogInfo << "Not found execution param, setting 2000 times." << endl;
        execution_time = 2000;
    }
    else
    {
        execution_time = std::atoi(argv[1]);
        if (execution_time == 0)
        {
            gLogError << __CXX_PREFIX << "Execution times param error, usage: [program] <execution-time>" << endl;
            gLogInfo << "setting 2000 times." << endl;
            execution_time = 2000;
        }
    }

    // 先读取类目表
    vector<string> cat;
    int read = readCategory("synset.txt", "/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/", cat);
    if (read != 0 || cat.size() != 1000)
    {
        gLogFatal << "Can not read category for ResNet-50, exiting..." << endl;
        return -1;
    }
    gLogInfo << "Read category for ResNet-50 done." << endl;

    std::vector<std::vector<char>> input;
    std::vector<std::vector<char>> h_output;

    // 输入预处理
    // ResNet-50的输入为图片，所以需要引入opencv
    const string prefix = "/home/lijiakang/TensorRT-6.0.1.5/data/resnet50/";
    for (auto pic : {"tabby_tiger_cat.jpg", "cat/cat_1.jpg", "cat/cat_2.jpg"})
    {
        input.clear();
        h_output.clear();
        Mat img = imread(prefix + pic);
        if (img.empty())
        {
            gLogFatal << "Cannot read input image." << endl;
            return -1;
        }
        if (img.channels() != 3)
        {
            gLogFatal << "Read image, but not 3 channels image." << endl;
            return -1;
        }
        resize(img, img, Size(224, 224));

        char *input_0 = new char[input_size];
        ProcessBGRImage(img, ef.InputDim[0], input_0);
        preProcessHostInput(input, input_0, 3 * 224 * 224, nvinfer1::DataType::kFLOAT);
        delete[] input_0;

        // Cold start
        execution = computation_worker.Compute("resnet_50", input, h_output);
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }

        uint64_t OutputDim = 0;
        int executed = computation_worker.GetModelOutputSize("resnet_50", 0, &OutputDim);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Cannot get output size of output[0]" << endl;
            return -1;
        }
        float *output = (float *)new char[OutputDim];
        preProcessHostOutput(h_output, 0, (void **)&output, 1000, nvinfer1::DataType::kFLOAT);
        // 走一轮softmax
        {
            OutputDim /= sizeof(float);
            Softmax(output, OutputDim);
            vector<std::pair<float, string>> prob;
            // 记录下这些category里面最长的
            size_t max_length = 0;
            for (unsigned int i = 0; i < OutputDim; i++)
            {
                prob.push_back(pair<float, string>(output[i], cat.at(i)));
                if (cat.at(i).length() > max_length)
                    max_length = cat.at(i).length();
            }
            // 排序prob
            std::sort(prob.begin(), prob.end(), [](pair<float, string> a, pair<float, string> b) { return a.first > b.first; });
            gLogInfo << "For pic: " << pic << endl;
            gLogInfo << "The detect result TOP-5 is: " << endl;
            // 输出前五个
            for (auto iter = prob.begin(); iter != prob.end() && iter != prob.begin() + 5; ++iter)
            {
                stringstream ss;
                ss << std::left << std::setw(max_length) << iter->second << " : " << std::fixed << std::setprecision(3) << std::left << std::setw(6) << setfill('0')
                   << iter->first * 100 << "%";
                cout << ss.str() << endl;
            }
        }
        gLogInfo << "Detect finished." << endl;
        delete[] output;
    }

    for (int i = 1; i <= execution_time; i++)
    {
        _CXX_MEASURE_TIME(execution = computation_worker.Compute("resnet_50", input, h_output), fout_1);
        h_output.clear();
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }

        // _CXX_MEASURE_TIME(execution = computation_worker.Compute("resnet_50_default", input, h_output), fout_3);
        // h_output.clear();
        // if (execution != 0)
        // {
        //     gLogFatal << "Model execution failed, current memory pool info: " << endl;
        //     MemPoolInfo();
        //     throw "";
        // }

        _CXX_MEASURE_TIME(execution = computation_worker.ComputeWithStream("resnet_50", input, h_output), fout_2);
        h_output.clear();
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }
    }
    fout_1.close();
    fout_2.close();
    // fout_3.close();
    return 0;
}

int ProcessBGRImage(cv::Mat img, Dims input_dims, char *(&output))
{
    const int inputC = input_dims.d[0];
    const int inputH = input_dims.d[1];
    const int inputW = input_dims.d[2];

    // 申请一个float数组
    float *img_buffer = new float[inputC * inputH * inputW];
    if (img.channels() != inputC)
    {
        gLogError << __CXX_PREFIX << "Image channel not vailed, got: " << img.channels() << " expected: " << inputC;
        return -1;
    }
    if (img.rows != inputH || img.cols != inputW)
    {
        resize(img, img, Size(inputW, inputH));
    }

    const float mean_vec[3] = {0.485f, 0.456f, 0.406f};
    const float stddev_vec[3] = {0.229f, 0.224f, 0.225f};
    // Pixel mean used by the Faster R-CNN's author
    for (int i = 0, volImg = inputC * inputH * inputW; i < 1; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                // How to preprocess? See:https://github.com/onnx/models/tree/master/vision/classification/resnet#preprocessing
                img_buffer[i * volImg + c * volChl + j] = (float(img.data[j * inputC + 2 - c]) / 255 - mean_vec[c]) / stddev_vec[c];
            }
        }
    }
    std::memmove((void *)output, img_buffer, sizeof(float) * inputC * inputH * inputW);
    delete[] img_buffer;
    return 0;
}

int readCategory(std::string cate_file, std::string cate_path, std::vector<std::string> &output)
{
    std::ifstream fin(cate_path + cate_file);
    if (!fin.is_open())
    {
        gLogError << __CXX_PREFIX << "Can not read category file." << endl;
        return -1;
    }
    std::string temp;
    while (fin.peek() != EOF)
    {
        getline(fin, temp, '\n');
        output.push_back(temp);
        temp.clear();
    }
    fin.close();
    return 0;
}

int Softmax(float *(&input), size_t size)
{
    // reference: https://blog.csdn.net/m0_37477175/article/details/79686164
    double sum{0.0f};
    // 防止上下溢
    float M = numeric_limits<float>::min();
    for (int i = 0; i < size; i++)
    {
        M = std::max(M, input[i]);
    }
    for (int i = 0; i < size; i++)
    {
        sum += exp(input[i] - M);
    }
    for (int i = 0; i < size; i++)
    {
        input[i] = exp((input[i] - M) - log(sum));
    }
    return 0;
}