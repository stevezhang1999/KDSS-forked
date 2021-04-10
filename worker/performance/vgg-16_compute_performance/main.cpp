#include "worker.hpp"
#include "common/common.h"
#include "kgmalloc.hpp"
#include "common.hpp"
#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

using namespace cv;
using namespace std;

int ProcessBGRImage(cv::Mat img, Dims input_dims, char *(&output));
int readCategory(std::string cate_file, std::string cate_path, std::vector<std::string> &output);
int Softmax(float *(&input), size_t size);

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        cout << "Usage: ./main <allocator_name> [execution_times]" << endl;
        return -1;
    }
    ALLOCATOR_TYPE type;
#if CV_VERSION_MAJOR >= 3 && CV_VERSION_MINOR >= 4 && CV_VERSION_PATCH <= 14
    string type_string = cv::String(argv[1]).toLowerCase();
#else
    string type_string = toLowerCase(argv[1]);
#endif
    if (type_string == "default")
        type = DEFAULT_ALLOCATOR;
    else if (type_string == "kgmalloc")
        type = KGMALLOC_ALLOCATOR;
    else
        type = KGMALLOCV2_ALLOCATOR;
    TransferWorker transfer_worker(type);
    ComputationWorker computation_worker;

    int loaded;
    ifstream fin(std::string("/home/lijiakang/TensorRT-6.0.1.5/data/vgg/") + std::string("vgg-16.tengine"));
    if (!fin)
    {
        DefaultAllocator *df = new DefaultAllocator();
        loaded = transfer_worker.LoadModel("vgg-16", "vgg16-7.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/vgg/", ONNX_FILE, df, 1_GiB);
        delete df;
        if (loaded == -1)
        {
            gLogFatal << "Loading vgg-16 model into memory failed." << endl;
            return loaded;
        }
        gLogInfo << "Loading vgg-16 model into memory successfully." << endl;
        int saved = transfer_worker.SaveModel("vgg-16", "/home/lijiakang/TensorRT-6.0.1.5/data/vgg/", "vgg-16.tengine");
        if (saved != 0)
        {
            gLogFatal << "Saving vgg-16 model into disk failed." << endl;
            return saved;
        }
        gLogInfo << "Saving vgg-16 model into disk successfully." << endl;
    }
    else
    {
        fin.close();
        loaded = transfer_worker.LoadFromEngineFile("vgg-16", "vgg-16.tengine", "/home/lijiakang/TensorRT-6.0.1.5/data/vgg/", vector<string>{"data"}, vector<string>{"vgg0_dense2_fwd"});
        if (loaded == -1)
        {
            gLogFatal << "Loading vgg-16 model into memory failed." << endl;
            return loaded;
        }
    }

    gLogInfo << "Running model vgg-16 performance test..." << endl;
    EngineInfo ef;
    int got = GetModel("vgg-16", &ef);
    if (got != 0)
    {
        gLogFatal << "Loading vgg-16 model from memory failed." << endl;
        return got;
    }

    gLogInfo << "vgg-16 input:" << endl;
    for (int i = 0; i < ef.InputName.size(); i++)
    {
        uint64_t size;
        GetModelInputSize("vgg-16", ef.InputName.at(i), &size);
        gLogInfo
            << ef.InputName.at(i) << " Dims: " << ef.InputDim.at(i) << " Size: " << size
            << " bytes" << endl;
    }
    gLogInfo << "vgg-16 output:" << endl;
    for (int i = 0; i < ef.OutputName.size(); i++)
    {
        uint64_t size;
        GetModelOutputSize("vgg-16", ef.OutputName.at(i), &size);
        gLogInfo << ef.OutputName.at(i) << " Dims: " << ef.OutputDim.at(i) << " Size: " << size
                 << " bytes" << endl;
    }

    uint64_t input_size;
    if (GetModelInputSize("vgg-16", 0, &input_size) != 0)
    {
        gLogError << "Can not get model input size of 0." << endl;
        return -1;
    }

    int execution = 0;
    int execution_time = 0;
    if (argc < 3)
    {
        gLogInfo << "Not found execution param, setting 2000 times." << endl;
        execution_time = 2000;
    }
    else
    {
        execution_time = std::atoi(argv[2]);
        if (execution_time == 0)
        {
            gLogError << __CXX_PREFIX << "Execution times param error, usage: [program] <allocator> <execution-time>" << endl;
            gLogInfo << "setting 2000 times." << endl;
            execution_time = 2000;
        }
    }

    // 先读取类目表
    vector<string> cat;
    int read = readCategory("synset.txt", "/home/lijiakang/TensorRT-6.0.1.5/data/vgg/", cat);
    if (read != 0 || cat.size() != 1000)
    {
        gLogFatal << "Can not read category for vgg-16, exiting..." << endl;
        return -1;
    }
    gLogInfo << "Read category for vgg-16 done." << endl;

    GPUMemoryUniquePtr<void *> d_input(new void *[ef.InputName.size()]);
    GPUMemoryUniquePtr<void *> d_output(new void *[ef.OutputName.size()]);

    if (!d_input || !d_output)
    {
        gLogError << __CXX_PREFIX << "Allocate host memory for vgg-16 input and output failed."
                  << endl;
        return -1;
    }

    memset(d_input.get(), 0, sizeof(void *) * ef.InputName.size());
    memset(d_output.get(), 0, sizeof(void *) * ef.OutputName.size());

    std::vector<std::vector<char>> input_data;
    std::vector<std::vector<char>> output_data;

    // 输入预处理
    // ResNet-50的输入为图片，所以需要引入opencv
    const string prefix = "/home/lijiakang/TensorRT-6.0.1.5/data/vgg/";
    for (auto pic : {"tabby_tiger_cat.jpg", "cat/cat_1.jpg", "cat/cat_2.jpg"})
    {
        // 申请device端output空间，该空间会在TransferOutput时被释放掉
        for (int i = 0; i < ef.OutputName.size(); i++)
        {
            uint64_t size;
            int executed = GetModelOutputSize("vgg-16", i, &size);
            if (executed != 0)
            {
                gLogError << __CXX_PREFIX << "Can not get vgg-16 output size"
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
        global_allocator->free(d_input.get()[0]);
        input_data.clear();
        output_data.clear();
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
        preProcessHostInput(input_data, input_0, 3 * 224 * 224, nvinfer1::DataType::kFLOAT);
        delete[] input_0;

        // 处理host端input
        // 暂时解除智能指针的托管
        auto d_input_ptr = d_input.release();
        int executed = transfer_worker.TransferInput("vgg-16", input_data, d_input_ptr, global_allocator.get());
        // 恢复
        d_input.reset(d_input_ptr);
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Invaild input, exit..."
                      << endl;
            return -1;
        }

        // Cold start
        // 暂时解除智能指针的托管
        auto d_output_ptr = d_output.release();
        executed = computation_worker.Compute("vgg-16", d_input.get(), d_output_ptr);
        // 恢复
        d_output.reset(d_output_ptr);
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            MemPoolInfo();
            throw "";
        }
        executed = transfer_worker.TransferOutput("vgg-16", d_output.get(), output_data, global_allocator.get());
        if (executed != 0)
        {
            gLogError << __CXX_PREFIX << "Invaild output, exit..."
                      << endl;
            return -1;
        }
        float *output((float *)new char[output_data[0].size()]);
        if (!output)
        {
            gLogError << __CXX_PREFIX << "Can not allocate memory for output data." << endl;
            return -1;
        }
        memcpy(output, output_data[0].data(), output_data[0].size());
        // 走一轮softmax
        {
            uint64_t OutputDim = 0;
            int executed = GetModelOutputSize("vgg-16", 0, &OutputDim);
            if (executed)
            {
                gLogError << __CXX_PREFIX << "Cannot get output size of output[0]"
                          << endl;
                return -1;
            }
            OutputDim /= sizeof(float);
            Softmax(output, OutputDim);
            vector<std::pair<float, string>> prob;
            // 记录下这些category里面最长的
            size_t max_length = 0;
            for (unsigned int i = 0; i < OutputDim; i++)
            {
                prob.push_back(pair<float, string>(output[i], cat.at(i)));
            }
            // 排序prob
            std::sort(prob.begin(), prob.end(), [](pair<float, string> a, pair<float, string> b) { return a.first > b.first; });
            gLogInfo << "For pic: " << pic << endl;
            gLogInfo << "The detect result TOP-5 is: " << endl;
            // 输出前五个
            for (int i = 0; i < 5; i++)
            {
                if (cat.at(i).length() > max_length)
                    max_length = cat.at(i).length();
            }
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

    printCurrentPool(dynamic_cast<KGAllocatorV2 *>(global_allocator.get()));
    // 性能测试开始
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

    // 创建上下文
    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter>
        ctx1(ef.engine->createExecutionContextWithoutDeviceMemory());
    if (!ctx1)
    {
        gLogFatal << __CXX_PREFIX << "Can not create execution context of vgg-16" << endl;
        return -1;
    }

    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter> ctx2(ef.engine->createExecutionContextWithoutDeviceMemory());
    if (!ctx2)
    {
        gLogFatal << __CXX_PREFIX << "Can not create execution context of vgg-16" << endl;
        return -1;
    }

    int executed = 0;
    for (int i = 1; i <= execution_time; i++)
    {
        if (i % 100 == 0)
        {
            gLogInfo << "Test " << i << " times" << endl;
        }

        // 暂时解除智能指针的托管
        auto d_output_ptr = d_output.release();
        _CXX_MEASURE_TIME(executed = computation_worker.Compute("vgg-16", d_input.get(), d_output_ptr, global_allocator.get(), ctx1.get(), &ef), fout[0]);
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            switch (type)
            {
            case KGMALLOC_ALLOCATOR:
                MemPoolInfo();
                break;
            case KGMALLOCV2_ALLOCATOR:
                printCurrentPool(dynamic_cast<KGAllocatorV2 *>(global_allocator.get()));
                break;
            default:
                break;
            }
            throw "";
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);

#if NV_TENSORRT_MAJOR <= 6 // TensorRT 7好像并不支持流式传输重用上下文
        // 暂时解除智能指针的托管
        d_output_ptr = d_output.release();
        _CXX_MEASURE_TIME(executed = computation_worker.ComputeWithStream("vgg-16", d_input.get(), d_output_ptr, global_allocator.get(), ctx2.get(), &ef), fout[1]);
        if (execution != 0)
        {
            gLogFatal << __CXX_PREFIX << "Model execution failed, current memory pool info: " << endl;
            switch (type)
            {
            case KGMALLOC_ALLOCATOR:
                MemPoolInfo();
                break;
            case KGMALLOCV2_ALLOCATOR:
                printCurrentPool(dynamic_cast<KGAllocatorV2 *>(global_allocator.get()));
                break;
            default:
                break;
            }
            throw "";
        }
        output_data.clear();
        // 恢复
        d_output.reset(d_output_ptr);
#endif
    }
    for (int i = 0; i < 2; i++)
        fout[i].close();

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