#pragma once

#include "base.hpp"
#include "trt_allocator.hpp"
#include <exception>
#include <unordered_map>
#include <mutex>

using namespace nvinfer1;
extern std::shared_ptr<nvinfer1::IGpuAllocator> global_allocator;

enum ALLOCATOR_TYPE
{
    DEFAULT_ALLOCATOR,
    KGMALLOC_ALLOCATOR,
    KGMALLOCV2_ALLOCATOR,
};

class TransferWorker final : public IWorker
{
public:
    TransferWorker(ALLOCATOR_TYPE type);

    virtual ~TransferWorker();

    // LoadModel 加载模型到显存中
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，为ONNX文件
    // \param file_path 模型文件的路径
    // \param type 输入流代表的实际类型
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    virtual int LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type);

    // LoadModel 加载模型到显存中
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，为ONNX文件
    // \param file_path 模型文件的路径
    // \param type 输入流代表的实际类型
    // \param allocator 构建引擎时为工作区指定的allocator
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    int LoadModel(std::string model_name, std::string model_file, std::string file_path, ModelType type, IGpuAllocator *allocator);

    // UnLoadModel 从显存中卸载模型
    // \param model_name 模型名称，该名称是在LoadModel时指定的。
    // \returns 如果卸载不成功或模型不存在，将会返回-1，并在logger中输出错误信息
    virtual int UnloadModel(std::string model_name);

    // TransferInput 将输入从内存转移到显存，并申请对应的显存
    // \param model_name 该输入对应的模型名称
    // \param input_data 输入的字节流载荷
    // \param input_ptr 该输入对应的显存地址数组指针
    // \param allocator 转移到显存时需要用的分配器
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferInput(std::string model_name, const std::vector<std::vector<char>> input_data, void **(&input_ptr), nvinfer1::IGpuAllocator *allocator);

    // TransferOutput 将输出从显存转移到内存，并释放对应的显存
    // \param model_name 该输出对应的模型名称
    // \param output_ptr 该输出对应的显存地址数组指针
    // \param output_data 输出的字节流载荷
    // \param allocator 分配时用的allocator，必须是Compute使用的allocator
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int TransferOutput(std::string model_name, void **output_ptr, std::vector<std::vector<char>> &output_data, nvinfer1::IGpuAllocator *allocator);

    // GetModelName 获得指定索引的模型的名称
    // \param index 模型在全局的唯一索引
    // \returns 该模型的名称，如果该模型的索引不存在，将会返回空串
    virtual std::string GetModelName(int index) const;

    // Compute 开始根据模型执行计算。
    //
    // TransferWorker并不负责计算，调用Compute会直接throw。
    // \param model_name 需要调用的模型的名称
    // \param input 输入对应的显存数组指针
    // \param output 输出对应的显存指针数组的引用
    // \param input_allocator 输入使用的allocator
    // \param output_allocator 输出使用的allocator
    // \returns 执行成功则返回0，否则返回一个非0的数。
    virtual int Compute(std::string model_name, void **input, void **(&output));

    // LoadFromEngineFile 从TensorRT引擎文件加载模型
    // \param model_name 模型名称，该名称需要与目前已被加载的模型均不同，是一个唯一标识模型的名称
    // \param model_file 模型文件，为本机当前GPU生成的TensorRT引擎文件
    // \param file_path 模型文件的路径
    // \param inTensorVec 输入张量名称集合
    // \param outTensorVec 输出张量名称集合
    // \returns 该模型在全局的唯一索引，如果导入不成功，将会返回-1，并在logger中输出错误信息
    int LoadFromEngineFile(std::string model_name, std::string model_file, std::string file_path, std::vector<std::string> inTensorVec, std::vector<std::string> outTensorVec);

    // SaveModel 保存已经加载的模型
    // \param model_name 需要保存的模型的名称
    // \param model_path 需要保存的模型的路径
    // \param file_name 需要保存的模型的文件名
    // \returns 执行成功则返回0，否则返回一个非0的数。
    int SaveModel(std::string model_name, std::string model_path, std::string file_name);

};

extern uint64_t alignment;

// preProcessHostInput 对已经写好数据的input转移到vector中
//
// 需要对每个input分别操作
// \param input_vec 需要传入Compute做计算的数组
// \param input 需要写入的input
// \param size 该input包含元素的个数
// \param type 该input的类型（float传入kFLOAT，uint8_t传入kINT8，uint32_t传入kINT32）
// \returns 执行成功则返回0，否则返回一个非0的数。
int preProcessHostInput(std::vector<std::vector<char>> &input_vec, void *input, uint64_t num, nvinfer1::DataType type);

// preProcessHostInput 对已经写好数据的output_vec提取数据到output中
//
// 需要对每个output分别操作
// \param output_vec 从Compute计算得到的output数组
// \param index 需要取得的第index个output
// \param output 指向output的指针
// \param size 该output包含元素的个数
// \param type 该output的类型（float传入kFLOAT，uint8_t传入kINT8，uint32_t传入kINT32）
// \returns 执行成功则返回0，否则返回一个非0的数。
int preProcessHostOutput(const std::vector<std::vector<char>> &output_vec, int index, void **output, uint64_t num, nvinfer1::DataType type);

// WrapInput 将数据从host_memory通过allocator迁移到device_memory
// \param host_memory 指向内存的数据指针
// \param size 该段数据的大小，host_memory必须至少分配了size字节的大小，否则会导致未定义行为
// \param allocator 分配device_memory用的allocator
// \returns 执行成功则返回指向device_memory的指针，否则返回nullptr。
void *WrapInput(void *host_memory, uint64_t size, nvinfer1::IGpuAllocator *allocator);

// WrapInput 将“数据从host_memory通过allocator迁移到device_memory”事件加入到CUDA流中
// \param host_memory 指向内存的数据指针
// \param size 该段数据的大小，host_memory必须至少分配了size字节的大小，否则会导致未定义行为
// \param allocator 分配device_memory用的allocator
// \param stream 用于构建流式过程的CUDA流，必须由cudaCreateSteram生成，否则会导致未定义行为
// \returns 执行成功则返回指向device_memory的指针，否则返回nullptr。
void *WrapInputAsync(void *host_memory, uint64_t size, nvinfer1::IGpuAllocator *allocator, cudaStream_t stream);

// UnwrapOutput 将数据从device_memory迁移回host_memory
// \param device_memory 指向显存的数据指针
// \param size 该段数据的大小，device_memory必须至少分配了size字节的大小，否则会导致未定义行为
// \returns 执行成功则返回指向host_memory的指针，否则返回nullptr。
void *UnwrapOutput(void *device_memory, uint64_t size);

// UnwrapOutputAsync 将“数据从device_memory迁移回host_memory”事件加入到CUDA流中
// \param device_memory 指向显存的数据指针
// \param size 该段数据的大小，device_memory必须至少分配了size字节的大小，否则会导致未定义行为
// \param stream 用于构建流式过程的CUDA流，必须由cudaCreateSteram生成，否则会导致未定义行为
// \returns 执行成功则返回指向host_memory的指针（在stream执行完后数据可用），否则返回nullptr。
void *UnwrapOutputAsync(void *device_memory, uint64_t size, cudaStream_t stream);

// end of transfer_worker.hpp