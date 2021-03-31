#include "computation_worker.hpp"
#include "common.hpp"
#include <NvInfer.h>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

std::string ComputationWorker::GetModelName(int index)
{
    mt_rw_mu.rlock();
    auto iter = model_table.find(index);
    mt_rw_mu.runlock();
    if (iter == model_table.end())
        return "";
    return iter->second;
}

void *ComputationWorker::Compute(std::string model_name, void *input)
{
    // 从table中取得已注册的引擎
    et_rw_mu.rlock();
    auto iter = engine_table.find(model_name);
    et_rw_mu.runlock();
    if (iter == engine_table.end())
    {
        cerr << __CXX_PREFIX << "engine not vaild." << endl;
        return nullptr;
    }
    nvinfer1::ICudaEngine* engine = iter->second.engine;
    if (!engine)
    {
        cerr << __CXX_PREFIX << "engine not vaild." << endl;
        return nullptr;
    }
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    if (!context)
    {
        cerr << __CXX_PREFIX << "engine start failed, context error." << endl;
        return nullptr;
    }
    int input_index = engine->getBindingIndex(iter->second.InputName.c_str());
    int output_index = engine->getBindingIndex(iter->second.OutputName.c_str());

    void *buffers[2];
    buffers[input_index] = input;
    void * output = allocator->allocate(iter->second.OutputSize, 0, 0);
    if (!output)
    {
        cerr << __CXX_PREFIX << "allocate output memory failed." << endl;
        return nullptr;
    }
    buffers[output_index] = output;
    context->executeV2(buffers);
    return output;
}

// end of computation_worker.cpp