#pragma once

#include "base.hpp"
#include <exception>
#include <unordered_map>
#include <mutex>

static std::mutex models_mu;

class TransferWorker final : public IWorker
{
public:
    int Load(std::string model_name, std::istream model_file, ModelType type);
    int Unload(std::string model_name);
    std::string GetModelName(int index);
    void *Compute(std::string model_name, void *input)
    {
#pragma message("Transfer_worker doesn't not support Compute method.")
        throw "Compute method not supported.";
        return nullptr;
    }
};

// end of transfer_worker.hpp