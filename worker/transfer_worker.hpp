#pragma once

#include "base.hpp"
#include <exception>
#include <unordered_map>
#include <mutex>

class TransferWorker final : public IWorker
{
public:
    int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type);
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