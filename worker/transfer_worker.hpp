#pragma once

#include "base.hpp"
#include <exception>
#include <unordered_map>
#include <mutex>

class TransferWorker final : public IWorker
{
public:
    TransferWorker(){};
    virtual ~TransferWorker();
    virtual int Load(std::string model_name, std::string model_file, std::string file_path, ModelType type);
    virtual int Unload(std::string model_name);
    virtual std::string GetModelName(int index) const;
    virtual void *Compute(std::string model_name, void *input);
};

// end of transfer_worker.hpp