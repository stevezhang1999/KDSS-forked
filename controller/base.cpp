#include "base.hpp"

#include <vector>
#include <string>

using namespace std;

ComputeTask::ComputeTask(string model_name, unsigned int batch_size, vector<pair<string, vector<char>>> input, uint64_t begin_timestamp, uint64_t end_timestamp, char *checksum)
{
    this->magic_number = COMPUTE_TASK_MAGIC_NUMBER;
    this->task_id = 0;
    this->model_name = model_name;
    this->batch_size = batch_size;
    for (auto i : input)
    {
        this->input.push_back(i);
    }
    for (auto is : input_size)
    {
        this->input_size.push_back(is);
    }
    output.clear();
    for (auto os : output_size)
    {
        this->output_size.push_back(os);
    }
    this->begin_timestamp = begin_timestamp;
    this->end_timestamp = end_timestamp;
    this->status = INVAILD;
    memcpy(this->checksum, checksum, sizeof(char) * 16);
    this->execute_func = EmptyTaskFunc;
    this->callback_func = EmptyTaskFunc;
    return;
}

// end of base.cpp