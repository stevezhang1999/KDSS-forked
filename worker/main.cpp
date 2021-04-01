#include "worker.hpp"
// #include <NvInfer.h>

int main()
{
    TransferWorker transfer_worker;
    ComputationWorker computaion_worker;
    transfer_worker.Load("mnist", "mnist.onnx", "/home/lijiakang/TensorRT-6.0.1.5/data/mnist/",ONNX_FILE);
    return 0;
}