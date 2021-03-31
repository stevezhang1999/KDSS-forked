#include "trt_allocator.hpp"
#include "common.hpp"
#include "kgmalloc.h"
#include "hash/hash.h"
#include "common/logger.h" // On TensorRT/samples
#include <string>
#include <sstream>
#include <iostream>

using std::cerr;
using std::endl;
using std::ostringstream;
using std::string;

// KGAllocator 执行底层kgmalloc初始化的构造函数
KGAllocator::KGAllocator()
{
    KGErrCode err = KGInit(FIRST_FIT);
    if (err != KGMALLOC_SUCCESS)
    {
        string err_msg;
        ostringstream oss(err_msg);
        oss << __CXX_PREFIX << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        gLogError << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        throw oss.str().c_str();
    }
#ifdef __DEBUG
    MemPoolInfo();
#endif
}

void *KGAllocator::allocate(uint64_t size, uint64_t alignment, uint32_t flags)
{
    if (size == 0)
    {
        return nullptr;
    }
    alloc_mu.lock();
    if (alignment > 0)
    {
        gLogError << "KGAllocator does not enable alignment." << endl;
    }
    CudaMemNode **node = new (CudaMemNode *);
    if (!node)
    {
        gLogError << "node memory allocate failed." << endl;
        alloc_mu.unlock();
        return nullptr;
    }
    KGErrCode err;
    unsigned int hash = 0;
    err = GetHash(&hash);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << "allocate failed, err: " << err;
        alloc_mu.unlock();
        return nullptr;
    }
    err = KGAllocMem(node, size, hash, -1);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << "allocate failed, err: " << err;
        alloc_mu.unlock();
        return nullptr;
    }
    node_pool.insert(std::pair<void *, void *>((*node)->d_ptr, node));
    alloc_mu.unlock();
    return (*node)->d_ptr;
}

void KGAllocator::free(void *memory)
{
    if (!memory)
        return;
    alloc_mu.lock();
    auto iter = node_pool.find(memory);
    if (iter == node_pool.end())
    {
        alloc_mu.unlock();
        return;
    }
    KGErrCode err = KGReleaseMem((CudaMemNode **)iter->second);
    if (err != KGMALLOC_SUCCESS)
        gLogError << "free failed, err: " << err;
    alloc_mu.unlock();
    return;
}

KGAllocator::~KGAllocator()
{
    KGErrCode err = KGDestroy();
    if (err != KGMALLOC_SUCCESS)
        gLogError << "recycle kgmalloc memory failed, err: " << err;
    return;
}

// end of trt_allocator.cpp