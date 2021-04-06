#include "trt_allocator.hpp"
#include "common.hpp"
#include "kgmalloc.hpp"
#include "umap/umap.hpp"
#include "hash/hash.hpp"
#include "common/logger.h" // On TensorRT/samples
#include "common/common.h"
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <memory>

using std::cerr;
using std::cout;
using std::endl;
using std::ostringstream;
using std::string;

// kg_allocator - 全局唯一allocator
std::shared_ptr<nvinfer1::IGpuAllocator> kg_allocator(new KGAllocator());
// std::shared_ptr<nvinfer1::IGpuAllocator> kg_allocator = nullptr;

// KGAllocator 执行底层kgmalloc初始化的构造函数
KGAllocator::KGAllocator()
{
    // 申请最多8GB的显存
    // 小结点申请512字节一个node，在本机上是以512为单位对齐的
    KGErrCode err = KGInit(FIRST_FIT, 0, 8_MiB, 512, 8_GiB);
    if (err != KGMALLOC_SUCCESS)
    {
        string err_msg;
        ostringstream oss(err_msg);
        oss << __CXX_PREFIX << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        gLogError << __CXX_PREFIX << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        throw oss.str().c_str();
    }
}

void *KGAllocator::allocate(uint64_t size, uint64_t alignment, uint32_t flags)
{
    if (size == 0)
    {
        return nullptr;
    }
    uint64_t alignment_size;
    alloc_mu.lock();
    if (alignment != 0)
    {
        // gLogInfo << "KGAllocator does not enable alignment." << endl;
        // 实现alignment，不然有可能会导致数据有覆盖的情况
        alignment_size = alignment;
        while (alignment_size < size)
            alignment_size += alignment;
    }
    else
    {
        alignment_size = size;
    }
    CudaMemNode **node = new (CudaMemNode *);
    if (!node)
    {
        gLogError << __CXX_PREFIX << "node memory allocate failed." << endl;
        alloc_mu.unlock();
        return nullptr;
    }
    KGErrCode err;
    unsigned int hash = 0;
    err = GetHash(&hash);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << __CXX_PREFIX << "allocate failed, err: " << err << endl;
        alloc_mu.unlock();
        return nullptr;
    }
    err = KGAllocMem(node, alignment_size, hash, -1);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << __CXX_PREFIX << "allocate failed, err: " << err << endl;
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
    // TODO：修复UMAP erase的错误
    CudaMemNode **node = static_cast<CudaMemNode **>(iter->second);
    KGErrCode err = KGReleaseMem(node);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << __CXX_PREFIX << "free failed, err: " << err << endl;
    }
    delete node;
    node_pool.erase(iter);
    alloc_mu.unlock();
    return;
}

KGAllocator::~KGAllocator()
{
    KGErrCode err = KGDestroy();
    if (err != KGMALLOC_SUCCESS)
        gLogError << __CXX_PREFIX << "recycle kgmalloc memory failed, err: " << err << endl;
    gLogInfo << "Memory pool destroyed." << endl;
    return;
}

// end of trt_allocator.cpp