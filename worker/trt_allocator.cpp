#include "trt_allocator.hpp"
#include "common.hpp"
#include "kgmalloc.hpp"
#include "umap/umap.hpp"
#include "hash/hash.hpp"
#include "common/logger.h" // On TensorRT/samples
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
    // 大结点2MB一个，小结点4KB一个
    KGErrCode err = KGInit(FIRST_FIT, 0, static_cast<size_t>(1 << 21), static_cast<size_t>(1 << 12));
    if (err != KGMALLOC_SUCCESS)
    {
        string err_msg;
        ostringstream oss(err_msg);
        oss << __CXX_PREFIX << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        gLogError << "kgmalloc init failed. Error code: " << static_cast<int>(err);
        throw oss.str().c_str();
    }
}

void *KGAllocator::allocate(uint64_t size, uint64_t alignment, uint32_t flags)
{
    if (size == 0)
    {
        return nullptr;
    }
    alloc_mu.lock();
    if (alignment != 0 && (alignment & (alignment - 1)))
    {
        gLogInfo << "KGAllocator does not enable alignment." << endl;
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
        gLogError << "allocate failed, err: " << err << endl;
        alloc_mu.unlock();
        return nullptr;
    }
    err = KGAllocMem(node, size, hash, -1);
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << "allocate failed, err: " << err << endl;
        alloc_mu.unlock();
        return nullptr;
    }
    // gLogInfo << "Executing func: "
    //          << "InsertUMap(" << node << ", " << (*node)->self << ");" << endl;
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << "register node on umap failed, err: " << err << endl;
    }
    // if (PrintUMap(UMapPtrToAddr).length() != 0)
    // {
    //     gLogInfo << "Current UMapPtrToAddr:" << endl;
    //     cout << PrintUMap(UMapPtrToAddr) << endl;
    // }
    // if (PrintUMap(UMapAddrToPtr).length() != 0)
    // {
    //     gLogInfo << "Current UMapAddrToPtr:" << endl;
    //     cout << PrintUMap(UMapAddrToPtr) << endl;
    // }
    node_pool.insert(std::pair<void *, void *>((*node)->d_ptr, node));
    // if (PrintUMap(node_pool).length() != 0)
    // {
    //     gLogInfo << "Current node_pool:" << endl;
    //     cout << PrintUMap(node_pool) << endl;
    // }
    alloc_mu.unlock();
    // MemPoolInfo();
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
    // gLogInfo << "Executing func: "
    //          << "KGReleaseMem(" << node << ");" << endl;
    if (err != KGMALLOC_SUCCESS)
    {
        gLogError << "free failed, err: " << err << endl;
    }
    // if (PrintUMap(UMapPtrToAddr).length() != 0)
    // {
    //     gLogInfo << "Current UMapPtrToAddr:" << endl;
    //     cout << PrintUMap(UMapPtrToAddr) << endl;
    // }
    // if (PrintUMap(UMapAddrToPtr).length() != 0)
    // {
    //     gLogInfo << "Current UMapAddrToPtr:" << endl;
    //     cout << PrintUMap(UMapAddrToPtr) << endl;
    // }
    // if (PrintUMap(node_pool).length() != 0)
    // {
    //     gLogInfo << "Current node_pool:" << endl;
    //     cout << PrintUMap(node_pool) << endl;
    // }
    node_pool.erase(iter);
    alloc_mu.unlock();
    // MemPoolInfo();
    return;
}

KGAllocator::~KGAllocator()
{
    KGErrCode err = KGDestroy();
    if (err != KGMALLOC_SUCCESS)
        gLogError << "recycle kgmalloc memory failed, err: " << err << endl;
    gLogInfo << "Memory pool destroyed." << endl;
    return;
}

// end of trt_allocator.cpp