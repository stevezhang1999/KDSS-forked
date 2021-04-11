#include "trt_allocator.hpp"
#include "common.hpp"
#include "kgmalloc.hpp"
#include "umap/umap.hpp"
#include "hash/hash.hpp"
#include "common/logger.h" // On TensorRT/samples
#include "common/common.h"
#include <nvml.h> // for detect GPU running process and power
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <thread>
#include <algorithm>

#if NV_TENSORRT_MAJOR >= 7
using namespace sample;
#endif

using namespace std;

// global_allocator - 全局唯一allocator
std::shared_ptr<nvinfer1::IGpuAllocator> global_allocator = nullptr;

void AllocatorInit()
{
#if defined _WIN32
    cerr << "We don't implemented NVML support for Windows, using device on index 0." << endl;
    return;
#endif
    int nvml_running = 0;
    int result = 0;
    check_nvml_success(nvmlInit_v2(), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Can not start NVML, which means we can not choose the device which is not running job properly." << endl;
    }
    else
    {
        nvml_running = 1;
    }
    unsigned int current_device_index = 32767;
    if (nvml_running == 1)
    {
        // Find the device which is not using.
        // Get device count first.
        unsigned int device_count;
        // NVMLShutter promise that every exit can call NvmlShutdown()
        std::unique_ptr<void *, NVMLShutter> nvml_guard(nullptr);
        check_nvml_success(nvmlDeviceGetCount_v2(&device_count), result);
        if (result != 0)
        {
            gLogFatal << __CXX_PREFIX << "[NVML_ERROR] Get device count failed. Exiting..." << endl;
            exit(1);
        }
        unsigned int i = 0;
        for (; i < device_count; i++)
        {
            nvmlPstates_t pstate;
            nvmlDevice_t device;
            check_nvml_success(nvmlDeviceGetHandleByIndex_v2(i, &device), result);
            if (result != 0)
            {
                gLogFatal << __CXX_PREFIX << "[NVML_ERROR] Get device handle failed. Exiting..." << endl;
                exit(1);
            }
            check_nvml_success(nvmlDeviceGetPerformanceState(device, &pstate), result);
            if (result != 0)
            {
                gLogFatal << __CXX_PREFIX << "[NVML_ERROR] Get device performance state failed. Exiting..." << endl;
                exit(1);
            }
            if (pstate < nvmlPStates_enum::NVML_PSTATE_8)
            {
                gLogInfo << "Device " << i << " is using, switch to next device." << endl;
                continue;
            }
            // detect if it's already in exclusive mode.
            nvmlComputeMode_t cmode;
            check_nvml_success(nvmlDeviceGetComputeMode(device, &cmode), result);
            if (result != 0)
            {
                gLogError << __CXX_PREFIX << " [NVML_ERROR] Can not get device compute mode on index " << i << "." << endl;
            }
            if (cmode != nvmlComputeMode_enum::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS)
            {
                // try to set exclusive mode, if failed, just set CUDA device.
                check_nvml_success(nvmlDeviceSetComputeMode(device, nvmlComputeMode_enum::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS), result);
                if (result != 0)
                {
                    gLogError << __CXX_PREFIX << " [NVML_INFO] Set device on index " << i << " to exclusive compute mode failed, maybe you need to rerun your program on sudo (Linux) / Administrator (Windows)." << endl;
                }
            }
            else
            {
                gLogInfo << "Device " <<i <<"  is on exclusive compute mode." << endl;
            }
            check_cuda_success(cudaSetDevice(i), result);
            if (result != 0)
            {
                gLogFatal << __CXX_PREFIX << "[CUDA_ERROR] set device on index " << i << "for CUDA thread running failed." << endl;
                exit(1);
            }
            current_device_index = i;
            size_t freeMem, totalMem;
            check_cuda_success(cudaMemGetInfo(&freeMem, &totalMem), result);
            if (result != 0)
            {
                gLogInfo << "[CUDA_ERROR] Device " << i << "is on using. But can not get its memory information." << endl;
                continue;
            }
            if (freeMem < (size_t)(1 << 30))
            {
                gLogError << __CXX_PREFIX << "[CUDA_ERROR] Device " << i << " free memory is less than 1 GiB, can not used for TensorRT execution." << endl;
                continue;
            }
            gLogInfo.setf(ios::fixed, ios::floatfield);
            gLogInfo << "Device " << i << " ,free memory: " << setprecision(3) << freeMem * 1.0f / (1 << 20) << " MiB,"
                     << " total memory: " << setprecision(3) << totalMem * 1.0f / (1 << 20) << " MiB." << endl;
            break;
        }
        if (i == device_count)
        {
            gLogFatal << __CXX_PREFIX << "[NVML_ERROR] No device can be used for execution." << endl;
            exit(1);
        }
    }
    struct cudaDeviceProp prop;
    check_cuda_success(cudaGetDevice((int *)&current_device_index), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "[CUDA_ERROR] Can not get current device index." << endl;
        return;
    }
    check_cuda_success(cudaGetDeviceProperties(&prop, current_device_index), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "[CUDA_ERROR] Can not get device properties on device " << current_device_index << "." << endl;
        return;
    }
    gLogInfo << "Using CUDA device " << prop.name << " on index " << current_device_index << "." << endl;
}
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
        oss << __CXX_PREFIX << "kgmalloc init failed. Error code: " << static_cast<int>(err) << endl;
        cerr << oss.str();
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

// do not release global_allocator after main()
// call destroy() before return 0;
KGAllocator::~KGAllocator()
{
}

KGErrCode KGAllocator::destroy()
{
    // destroy global_allocator
    if (global_allocator != nullptr && dynamic_cast<KGAllocator *>(global_allocator.get()) != nullptr)
    {
        KGErrCode err = KGDestroy();
        if (err != KGMALLOC_SUCCESS)
            gLogError << __CXX_PREFIX << "recycle kgmalloc memory failed, err: " << err << endl;
        gLogInfo << "Memory pool destroyed." << endl;
        global_allocator = nullptr;
    }
    return KGMALLOC_SUCCESS;
}

DefaultAllocator::DefaultAllocator()
{
    AllocatorInit();
}

void *DefaultAllocator::allocate(uint64_t size, uint64_t alignment, uint32_t flags)
{
    void *d_ptr = nullptr;
    int result;
    check_cuda_success(cudaMalloc(&d_ptr, size), result);
    if (result == -1)
        return nullptr;
    return d_ptr;
}

void DefaultAllocator::free(void *memory)
{
    int result;
    check_cuda_success(cudaFree(memory), result);
}

DefaultAllocator::~DefaultAllocator()
{
}

KGAllocatorV2Chunk::KGAllocatorV2Chunk(uint64_t size)
{
    int result;
    check_cuda_success(cudaMalloc(&this->d_ptr, size), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "CUDA malloc error." << endl;
        this->d_ptr = nullptr;
        this->flag = false;
        this->size = 0;
        return;
    }
    this->flag = true;
    this->size = size;
    return;
};

KGAllocatorV2Chunk::~KGAllocatorV2Chunk()
{
    int result = 0;
    if (flag == true)
    {
        gLogError << "Warning: device memory " << d_ptr << " is still in memory pool.";
        gLogError << "Your device memory may leaked." << endl;
    }
    // do not try to free device memory on destructor, driver is shutting down.
    // check_cuda_success(cudaFree(d_ptr), result);
    // if (result != 0)
    // {
    //     gLogError << "Release device memory failed." << endl;
    // }
}

KGAllocatorV2::KGAllocatorV2()
{
    AllocatorInit();
}

KGAllocatorV2::~KGAllocatorV2()
{
    std::lock_guard<std::mutex>(this->mu);
    int result = 0;
    for (auto n : this->memory_pool)
    {
        // find chunk
        auto slab = n.second;
        for (auto iter = slab->chunks.begin(); iter != slab->chunks.end(); ++iter)
        {
            auto chunk = *iter;
            // free chunk
            delete chunk;
        }
        // free slab
        delete slab;
    }
}

void *KGAllocatorV2::allocate(uint64_t size, uint64_t alignment, uint32_t flags)
{
#ifdef __DEBUG
    INSERT_ALLOCATOR_V2_DEBUG_INFO("[DEBUG] Before kgallocator v2 allocate:");
#endif
    int result = 0;
    this->mu.lock();
    auto iter = this->memory_pool.find(size);
    if (iter == this->memory_pool.end())
    {
        this->memory_pool.insert(std::pair<uint64_t, V2Slab *>(size, new V2Slab()));
        iter = this->memory_pool.find(size);
    }
    auto slab = iter->second;
    this->mu.unlock();

    // lock the slab, lock guard can guarantee that we don't need to call slab->SlabMu->unlock();
    std::lock_guard<std::mutex> lock(slab->SlabMu);
    if (slab->chunks.size() != 0)
    {
        // find available chunk
        for (auto iter = slab->chunks.rbegin(); iter != slab->chunks.rend(); ++iter)
        {
            if ((*iter)->flag == true)
            {
                // get available chunk
                (*iter)->flag = false;
#ifdef __DEBUG
                INSERT_ALLOCATOR_V2_DEBUG_INFO("[DEBUG] After kgallocator v2 allocate:");
#endif
                this->mu.lock();
                this->mapping.insert({slab->chunks.back()->d_ptr, slab->chunks.back()});
                this->mu.unlock();
                slab->free_chunk_num--;
                slab->using_chunk_num++;
                return (*iter)->d_ptr;
            }
        }
        // no available chunk.
    }

    // allocate for new chunk
    // 4MiB<size<=8MiB的显存分配俩块。
    // size > 8MiB的分配一块。
    // <=4MiB的分配四块。
    int loop_times = 1;
    if (size > (uint64_t)(1 << 22) && size <= (uint64_t)(1 << 23))
    {
        loop_times = 2;
    }
    else if (size > (uint64_t)(1 << 23))
    {
        loop_times = 1;
    }
    else
    {
        loop_times = 4;
    }

    for (int i = 0; i < loop_times; i++)
    {
        V2Chunk *chunk = new V2Chunk(size);
        if (chunk->d_ptr == nullptr)
            return nullptr;
        slab->chunks.push_back(chunk);
        slab->free_chunk_num++;
    }

    // get recent allocated chunk
    slab->chunks.back()->flag = false;
    this->mu.lock();
    this->mapping.insert({slab->chunks.back()->d_ptr, slab->chunks.back()});
    this->mu.unlock();
    slab->free_chunk_num--;
    slab->using_chunk_num++;
#ifdef __DEBUG
    INSERT_ALLOCATOR_V2_DEBUG_INFO("[DEBUG] After kgallocator v2 allocate:");
#endif
    return slab->chunks.back()->d_ptr;
}

void KGAllocatorV2::free(void *memory)
{
#ifdef __DEBUG
    INSERT_ALLOCATOR_V2_DEBUG_INFO("[DEBUG] Before kgallocator v2 free:");
#endif
    if (memory == nullptr)
        return;
    // mark as free and sort if there is too many idle chunk
    this->mu.lock();
    auto iter = this->mapping.find(memory);
    if (iter == this->mapping.end() || iter->second->flag == true)
    {
        gLogError << __CXX_PREFIX << "memory invaild." << endl;
        this->mu.unlock();
        return;
    }
    auto chunk = iter->second;
    // get chunk size and get the slab
    uint64_t size = iter->second->size;
    auto slab_iter = this->memory_pool.find(size);
    if (slab_iter == this->memory_pool.end())
    {
        gLogError << __CXX_PREFIX << "memory invaild." << endl;
        this->mu.unlock();
        return;
    }
    auto slab = slab_iter->second;
    // erase on mapping
    mapping.erase(memory);
    this->mu.unlock();

    int result = 0;
    // lock the slab, lock guard can guarantee that we don't need to call slab->SlabMu->unlock();
    std::lock_guard<std::mutex> lock(slab->SlabMu);
    // free the chunk
    chunk->flag = true;
    // protect user input, output and model information
    check_cuda_success(cudaMemset(chunk->d_ptr, 0, chunk->size), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Unable to clear the device memory data." << endl;
    }
    slab->using_chunk_num--;
    slab->free_chunk_num++;
#ifdef __DEBUG
    // free this chunk intermediately.

    // find the chunk pos in slab
    auto chunk_iter = std::find(slab->chunks.begin(), slab->chunks.end(), chunk);
    if (chunk_iter == (slab->chunks.end()))
    {
        // it should not happen
        gLogError << __CXX_PREFIX << "Can not find chunk on SLAB." << endl;
        return;
    }
    // Since the destructor of chunk will not release the device memory,
    // We need to do it by ourselves.
    check_cuda_success(cudaFree((*chunk_iter)->d_ptr), result);
    if (result != 0)
    {
        gLogError << __CXX_PREFIX << "Device memory " << (*chunk_iter)->d_ptr << " release failed." << endl;
        return;
    }
    slab->chunks.erase(chunk_iter);
    return;
#endif
    // sort in order that first available chunk is at the end.
    std::sort(slab->chunks.begin(), slab->chunks.end(), [](V2Chunk *c1, V2Chunk *v2) {
        return (c1->flag == false) ? true : false;
    });
    // if there is too much free chunk? (bigger than 32MiB) release half of it.
    if (slab->free_chunk_num + slab->using_chunk_num > 5 && slab->free_chunk_num >= slab->using_chunk_num && slab->free_chunk_num * size >= (uint64_t)(1 << 25))
    {
        for (int i = 0; i < slab->free_chunk_num / 2; i++)
        {
            auto iter = slab->chunks.back();
            if (iter->flag == false)
            {
                gLogError << __CXX_PREFIX << "KGAllocatorV2 internal error!" << endl;
                gLogError << __CXX_PREFIX << "The compress of memory pool operates on using chunk." << endl;
                abort();
            }
            check_cuda_success(cudaFree(iter->d_ptr), result);
            slab->free_chunk_num--;
            if (result != 0)
            {
                gLogError << __CXX_PREFIX << "memory invaild." << endl;
                return;
            }
            slab->chunks.pop_back();
        }
    }
#ifdef __DEBUG
    INSERT_ALLOCATOR_V2_DEBUG_INFO("[DEBUG] After kgallocator v2 allocate:");
#endif
    return;
}

void printCurrentPool(KGAllocatorV2 *allocator)
{
    if (!allocator)
        return;
    if (allocator->memory_pool.size() == 0)
    {
        cout << "no any device memory on memory pool." << endl;
        return;
    }
    std::lock_guard<std::mutex>(allocator->mu);
    gLogInfo << "KGAllocator V2 current memory pool:" << endl;
    cout << left << setfill(' ') << setw(20) << "address" << left << setfill(' ') << setw(20) << "size (Bytes)" << left << setfill(' ') << setw(20) << "status" << endl;
    for (auto n : allocator->memory_pool)
    {
        for (auto k : n.second->chunks)
        {
            cout << left << setfill(' ') << setw(20) << k->d_ptr << left << setfill(' ') << setw(20) << k->size << left << setfill(' ') << setw(20);
            if (k->flag == true)
                cout << "idle" << endl;
            else
                cout << "using" << endl;
        }
        cout << "Slab " << n.first << " free chunk num: " << n.second->free_chunk_num << ", using chunk num: " << n.second->using_chunk_num << endl;
    }
    return;
}

int KGAllocatorV2::CompressMemoryPool()
{
    std::lock_guard<std::mutex> lock(this->mu);
    for (auto iter = this->mapping.begin(); iter != this->mapping.end(); ++iter)
    {
        // Search for each idle chunk.
        auto chunk = iter->second;
        if (chunk->flag == true)
        {
            // find the slab
            auto iter_2 = this->memory_pool.find(iter->second->size);
            if (iter_2 == this->memory_pool.end())
            {
                // It should not happen.
                gLogError << __CXX_PREFIX << "kgmallocV2 internal error." << endl;
                gLogError << __CXX_PREFIX << "Not found address of mapping in memory pool." << endl;
                return -1;
            }
            auto slab = iter_2->second;
            auto slab_iter = slab->chunks.begin();
            for (; slab_iter != slab->chunks.end(); ++slab_iter)
            {
                if (*slab_iter == chunk)
                {
                    slab->chunks.erase(slab_iter);
                    break;
                }
            }
            if (slab_iter == slab->chunks.end())
            {
                // It should not happen.
                gLogError << __CXX_PREFIX << "kgmallocV2 internal error." << endl;
                gLogError << __CXX_PREFIX << "Not found address of mapping in slab." << endl;
                return -1;
            }
        }
    }
    return 0;
}
// end of trt_allocator.cpp