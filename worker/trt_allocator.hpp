#pragma once

#include "kgmalloc.hpp"
#include "NvInferRuntimeCommon.h"
#include "common.hpp"
#include "common/logger.h" // On TensorRT/samples

#include <nvml.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include <sstream>
#include <memory>
#include <atomic>

#if defined(_WIN32) || defined(_MSC_VER)
typedef unsigned int uint;
#endif

struct NVMLShutter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            int result;
            check_nvml_success(nvmlShutdown(), result);
            if (result != 0)
            {
#if NV_TENSORRT_MAJOR < 7
                gLogFatal << __CXX_PREFIX << " [NVML_ERROR] shutting down NVML failed." << std::endl;
#else
                sample::gLogFatal << __CXX_PREFIX << " [NVML_ERROR] shutting down NVML failed." << std::endl;
#endif
                exit(1);
            }
        }
    }
};

class KGAllocator final : public nvinfer1::IGpuAllocator
{
public:
    KGAllocator();
    //!
    //! A callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory required.
    //! \param alignment The required alignment of memory. Alignment will zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! If an allocation request of size 0 is made, nullptr should be returned.
    //!
    //! If an allocation request cannot be satisfied, nullptr should be returned.
    //!
    void *allocate(uint64_t size, uint64_t alignment, uint32_t flags);

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    void free(void *memory);

    virtual ~KGAllocator();

    uint64_t GetDeviceMemorySize(void *device_memory)
    {
        auto iter = node_pool.find(device_memory);
        if (iter == node_pool.end())
        {
            getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "device_memory invaild!\n");
            return 0;
        }
        CudaMemNode **node = static_cast<CudaMemNode **>(iter->second);
        // fix:???chunk???????????????????????????
        uint64_t chunk_size = 0;
        uint hash = (*node)->meta.hash;
        List *head = static_cast<List *>((void *)(*node));
        while (head && head->data.meta.hash == hash)
        {
            chunk_size += head->data.meta.length;
            head = head->next;
        }
        // return (*node)->meta.length;
        return chunk_size;
    }

    static KGErrCode destroy();

private:
    // node_pool ?????????d_ptr???node_ptr??????????????????
    std::unordered_map<void *, void *> node_pool;
    // alloc_mu allocator???????????????????????????memory_pool?????????????????????????????????mu??????
    std::mutex alloc_mu;

    // ??????UMapPtrToAddr/UMapAddrToPtr/node_pool
    std::string PrintUMap(std::unordered_map<void *, void *> umap)
    {
        std::ostringstream oss;
        for (auto n : umap)
        {
            oss << n.first << " -> " << n.second << std::endl;
        }
        return oss.str();
    }
};

class DefaultAllocator final : public nvinfer1::IGpuAllocator
{
public:
    DefaultAllocator();
    //!
    //! A callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory required.
    //! \param alignment The required alignment of memory. Alignment will zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! If an allocation request of size 0 is made, nullptr should be returned.
    //!
    //! If an allocation request cannot be satisfied, nullptr should be returned.
    //!
    void *allocate(uint64_t size, uint64_t alignment, uint32_t flags);

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    void free(void *memory);

    virtual ~DefaultAllocator();
};

typedef struct KGAllocatorV2Chunk
{
    KGAllocatorV2Chunk(uint64_t size);
    ~KGAllocatorV2Chunk();
    void *d_ptr;   // device memory address
    uint64_t size; // this address pointed chunk's size
    bool flag;     // false - unavailable, true - available
} V2Chunk;

typedef struct KGAllocatorV2Slab
{
    KGAllocatorV2Slab() : free_chunk_num(0), using_chunk_num(0) { chunks.clear(); };
    std::vector<V2Chunk *> chunks;         // chunk queue
    std::atomic<uint32_t> free_chunk_num;  // number of available chunk
    std::atomic<uint32_t> using_chunk_num; // number of using chunk
    std::mutex SlabMu;                     // mutex for chunk queue
} V2Slab;

class KGAllocatorV2 final : public nvinfer1::IGpuAllocator
{
public:
    KGAllocatorV2();
    //!
    //! A callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory required.
    //! \param alignment The required alignment of memory. Alignment will zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! If an allocation request of size 0 is made, nullptr should be returned.
    //!
    //! If an allocation request cannot be satisfied, nullptr should be returned.
    //!
    void *allocate(uint64_t size, uint64_t alignment, uint32_t flags);

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    void free(void *memory);

    friend void printCurrentPool(KGAllocatorV2 *allocator);

    //
    // Release all idle node immediately.
    // 0 should be returned if CompressMemoryPool execute successfully.
    int CompressMemoryPool();

    virtual ~KGAllocatorV2();

private:
    std::unordered_map<uint64_t, V2Slab *> memory_pool; // each size allocated has a SLAB.
    std::unordered_map<void *, V2Chunk *> mapping;      // for d_ptr -> chunk_ptr mapping
    std::mutex mu;                                      // allocate global lock
};

extern std::shared_ptr<nvinfer1::IGpuAllocator> global_allocator;

extern void printCurrentPool(KGAllocatorV2 *allocator);

#ifndef INSERT_ALLOCATOR_V2_DEBUG_INFO
#define INSERT_ALLOCATOR_V2_DEBUG_INFO(INFO)                                     \
    do                                                                           \
    {                                                                            \
        gLogInfo << __CXX_PREFIX << (INFO) << endl;                              \
        printCurrentPool(dynamic_cast<KGAllocatorV2 *>(global_allocator.get())); \
    } while (0);
#endif
// end of trt_allocator.hpp