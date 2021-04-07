#pragma once

#include "kgmalloc.hpp"
#include <NvInferRuntimeCommon.h>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <sstream>
#include <memory>

#if defined(_WIN32) || defined(_MSC_VER)
typedef unsigned int uint;
#endif

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
        // fix:该chunk不一定只有一个结点
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

private:
    // node_pool 建立从d_ptr到node_ptr的地址的连接
    std::unordered_map<void *, void *> node_pool;
    // alloc_mu allocator全局锁，所有可能对memory_pool产生读写冲突的地方都由mu控制
    std::mutex alloc_mu;

    // 打印UMapPtrToAddr/UMapAddrToPtr/node_pool
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
    void *allocate(uint64_t size, uint64_t alignment, uint32_t flags)
    {
        void *d_ptr;
        int result;
        check_cuda_success(cudaMalloc(&d_ptr, size), result);
        if (!result)
            return nullptr;
        return d_ptr;
    }

    //!
    //! A callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    void free(void *memory)
    {
        int result;
        check_cuda_success(cudaFree(memory), result);
    }

    virtual ~DefaultAllocator();
};

extern std::shared_ptr<nvinfer1::IGpuAllocator> kg_allocator;

// end of trt_allocator.hpp