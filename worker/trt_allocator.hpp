#pragma once

#include <NvInferRuntimeCommon.h>
#include <unordered_map>
#include <mutex>

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

private:
    // node_pool 建立从d_ptr到node_ptr的地址的连接
    std::unordered_map<void *, void *> node_pool;
    // alloc_mu allocator全局锁，所有可能对memory_pool产生读写冲突的地方都由mu控制
    std::mutex alloc_mu;
};

extern nvinfer1::IGpuAllocator *kg_allocator;

// end of trt_allocator.hpp