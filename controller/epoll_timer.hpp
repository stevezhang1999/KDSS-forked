#pragma once
#include "base.hpp"
#include "../util/RWMutex/rwmutex.hpp"
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <list>
#include <unordered_map>

#include <pthread.h>
#include <semaphore.h>

// We use Linux epoll to implement our timer
#if !defined(__linux__)
#error "epoll_timer only support Linux system."
#endif

// Concurrent_atomic atomic的并发安全版本
template <typename T>
class Concurrent_atomic
{
private:
    std::mutex mu;
    std::atomic<T> data;

protected:
    // no duplicate
    Concurrent_atomic(const Concurrent_atomic &){};

public:
    Concurrent_atomic()
    {
        this->data = 0;
    }

    Concurrent_atomic &operator++()
    {
        std::lock_guard<std::mutex>(this->mu);
        this->data++;
        return *this;
    }
    Concurrent_atomic &operator--()
    {
        std::lock_guard<std::mutex>(this->mu);
        this->data--;
        return *this;
    }
    T load()
    {
        std::lock_guard<std::mutex>(this->mu);
        return this->data.load();
    }
    T operator=(const Concurrent_atomic &data)
    {
        return data->load();
    }
    Concurrent_atomic &operator=(const T &t)
    {
        this->store(t);
        return *this;
    }
    void store(const T data)
    {
        std::lock_guard<std::mutex>(this->mu);
        this->data.store(data);
    }
};

// Concurrent_unordered_map unordered_map的并发安全版本
template <typename _Key, typename _Tp>
class Concurrent_unordered_map
{
private:
    RWMutex mu;
    std::unordered_map<_Key, _Tp> umap;

public:
    Concurrent_unordered_map()
    {
    }

    ~Concurrent_unordered_map()
    {
    }

    void insert(pair<_Key, _Tp> &_p)
    {
        std::lock_guard<mutex>(mu);
        return umap.insert(_p);
    }

protected:
    Concurrent_unordered_map(const Concurrent_unordered_map &){};
    Concurrent_unordered_map &operator=(const Concurrent_unordered_map &){};
};

class EpollTaskHandler final : public IEventHandler
{
public:
    EpollTaskHandler();
    virtual ~EpollTaskHandler();

    // AddEvent 将一个事件增加到定时器队列中，当时间到达的时候执行事件
    // \param ts 该事件需要执行的UNIX时间戳
    // \param event 指向需要执行的事件指针。
    // \returns 如果插入成功，则返回该事件的一个对应id，该id恒大于0。否则返回0。
    virtual uint64_t AddEvent(uint64_t ts, void *event);

    uint64_t AddTask(uint64_t ts, ComputeTask *task);

    // DeleteEvent 将一个事件从定时器队列中删除。
    // 删除操作可能会影响性能。
    // \param event_id 该事件对应的ID
    // \returns 如果试图删除一个不存在的event_id，则返回一个非0的数，否则返回0。
    virtual int DeleteEvent(uint64_t event_id);

    // WaitEvent 等待一个事件执行完成。
    // 该函数会被阻塞，直到事件执行完毕。如果该事件存在callback，该callback将被自动调用。
    // \param event_id 需要等待的事件ID
    // \returns 不返回任何值。
    virtual void WaitEvent(uint64_t event_id);

    // CancelEvent 试图取消一个正在执行的事件。
    // 该函数可能会导致事件的异常，这取决于事件是否可以被取消，以及被取消的后果如何。
    // 对于EpollTaskHandler来说，已开始执行的事件，不可取消。
    // \param event_id 需要取消的事件ID
    // \returns 不返回任何值。
    virtual void CancelEvent(uint64_t event_id);

    // GetNextEvent 获取下一个将要被执行的事件ID。
    // \returns 返回下一个将要被执行的事件ID。如果当前没有需要被执行的事件，返回0。
    virtual uint64_t GetNextEvent();

private:
    // int - 底层的timer_id
    // ComputeTask - 待执行的事件指针
    std::vector<pair<int, ComputeTask *>> event_pool;
    // ep_locker event_poll线程锁
    std::mutex ep_locker;
    // ep_sem event_poll信号量，当event_poll没有任务时，ep_sem负责阻塞epoll_wait。
    sem_t ep_sem;
    // 全局唯一epoll file description
    int epfd = -1;
    // 全局唯一EpollTaskHandler event_id，并发安全
    Concurrent_atomic<uint64_t> event_id;
    // cancel_point，用于优雅地杀死listen_thread
    Concurrent_atomic<bool> cancel_point;
    // task_queue 当前需要计算的任务的队列
    std::queue<ComputeTask *> task_queue;
    // task_queue_locker
    std::mutex tq_locker;
    // task_queue_sem 当前队列信号量
    sem_t tq_sem;
    // last_executing_task_id 当前或最后一个被执行的task的task_id，并发安全
    Concurrent_atomic<uint64_t> last_executing_task_id;
    // delete_task_poll 已经被取消的task_id的集合
    std::unordered_map<uint64_t, bool> delete_task_poll;
    // wait_poll 正在等待完成的等待池
    std::unordered_map<uint64_t, std::condition_variable *> wait_poll;
    // wp_locker 等待池线程锁
    std::mutex wp_locker;
    // listen thread，用于监听epoll
    std::thread *listen_thread;

public:
    // listen_func，用于listen_thread的处理函数
    std::function<void(EpollTaskHandler *)> listen_func();
    // execute thread，用于执行任务
    std::thread *execute_thread;
    // execute_func，用于execute_thread的处理函数
    std::function<void(EpollTaskHandler *)> execute_func();
};

// end of epoll_timer.hpp