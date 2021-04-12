// We use Linux epoll to implement our timer
#if !defined(__linux__)
#error "epoll_timer only support Linux system."
#endif

#include "epoll_timer.hpp"
#include "util.hpp"

#include <algorithm>
#include <exception>

#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <sys/unistd.h>

// \brief PthreadDelegate 将pthread相关变量的初始化和销毁工作交由托管类进行
template <typename _Pthread_type, typename _Pthread_attr_type = pthread_mutexattr_t>
class PthreadDelegate
{
};

template <>
class PthreadDelegate<pthread_cond_t, pthread_condattr_t>
{
public:
    PthreadDelegate(pthread_cond_t &pt)
    {
        pthread_cond_init(&pt, NULL);
        this->pt = pt;
        this->has_pat = false;
    }

    PthreadDelegate(pthread_cond_t &pt, pthread_condattr_t pat)
    {
        pthread_cond_init(&pt, &pat);
        this->pt = pt;
        this->has_pat = true;
    }

    ~PthreadDelegate()
    {
        pthread_cond_destroy(&pt);
        if (has_pat)
            pthread_condattr_destroy(&pat);
    }

    pthread_cond_t GetPthreadVar() const { return pt; }
    int GetPthreadAttrVar(pthread_condattr_t *pat) const
    {
        if (this->has_pat)
        {
            *pat = this->pat;
            return 0;
        }
        else
        {
            return -1;
        }
    }

    // non duplicate
protected:
    PthreadDelegate(const PthreadDelegate &) {}
    PthreadDelegate &operator=(const PthreadDelegate &) {}

private:
    pthread_cond_t pt;
    bool has_pat;
    pthread_condattr_t pat;
};

template <>
class PthreadDelegate<pthread_mutex_t>
{
public:
    PthreadDelegate(pthread_mutex_t &pt)
    {
        pthread_mutex_init(&pt, NULL);
        this->pt = pt;
        this->has_pat = false;
    }

    PthreadDelegate(pthread_mutex_t &pt, pthread_mutexattr_t pat)
    {
        pthread_mutex_init(&pt, &pat);
        this->pt = pt;
        this->has_pat = true;
    }

    ~PthreadDelegate()
    {
        pthread_mutex_destroy(&pt);
        if (has_pat)
            pthread_mutexattr_destroy(&pat);
    }

    pthread_mutex_t GetPthreadVar() const { return pt; }
    int GetPthreadAttrVar(pthread_mutexattr_t *pat) const
    {
        if (this->has_pat)
        {
            *pat = this->pat;
            return 0;
        }
        else
        {
            return -1;
        }
    }

    // non duplicate
protected:
    PthreadDelegate(const PthreadDelegate &) {}
    PthreadDelegate &operator=(const PthreadDelegate &) {}

private:
    pthread_mutex_t pt;
    bool has_pat;
    pthread_mutexattr_t pat;
};

template <>
class PthreadDelegate<sem_t>
{
public:
    PthreadDelegate(sem_t sem, int init_value)
    {
        if (sem_init(&sem, 0, init_value) < 0)
        {
            LOGFATAL("Can not delegate semaphore.");
            std::terminate();
        }
        this->sem = sem;
    }

    ~PthreadDelegate()
    {
        if (sem_destroy(&sem) < 0)
        {
            LOGFATAL("Can not destruct delegated semaphore.");
            std::terminate();
        }
    }

    sem_t GetPthreadVar() const { return sem; }

    // non duplicate
protected:
    PthreadDelegate(const PthreadDelegate &) {}
    PthreadDelegate &operator=(const PthreadDelegate &) {}

private:
    sem_t sem;
};

std::function<void(EpollTaskHandler *)> EpollTaskHandler::listen_func()
{
    return function<void(EpollTaskHandler *)>([](EpollTaskHandler *eeh) {
        struct epoll_event *events = (struct epoll_event *)malloc(sizeof(struct epoll_event) * 32);
        if (NULL == events)
        {
            LOGERROR("Can not allocate memory for epoll events.");
            std::terminate();
            return;
        }
        memset(events, 0, sizeof(struct epoll_event) * 32);
        while (!eeh->cancel_point.load())
        {
            // 队列根本就没任务的时候就别监听了
            sem_wait(&(eeh->ep_sem));
            // 醒来之后先看看是不是被cancel_point打醒的
            if (eeh->cancel_point.load())
            {
                break;
            }
            // 不能无限监听，否则cancel_point起不了打断作用
            int nfds = epoll_wait(eeh->epfd, events, 32, 1000);
            if (nfds < 0)
            {
                LOGFATAL("epoll_wait failed.");
                LOGFATAL("errno: %d", errno);
            }
            if (nfds == 0)
            {
                static struct timespec now;
                if (clock_gettime(CLOCK_REALTIME, &now) < 0)
                {
                    LOGFATAL("Can not get current time.");
                    continue;
                }
#ifdef __EPOLL_LISTEN_DEBUG
                LOGERROR("Timestamp %ld, no event has read.", now.tv_sec);
#endif
                // 还原被wait掉的ep_sem
                sem_post(&(eeh->ep_sem));
                continue;
            }
            for (int i = 0; i < nfds; i++)
            {
                if (events[i].events & EPOLLIN)
                {
                    // find the task
                    eeh->ep_locker.lock();
                    for (auto iter = eeh->event_pool.begin(); iter != eeh->event_pool.end(); ++iter)
                    {
                        if ((*iter).first == events[i].data.fd)
                        {
                            // extract the task
                            auto task = (*iter).second;
                            if (task->magic_number != COMPUTE_TASK_MAGIC_NUMBER)
                            {
                                LOGERROR("This task not vaild, task pointer may corrupted. task_id: %ld", task->task_id);
                                continue;
                            }
                            uint64_t exp;
                            if (read(events[i].data.fd, &exp, sizeof(uint64_t)) != sizeof(uint64_t))
                            {
                                LOGERROR("Can not read data from timerfd for task %ld", task->task_id);
                                continue;
                            }
                            LOGINFO("Task %ld push into execute queue.", task->task_id);
                            if (close(events[i].data.fd) < 0)
                            {
                                LOGERROR("Can not close the timerfd for task %ld", task->task_id);
                                std::terminate();
                            }
                            // put the task into execute queue (producer)
                            eeh->tq_locker.lock();
                            eeh->task_queue.push(task);
                            sem_post(&eeh->tq_sem);
                            eeh->tq_locker.unlock();
                            eeh->event_pool.erase(iter);
                            make_heap(eeh->event_pool.begin(), eeh->event_pool.end());
                            break;
                        }
                    }
                    eeh->ep_locker.unlock();
                }
            }
        }
        LOGINFO("listen_thread exiting...");
        free(events);
    });
}

std::function<void(EpollTaskHandler *)> EpollTaskHandler::execute_func()
{
    return function<void(EpollTaskHandler *)>([](EpollTaskHandler *eeh) {
        while (!eeh->cancel_point.load())
        {
            // Get task (consumer)
            if (sem_wait(&eeh->tq_sem) < 0)
            {
                LOGERROR("Wait for ep_sem failed.");
                std::terminate();
            }
            if (eeh->cancel_point.load())
            {
                // cancel point, before close sem, we need to set eeh->cancel_point and then call sem_post in order to unblock the execute_func.
                break;
            }
            eeh->tq_locker.lock();
            auto task = eeh->task_queue.front();
            eeh->task_queue.pop();
            eeh->tq_locker.unlock();
            auto iter = eeh->delete_task_poll.find(task->task_id);
            if (iter != eeh->delete_task_poll.end())
            {
                // This task has been deleted by DeleteEvent
                eeh->delete_task_poll.erase(iter);
                continue;
            }
            if (task->magic_number != COMPUTE_TASK_MAGIC_NUMBER)
            {
                LOGERROR("This task not vaild, task pointer may corrupted. task_id: %ld", task->task_id);
                continue;
            }
            eeh->last_executing_task_id = task->task_id;
            LOGINFO("Task %ld begin execute.", task->task_id);
            task->execute_func(&task, &task);
            task->callback_func(&task, &task);
            struct timespec now;
            if (clock_gettime(CLOCK_REALTIME, &now) < 0)
            {
                LOGFATAL("Can not get current time.");
                return -1;
            }
            if (now.tv_sec > task->end_timestamp)
            {
                LOGWARNING("Warning: task %ld may exceed deadline.", task->task_id);
            }
            eeh->wp_locker.lock();
            auto wp_iter = eeh->wait_poll.find(task->task_id);
            if (wp_iter != eeh->wait_poll.end())
            {
                wp_iter->second->notify_one();
            }
            eeh->wp_locker.unlock();
            LOGINFO("Task %ld executed.", task->task_id);
        }
        LOGINFO("exeucte_thread exiting...");
    });
}

EpollTaskHandler::EpollTaskHandler()
{
    // 这里不能委托ep_sem，是因为ep_sem的委托类会在EpollTaskHandler()执行完毕之后直接被析构。
    if (sem_init(&this->ep_sem, 0, 0) < 0)
    {
        LOGFATAL("Can not init semaphore.");
        std::terminate();
    }
    this->event_id = 1;
    this->epfd = epoll_create1(0);
    if (this->epfd == -1)
    {
        LOGERROR("Could not create epoll fd %d", this->epfd);
        LOGERROR("errno: %d", errno);
        std::terminate();
    }
    this->cancel_point = false;
    // init semaphore
    if (sem_init(&this->tq_sem, 0, 0) != 0)
    {
        LOGFATAL("Can not init semaphore for queue.");
        std::terminate();
        return;
    }

    this->last_executing_task_id = 0;

    // Create a thread with listening this epfd
    this->listen_thread = new thread(this->listen_func(), this);
    if (!this->listen_thread)
    {
        LOGFATAL("Can not start listen_thread.");
        std::terminate();
        return;
    }
    LOGINFO("listen_thread init finished, start listening.");
    this->execute_thread = new thread(this->execute_func(), this);
    if (!this->execute_thread)
    {
        LOGFATAL("Can not start execute_thread.");
        std::terminate();
        return;
    }
    LOGINFO("execute_thread init finished, start waiting.");
}

EpollTaskHandler::~EpollTaskHandler()
{
    // Cancel execute thread
    this->cancel_point.store(true);
    if (sem_post(&this->tq_sem) < 0)
    {
        LOGERROR("Can not post this->tq_sem.");
        std::terminate();
    }
    this->execute_thread->join();
    delete this->execute_thread;
    // Cancel listen thread
    if (sem_post(&this->ep_sem) < 0)
    {
        LOGERROR("Can not post this->tq_sem.");
        std::terminate();
    }
    this->listen_thread->join();
    delete this->listen_thread;
    this->wait_poll.clear();
    this->delete_task_poll.clear();
    this->last_executing_task_id = 0;
    // destroy task queue semaphore
    if (sem_destroy(&this->tq_sem) < 0)
    {
        LOGERROR("Could not destroy task queue semaphore.");
        LOGERROR("errno: %d", errno);
    }
    // this->cancel_point = false;
    this->event_id = 0;
    if (this->epfd != -1)
    {
        int res = close(this->epfd);
        if (res < 0)
        {
            LOGERROR("Could not destroy epoll fd %d", this->epfd);
            LOGERROR("errno: %d", errno);
        }
    }
    this->event_id = 0;
    if (sem_post(&this->ep_sem) < 0)
    {
        LOGERROR("Can not post to this->ep_sem.");
        std::terminate();
    }
    this->event_pool.clear();
}

uint64_t EpollTaskHandler::AddEvent(uint64_t ts, void *event)
{
    LOGWARNING("Use EpollTaskHandler::AddTask instead for type cast of event pointer.");
    return this->AddTask(ts, (ComputeTask *)event);
}

uint64_t EpollTaskHandler::AddTask(uint64_t ts, ComputeTask *task)
{
    // See: https://www.cnblogs.com/wenqiang/p/6698371.html
    // Also see: https://www.cnblogs.com/zhanggaofeng/p/9410639.html
    // We use timer fd for epoll_wait
    int tfd = timerfd_create(CLOCK_REALTIME, TFD_NONBLOCK);
    if (tfd < 0)
    {
        LOGERROR("Could not create timer fd %d", this->epfd);
        LOGERROR("errno: %d", errno);
        return 0;
    }
    struct itimerspec new_value;
    new_value.it_interval = timespec{1, 0};
    new_value.it_value = timespec{ts, 0};
    struct epoll_event eevent;
    memset(&eevent, 0, sizeof(eevent));
    eevent.data.fd = tfd;
    eevent.events = EPOLLIN | EPOLLET;
    if (timerfd_settime(tfd, 1, &new_value, NULL) < 0)
    {
        LOGERROR("Could not set time to timerfd.");
        LOGERROR("errno: %d", errno);
        std::terminate();
    }
    if (epoll_ctl(this->epfd, EPOLL_CTL_ADD, tfd, &eevent) < 0)
    {
        LOGERROR("Could not modify epoll fd %d", this->epfd);
        LOGERROR("errno: %d", errno);
        return 0;
    }
    uint64_t ret_id = this->event_id.load();
    task->task_id = ret_id;
    this->event_id = ((ret_id + 1) % UINT64_MAX);
    this->ep_locker.lock();
    this->event_pool.push_back(pair<int, ComputeTask *>(tfd, task));
    // construct min-heap
    make_heap(this->event_pool.begin(), this->event_pool.end(), [](pair<int, ComputeTask *> e1, pair<int, ComputeTask *> e2) { return e1.second->end_timestamp < e2.second->end_timestamp; });
    this->ep_locker.unlock();
    if (sem_post(&(this->ep_sem)) < 0)
    {
        LOGERROR("Post to ep_sem failed.");
        std::terminate();
    }
    LOGINFO("Add task %ld, will be executed on timestamp %ld", ret_id, task->begin_timestamp);
    return ret_id;
}

int EpollTaskHandler::DeleteEvent(uint64_t event_id)
{
    if (event_id > this->last_executing_task_id.load())
    {
        this->delete_task_poll.insert(pair<uint64_t, bool>(event_id, true));
        return 0;
    }
    return -1;
}

void EpollTaskHandler::WaitEvent(uint64_t event_id)
{
    std::condition_variable cv;
    std::mutex mu;
    std::unique_lock<std::mutex> lk(mu);
    while (this->last_executing_task_id.load() < event_id)
    {
        this->wp_locker.lock();
        this->wait_poll.insert(pair<uint64_t, condition_variable *>(event_id, &cv));
        this->wp_locker.unlock();
        cv.wait(lk);
    }
    lk.unlock();
    return;
}

void EpollTaskHandler::CancelEvent(uint64_t event_id)
{
    // do nothing
}

uint64_t EpollTaskHandler::GetNextEvent()
{
    uint64_t res = 0;
    this->tq_locker.lock();
    if (this->task_queue.size() != 0)
        res = this->task_queue.front()->task_id;
    this->tq_locker.unlock();
    if (res != 0)
        return res;
    std::lock_guard<std::mutex>(this->ep_locker);
    if (this->event_pool.size() == 0)
        return 0;
    return this->event_pool.front().second->task_id;
}