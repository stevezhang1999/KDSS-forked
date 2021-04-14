#pragma once

#include <mutex>
#include <atomic>

using namespace std;

class RWMutex
{
private:
    mutex write;
    atomic<int> readers;
    pthread_mutex_t cond_mu;
    pthread_cond_t cond;

public:
    RWMutex();
    void lock();
    void unlock();
    void rlock();
    void runlock();
};

// end of rwmutex.hpp