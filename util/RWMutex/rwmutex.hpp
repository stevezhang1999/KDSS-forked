#pragma once

#include <mutex>
#include <atomic>

using namespace std;

class RWMutex
{
private:
    mutex write;
    atomic<int> readers;

public:
    void lock();
    void unlock();
    void rlock();
    void runlock();
};

// end of rwmutex.hpp