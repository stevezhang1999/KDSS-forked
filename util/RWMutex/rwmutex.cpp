#include "rwmutex.hpp"

void RWMutex::lock()
{
    write.lock();
    while (readers > 0) {
        // loop
    }
}

void RWMutex::unlock()
{
    write.unlock();
}

void RWMutex::rlock()
{
    write.lock();
    readers++;
    write.unlock();
}

void RWMutex::runlock()
{
    readers--;
}

// end of rwmutex.cpp