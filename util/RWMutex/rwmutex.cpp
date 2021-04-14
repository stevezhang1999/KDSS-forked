#include "rwmutex.hpp"

RWMutex::RWMutex()
{
    readers = 0;
    if (pthread_mutex_init(&cond_mu, NULL) < 0)
        std::terminate();
    if (pthread_cond_init(&cond, NULL) < 0)
        std::terminate();
}

void RWMutex::lock()
{
    write.lock();
    while (readers > 0)
    {
        // sleep
        pthread_mutex_lock(&cond_mu);
        pthread_cond_wait(&cond, &cond_mu);
        pthread_mutex_unlock(&cond_mu);
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
    if (readers.load() == 0)
        pthread_cond_signal(&cond);
}

// end of rwmutex.cpp