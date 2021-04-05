#pragma once

#define __CXX_PREFIX __FILE__ << ":" << __LINE__ << " "

#if defined __unix || __linux
#include <sys/time.h>
#include <iomanip>
#define _CXX_MEASURE_TIME(expr, ofstream)                                       \
    do                                                                          \
    {                                                                           \
        struct timeval t1, t2;                                                  \
        gettimeofday(&t1, NULL);                                                \
        (expr);                                                                 \
        gettimeofday(&t2, NULL);                                                \
        double time = (t2.tv_sec - t1.tv_sec) * 1000;                           \
        time += (t2.tv_usec - t1.tv_usec) * 1.0f / 1000;                        \
        ofstream << setiosflags(ios::fixed) << setprecision(3) << time << endl; \
    } while (0)
#endif
// end of common.hpp