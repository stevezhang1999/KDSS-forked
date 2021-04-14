#pragma once

#include <time.h>
#include <string.h>
#include <stdio.h>

#ifndef __CXX_PREFIX
#define __CXX_PREFIX (__FILE__ << ":" << __LINE__ << " ")
#endif

#ifndef __FATAL
#define __FATAL (" [F] ")
#endif

#ifndef __ERROR
#define __ERROR (" [E] ")
#endif

#ifndef __WARNING
#define __WARNING (" [W] ")
#endif

#ifndef __INFO
#define __INFO (" [I] ")
#endif

#ifndef __VERBOSE
#define __VERBOSE (" [V] ")
#endif

#ifdef __DEBUG
#ifndef LOGFATAL
#define LOGFATAL(str, args...)                                               \
    do                                                                       \
    {                                                                        \
        time_t timep = time(NULL);                                           \
        char *ltime = ctime(&timep);                                         \
        ltime[strlen(ltime) - 1] = 0;                                        \
        fprintf(stderr, "%s:%d [%s]%s", __FILE__, __LINE__, ltime, __FATAL); \
        fprintf(stderr, str, ##args);                                        \
        fprintf(stderr, "\n");                                               \
    } while (0)
#endif

#ifndef LOGERROR
#define LOGERROR(str, args...)                                               \
    do                                                                       \
    {                                                                        \
        time_t timep = time(NULL);                                           \
        char *ltime = ctime(&timep);                                         \
        ltime[strlen(ltime) - 1] = 0;                                        \
        fprintf(stderr, "%s:%d [%s]%s", __FILE__, __LINE__, ltime, __ERROR); \
        fprintf(stderr, str, ##args);                                        \
        fprintf(stderr, "\n");                                               \
    } while (0)
#endif

#ifndef LOGWARNING
#define LOGWARNING(str, args...)                                               \
    do                                                                         \
    {                                                                          \
        time_t timep = time(NULL);                                             \
        char *ltime = ctime(&timep);                                           \
        ltime[strlen(ltime) - 1] = 0;                                          \
        fprintf(stderr, "%s:%d [%s]%s", __FILE__, __LINE__, ltime, __WARNING); \
        fprintf(stderr, str, ##args);                                          \
        fprintf(stderr, "\n");                                                 \
    } while (0)
#endif

#ifndef LOGINFO
#define LOGINFO(str, args...)                                               \
    do                                                                      \
    {                                                                       \
        time_t timep = time(NULL);                                          \
        char *ltime = ctime(&timep);                                        \
        ltime[strlen(ltime) - 1] = 0;                                       \
        fprintf(stderr, "%s:%d [%s]%s", __FILE__, __LINE__, ltime, __INFO); \
        fprintf(stderr, str, ##args);                                       \
        fprintf(stderr, "\n");                                              \
    } while (0)
#endif

#ifndef LOGVERBOSE
#define LOGVERBOSE(str, args...)                                               \
    do                                                                         \
    {                                                                          \
        time_t timep = time(NULL);                                             \
        char *ltime = ctime(&timep);                                           \
        ltime[strlen(ltime) - 1] = 0;                                          \
        fprintf(stderr, "%s:%d [%s]%s", __FILE__, __LINE__, ltime, __VERBOSE); \
        fprintf(stderr, str, ##args);                                          \
        fprintf(stderr, "\n");                                                 \
    } while (0)
#endif

#else
#ifndef LOGFATAL
#define LOGFATAL(str, args...)                     \
    do                                             \
    {                                              \
        time_t timep = time(NULL);                 \
        char *ltime = ctime(&timep);               \
        ltime[strlen(ltime) - 1] = 0;              \
        fprintf(stderr, "[%s]%s", ltime, __FATAL); \
        fprintf(stderr, str, ##args);              \
        fprintf(stderr, "\n");                     \
    } while (0)
#endif

#ifndef LOGERROR
#define LOGERROR(str, args...)                     \
    do                                             \
    {                                              \
        time_t timep = time(NULL);                 \
        char *ltime = ctime(&timep);               \
        ltime[strlen(ltime) - 1] = 0;              \
        fprintf(stderr, "[%s]%s", ltime, __ERROR); \
        fprintf(stderr, str, ##args);              \
        fprintf(stderr, "\n");                     \
    } while (0)
#endif

#ifndef LOGWARNING
#define LOGWARNING(str, args...)                     \
    do                                               \
    {                                                \
        time_t timep = time(NULL);                   \
        char *ltime = ctime(&timep);                 \
        ltime[strlen(ltime) - 1] = 0;                \
        fprintf(stderr, "[%s]%s", ltime, __WARNING); \
        fprintf(stderr, str, ##args);                \
        fprintf(stderr, "\n");                       \
    } while (0)
#endif

#ifndef LOGINFO
#define LOGINFO(str, args...)                     \
    do                                            \
    {                                             \
        time_t timep = time(NULL);                \
        char *ltime = ctime(&timep);              \
        ltime[strlen(ltime) - 1] = 0;             \
        fprintf(stdout, "[%s]%s", ltime, __INFO); \
        fprintf(stdout, str, ##args);             \
        fprintf(stdout, "\n");                    \
    } while (0)
#endif

#ifndef LOGVERBOSE
#define LOGVERBOSE(str, args...)                     \
    do                                               \
    {                                                \
        time_t timep = time(NULL);                   \
        char *ltime = ctime(&timep);                 \
        ltime[strlen(ltime) - 1] = 0;                \
        fprintf(stdout, "[%s]%s", ltime, __VERBOSE); \
        fprintf(stdout, str, ##args);                \
        fprintf(stdout, "\n");                       \
    } while (0)
#endif
#endif