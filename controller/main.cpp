#include "epoll_timer.hpp"
#include "util.hpp"
#include <sys/unistd.h>

using namespace std;
int main()
{
    EpollTaskHandler handler;
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) < 0)
    {
        LOGFATAL("Can not get current time.");
        return -1;
    }
    char *checksum = "0874dd37ed0d164c";
    struct ComputeTask task_1("resnet-50", 1, vector<pair<string, vector<char>>>(), now.tv_sec + 5, now.tv_sec + 6, checksum);
    task_1.execute_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 1!");
        return;
    });
    task_1.callback_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 1 callback!");
        return;
    });
    uint64_t id = handler.AddTask(task_1.begin_timestamp, &task_1);
    if (id == 0)
    {
        LOGFATAL("Can not add task.");
        return -1;
    }
    struct ComputeTask task_2("resnet-50", 1, vector<pair<string, vector<char>>>(), now.tv_sec + 5, now.tv_sec + 6, checksum);
    task_2.execute_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 2!");
        return;
    });
    task_2.callback_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 2 callback!");
        return;
    });
    id = handler.AddTask(task_2.begin_timestamp, &task_2);
    if (id == 0)
    {
        LOGFATAL("Can not add task.");
        return -1;
    }

    struct ComputeTask task_3("resnet-50", 1, vector<pair<string, vector<char>>>(), now.tv_sec + 15, now.tv_sec + 16, checksum);
    task_3.execute_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 3!");
        return;
    });
    task_3.callback_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 3 callback!");
        return;
    });
    id = handler.AddTask(task_3.begin_timestamp, &task_3);
    if (id == 0)
    {
        LOGFATAL("Can not add task.");
        return -1;
    }

    struct ComputeTask task_4("resnet-50", 1, vector<pair<string, vector<char>>>(), now.tv_sec + 15, now.tv_sec + 16, checksum);
    task_4.execute_func = std::function<void(void *, void *)>([](void *, void *) {
        LOGINFO("This is task 4!");
        return;
    });
    task_4.callback_func = std::function<void(void *, void *)>([](void *, void *) {
        sleep(2);
        LOGINFO("After sleep 2 seconds, this is task 4 callback!");
        return;
    });
    id = handler.AddTask(task_4.begin_timestamp, &task_4);
    if (id == 0)
    {
        LOGFATAL("Can not add task.");
        return -1;
    }

    handler.WaitEvent(id);

    // LOGERROR("Test");
    return 0;
}