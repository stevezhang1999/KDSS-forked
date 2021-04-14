#pragma once

#if __cplusplus < 201103L
#error "Controller and Timer only support C++ 11 standard or later, please update your compiler."
#endif
#include "../worker/transfer_worker.hpp"
#include "../worker/computation_worker.hpp"

#include <vector>
#include <string>
#include <functional>

using namespace std;

enum TaskStatus
{
    WAITING,   // 在队列等待中
    COMPUTING, // 处于计算状态中
    FINISHED,  // 已计算完毕
    INVAILD,   // 此任务已失效（数据不可用）
};

const uint32_t COMPUTE_TASK_MAGIC_NUMBER = ('c' << 8) | ('t'); //'ct', 0x6374

template <typename _Res, typename... _ArgTypes>
using TaskFunc = std::function<_Res(_ArgTypes...)>;

const auto EmptyTaskFunc = TaskFunc<void, void *, void *>([](void *, void *) { return; });

typedef struct ComputeTask
{
public:
    ComputeTask(string model_name, unsigned int batch_size, vector<pair<string, vector<char>>> input, uint64_t begin_timestamp, uint64_t end_timestamp, char *checksum);
    uint32_t magic_number;                        // 用于判断当前ComputeTask是否为有效结构，该值应恒等于COMPUTE_TASK_MAGIC_NUMBER
    uint64_t task_id;                             // 唯一标识ID，该值恒大于0
    string model_name;                            // 当前计算任务需要调用的模型名称
    unsigned int batch_size;                      // 此任务的batch_size
    vector<pair<string, vector<char>>> input;     // 存放着输入名称与输入载荷的pair的集合
    vector<pair<string, uint32_t>> input_size;    // 存放着输入名称与输入长度的集合
    vector<pair<string, vector<char>>> output;    // 存放着输出名称与输出载荷的pair的集合
    vector<pair<string, uint32_t>> output_size;   // 存放着输入名称与输入长度的集合
    uint64_t begin_timestamp;                     // 该任务生成时的timestamp
    uint64_t end_timestamp;                       // 该任务的超时timestamp
    TaskStatus status;                            // 当前任务的状态
    char checksum[16];                            // 该任务的16位定长校验码
    TaskFunc<void, void *, void *> execute_func;  // 该任务对应的执行函数
    TaskFunc<void, void *, void *> callback_func; // 该任务执行完毕后对应的回调函数

protected:
    // No duplicate
    ComputeTask(const ComputeTask &);
    // No duplicate
    ComputeTask &operator=(const ComputeTask &);
} ComputeTask;

// A controller should have the functions of manipulating the computation task.
// IController definition of controller interface.
class IController
{
public:
    IController(){};
    virtual ~IController(){};

    // Enqueue 将计算任务放进等待队列中
    // \param task 需要进行计算的任务
    // \returns task对应的task_id，用于寻找task。
    //
    // 如果插入任务到队列失败，则返回0。
    virtual uint64_t Enqueue(ComputeTask task) = 0;

    // Dequeue 弹出下一个将要被计算的任务
    // \returns 将要被计算的task
    virtual ComputeTask &Dequeue() const = 0;

    // Cancel 将一个计算任务从等待队列中剔除或停止计算
    // \param task_id 待Cancel的任务
    // \param checksum 该task_id对应的checksum
    // \returns 如果删除成功，返回对应的task_id。否则返回0
    virtual uint64_t Cancel(const uint64_t task_id, const char checksum[16]) = 0;

    // Execute 执行一个计算任务
    // \param task 需要被执行的任务
    // \returns 如果执行成功，返回0。否则返回一个非0的数。
    virtual int Execute(ComputeTask &task) = 0;

    // GetTaskStatus 获得当前计算任务的status
    // \param task_id 需要获取status的task_id
    // \returns 如果该任务已失效或不存在，返回INVAILD。否则返回其它三种状态中的一种
    virtual TaskStatus GetTaskStatus(const uint64_t task_id) const = 0;

    // GetTaskOutput 获取一个任务的输出，该函数仅当任务status=FINISHED时返回长度大于0的vector容器
    // \param task_id 需要获取输出的任务task_id
    // \param checksum 该task_id对应的checksum
    // \returns 如果任务有效且计算完毕，返回一个与task_id对应任务输出size一致的vector。
    //
    // 否则返回一个空vector，可再次调用GetTaskStatus查看当前任务状态。
    virtual vector<pair<string, vector<char>>> GetTaskOutput(uint64_t task_id, const char checksum[16]) const = 0;
};

// A timer should have the control of a series of timing tasks.
class IEventHandler
{
public:
    IEventHandler(){};
    virtual ~IEventHandler() {}

    // AddEvent 将一个事件增加到定时器队列中，当时间到达的时候执行事件
    // \param ts 该事件需要执行的UNIX时间戳
    // \param event 指向需要执行的事件指针。
    // \returns 如果插入成功，则返回该事件的一个对应id，该id恒大于0。否则返回0。
    virtual uint64_t AddEvent(uint64_t ts, void *event) = 0;

    // DeleteEvent 将一个事件从定时器队列中删除。
    // \param event_id 该事件对应的ID
    // \returns 如果删除成功，则返回0，否则返回一个非0的数。
    virtual int DeleteEvent(uint64_t event_id) = 0;

    // WaitEvent 等待一个事件执行完成。
    // 该函数会被阻塞，直到事件执行完毕。如果该事件存在callback，该callback将被自动调用。
    // \param event_id 需要等待的事件ID
    // \returns 不返回任何值。
    virtual void WaitEvent(uint64_t event_id) = 0;

    // CancelEvent 试图取消一个正在执行的事件。
    // 该函数可能会导致事件的异常，这取决于事件是否可以被取消，以及被取消的后果如何。
    // \param event_id 需要取消的事件ID
    // \returns 不返回任何值。
    virtual void CancelEvent(uint64_t event_id) = 0;

    // GetNextEvent 获取下一个将要被执行的事件ID。
    // \returns 返回下一个将要被执行的事件ID。如果当前没有需要被执行的事件，返回0。
    virtual uint64_t GetNextEvent() = 0;
};

// end of base.hpp