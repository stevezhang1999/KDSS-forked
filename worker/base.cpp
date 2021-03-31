#include "base.hpp"

// model_table 全局唯一索引与模型名称对照表
std::unordered_map<int, std::string> model_table;

// max_index 当前最大index
atomic<int> max_index(-1);

// mt_rw_mu model_table配套RW锁
RWMutex mt_rw_mu;

// engine_table 全局唯一模型名称与引擎对照表
std::unordered_map<std::string, EngineInfo> engine_table;

// et_rw_mu engine_table配套RW锁
RWMutex et_rw_mu;