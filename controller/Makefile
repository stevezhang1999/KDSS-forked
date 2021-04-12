CXX := g++
CC := gcc

TRT_PATH := $(TENSORRT_PATH)
KGMALLOC_PATH := /home/lijiakang/kgmalloc
CUDA_INCLUDE_DIR := /usr/local/cuda/include
CUDA_LIB_DIR := /usr/local/cuda/lib64
TRT_INCLUDE_DIR := $(TRT_PATH)/include
HASH_INCLUDE_DIR := $(KGMALLOC_PATH)/hash
TRT_SAMPLE_INCLUDE_DIR := $(TRT_PATH)/samples
TRT_LIB_DIR := $(TRT_PATH)/lib
KGMALLOC_LIB_DIR := $(KGMALLOC_PATH)/lib
CUDA_LIBS := -lcudnn -lcublas -lcudart -lcurand -lnvidia-ml
TRT_LIBS := -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
LIBS := -lrt -ldl -lpthread

INCLUDE_DIR := $(CUDA_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_INCLUDE_DIR)
INCLUDE_DIR += -I$(KGMALLOC_PATH)
INCLUDE_DIR += -I$(HASH_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_SAMPLE_INCLUDE_DIR)
INCLUDE_DIR += -I.
INCLUDE_DIR += -I../worker

LIB_DIR := $(CUDA_LIB_DIR)
LIB_DIR += -L$(TRT_LIB_DIR)
LIB_DIR += -L$(KGMALLOC_LIB_DIR)



CFLAGS := -g -std=c++11 -Wno-deprecated-declarations -I$(INCLUDE_DIR) -L$(LIB_DIR) $(CUDA_LIBS) $(TRT_LIBS) $(LIBS) -m64

CFLAGS += -D__DEBUG

TARGET := main
SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))

EXT_OBJS := $(TRT_PATH)/bin/common/logger.o 
EXT_OBJS += $(KGMALLOC_PATH)/*.o
EXT_OBJS += ../util/RWMutex/*.o 
EXT_OBJS += $(KGMALLOC_PATH)/hash/hash.o 
EXT_OBJS += $(KGMALLOC_PATH)/umap/umap.o
EXT_OBJS += ../worker/base.o ../worker/trt_allocator.o ../worker/transfer_worker.o ../worker/computation_worker.o

all: $(WORKER_DIR) $(TARGET)
%.o:%.cpp
	@echo "Compling" $<
	$(CXX) -c $< $(CFLAGS) 
$(TARGET):$(OBJS)
	@echo "Linking to demo..."
	$(CXX) $(CFLAGS) $(EXT_OBJS) $^ -o $@

WORKER_DIR = /home/lijiakang/KDSS/worker

$(WORKER_DIR):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY : clean
clean:
	rm -rf $(TARGET) *.o
	