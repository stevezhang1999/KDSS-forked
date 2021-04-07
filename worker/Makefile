#ARCH  
ARCH= "$(shell uname -s 2>/dev/null)"

LINUX="Linux"

CXX := g++
CC := gcc

ifeq ($(ARCH) , $(LINUX))
	TRT_PATH := /home/lijiakang/TensorRT-6.0.1.5
	KGMALLOC_PATH := /home/lijiakang/kgmalloc
	CUDA_INCLUDE_DIR := /usr/local/cuda/include
	CUDA_LIB_DIR := /usr/local/cuda/lib64
else
	TRT_PATH := $(WINDOWS_TENSORRT_PATH)
	KGMALLOC_PATH := $(WINDOWS_KGMALLOC_PATH)
	CUDA_INCLUDE_DIR := "$(CUDA_PATH)/include"
	CUDA_LIB_DIR := "$(CUDA_PATH)/lib/x64"
endif

TRT_INCLUDE_DIR := $(TRT_PATH)/include

HASH_INCLUDE_DIR := $(KGMALLOC_PATH)/hash
TRT_SAMPLE_INCLUDE_DIR := $(TRT_PATH)/samples

TRT_LIB_DIR := $(TRT_PATH)/lib
KGMALLOC_LIB_DIR := $(KGMALLOC_PATH)/lib


INCLUDE_DIR := $(CUDA_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_INCLUDE_DIR)
INCLUDE_DIR += -I$(KGMALLOC_PATH)
INCLUDE_DIR += -I$(HASH_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_SAMPLE_INCLUDE_DIR)

LIB_DIR := $(CUDA_LIB_DIR)
LIB_DIR += -L$(TRT_LIB_DIR)
LIB_DIR += -L$(KGMALLOC_LIB_DIR)

ifeq ($(ARCH) , $(LINUX))
	CUDA_LIBS := -lcudnn -lcublas -lcudart -lcurand
	TRT_LIBS := -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
	LIBS := -lrt -ldl -lpthread
else
	CUDA_LIBS := $(CUDA_LIB_PATH)/cudnn64_7.dll $(CUDA_LIB_PATH)/cublas64_100.dll $(CUDA_LIB_PATH)/cudart64_100.dll $(CUDA_LIB_PATH)/curand64_100.dll
	TRT_LIBS := $(TRT_LIB_DIR)/nvinfer.dll $(TRT_LIB_DIR)/nvparsers.dll $(TRT_LIB_DIR)/nvinfer_plugin.dll $(TRT_LIB_DIR)/nvonnxparser.dll
	LIBS := libgcc_s_seh-1.dll libwinpthread-1.dll
endif

KGMALLOC_LIBS := -lumap -lhash
LIBS += $(KGMALLOC_LIBS)
LIBS += $(CUDA_LIBS)
LIBS += $(TRT_LIBS)


CFLAGS := -g -std=c++11 -Wno-deprecated-declarations -I$(INCLUDE_DIR) -L$(LIB_DIR) $(LIBS)

TARGET := main
SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))

EXT_OBJS := $(TRT_PATH)/bin/common/logger.o 
EXT_OBJS += $(KGMALLOC_PATH)/*.o
EXT_OBJS += ../util/RWMutex/*.o 
EXT_OBJS += $(KGMALLOC_PATH)/hash/hash.o 
EXT_OBJS += $(KGMALLOC_PATH)/umap/umap.o

all:$(TARGET)
ifeq ($(ARCH),$(LINUX))
	@echo "Compiling on Linux"
else
	@echo "Compiling on Windows"
endif
%.o:%.cpp
	@echo "Compling" $<
	$(CXX) -c $< $(CFLAGS) 
$(TARGET):$(OBJS)
	@echo "Linking to demo..."
	$(CXX) $(CFLAGS) -o $@ $(EXT_OBJS) $^

.PHONY : clean
clean:
ifeq ($(ARCH),$(LINUX))
	rm -rf $(TARGET) *.o
else
	rm -rf $(TARGET).exe *.o
endif
	