CXX := g++
CC := gcc

TRT_PATH := TRT_PATH := $(TENSORRT_PATH)
KGMALLOC_PATH := /home/lijiakang/kgmalloc

CUDA_INCLUDE_DIR := /usr/local/cuda/include
TRT_INCLUDE_DIR := $(TRT_PATH)/include

HASH_INCLUDE_DIR := $(KGMALLOC_PATH)/hash
TRT_SAMPLE_INCLUDE_DIR := $(TRT_PATH)/samples

CUDA_LIB_DIR := /usr/local/cuda/lib64
TRT_LIB_DIR := $(TRT_PATH)/lib
KGMALLOC_LIB_DIR := $(KGMALLOC_PATH)/lib


INCLUDE_DIR := $(CUDA_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_INCLUDE_DIR)
INCLUDE_DIR += -I$(KGMALLOC_PATH)
INCLUDE_DIR += -I$(HASH_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_SAMPLE_INCLUDE_DIR)
INCLUDE_DIR += -I../../

LIB_DIR := $(CUDA_LIB_DIR)
LIB_DIR += -L$(TRT_LIB_DIR)
LIB_DIR += -L$(KGMALLOC_LIB_DIR)

CUDA_LIBS := -lcudnn -lcublas -lcudart -lcurand
TRT_LIBS := -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
KGMALLOC_LIBS := -lumap -lhash
LIBS := -lrt -ldl -lpthread
LIBS += $(KGMALLOC_LIBS)
LIBS += $(CUDA_LIBS)
LIBS += $(TRT_LIBS)

CFLAGS := -g -std=c++11 -Wno-deprecated-declarations -I$(INCLUDE_DIR) -L$(LIB_DIR) $(LIBS)

TARGET := main
SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))
MAIN_OBJS = $(filter-out ../../main.o, $(wildcard ../../*.o))
OBJS += $(MAIN_OBJS)

EXT_OBJS := $(TRT_PATH)/bin/common/logger.o 
EXT_OBJS += $(KGMALLOC_PATH)/*.o
EXT_OBJS += ../../../util/RWMutex/*.o 
EXT_OBJS += $(KGMALLOC_PATH)/hash/hash.o 
EXT_OBJS += $(KGMALLOC_PATH)/umap/umap.o

all:$(TARGET)
%.o:%.cpp
	@echo "Compling" $<
	$(CXX) -c $< $(CFLAGS) 
$(TARGET):$(OBJS)
	@echo "Linking to demo..."
	$(CXX) $(CFLAGS) -o $@ $(EXT_OBJS) $^

.PHONY : clean
clean:
	rm -rf $(TARGET) *.o
