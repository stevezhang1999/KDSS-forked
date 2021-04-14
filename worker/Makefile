#ARCH  
ARCH= "$(shell uname -s 2>/dev/null)"

LINUX="Linux"

CXX := g++
CC := gcc

ifeq ($(ARCH) , $(LINUX))
	TRT_PATH := $(TENSORRT_PATH)
	KGMALLOC_PATH := /home/lijiakang/kgmalloc
	CUDA_INCLUDE_DIR := /usr/local/cuda/include
	CUDA_LIB_DIR := /usr/local/cuda/lib64
	TRT_INCLUDE_DIR := $(TRT_PATH)/include
	HASH_INCLUDE_DIR := $(KGMALLOC_PATH)/hash
	TRT_SAMPLE_INCLUDE_DIR := $(TRT_PATH)/samples
	TRT_LIB_DIR := $(TRT_PATH)/lib
	KGMALLOC_LIB_DIR := $(KGMALLOC_PATH)/lib
	HASH_LIB_DIR += $(KGMALLOC_PATH)/hash
	UMAP_LIB_DIR += $(KGMALLOC_PATH)/umap
	CUDA_LIBS := -lcudnn -lcublas -lcudart -lcurand -lnvidia-ml
	TRT_LIBS := -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
	LIBS := -lrt -ldl -lpthread
else
	TRT_PATH := "$(subst \,/,$(WINDOWS_TENSORRT_PATH))"
	KGMALLOC_PATH :=  "$(subst \,/,$(WINDOWS_KGMALLOC_PATH))"
	WINDOWS_CUDA_PATH =  "$(subst \,/,$(CUDA_PATH))"

	CUDA_INCLUDE_DIR := $(WINDOWS_CUDA_PATH)/include
	CUDA_LIB_DIR := $(WINDOWS_CUDA_PATH)/bin
	TRT_INCLUDE_DIR := $(TRT_PATH)/include
	HASH_INCLUDE_DIR := $(WINDOWS_KGMALLOC_PATH)/hash
	
	TRT_SAMPLE_INCLUDE_DIR := $(TRT_PATH)/samples
	TRT_LIB_DIR := $(TRT_PATH)/lib
	KGMALLOC_LIB_DIR := "$(WINDOWS_KGMALLOC_PATH)"/hash -L"$(WINDOWS_KGMALLOC_PATH)"/umap

	CUDA_LIBS := $(CUDA_LIB_DIR)/cudnn64_7.dll $(CUDA_LIB_DIR)/cublas64_100.dll $(CUDA_LIB_DIR)/cudart64_100.dll $(CUDA_LIB_DIR)/curand64_100.dll
	TRT_LIBS := $(TRT_LIB_DIR)/nvinfer.dll $(TRT_LIB_DIR)/nvparsers.dll $(TRT_LIB_DIR)/nvinfer_plugin.dll  $(TRT_LIB_DIR)/nvonnxparser.dll
	LIBS := -lpthread
endif

INCLUDE_DIR := $(CUDA_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_INCLUDE_DIR)
INCLUDE_DIR += -I$(KGMALLOC_PATH)
INCLUDE_DIR += -I$(HASH_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_SAMPLE_INCLUDE_DIR)
INCLUDE_DIR += -I.

LIB_DIR := $(CUDA_LIB_DIR)
LIB_DIR += -L$(TRT_LIB_DIR)
LIB_DIR += -L$(KGMALLOC_LIB_DIR)
LIB_DIR += -L$(HASH_LIB_DIR)
LIB_DIR += -L$(UMAP_LIB_DIR)

KGMALLOC_LIBS := -lumap -lhash
LIBS += $(KGMALLOC_LIBS)


CFLAGS := -g -std=c++11 -Wno-deprecated-declarations -I$(INCLUDE_DIR) -L$(LIB_DIR) $(CUDA_LIBS) $(TRT_LIBS) $(LIBS) -m64

# CFLAGS += -D__DEBUG

ifneq ($(ARCH), $(LINUX))
	CFLAGS += -fexec-charset=GBK -finput-charset=UTF-8
endif

TARGET := main
SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))

EXT_OBJS := $(TRT_PATH)/bin/common/logger.o 
EXT_OBJS += $(KGMALLOC_PATH)/*.o
EXT_OBJS += ../util/RWMutex/*.o 
EXT_OBJS += $(KGMALLOC_PATH)/hash/hash.o 
EXT_OBJS += $(KGMALLOC_PATH)/umap/umap.o

all:rw_mutex $(TARGET)
%.o:%.cpp
	@echo "Compling" $<
	$(CXX) -g -c $< $(CFLAGS) 
$(TARGET):$(OBJS)
	@echo "Linking to demo..."
	$(CXX) $(CFLAGS) -o $@ $(EXT_OBJS) $^

rw_mutex:
	@echo "Compiling" ../util/RWMutex/rwmutex.cpp
	$(CXX) -g -c ../util/RWMutex/rwmutex.cpp -o ../util/RWMutex/rwmutex.o -std=c++11
.PHONY : clean
clean:
ifeq ($(ARCH),$(LINUX))
	rm -rf $(TARGET) *.o
else
	rm -rf $(TARGET).exe *.o
endif
	