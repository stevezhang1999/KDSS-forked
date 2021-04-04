CXX := g++
CC := gcc

CUDA_INCLUDE_DIR := /usr/local/cuda/include
TRT_INCLUDE_DIR := /home/lijiakang/TensorRT-6.0.1.5/include
KGMALLOC_INCLUDE_DIR := /home/lijiakang/kgmalloc
HASH_INCLUDE_DIR := /home/lijiakang/kgmalloc/hash
TRT_SAMPLE_INCLUDE_DIR := /home/lijiakang/TensorRT-6.0.1.5/samples

CUDA_LIB_DIR := /usr/local/cuda/lib64
TRT_LIB_DIR := /home/lijiakang/TensorRT-6.0.1.5/lib
KGMALLOC_LIB_DIR := /home/lijiakang/kgmalloc/lib


INCLUDE_DIR := $(CUDA_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_INCLUDE_DIR)
INCLUDE_DIR += -I$(KGMALLOC_INCLUDE_DIR)
INCLUDE_DIR += -I$(HASH_INCLUDE_DIR)
INCLUDE_DIR += -I$(TRT_SAMPLE_INCLUDE_DIR)

LIB_DIR := $(CUDA_LIB_DIR)
LIB_DIR += -L$(TRT_LIB_DIR)
LIB_DIR += -L$(KGMALLOC_LIB_DIR)

LIBS := -lumap -lhash -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart -lrt -ldl -lpthread -lcurand

CFLAGS := -g -std=c++11 -Wno-deprecated-declarations -I$(INCLUDE_DIR) -L$(LIB_DIR) $(LIBS)

TARGET := main
SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))

EXT_OBJS := /home/lijiakang/TensorRT-6.0.1.5/bin/common/logger.o /home/lijiakang/kgmalloc/*.o ../util/RWMutex/*.o /home/lijiakang/kgmalloc/hash/hash.o /home/lijiakang/kgmalloc/umap/umap.o

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