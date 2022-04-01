#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CUDA=/usr/local/cuda

DEBERTA_PLUGIN=./disentangledAttentionPlugin
DEBERTA_HEADERS= $(DEBERTA_PLUGIN)/disentangledAttentionPlugin.h
COMMON=./common # TRT plugin common. has serialize.hpp, plugin.h, etc.

IDIR=./
CC=g++

LIBS_AND_INC= -I$(IDIR) -I$(DEBERTA_PLUGIN) -I$(COMMON) -I$(CUDA)/include -L$(CUDA)/lib64 -ldl

CFLAGS=-O3 -Wall -Wno-deprecated-declarations -Wl,--no-undefined -std=c++17 $(LIBS_AND_INC)
NVCC_FLAGS=-ccbin $(CC) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -O3 -Wno-deprecated-declarations $(LIBS_AND_INC)

ODIR=./build/obj
EXEC_OUT=./build/bin

LIBS=-lcudart -lnvinfer -lnvinfer_plugin -lstdc++fs -lstdc++ -lcublas -lcublasLt

# Get object file list for cpp files in plugin folders.
CPP_PLUGIN_SRCS = $(wildcard $(DEBERTA_PLUGIN)/*.cpp)
PLUGIN_OBJ = $(patsubst %,$(ODIR)/%, $(notdir $(patsubst %.cpp, $(ODIR)/%.o, $(CPP_PLUGIN_SRCS)))) 

# Get object file list for cu files in plugin folders.
CUDA_PLUGIN_SRCS = $(wildcard $(DEBERTA_PLUGIN)/*.cu)
CUDA_OBJ = $(patsubst %,$(ODIR)/%, $(notdir $(patsubst %.cu, $(ODIR)/%.o, $(CUDA_PLUGIN_SRCS))))

.PHONY: all
all : $(EXEC_OUT)/libcustomplugins.so

.PHONY: test
test : $(EXEC_OUT)/test test_run
$(EXEC_OUT)/test : $(DEBERTA_PLUGIN)/kernels.cu | $(EXEC_OUT)
	$(CUDA)/bin/nvcc -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -O3 -o $@ $< 
#	$(CUDA)/bin/nvcc -Xcompiler -O3 -E $< # show precomile code
	
test_run :
	$(EXEC_OUT)/test

# Build binaries and shared object files
$(EXEC_OUT)/libcustomplugins.so: $(PLUGIN_OBJ) $(CUDA_OBJ) | $(EXEC_OUT)
	$(CC) -shared -fPIC -o $@ $^ $(CFLAGS) $(LIBS) 

# Build Object files
$(ODIR)/%.o: $(DEBERTA_PLUGIN)/%.cpp $(DEBERTA_HEADERS) | $(ODIR)
	$(CC) -fpic -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(DEBERTA_PLUGIN)/%.cu $(DEBERTA_HEADERS) | $(ODIR)
	$(CUDA)/bin/nvcc -Xcompiler -fPIC -c -o $@ $< $(NVCC_FLAGS)
#	$(CUDA)/bin/nvcc -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -O3 -fPIC -c -o $@ $<


# Directory creation rules
$(EXEC_OUT):
	if [ ! -d $(EXEC_OUT) ]; then mkdir -p $(EXEC_OUT); fi

$(ODIR):
	if [ ! -d $(ODIR) ]; then mkdir -p $(ODIR); fi

# Rule to clean
.PHONY: clean
clean:
	rm -rf $(ODIR) 
	rm -rf $(EXEC_OUT)
