NVCC       ?= nvcc
CXX        ?= g++
CUDA_ARCH  ?= sm_89
NVCCFLAGS  ?= -O2 -Xptxas -v -arch=$(CUDA_ARCH)
CXXFLAGS   ?= -O2 -std=c++17

GPU_TARGET = main
CPU_TARGET = main_cpu

.PHONY: all gpu cpu run-gpu run-cpu clean

all: gpu cpu

gpu: $(GPU_TARGET)

cpu: $(CPU_TARGET)

$(GPU_TARGET): main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(CPU_TARGET): main_cpu.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run-gpu: $(GPU_TARGET)
	./$(GPU_TARGET)

run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET)
