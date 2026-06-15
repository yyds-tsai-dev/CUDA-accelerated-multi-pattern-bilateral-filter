NVCC ?= nvcc
ARCH ?= sm_70
NVCCFLAGS ?= -O2 -std=c++17 -Xptxas -v -arch=$(ARCH)
TARGET := main

.PHONY: all run ptx profile clean

all:
	$(NVCC) $(NVCCFLAGS) main.cu -o $(TARGET)

run: all
	cd .. && ./P5/$(TARGET) 1024 1024 1 256 5
	cd .. && ./P5/$(TARGET) 1024 1024 2 256 5
	cd .. && ./P5/$(TARGET) 1024 1024 4 256 5
	cd .. && ./P5/$(TARGET) 1024 1024 8 256 5
	cd .. && ./P5/$(TARGET) 1024 1024 16 256 5

ptx:
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(ARCH) main.cu -o main.ptx

profile: all
	cd .. && ncu --set basic ./P5/$(TARGET) 1024 1024 16 256 5

clean:
	rm -f $(TARGET) main.ptx
