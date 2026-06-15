NVCC ?= nvcc
ARCH ?= sm_70
NVCCFLAGS ?= -O2 -std=c++17 -Xptxas -v -arch=$(ARCH)
TARGET := main

.PHONY: all run ptx profile clean

all:
	$(NVCC) $(NVCCFLAGS) main.cu -o $(TARGET)

run: all
	cd .. && ./P4/$(TARGET) 512 512 3 16 16 0 5
	cd .. && ./P4/$(TARGET) 512 512 3 16 16 1 5
	cd .. && ./P4/$(TARGET) 1024 1024 5 16 16 0 5
	cd .. && ./P4/$(TARGET) 1024 1024 5 16 16 1 5

ptx:
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(ARCH) main.cu -o main.ptx

profile: all
	cd .. && ncu --set basic ./P4/$(TARGET) 1024 1024 3 16 16 1 5

clean:
	rm -f $(TARGET) main.ptx
