CXX ?= g++
RISCV_CXX ?= riscv64-linux-gnu-g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I..
RISCV_FLAGS ?= -std=c++17 -O2 -Wall -Wextra -I.. -march=rv64gcv -mabi=lp64d
TARGET := main

.PHONY: all host riscv run clean

all: host

host:
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

riscv:
	$(RISCV_CXX) $(RISCV_FLAGS) main.cpp -o $(TARGET)

run: host
	cd .. && ./P3/$(TARGET) 32 32 2
	cd .. && ./P3/$(TARGET) 32 32 4
	cd .. && ./P3/$(TARGET) 32 32 8
	cd .. && ./P3/$(TARGET) 64 64 4

clean:
	rm -f $(TARGET)
	rm -rf m5out
