CXX ?= g++
RISCV_CXX ?= riscv64-linux-gnu-g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I..
TARGET := main

.PHONY: all host riscv run clean

all: host

host:
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

riscv:
	$(RISCV_CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

run: host
	cd .. && ./P1/$(TARGET) 32 32 3
	cd .. && ./P1/$(TARGET) 64 64 3

clean:
	rm -f $(TARGET)
	rm -rf m5out
