# Bilateral Filter Final Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a course-ready Computer Architecture final project that compares one arithmetic bilateral-style image filter across scalar C++, RVV reduction, SIMD-like RVV, CUDA SIMT, and multi-pattern CUDA GPU parallelism.

**Architecture:** The project uses small deterministic grayscale arrays for gem5/RVV and larger deterministic generated images for CUDA. Shared host-side utilities live in `common/`, each course part is independently buildable, and every executable prints checksums, selected pixel values, max-difference validation, and timing or simulator guidance.

**Tech Stack:** C++17, CUDA C++, RISC-V GNU toolchain inside the course gem5 Docker image, CUDA 12.x, Nsight Compute, GNU Make, shell scripts.

---

## File Structure

Create these files:

- `common/bilateral_common.h`: header-only constants, deterministic image generation, arithmetic bilateral weights, scalar reference filter, checksum, max-difference, selected-pixel printing, PGM writer.
- `common/result_io.h`: small helpers for consistent result directory creation and CSV line writing.
- `scripts/check_environment.sh`: checks Git LFS, Docker, GPU, CUDA, `nvcc`, `ncu`, and RISC-V/gem5 container availability.
- `scripts/collect_cuda_results.sh`: builds and runs P4/P5 CUDA experiments on the host or inside CUDA container.
- `P1/Makefile`: host and gem5 build commands for scalar baseline.
- `P1/main.cpp`: scalar baseline entry point.
- `P2/Makefile`: host fallback and RISC-V vector build commands for RVV reduction.
- `P2/main.cpp`: RVV reduction entry point with scalar fallback for non-RISC-V host smoke tests.
- `P3/Makefile`: host fallback and RISC-V vector build commands for SIMD-like RVV.
- `P3/main.cpp`: SIMD-like RVV entry point with scalar fallback.
- `P4/Makefile`: CUDA build, PTX generation, and Nsight Compute command helpers.
- `P4/main.cu`: CUDA SIMT global-memory and shared-memory tiled implementations.
- `P5/Makefile`: CUDA build, PTX generation, and Nsight Compute command helpers for multi-pattern GPU.
- `P5/main.cu`: multi-pattern CUDA implementation with `grid.y` as pattern index.
- `results/.gitkeep`: keep results directory present.
- `data/.gitkeep`: keep data directory present for generated images and PGM outputs.
- `Readme.txt`: exact build/run instructions for all parts.
- `report/Report_outline.md`: report source outline with fixed section text and tables for measured experiment rows.

Modify these files:

- `.gitignore`: ignore build binaries, generated PTX, gem5 `m5out/`, CSV results, and generated PGM files while keeping `results/.gitkeep` and `data/.gitkeep`.

Do not modify these files:

- `docs/CA_Final_Project.pdf`
- `docs/CA_Final_Project_example.pdf`
- `docs/環境Setup.pdf`

## Implementation Rules

- Use deterministic generated data; do not require external image libraries for correctness.
- Use clamp-to-edge boundary handling in every part.
- Use the arithmetic bilateral-style weight from the approved spec.
- Print output checksums and selected pixels in every part to prevent dead-code elimination.
- Keep gem5 sizes at 32x32 and 64x64 with radius 3.
- Keep CUDA sizes at 512x512 and 1024x1024 with radius 3 and 5 for P4.
- Keep P5 at 1024x1024, radius 3, pattern counts 1, 2, 4, 8, and 16.
- Use `-arch=sm_70` for the detected Tesla V100 host GPU.

---

### Task 1: Environment Recovery And Checks

**Files:**
- Create: `scripts/check_environment.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Write the environment checker**

Create `scripts/check_environment.sh`:

```bash
#!/usr/bin/env bash
set -u

echo "== Host =="
uname -a

echo
echo "== Git LFS =="
if git lfs version >/dev/null 2>&1; then
  git lfs version
else
  echo "missing: git-lfs"
fi

echo
echo "== Docker =="
if command -v docker >/dev/null 2>&1; then
  docker --version
  docker ps >/dev/null 2>&1 && echo "docker-access: ok" || echo "docker-access: blocked"
else
  echo "missing: docker"
fi

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "missing: nvidia-smi"
fi

echo
echo "== CUDA =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
else
  echo "missing: nvcc"
fi

echo
echo "== Nsight Compute =="
if command -v ncu >/dev/null 2>&1; then
  ncu --version | head -n 3
else
  echo "missing: ncu"
fi

echo
echo "== RISC-V toolchain on host =="
if command -v riscv64-linux-gnu-g++ >/dev/null 2>&1; then
  riscv64-linux-gnu-g++ --version | head -n 1
else
  echo "missing: riscv64-linux-gnu-g++ on host; use ca-fp-gem5 container"
fi
```

- [ ] **Step 2: Make the checker executable**

Run:

```bash
chmod +x scripts/check_environment.sh
```

Expected: command exits with status 0.

- [ ] **Step 3: Update `.gitignore`**

Append these lines to `.gitignore`:

```gitignore
# Build outputs
P1/main
P2/main
P3/main
P4/main
P5/main
*.o
*.ptx

# gem5 outputs
m5out/
P1/m5out/
P2/m5out/
P3/m5out/

# Generated experiment artifacts
results/*.csv
results/*.txt
data/*.pgm
```

- [ ] **Step 4: Run environment checker**

Run:

```bash
./scripts/check_environment.sh
```

Expected on the current host: it reports NVIDIA V100 GPU, CUDA 12.9, `nvcc`, and `ncu`; it reports Docker and Git LFS as missing until system setup is completed.

- [ ] **Step 5: Install Git LFS and pull starter package when system package access works**

Run:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
file final_project-20260522T170812Z-3-001.zip
```

Expected after success: `file` reports a Zip archive or compressed data, not ASCII text.

- [ ] **Step 6: Install Docker when system package access works**

Run:

```bash
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh
sudo service docker start
sudo docker run hello-world
```

Expected: `hello-world` prints Docker's success message.

- [ ] **Step 7: Download course containers**

Run:

```bash
sudo docker run -it --name ca-fp-gem5 \
  -v "$PWD":/workspace -w /workspace \
  weisheng505/gem5-rvv-image:v1
```

Inside the container, run:

```bash
exit
```

Then run:

```bash
sudo docker run -it --gpus all --name ca-fp-cuda \
  -v "$PWD":/workspace -w /workspace \
  weisheng505/cuda-env:v1
```

Inside the CUDA container, run:

```bash
nvidia-smi
exit
```

Expected: both images download, both containers start, and the CUDA container sees the GPU.

- [ ] **Step 8: Commit**

Run:

```bash
git add .gitignore scripts/check_environment.sh
git commit -m "chore: add environment checks"
```

---

### Task 2: Shared Bilateral Utilities

**Files:**
- Create: `common/bilateral_common.h`
- Create: `common/result_io.h`
- Create: `data/.gitkeep`
- Create: `results/.gitkeep`

- [ ] **Step 1: Create shared bilateral utilities**

Create `common/bilateral_common.h` with these APIs and matching implementations:

```cpp
#ifndef BILATERAL_COMMON_H
#define BILATERAL_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct BilateralParams {
  int width;
  int height;
  int radius;
  float sigma_s2;
  float sigma_r2;
};

inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline float clean_pixel_value(int x, int y, int pattern) {
  const int base = (x * 13 + y * 17 + pattern * 19) & 255;
  const int edge = (x > 0 && ((x / 16) & 1)) ? 42 : 0;
  return static_cast<float>((base + edge) & 255);
}

inline float noisy_pixel_value(int x, int y, int pattern) {
  const int noise = ((x * 31 + y * 7 + pattern * 23) % 21) - 10;
  const int value = static_cast<int>(clean_pixel_value(x, y, pattern)) + noise;
  return static_cast<float>(clamp_int(value, 0, 255));
}

inline void generate_image(std::vector<float>& image, int width, int height, int pattern) {
  image.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image[static_cast<size_t>(y) * width + x] = noisy_pixel_value(x, y, pattern);
    }
  }
}

inline float spatial_weight(int dx, int dy, float sigma_s2) {
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  return 1.0f / (1.0f + d2 / sigma_s2);
}

inline float range_weight(float center, float neighbor, float sigma_r2) {
  const float diff = center - neighbor;
  return 1.0f / (1.0f + (diff * diff) / sigma_r2);
}

inline float bilateral_weight(int dx, int dy, float center, float neighbor, const BilateralParams& params) {
  return spatial_weight(dx, dy, params.sigma_s2) * range_weight(center, neighbor, params.sigma_r2);
}

inline float scalar_filter_one(const std::vector<float>& input, int x, int y, const BilateralParams& params) {
  const int width = params.width;
  const int height = params.height;
  const float center = input[static_cast<size_t>(y) * width + x];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -params.radius; dy <= params.radius; ++dy) {
    const int yy = clamp_int(y + dy, 0, height - 1);
    for (int dx = -params.radius; dx <= params.radius; ++dx) {
      const int xx = clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[static_cast<size_t>(yy) * width + xx];
      const float weight = bilateral_weight(dx, dy, center, neighbor, params);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }

  return numerator / denominator;
}

inline void scalar_bilateral_filter(const std::vector<float>& input,
                                    std::vector<float>& output,
                                    const BilateralParams& params) {
  output.assign(input.size(), 0.0f);
  for (int y = 0; y < params.height; ++y) {
    for (int x = 0; x < params.width; ++x) {
      output[static_cast<size_t>(y) * params.width + x] = scalar_filter_one(input, x, y, params);
    }
  }
}

inline double checksum_image(const std::vector<float>& image) {
  double checksum = 0.0;
  for (size_t i = 0; i < image.size(); ++i) {
    checksum += static_cast<double>(image[i]) * static_cast<double>((i % 97) + 1);
  }
  return checksum;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

inline void print_selected_pixels(const char* label, const std::vector<float>& image, int width, int height) {
  const int xs[4] = {0, width / 3, (2 * width) / 3, width - 1};
  const int ys[4] = {0, height / 3, (2 * height) / 3, height - 1};
  std::cout << label;
  for (int i = 0; i < 4; ++i) {
    const int x = clamp_int(xs[i], 0, width - 1);
    const int y = clamp_int(ys[i], 0, height - 1);
    std::cout << " p" << i << "=" << std::fixed << std::setprecision(4)
              << image[static_cast<size_t>(y) * width + x];
  }
  std::cout << "\n";
}

inline void write_pgm(const std::string& path, const std::vector<float>& image, int width, int height) {
  std::ofstream out(path, std::ios::binary);
  out << "P5\n" << width << " " << height << "\n255\n";
  for (float value : image) {
    const int rounded = clamp_int(static_cast<int>(std::lround(value)), 0, 255);
    const unsigned char byte = static_cast<unsigned char>(rounded);
    out.write(reinterpret_cast<const char*>(&byte), 1);
  }
}

#endif
```

- [ ] **Step 2: Create result I/O helper**

Create `common/result_io.h`:

```cpp
#ifndef RESULT_IO_H
#define RESULT_IO_H

#include <filesystem>
#include <fstream>
#include <string>

inline void ensure_results_dir() {
  std::filesystem::create_directories("results");
}

inline void append_csv_line(const std::string& path, const std::string& line) {
  std::ofstream out(path, std::ios::app);
  out << line << "\n";
}

#endif
```

- [ ] **Step 3: Create tracked data and results directories**

Run:

```bash
mkdir -p data results
touch data/.gitkeep results/.gitkeep
```

Expected: both `.gitkeep` files exist.

- [ ] **Step 4: Write a temporary smoke test for shared utilities**

Create `/tmp/test_bilateral_common.cpp`:

```cpp
#include "common/bilateral_common.h"
#include <iostream>

int main() {
  BilateralParams params{32, 32, 3, 9.0f, 900.0f};
  std::vector<float> input;
  std::vector<float> output;
  generate_image(input, params.width, params.height, 0);
  scalar_bilateral_filter(input, output, params);
  std::cout << "checksum=" << checksum_image(output) << "\n";
  print_selected_pixels("selected", output, params.width, params.height);
  return output.size() == 1024 ? 0 : 1;
}
```

- [ ] **Step 5: Run shared utility smoke test**

Run:

```bash
g++ -std=c++17 -O2 -I. /tmp/test_bilateral_common.cpp -o /tmp/test_bilateral_common
/tmp/test_bilateral_common
```

Expected: command exits 0 and prints a `checksum=` line plus selected pixel values.

- [ ] **Step 6: Commit**

Run:

```bash
git add common/bilateral_common.h common/result_io.h data/.gitkeep results/.gitkeep
git commit -m "feat: add shared bilateral utilities"
```

---

### Task 3: P1 Scalar Baseline

**Files:**
- Create: `P1/Makefile`
- Create: `P1/main.cpp`

- [ ] **Step 1: Create P1 entry point**

Create `P1/main.cpp`:

```cpp
#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int radius = argc > 3 ? std::atoi(argv[3]) : 3;
  BilateralParams params{width, height, radius, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> output;
  generate_image(input, width, height, 0);

  const auto start = std::chrono::high_resolution_clock::now();
  scalar_bilateral_filter(input, output, params);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  std::cout << "part=P1 scalar\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "host_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << "P1," << width << "," << height << "," << radius << "," << checksum << "," << ms;
  append_csv_line("results/p1_scalar.csv", row.str());
  write_pgm("data/p1_scalar_output.pgm", output, width, height);
  return 0;
}
```

- [ ] **Step 2: Create P1 Makefile**

Create `P1/Makefile`:

```makefile
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
	./$(TARGET) 32 32 3
	./$(TARGET) 64 64 3

clean:
	rm -f $(TARGET)
	rm -rf m5out
```

- [ ] **Step 3: Build P1 on host**

Run:

```bash
make -C P1 clean
make -C P1 host
```

Expected: `P1/main` exists.

- [ ] **Step 4: Run P1 smoke tests**

Run:

```bash
./P1/main 32 32 3
./P1/main 64 64 3
```

Expected: both runs print `part=P1 scalar`, `checksum=`, `host_ms=`, and `selected`.

- [ ] **Step 5: Build P1 inside gem5 container**

Run after `ca-fp-gem5` exists:

```bash
sudo docker start -ai ca-fp-gem5
```

Inside the container:

```bash
cd /workspace
make -C P1 clean
make -C P1 riscv
```

Expected: `P1/main` is a RISC-V executable.

- [ ] **Step 6: Run P1 with gem5**

Inside the gem5 container:

```bash
cd /workspace
GEM5_BIN="$(find / -path '*/build/RISCV/gem5.opt' -print -quit)"
GEM5_CFG="$(find / -path '*/configs/example/se.py' -print -quit)"
"$GEM5_BIN" "$GEM5_CFG" -c ./P1/main -o "32 32 3"
cp m5out/stats.txt results/p1_32_stats.txt
"$GEM5_BIN" "$GEM5_CFG" -c ./P1/main -o "64 64 3"
cp m5out/stats.txt results/p1_64_stats.txt
```

Expected: `results/p1_32_stats.txt` and `results/p1_64_stats.txt` contain `simSeconds`, `simInsts`, `numCycles`, `cpi` or `CPI`, and cache miss-rate fields.

- [ ] **Step 7: Commit**

Run:

```bash
git add P1/Makefile P1/main.cpp
git commit -m "feat: add scalar baseline"
```

---

### Task 4: P2 RVV Vector Reduction

**Files:**
- Create: `P2/Makefile`
- Create: `P2/main.cpp`

- [ ] **Step 1: Create P2 entry point with scalar fallback**

Create `P2/main.cpp` with this structure:

```cpp
#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static float reduce_sum_scalar(const float* values, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += values[i];
  }
  return sum;
}

static float reduce_sum_rvv(const float* values, int n) {
#if defined(__riscv_vector)
  float sum = 0.0f;
  int offset = 0;
  while (offset < n) {
    size_t vl = __riscv_vsetvl_e32m1(n - offset);
    vfloat32m1_t x = __riscv_vle32_v_f32m1(values + offset, vl);
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(x, zero, vl);
    sum += __riscv_vfmv_f_s_f32m1_f32(red);
    offset += static_cast<int>(vl);
  }
  return sum;
#else
  return reduce_sum_scalar(values, n);
#endif
}

static float rvv_reduction_filter_one(const std::vector<float>& input,
                                      int x,
                                      int y,
                                      const BilateralParams& params) {
  const int width = params.width;
  const int height = params.height;
  const int window = (2 * params.radius + 1) * (2 * params.radius + 1);
  std::vector<float> numerator_terms(static_cast<size_t>(window));
  std::vector<float> denominator_terms(static_cast<size_t>(window));
  const float center = input[static_cast<size_t>(y) * width + x];
  int term = 0;

  for (int dy = -params.radius; dy <= params.radius; ++dy) {
    const int yy = clamp_int(y + dy, 0, height - 1);
    for (int dx = -params.radius; dx <= params.radius; ++dx) {
      const int xx = clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[static_cast<size_t>(yy) * width + xx];
      const float weight = bilateral_weight(dx, dy, center, neighbor, params);
      numerator_terms[static_cast<size_t>(term)] = neighbor * weight;
      denominator_terms[static_cast<size_t>(term)] = weight;
      ++term;
    }
  }

  const float numerator = reduce_sum_rvv(numerator_terms.data(), window);
  const float denominator = reduce_sum_rvv(denominator_terms.data(), window);
  return numerator / denominator;
}

static void rvv_reduction_filter(const std::vector<float>& input,
                                 std::vector<float>& output,
                                 const BilateralParams& params) {
  output.assign(input.size(), 0.0f);
  for (int y = 0; y < params.height; ++y) {
    for (int x = 0; x < params.width; ++x) {
      output[static_cast<size_t>(y) * params.width + x] = rvv_reduction_filter_one(input, x, y, params);
    }
  }
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  BilateralParams params{width, height, 3, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> reference;
  std::vector<float> output;
  generate_image(input, width, height, 0);
  scalar_bilateral_filter(input, reference, params);

  const auto start = std::chrono::high_resolution_clock::now();
  rvv_reduction_filter(input, output, params);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(reference, output);
  std::cout << "part=P2 rvv_reduction\n";
  std::cout << "width=" << width << " height=" << height << " radius=3\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "host_fallback_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << "P2," << width << "," << height << ",3," << checksum << "," << diff << "," << ms;
  append_csv_line("results/p2_rvv_reduction.csv", row.str());
  return diff <= 0.01f ? 0 : 2;
}
```

- [ ] **Step 2: Create P2 Makefile**

Create `P2/Makefile`:

```makefile
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
	./$(TARGET) 32 32
	./$(TARGET) 64 64

clean:
	rm -f $(TARGET)
	rm -rf m5out
```

- [ ] **Step 3: Build and run P2 host fallback**

Run:

```bash
make -C P2 clean
make -C P2 host
./P2/main 32 32
./P2/main 64 64
```

Expected: both runs print `part=P2 rvv_reduction` and `max_abs_diff=0.000000` or another value below `0.010000`.

- [ ] **Step 4: Build P2 with RVV inside gem5 container**

Inside `ca-fp-gem5`:

```bash
cd /workspace
make -C P2 clean
make -C P2 riscv
```

Expected: P2 builds with `-march=rv64gcv`.

- [ ] **Step 5: Run P2 with gem5**

Inside the gem5 container:

```bash
cd /workspace
GEM5_BIN="$(find / -path '*/build/RISCV/gem5.opt' -print -quit)"
GEM5_CFG="$(find / -path '*/configs/example/se.py' -print -quit)"
"$GEM5_BIN" "$GEM5_CFG" -c ./P2/main -o "32 32"
cp m5out/stats.txt results/p2_32_stats.txt
"$GEM5_BIN" "$GEM5_CFG" -c ./P2/main -o "64 64"
cp m5out/stats.txt results/p2_64_stats.txt
```

Expected: `results/p2_32_stats.txt` and `results/p2_64_stats.txt` exist, and stdout includes `max_abs_diff` below `0.010000`.

- [ ] **Step 6: Commit**

Run:

```bash
git add P2/Makefile P2/main.cpp
git commit -m "feat: add rvv reduction version"
```

---

### Task 5: P3 SIMD-Like RVV Parallelization

**Files:**
- Create: `P3/Makefile`
- Create: `P3/main.cpp`

- [ ] **Step 1: Create P3 entry point**

Create `P3/main.cpp` with this implementation plan:

```cpp
#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static void simd_like_filter_scalar_fallback(const std::vector<float>& input,
                                             std::vector<float>& output,
                                             const BilateralParams& params,
                                             int k) {
  output.assign(input.size(), 0.0f);
  for (int x = 0; x < params.width; ++x) {
    for (int y = 0; y < params.height; y += k) {
      const int lanes = std::min(k, params.height - y);
      for (int lane = 0; lane < lanes; ++lane) {
        output[static_cast<size_t>(y + lane) * params.width + x] =
            scalar_filter_one(input, x, y + lane, params);
      }
    }
  }
}

static void simd_like_filter_rvv(const std::vector<float>& input,
                                 std::vector<float>& output,
                                 const BilateralParams& params,
                                 int k) {
#if defined(__riscv_vector)
  scalar_bilateral_filter(input, output, params);
  const int width = params.width;
  const int height = params.height;
  const ptrdiff_t stride = static_cast<ptrdiff_t>(width * sizeof(float));

  for (int x = params.radius; x < width - params.radius; ++x) {
    int y = params.radius;
    for (; y + k + params.radius <= height; y += k) {
      const size_t vl = __riscv_vsetvl_e32m1(k);
      const float* center_ptr = input.data() + static_cast<size_t>(y) * width + x;
      vfloat32m1_t center = __riscv_vlse32_v_f32m1(center_ptr, stride, vl);
      vfloat32m1_t numerator = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vfloat32m1_t denominator = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      for (int dy = -params.radius; dy <= params.radius; ++dy) {
        const int yy = clamp_int(y + dy, 0, height - 1);
        for (int dx = -params.radius; dx <= params.radius; ++dx) {
          const int xx = x + dx;
          const float* neighbor_ptr = input.data() + static_cast<size_t>(yy) * width + xx;
          vfloat32m1_t neighbor = __riscv_vlse32_v_f32m1(neighbor_ptr, stride, vl);
          vfloat32m1_t diff = __riscv_vfsub_vv_f32m1(center, neighbor, vl);
          vfloat32m1_t diff2 = __riscv_vfmul_vv_f32m1(diff, diff, vl);
          vfloat32m1_t range_den = __riscv_vfadd_vf_f32m1(
              __riscv_vfdiv_vf_f32m1(diff2, params.sigma_r2, vl), 1.0f, vl);
          vfloat32m1_t range = __riscv_vfrdiv_vf_f32m1(range_den, 1.0f, vl);
          const float spatial = spatial_weight(dx, dy, params.sigma_s2);
          vfloat32m1_t weight = __riscv_vfmul_vf_f32m1(range, spatial, vl);
          numerator = __riscv_vfmacc_vv_f32m1(numerator, neighbor, weight, vl);
          denominator = __riscv_vfadd_vv_f32m1(denominator, weight, vl);
        }
      }

      vfloat32m1_t result = __riscv_vfdiv_vv_f32m1(numerator, denominator, vl);
      float* output_ptr = output.data() + static_cast<size_t>(y) * width + x;
      __riscv_vsse32_v_f32m1(output_ptr, stride, result, vl);
    }

  }
#else
  simd_like_filter_scalar_fallback(input, output, params, k);
#endif
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 32;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int k = argc > 3 ? std::atoi(argv[3]) : 4;
  BilateralParams params{width, height, 3, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> reference;
  std::vector<float> output;
  generate_image(input, width, height, 0);
  scalar_bilateral_filter(input, reference, params);

  const auto start = std::chrono::high_resolution_clock::now();
  simd_like_filter_rvv(input, output, params, k);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(stop - start).count();

  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(reference, output);
  std::cout << "part=P3 simd_like_rvv\n";
  std::cout << "width=" << width << " height=" << height << " radius=3 k=" << k << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "host_fallback_ms=" << std::fixed << std::setprecision(6) << ms << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << "P3," << width << "," << height << ",3," << k << "," << checksum << "," << diff << "," << ms;
  append_csv_line("results/p3_simd_like_rvv.csv", row.str());
  return diff <= 0.01f ? 0 : 2;
}
```

- [ ] **Step 2: Create P3 Makefile**

Create `P3/Makefile`:

```makefile
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
	./$(TARGET) 32 32 2
	./$(TARGET) 32 32 4
	./$(TARGET) 32 32 8
	./$(TARGET) 64 64 4

clean:
	rm -f $(TARGET)
	rm -rf m5out
```

- [ ] **Step 3: Build and run P3 host fallback**

Run:

```bash
make -C P3 clean
make -C P3 host
./P3/main 32 32 2
./P3/main 32 32 4
./P3/main 32 32 8
./P3/main 64 64 4
```

Expected: every run prints `part=P3 simd_like_rvv` and `max_abs_diff` below `0.010000`.

- [ ] **Step 4: Build P3 with RVV inside gem5 container**

Inside `ca-fp-gem5`:

```bash
cd /workspace
make -C P3 clean
make -C P3 riscv
```

Expected: P3 builds with `-march=rv64gcv`.

- [ ] **Step 5: Run P3 with gem5**

Inside the gem5 container:

```bash
cd /workspace
GEM5_BIN="$(find / -path '*/build/RISCV/gem5.opt' -print -quit)"
GEM5_CFG="$(find / -path '*/configs/example/se.py' -print -quit)"
"$GEM5_BIN" "$GEM5_CFG" -c ./P3/main -o "32 32 2"
cp m5out/stats.txt results/p3_32_k2_stats.txt
"$GEM5_BIN" "$GEM5_CFG" -c ./P3/main -o "32 32 4"
cp m5out/stats.txt results/p3_32_k4_stats.txt
"$GEM5_BIN" "$GEM5_CFG" -c ./P3/main -o "32 32 8"
cp m5out/stats.txt results/p3_32_k8_stats.txt
"$GEM5_BIN" "$GEM5_CFG" -c ./P3/main -o "64 64 4"
cp m5out/stats.txt results/p3_64_k4_stats.txt
```

Expected: the copied stats files exist, and stdout includes `max_abs_diff` below `0.010000`.

- [ ] **Step 6: Commit**

Run:

```bash
git add P3/Makefile P3/main.cpp
git commit -m "feat: add simd-like rvv version"
```

---

### Task 6: P4 CUDA SIMT Naive And Shared-Memory Tiling

**Files:**
- Create: `P4/Makefile`
- Create: `P4/main.cu`

- [ ] **Step 1: Create P4 CUDA implementation**

Create `P4/main.cu` with these required components:

```cpp
#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while (0)

__device__ int d_clamp_int(int v, int lo, int hi) {
  return max(lo, min(v, hi));
}

__device__ float d_spatial_weight(int dx, int dy, float sigma_s2) {
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  return 1.0f / (1.0f + d2 / sigma_s2);
}

__device__ float d_range_weight(float center, float neighbor, float sigma_r2) {
  const float diff = center - neighbor;
  return 1.0f / (1.0f + (diff * diff) / sigma_r2);
}

__global__ void bilateral_naive_kernel(const float* input,
                                       float* output,
                                       int width,
                                       int height,
                                       int radius,
                                       float sigma_s2,
                                       float sigma_r2) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const float center = input[y * width + x];
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = d_clamp_int(y + dy, 0, height - 1);
    for (int dx = -radius; dx <= radius; ++dx) {
      const int xx = d_clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[yy * width + xx];
      const float weight = d_spatial_weight(dx, dy, sigma_s2) * d_range_weight(center, neighbor, sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }
  output[y * width + x] = numerator / denominator;
}

__global__ void bilateral_shared_kernel(const float* input,
                                        float* output,
                                        int width,
                                        int height,
                                        int radius,
                                        float sigma_s2,
                                        float sigma_r2) {
  extern __shared__ float tile[];
  const int tile_w = blockDim.x + 2 * radius;
  const int tile_h = blockDim.y + 2 * radius;
  const int threads = blockDim.x * blockDim.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_x = blockIdx.x * blockDim.x;
  const int block_y = blockIdx.y * blockDim.y;

  for (int linear = tid; linear < tile_w * tile_h; linear += threads) {
    const int tx = linear % tile_w;
    const int ty = linear / tile_w;
    const int gx = d_clamp_int(block_x + tx - radius, 0, width - 1);
    const int gy = d_clamp_int(block_y + ty - radius, 0, height - 1);
    tile[linear] = input[gy * width + gx];
  }
  __syncthreads();

  const int x = block_x + threadIdx.x;
  const int y = block_y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int local_x = threadIdx.x + radius;
  const int local_y = threadIdx.y + radius;
  const float center = tile[local_y * tile_w + local_x];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const float neighbor = tile[(local_y + dy) * tile_w + (local_x + dx)];
      const float weight = d_spatial_weight(dx, dy, sigma_s2) * d_range_weight(center, neighbor, sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }
  output[y * width + x] = numerator / denominator;
}

static float run_kernel(bool shared,
                        const std::vector<float>& input,
                        std::vector<float>& output,
                        const BilateralParams& params,
                        dim3 block) {
  const size_t bytes = input.size() * sizeof(float);
  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  cudaEvent_t h2d_start;
  cudaEvent_t h2d_stop;
  cudaEvent_t kernel_start;
  cudaEvent_t kernel_stop;
  cudaEvent_t d2h_start;
  cudaEvent_t d2h_stop;
  CUDA_CHECK(cudaEventCreate(&h2d_start));
  CUDA_CHECK(cudaEventCreate(&h2d_stop));
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventCreate(&d2h_start));
  CUDA_CHECK(cudaEventCreate(&d2h_stop));

  CUDA_CHECK(cudaEventRecord(h2d_start));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d_stop));
  CUDA_CHECK(cudaEventSynchronize(h2d_stop));

  const dim3 grid((params.width + block.x - 1) / block.x,
                  (params.height + block.y - 1) / block.y);

  CUDA_CHECK(cudaEventRecord(kernel_start));
  if (shared) {
    const int tile_w = static_cast<int>(block.x) + 2 * params.radius;
    const int tile_h = static_cast<int>(block.y) + 2 * params.radius;
    const size_t shared_bytes = static_cast<size_t>(tile_w) * tile_h * sizeof(float);
    bilateral_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, params.width, params.height,
                                                           params.radius, params.sigma_s2, params.sigma_r2);
  } else {
    bilateral_naive_kernel<<<grid, block>>>(d_input, d_output, params.width, params.height,
                                            params.radius, params.sigma_s2, params.sigma_r2);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  CUDA_CHECK(cudaEventRecord(d2h_start));
  output.resize(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h_stop));
  CUDA_CHECK(cudaEventSynchronize(d2h_stop));

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

  std::cout << "h2d_ms=" << h2d_ms << " kernel_ms=" << kernel_ms << " d2h_ms=" << d2h_ms << "\n";

  CUDA_CHECK(cudaEventDestroy(h2d_start));
  CUDA_CHECK(cudaEventDestroy(h2d_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));
  CUDA_CHECK(cudaEventDestroy(d2h_start));
  CUDA_CHECK(cudaEventDestroy(d2h_stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return kernel_ms;
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 512;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int radius = argc > 3 ? std::atoi(argv[3]) : 3;
  const int block_x = argc > 4 ? std::atoi(argv[4]) : 16;
  const int block_y = argc > 5 ? std::atoi(argv[5]) : 16;
  const bool shared = argc > 6 ? std::atoi(argv[6]) != 0 : false;
  BilateralParams params{width, height, radius, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> reference;
  std::vector<float> output;
  generate_image(input, width, height, 0);
  scalar_bilateral_filter(input, reference, params);

  const float kernel_ms = run_kernel(shared, input, output, params, dim3(block_x, block_y));
  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(reference, output);
  std::cout << "part=P4 cuda_simt variant=" << (shared ? "shared" : "naive") << "\n";
  std::cout << "width=" << width << " height=" << height << " radius=" << radius
            << " block=" << block_x << "x" << block_y << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  print_selected_pixels("selected", output, width, height);

  ensure_results_dir();
  std::ostringstream row;
  row << "P4," << (shared ? "shared" : "naive") << "," << width << "," << height << ","
      << radius << "," << block_x << "x" << block_y << "," << checksum << "," << diff << "," << kernel_ms;
  append_csv_line("results/p4_cuda.csv", row.str());
  return diff <= 0.05f ? 0 : 2;
}
```

- [ ] **Step 2: Create P4 Makefile**

Create `P4/Makefile`:

```makefile
NVCC ?= nvcc
ARCH ?= sm_70
NVCCFLAGS ?= -O2 -std=c++17 -Xptxas -v -arch=$(ARCH)
TARGET := main

.PHONY: all run ptx profile clean

all:
	$(NVCC) $(NVCCFLAGS) main.cu -o $(TARGET)

run: all
	./$(TARGET) 512 512 3 16 16 0
	./$(TARGET) 512 512 3 16 16 1
	./$(TARGET) 1024 1024 5 16 16 0
	./$(TARGET) 1024 1024 5 16 16 1

ptx:
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(ARCH) main.cu -o main.ptx

profile: all
	ncu --set basic ./$(TARGET) 1024 1024 3 16 16 1

clean:
	rm -f $(TARGET) main.ptx
```

- [ ] **Step 3: Build P4**

Run:

```bash
make -C P4 clean
make -C P4 all
```

Expected: PTXAS prints register information and `P4/main` exists.

- [ ] **Step 4: Run P4 experiments**

Run:

```bash
./P4/main 512 512 3 8 8 0
./P4/main 512 512 3 8 8 1
./P4/main 512 512 3 16 16 0
./P4/main 512 512 3 16 16 1
./P4/main 1024 1024 5 16 16 0
./P4/main 1024 1024 5 16 16 1
./P4/main 1024 1024 5 32 8 0
./P4/main 1024 1024 5 32 8 1
```

Expected: every run prints `part=P4 cuda_simt`, `max_abs_diff` below `0.050000`, and `kernel_ms`.

- [ ] **Step 5: Generate PTX**

Run:

```bash
make -C P4 ptx
```

Expected: `P4/main.ptx` exists and includes global load instructions and shared-memory instructions for the shared kernel.

- [ ] **Step 6: Run Nsight Compute basic profile**

Run:

```bash
ncu --set basic ./P4/main 1024 1024 3 16 16 1
```

Expected: `ncu` reports SM utilization, occupancy, active warps, and memory throughput. If profiling permission blocks counters, record the permission error text in `results/p4_ncu_permission.txt`.

- [ ] **Step 7: Commit**

Run:

```bash
git add P4/Makefile P4/main.cu
git commit -m "feat: add cuda simt variants"
```

---

### Task 7: P5 Multi-Pattern GPU Parallelism

**Files:**
- Create: `P5/Makefile`
- Create: `P5/main.cu`

- [ ] **Step 1: Create P5 CUDA implementation**

Create `P5/main.cu` with this structure:

```cpp
#include "../common/bilateral_common.h"
#include "../common/result_io.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while (0)

__device__ int d_clamp_int(int v, int lo, int hi) {
  return max(lo, min(v, hi));
}

__device__ float d_spatial_weight(int dx, int dy, float sigma_s2) {
  const float d2 = static_cast<float>(dx * dx + dy * dy);
  return 1.0f / (1.0f + d2 / sigma_s2);
}

__device__ float d_range_weight(float center, float neighbor, float sigma_r2) {
  const float diff = center - neighbor;
  return 1.0f / (1.0f + (diff * diff) / sigma_r2);
}

__global__ void bilateral_multi_pattern_kernel(const float* input,
                                               float* output,
                                               int width,
                                               int height,
                                               int patterns,
                                               int radius,
                                               float sigma_s2,
                                               float sigma_r2) {
  const int image_size = width * height;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pattern = blockIdx.y;
  if (index >= image_size || pattern >= patterns) {
    return;
  }

  const int x = index % width;
  const int y = index / width;
  const int base = pattern * image_size;
  const float center = input[base + index];
  float numerator = 0.0f;
  float denominator = 0.0f;

  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = d_clamp_int(y + dy, 0, height - 1);
    for (int dx = -radius; dx <= radius; ++dx) {
      const int xx = d_clamp_int(x + dx, 0, width - 1);
      const float neighbor = input[base + yy * width + xx];
      const float weight = d_spatial_weight(dx, dy, sigma_s2) * d_range_weight(center, neighbor, sigma_r2);
      numerator += neighbor * weight;
      denominator += weight;
    }
  }
  output[base + index] = numerator / denominator;
}

static void generate_patterns(std::vector<float>& images, int width, int height, int patterns) {
  images.resize(static_cast<size_t>(width) * height * patterns);
  std::vector<float> one;
  for (int pattern = 0; pattern < patterns; ++pattern) {
    generate_image(one, width, height, pattern);
    std::copy(one.begin(), one.end(), images.begin() + static_cast<size_t>(pattern) * width * height);
  }
}

static void scalar_patterns_reference(const std::vector<float>& input,
                                      std::vector<float>& reference,
                                      const BilateralParams& params,
                                      int patterns) {
  const int image_size = params.width * params.height;
  reference.resize(input.size());
  std::vector<float> one_input(static_cast<size_t>(image_size));
  std::vector<float> one_output;
  for (int pattern = 0; pattern < patterns; ++pattern) {
    std::copy(input.begin() + static_cast<size_t>(pattern) * image_size,
              input.begin() + static_cast<size_t>(pattern + 1) * image_size,
              one_input.begin());
    scalar_bilateral_filter(one_input, one_output, params);
    std::copy(one_output.begin(), one_output.end(),
              reference.begin() + static_cast<size_t>(pattern) * image_size);
  }
}

int main(int argc, char** argv) {
  const int width = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int height = argc > 2 ? std::atoi(argv[2]) : width;
  const int patterns = argc > 3 ? std::atoi(argv[3]) : 4;
  const int threads_per_block = argc > 4 ? std::atoi(argv[4]) : 256;
  BilateralParams params{width, height, 3, 9.0f, 900.0f};

  std::vector<float> input;
  std::vector<float> reference;
  std::vector<float> output;
  generate_patterns(input, width, height, patterns);
  scalar_patterns_reference(input, reference, params, patterns);
  output.resize(input.size());

  const size_t bytes = input.size() * sizeof(float);
  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  dim3 block(threads_per_block);
  dim3 grid((width * height + block.x - 1) / block.x, patterns);

  CUDA_CHECK(cudaEventRecord(start));
  bilateral_multi_pattern_kernel<<<grid, block>>>(d_input, d_output, width, height, patterns,
                                                  params.radius, params.sigma_s2, params.sigma_r2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  const double checksum = checksum_image(output);
  const float diff = max_abs_diff(reference, output);
  std::cout << "part=P5 multi_pattern_cuda\n";
  std::cout << "width=" << width << " height=" << height << " radius=3 patterns=" << patterns
            << " threads_per_block=" << threads_per_block << "\n";
  std::cout << "checksum=" << std::fixed << std::setprecision(6) << checksum << "\n";
  std::cout << "max_abs_diff=" << std::fixed << std::setprecision(6) << diff << "\n";
  std::cout << "kernel_ms=" << std::fixed << std::setprecision(6) << kernel_ms << "\n";
  std::cout << "ms_per_pattern=" << std::fixed << std::setprecision(6) << (kernel_ms / patterns) << "\n";

  ensure_results_dir();
  std::ostringstream row;
  row << "P5," << width << "," << height << ",3," << patterns << "," << threads_per_block
      << "," << checksum << "," << diff << "," << kernel_ms << "," << (kernel_ms / patterns);
  append_csv_line("results/p5_multi_pattern.csv", row.str());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return diff <= 0.05f ? 0 : 2;
}
```

- [ ] **Step 2: Create P5 Makefile**

Create `P5/Makefile`:

```makefile
NVCC ?= nvcc
ARCH ?= sm_70
NVCCFLAGS ?= -O2 -std=c++17 -Xptxas -v -arch=$(ARCH)
TARGET := main

.PHONY: all run ptx profile clean

all:
	$(NVCC) $(NVCCFLAGS) main.cu -o $(TARGET)

run: all
	./$(TARGET) 1024 1024 1 256
	./$(TARGET) 1024 1024 2 256
	./$(TARGET) 1024 1024 4 256
	./$(TARGET) 1024 1024 8 256
	./$(TARGET) 1024 1024 16 256

ptx:
	$(NVCC) --ptx -O2 -std=c++17 -arch=$(ARCH) main.cu -o main.ptx

profile: all
	ncu --set basic ./$(TARGET) 1024 1024 16 256

clean:
	rm -f $(TARGET) main.ptx
```

- [ ] **Step 3: Build P5**

Run:

```bash
make -C P5 clean
make -C P5 all
```

Expected: `P5/main` exists.

- [ ] **Step 4: Run P5 scaling experiments**

Run:

```bash
./P5/main 1024 1024 1 256
./P5/main 1024 1024 2 256
./P5/main 1024 1024 4 256
./P5/main 1024 1024 8 256
./P5/main 1024 1024 16 256
```

Expected: every run prints `part=P5 multi_pattern_cuda`, `max_abs_diff` below `0.050000`, `kernel_ms`, and `ms_per_pattern`.

- [ ] **Step 5: Generate PTX and profile**

Run:

```bash
make -C P5 ptx
ncu --set basic ./P5/main 1024 1024 16 256
```

Expected: `P5/main.ptx` exists. `ncu` reports GPU metrics or prints a permission error that must be saved into `results/p5_ncu_permission.txt`.

- [ ] **Step 6: Commit**

Run:

```bash
git add P5/Makefile P5/main.cu
git commit -m "feat: add multi-pattern cuda version"
```

---

### Task 8: Result Collection Scripts And Submission Readme

**Files:**
- Create: `scripts/collect_cuda_results.sh`
- Create: `Readme.txt`
- Create: `report/Report_outline.md`

- [ ] **Step 1: Create CUDA result collection script**

Create `scripts/collect_cuda_results.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

make -C P4 clean
make -C P4 all
./P4/main 512 512 3 8 8 0
./P4/main 512 512 3 8 8 1
./P4/main 512 512 3 16 16 0
./P4/main 512 512 3 16 16 1
./P4/main 1024 1024 5 16 16 0
./P4/main 1024 1024 5 16 16 1
./P4/main 1024 1024 5 32 8 0
./P4/main 1024 1024 5 32 8 1

make -C P5 clean
make -C P5 all
./P5/main 1024 1024 1 256
./P5/main 1024 1024 2 256
./P5/main 1024 1024 4 256
./P5/main 1024 1024 8 256
./P5/main 1024 1024 16 256

echo "CUDA result CSV files:"
ls -lh results/p4_cuda.csv results/p5_multi_pattern.csv
```

- [ ] **Step 2: Make collection script executable**

Run:

```bash
chmod +x scripts/collect_cuda_results.sh
```

Expected: command exits 0.

- [ ] **Step 3: Create Readme**

Create `Readme.txt`:

```text
Computer Architecture Final Project
Architecture-Aware Bilateral Filter Acceleration

Required parts:
P1: Scalar C++ baseline
P2: RVV vector reduction
P3: SIMD-like RVV parallelization
P4: CUDA SIMT global-memory and shared-memory versions
P5: Multi-pattern CUDA GPU parallelism

Environment:
- gem5/RVV: use course Docker image weisheng505/gem5-rvv-image:v1
- CUDA: use host CUDA or course Docker image weisheng505/cuda-env:v1
- Detected host GPU during planning: Tesla V100-SXM2-32GB
- CUDA architecture flag for V100: sm_70

Check environment:
./scripts/check_environment.sh

P1 host smoke test:
make -C P1 clean
make -C P1 host
./P1/main 32 32 3
./P1/main 64 64 3

P1 gem5 build:
docker start -ai ca-fp-gem5
cd /workspace
make -C P1 clean
make -C P1 riscv

P2 host smoke test:
make -C P2 clean
make -C P2 host
./P2/main 32 32
./P2/main 64 64

P2 gem5 build:
docker start -ai ca-fp-gem5
cd /workspace
make -C P2 clean
make -C P2 riscv

P3 host smoke test:
make -C P3 clean
make -C P3 host
./P3/main 32 32 2
./P3/main 32 32 4
./P3/main 32 32 8

P3 gem5 build:
docker start -ai ca-fp-gem5
cd /workspace
make -C P3 clean
make -C P3 riscv

P4 CUDA:
make -C P4 clean
make -C P4 all
./P4/main 512 512 3 16 16 0
./P4/main 512 512 3 16 16 1

P5 CUDA:
make -C P5 clean
make -C P5 all
./P5/main 1024 1024 1 256
./P5/main 1024 1024 16 256

Collect CUDA results:
./scripts/collect_cuda_results.sh

Generated outputs:
- results/p1_scalar.csv
- results/p2_rvv_reduction.csv
- results/p3_simd_like_rvv.csv
- results/p4_cuda.csv
- results/p5_multi_pattern.csv
- data/*.pgm visualization images

Correctness:
Every executable prints checksum, selected pixels, and max_abs_diff when a reference comparison is available.
```

- [ ] **Step 4: Create report outline**

Create `report/Report_outline.md`:

```markdown
# Architecture-Aware Bilateral Filter Acceleration

## 1. Environment

- CPU: record from `lscpu | grep 'Model name'`
- GPU: Tesla V100-SXM2-32GB observed during planning; confirm with `nvidia-smi`
- CUDA version: CUDA 12.9 observed during planning; confirm with `nvcc --version`
- gem5 Docker image: `weisheng505/gem5-rvv-image:v1`
- CUDA environment: host CUDA or `weisheng505/cuda-env:v1`, whichever is used for the final measurements

## 2. File Structure

`common/` contains deterministic image generation, arithmetic filter helpers, scalar reference code, checksum, max-difference, and result I/O helpers.

`P1/` contains the scalar baseline. `P2/` contains RVV vector reduction. `P3/` contains SIMD-like RVV. `P4/` contains CUDA SIMT global-memory and shared-memory kernels. `P5/` contains multi-pattern CUDA with `grid.y` as pattern index.

`scripts/` contains environment checks and CUDA result collection. `data/` contains generated PGM visualization files. `results/` contains CSV measurements and copied gem5 stats files.

## 3. Algorithm

The grayscale arithmetic bilateral-style filter computes each output pixel from a clamped local window:

```text
spatial = 1 / (1 + (dx*dx + dy*dy) / sigma_s2)
range   = 1 / (1 + (center - neighbor)^2 / sigma_r2)
weight  = spatial * range
num    += neighbor * weight
den    += weight
out     = num / den
```

## 4. Implementation Methods

### P1 Scalar

Nested loops over pixels and window neighbors.

### P2 RVV Reduction

Vector lanes represent window terms for one output pixel. Vector reduction sums numerator and denominator terms.

### P3 SIMD-like RVV

Vector lanes represent different output pixels. Accumulation stays per lane and does not use reduction.

### P4 CUDA SIMT

One thread computes one output pixel. Compare global-memory direct reads with shared-memory tile plus halo.

### P5 Multi-pattern CUDA

`grid.y` maps independent input pattern index. More patterns increase independent GPU work.

## 5. Simulation And Profiling Results

### P1-P3 gem5

| Part | Size | Radius | k | simSeconds | simInsts | numCycles | CPI | overallMissRate |
|------|------|--------|---|------------|----------|-----------|-----|-----------------|

### P4 CUDA

| Variant | Size | Radius | Block | Kernel ms | H2D ms | D2H ms | Max diff |
|---------|------|--------|-------|-----------|--------|--------|----------|

### P5 CUDA

| Patterns | Size | Radius | Block | Kernel ms | ms per pattern | Max diff |
|----------|------|--------|-------|-----------|----------------|----------|

## 6. Discussion And Comparison

- Explain why P2 reduces instruction/cycle count for window summation.
- Explain why P3 does not use reduction and how k affects performance.
- Explain when shared memory helps in P4 and how halo size changes reuse.
- Explain why P5 exposes more GPU scalability and why scaling saturates.
- Keep gem5 simulated time separate from CPU/GPU real runtime.

## 7. Problems And Solutions

- Git LFS pointer file and recovery.
- Docker or container GPU access.
- RVV reduction versus SIMD-like misunderstanding.
- Shared-memory halo boundary validation.
- Nsight Compute profiling permission.

## 8. Conclusion

Summarize the architectural lesson across scalar, vector, SIMD-like, SIMT, and multi-pattern GPU execution.
```

- [ ] **Step 5: Run script syntax checks**

Run:

```bash
bash -n scripts/check_environment.sh
bash -n scripts/collect_cuda_results.sh
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/collect_cuda_results.sh Readme.txt report/Report_outline.md
git commit -m "docs: add run instructions and report outline"
```

---

### Task 9: Final Verification Pass

**Files:**
- Modify: `Readme.txt` if command output differs from documented commands.
- Modify: `report/Report_outline.md` with measured result summaries after experiments.

- [ ] **Step 1: Verify host smoke builds**

Run:

```bash
make -C P1 clean && make -C P1 host && ./P1/main 32 32 3
make -C P2 clean && make -C P2 host && ./P2/main 32 32
make -C P3 clean && make -C P3 host && ./P3/main 32 32 4
```

Expected: all commands exit 0.

- [ ] **Step 2: Verify CUDA builds**

Run:

```bash
make -C P4 clean && make -C P4 all && ./P4/main 512 512 3 16 16 0 && ./P4/main 512 512 3 16 16 1
make -C P5 clean && make -C P5 all && ./P5/main 1024 1024 1 256
```

Expected: all commands exit 0 and print `max_abs_diff` within thresholds.

- [ ] **Step 3: Verify gem5 builds inside container**

Inside `ca-fp-gem5`:

```bash
cd /workspace
make -C P1 clean && make -C P1 riscv
make -C P2 clean && make -C P2 riscv
make -C P3 clean && make -C P3 riscv
```

Expected: all RISC-V binaries build.

- [ ] **Step 4: Verify result files**

Run:

```bash
ls -lh results
head -n 5 results/p4_cuda.csv
head -n 5 results/p5_multi_pattern.csv
```

Expected: result CSV files contain rows from the experiments.

- [ ] **Step 5: Final commit**

Run:

```bash
git add Readme.txt report/Report_outline.md results data
git commit -m "docs: record final experiment results"
```
