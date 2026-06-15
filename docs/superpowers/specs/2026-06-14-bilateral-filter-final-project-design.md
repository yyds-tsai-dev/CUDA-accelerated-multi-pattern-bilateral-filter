# CUDA-Accelerated Multi-Pattern Bilateral Filter Final Project Design

## Decision

Adopt the project topic:

`Architecture-Aware Bilateral Filter Acceleration: Scalar, RVV Reduction, RVV SIMD-like, CUDA SIMT, and Multi-pattern GPU Parallelism`

The project will use one bilateral-style grayscale image denoising algorithm across all five required course parts:

- P1: scalar C/C++ baseline on gem5
- P2: RVV vector reduction on gem5
- P3: SIMD-like RVV parallelization on gem5
- P4: CUDA SIMT implementation on GPU
- P5: multi-pattern CUDA GPU parallelism

This replaces the older "CPU plus two CUDA versions" framing. The 2026 course PDF requires the P1-P5 architecture comparison, so the CUDA naive-vs-shared-memory comparison becomes the core of P4 rather than the whole project.

## Course Fit

The topic fits the final project requirements because bilateral filtering is computationally intensive, image-processing related, and naturally parallelizable. For each output pixel, the algorithm scans a local window and accumulates weighted sums, which matches the nested-loop requirement:

```text
sum f(ai[j], bi[j]) for j in the window
```

The original bilateral filter often uses `exp()`. To better match the PDF requirement that `f` use basic arithmetic operations, this project will use an arithmetic-friendly bilateral-style weight:

```text
spatial = 1 / (1 + (dx*dx + dy*dy) / sigma_s2)
range   = 1 / (1 + (center - neighbor)^2 / sigma_r2)
weight  = spatial * range
num    += neighbor * weight
den    += weight
out     = num / den
```

This keeps the important edge-preserving behavior while making P1-P3 feasible in C/C++ and RVV assembly/intrinsics.

## Scope

The implementation will use grayscale images only. RGB filtering is intentionally out of scope because it triples the data path and increases report/debugging risk without adding much architecture value.

The default experiment parameters are:

- image sizes: 32x32 and 64x64 generated arrays for P1-P3, 512x512 and 1024x1024 generated images for P4-P5
- radius: 3 for P1-P3 gem5 runs; radius 3 and 5 for P4 CUDA comparison; radius 3 for P5 multi-pattern scaling
- sigma values: fixed arithmetic constants for reproducibility
- input data: deterministic generated grayscale image plus deterministic noise pattern
- output validation: checksum, selected pixel values, and max absolute difference against scalar reference

The project will not depend on external image libraries for the correctness path. PGM output is a non-required visualization helper and will not be used as the primary validation mechanism.

TA clarification: there is no fixed test-data format, size, or content restriction. The project will use deterministic generated grayscale patterns as the required test data and will document the verification method in `Readme.txt`. If external image files are added later, they must live under `data/` and be described in `Readme.txt` and the report.

## Architecture

Shared code will live under `common/` and define image generation, clamp helpers, arithmetic weight computation, checksum utilities, and result comparison. Each course part will be independently buildable so report data can be collected part by part.

Recommended submission layout:

```text
StudentID1_StudentID2/
  P1/
    Makefile
    main.cpp
  P2/
    Makefile
    main.cpp
  P3/
    Makefile
    main.cpp
  P4/
    Makefile
    main.cu
  P5/
    Makefile
    main.cu
  common/
    bilateral_common.h
    image_data.h
  data/
  results/
  Readme.txt
  Report.pdf
```

In this repository, source work can use the same top-level folders before final packaging.

## Part Design

### P1 Scalar Baseline

P1 implements the reference scalar version in C/C++. It runs the full nested loop for each output pixel and records output checksum plus a small set of selected pixels. It will first be checked with host `g++`, then compiled with the course RISC-V toolchain and executed in gem5.

TA clarification: Makefiles may be changed for extra headers, source files, and compiler flags. gem5 cache size, associativity, and block-size parameters may be adjusted when the report explicitly discusses the comparison. CPU model settings should avoid `AtomicSimpleCPU`.

Report metrics:

- `simSeconds`
- `simInsts`
- `numCycles`
- `CPI`
- `overallMissRate`

### P2 RVV Vector Reduction

P2 vectorizes the per-pixel window summation. For one output pixel, RVV lanes hold neighboring pixel values and weights. Numerator and denominator are accumulated with vector arithmetic and finalized using vector reduction instructions.

This part must use RVV reduction operations. The report will compare instruction and cycle reductions against P1 and explain why reduction helps the window-sum pattern.

### P3 SIMD-Like RVV Parallelization

P3 computes `k` output pixels at the same time. RVV lanes correspond to different output pixels, and each lane accumulates its own scalar-like window sum. This is across-output-pixel SIMD rather than within-one-pixel reduction.

This part must not use vector reduction operations. It will use strided memory access for across-pixel lanes and test `k` values 2, 4, and 8. The report will explain the difference between P2's within-window reduction and P3's across-pixel parallelism. TA clarification confirms this interpretation: Part 3 places `k` independent loop iterations into one vector, so reduction is not applicable.

### P4 CUDA SIMT

P4 implements CUDA kernels with one thread computing one output pixel.

Two CUDA variants will be implemented:

- global-memory naive kernel: each thread reads its window directly from global memory
- shared-memory tiled kernel: each block loads a tile plus halo into shared memory, synchronizes, then computes outputs from shared memory

The shared-memory comparison directly supports the PDF prompts about memory access patterns, shared memory effectiveness, block size, occupancy, and latency hiding.

Report metrics:

- CUDA event kernel time from `cuda_runtime.h`
- H2D/D2H timing for GPU total-runtime measurements
- PTXAS registers per thread
- shared memory per block
- `ncu --set basic` metrics when profiling permission allows it

GPU timing will use CUDA events and event synchronization. If `chrono` is used for a CPU baseline or an end-to-end host measurement, the timed region must synchronize the CUDA work before reading the stop time. Each CUDA setting will be measured across five runs and reported as an average.

On the current host, the detected GPU is Tesla V100-SXM2-32GB, so the expected CUDA target is `-arch=sm_70`.

### P5 Multi-Pattern GPU Parallelism

P5 processes multiple independent input patterns in one CUDA launch. The x dimension maps output pixels; the y dimension maps pattern index:

```text
grid.x = output tile or output index space
grid.y = number of patterns
```

Patterns will be deterministic noisy variants of the same base image. Batch sizes will include 1, 2, 4, 8, and 16. The report will analyze whether throughput scales linearly and how more patterns affect occupancy, latency hiding, and utilization.

P5 timing follows the same CUDA timing policy as P4: use CUDA events from `cuda_runtime.h`, synchronize recorded events before reading elapsed time, and report five-run averages.

## Environment Plan

Current host findings:

- GPU available: 2 x Tesla V100-SXM2-32GB
- CUDA available: 12.9
- `nvcc` available
- `ncu` available
- Docker not currently installed
- gem5/RISC-V toolchain not currently installed on host
- Git LFS not currently installed
- `final_project-20260522T170812Z-3-001.zip` is currently a Git LFS pointer, not the real zip

Required setup:

```bash
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
```

Docker setup follows `docs/環境Setup.pdf` and the official Docker Linux guidance. The course images should be used for gem5 and may also be used for CUDA consistency:

```bash
docker run -it --name ca-fp-gem5 \
  -v "$PWD":/workspace -w /workspace \
  weisheng505/gem5-rvv-image:v1

docker run -it --gpus all --name ca-fp-cuda \
  -v "$PWD":/workspace -w /workspace \
  weisheng505/cuda-env:v1
```

If Docker GPU access fails, install and configure NVIDIA Container Toolkit, then retry the CUDA container.

## Data Flow

1. Generate deterministic grayscale input.
2. Generate deterministic noisy variants for P5.
3. Run scalar reference and save checksum/selected output pixels.
4. Run each optimized part.
5. Compare each result against scalar reference.
6. Record timing/profiling/statistics into `results/`.
7. Use results tables in `Report.pdf`; include PGM visual outputs only as supporting figures.

## Correctness

Every part must output a value that prevents compiler dead-code elimination. The minimum correctness contract is:

- output checksum is printed
- selected pixel outputs are printed
- max absolute difference versus scalar reference is reported
- acceptable max absolute difference is at most 1 for integer output paths or a small float tolerance before final rounding

Boundary handling will use clamp-to-edge consistently across all parts.

## Performance Experiments

P1-P3 gem5 experiments:

- image sizes 32x32 and 64x64
- radius 3
- compare scalar, RVV reduction, and SIMD-like RVV
- report simulated time separately from real CPU/GPU runtime
- keep the course CPU model unless a report section explicitly studies cache settings; do not switch to `AtomicSimpleCPU`

P4 CUDA experiments:

- image sizes 512x512 and 1024x1024
- radius 3 and 5
- block sizes such as 8x8, 16x16, and 32x8
- compare global-memory and shared-memory kernels
- run each setting five times and report the average CUDA event kernel time

P5 CUDA experiments:

- fixed 1024x1024 image size and radius 3
- pattern counts 1, 2, 4, 8, and 16
- compare throughput per pattern and total runtime
- run each setting five times and report the average CUDA event kernel time

## Error Handling And Risks

Docker may be missing or unavailable. If so, install Docker first; gem5 work depends on the course Docker image unless the RISC-V toolchain is separately installed.

Git LFS may be missing. If the starter package remains a pointer file, install Git LFS and pull the real archive before assuming the project has no starter code.

RVV implementation is the highest-risk part. The mitigation is to keep input sizes small, use a simple arithmetic kernel, and keep scalar reference outputs identical across P1-P3.

Shared-memory halo loading can introduce boundary bugs. The mitigation is to validate edge pixels and use the same clamp-to-edge function in CPU and CUDA paths.

Nsight Compute profiling may be blocked by permissions. If `ncu` cannot collect all metrics, CUDA event timing and PTXAS output remain the required fallback evidence.

## Testing

The test strategy is lightweight and direct:

- host scalar smoke test with `g++`
- P1 gem5 run produces `m5out/stats.txt`
- P2/P3 gem5 runs produce comparable stats and matching checksums
- P4/P5 CUDA runs produce matching checksums and timing output
- Makefile targets build cleanly from a fresh checkout or container
- `Readme.txt` includes exact build and run commands for each part

## Report Strategy

The report will follow the course PDF suggested structure:

1. Environment
2. File structure
3. Algorithm
4. Implementation methods
5. Simulation and profiling results
6. Discussion and comparison
7. Conclusion

The main analysis claims will be:

- P2 improves the within-window summation using vector reduction.
- P3 improves across-pixel throughput without reduction, but larger `k` may not always help due to memory access and vector length constraints.
- P4 shared memory helps when neighboring threads reuse overlapping windows, especially for larger radius, but block size and shared-memory footprint affect occupancy.
- P5 better exposes GPU scalability by increasing independent work and improving latency hiding, but speedup will eventually saturate.

## Approval State

The user approved this direction on 2026-06-14 with "採用". Implementation should begin only after this spec is reviewed and an implementation plan is written.
