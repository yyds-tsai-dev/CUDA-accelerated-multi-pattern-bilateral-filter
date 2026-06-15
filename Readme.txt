Computer Architecture Final Project / Architecture-Aware Bilateral Filter Acceleration

Overview
This project implements and compares architecture-aware bilateral filter acceleration across CPU, RVV, and CUDA targets. The required parts are:

- P1: scalar CPU bilateral filter baseline.
- P2: RVV reduction version for vectorized accumulation.
- P3: SIMD-like RVV version that processes multiple output pixels in a vector style.
- P4: CUDA SIMT implementation with global-memory and shared-memory modes.
- P5: multi-pattern CUDA implementation for batching several bilateral patterns in one run.

Environment
- gem5/RVV: use the course Docker image `weisheng505/gem5-rvv-image:v1`.
- CUDA: use a host CUDA installation or the course CUDA Docker image `weisheng505/cuda-env:v1`.
- Observed GPU for CUDA runs: Tesla V100-SXM2-32GB.
- V100 CUDA architecture flag: `sm_70`. The P4/P5 Makefiles default to `ARCH=sm_70`.
- Bare hosts may not have Docker, RVV tools, or CUDA tools installed. RVV/gem5 work should be run inside the course gem5 container when those tools are unavailable locally.

Check the local environment:

```sh
./scripts/check_environment.sh
```

Host Smoke Runs
These commands build and run small native host checks for P1/P2/P3:

```sh
make -C P1 clean
make -C P1 all
./P1/main 64 64 3

make -C P2 clean
make -C P2 all
./P2/main 64 64

make -C P3 clean
make -C P3 all
./P3/main 64 64 4
```

gem5/RVV Builds
Start the course gem5 container, enter the workspace, and build the RISC-V binaries:

```sh
docker start -ai ca-fp-gem5
cd /workspace
make -C P1 riscv
make -C P2 riscv
make -C P3 riscv
```

CUDA Builds And Runs
Build and run representative P4 CUDA cases:

```sh
make -C P4 clean
make -C P4 all
./P4/main 512 512 3 16 16 0 5
./P4/main 512 512 3 16 16 1 5
```

Build and run representative P5 multi-pattern CUDA cases:

```sh
make -C P5 clean
make -C P5 all
./P5/main 1024 1024 4 256 5
./P5/main 1024 1024 16 256 5
```

Collect the standard CUDA result set:

```sh
./scripts/collect_cuda_results.sh
```

Generated Outputs
The programs create generated artifacts under:

- `results/p1_scalar.csv`
- `results/p2_rvv_reduction.csv`
- `results/p3_simd_like_rvv.csv`
- `results/p4_cuda.csv`
- `results/p5_multi_pattern.csv`
- `data/p1_scalar_output.pgm`

The test images are deterministic generated grayscale data, so the same dimensions and parameters produce repeatable inputs without requiring external image files.

Correctness And Timing Notes
- Correctness is reported with checksum values, selected pixel samples, and `max_abs_diff` when a scalar reference exists.
- P4/P5 CUDA timing uses CUDA events for GPU kernel measurements.
- P1/P2/P3 host smoke timing uses C++ `chrono`.
- gem5 simulated time is separate from real CPU/GPU runtime and should be reported separately from host or CUDA timings.
