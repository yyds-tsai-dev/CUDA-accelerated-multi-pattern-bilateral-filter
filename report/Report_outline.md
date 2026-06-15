# Architecture-Aware Bilateral Filter Acceleration

## Environment

- Course gem5/RVV Docker image: `weisheng505/gem5-rvv-image:v1`
- CUDA environment: host CUDA installation or `weisheng505/cuda-env:v1`
- Observed CUDA GPU: Tesla V100-SXM2-32GB
- CUDA architecture flag for V100: `sm_70`
- Environment check: `./scripts/check_environment.sh`

## File Structure

| Path | Purpose |
| --- | --- |
| `common/bilateral_common.h` | Shared deterministic input generation, scalar reference helpers, checksum, diff, and PGM output utilities. |
| `common/result_io.h` | CSV append helper for generated result files. |
| `P1/main.cpp` | Scalar CPU baseline implementation. |
| `P2/main.cpp` | RVV reduction implementation. |
| `P3/main.cpp` | SIMD-like RVV implementation. |
| `P4/main.cu` | CUDA SIMT implementation with global-memory and shared-memory modes. |
| `P5/main.cu` | Multi-pattern CUDA implementation. |
| `scripts/check_environment.sh` | Host environment inspection script. |
| `scripts/collect_cuda_results.sh` | Standard CUDA result collection script. |
| `results/` | Generated CSV result files. |
| `data/` | Generated grayscale PGM test output. |

## Algorithm

Arithmetic bilateral-style filtering computes each output pixel from a weighted neighborhood:

```text
spatial = 1 / (1 + (dx*dx + dy*dy) / sigma_s2)
range   = 1 / (1 + (center - neighbor)^2 / sigma_r2)
weight  = spatial * range
num    += neighbor * weight
den    += weight
out     = num / den
```

The arithmetic form avoids `exp()` in RVV/gem5 while keeping the same architectural structure: spatial weighting, range weighting, and per-pixel weighted summation.

## Implementation Methods

### P1 Scalar

- Straight scalar C++ baseline.
- Generates deterministic grayscale input and writes `results/p1_scalar.csv`.
- Used as correctness reference for later parts.

### P2 RVV Reduction

- Uses RVV-style reduction across the filter neighborhood.
- Compares output against the scalar reference with `max_abs_diff`.
- Writes `results/p2_rvv_reduction.csv`.

### P3 SIMD-like RVV

- Processes multiple output pixels in a vector/SIMD-like style.
- Sweeps the vector grouping parameter `k`.
- Compares output against the scalar reference with `max_abs_diff`.
- Writes `results/p3_simd_like_rvv.csv`.

### P4 CUDA SIMT

- Maps output pixels to CUDA threads.
- Compares global-memory and shared-memory modes.
- Sweeps block dimensions such as `8x8`, `16x16`, and `32x8`.
- Uses CUDA events for kernel timing.
- Writes `results/p4_cuda.csv`.

### P5 Multi-pattern CUDA

- Evaluates multiple bilateral patterns in a single CUDA-oriented workflow.
- Sweeps pattern counts such as `1`, `2`, `4`, `8`, and `16`.
- Uses CUDA events for kernel timing.
- Writes `results/p5_multi_pattern.csv`.
- Records `avg_kernel_ms`, `avg_ms_per_pattern`, and final-field `total_kernel_ms`.

## Simulation And Profiling Results

### P1-P3 gem5

Final gem5/RVV runs must be completed inside the course container because this host did not have `docker` or `riscv64-linux-gnu-g++` available during final verification.

| Part | Host smoke command | Checksum | max_abs_diff | Host timing field | gem5 status |
| --- | --- | --- | --- | --- | --- |
| P1 scalar | `./P1/main 32 32 3` | 6238116.209287 | N/A | `host_ms=0.241593` | Blocked on missing Docker/RISC-V toolchain. |
| P2 RVV reduction fallback | `./P2/main 32 32` | 6238116.209287 | 0.000000 | `host_fallback_ms=0.274421` | Blocked on missing Docker/RISC-V toolchain. |
| P3 SIMD-like RVV fallback | `./P3/main 32 32 4` | 6238116.209287 | 0.000000 | `host_fallback_ms=0.178394` | Blocked on missing Docker/RISC-V toolchain. |

### P4 CUDA

| Width | Height | Radius | Block | Mode | Repeats | Checksum | max_abs_diff | avg_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 3 | 8x8 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.115885 |
| 512 | 512 | 3 | 8x8 | shared | 5 | 1637512415.620656 | 0.000061 | 0.109459 |
| 512 | 512 | 3 | 16x16 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.112691 |
| 512 | 512 | 3 | 16x16 | shared | 5 | 1637512415.620656 | 0.000061 | 0.107827 |
| 1024 | 1024 | 5 | 16x16 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.774765 |
| 1024 | 1024 | 5 | 16x16 | shared | 5 | 6550815804.927626 | 0.000076 | 0.721024 |
| 1024 | 1024 | 5 | 32x8 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.776499 |
| 1024 | 1024 | 5 | 32x8 | shared | 5 | 6550815804.927626 | 0.000076 | 0.720166 |

### P5 CUDA

| Width | Height | Radius | Patterns | Threads/block | Repeats | Checksum | max_abs_diff | avg_kernel_ms | avg_ms_per_pattern | Total kernel ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 3 | 1 | 256 | 5 | 6550831709.315681 | 0.000061 | 0.350208 | 0.350208 | 1.751040 |
| 1024 | 1024 | 3 | 2 | 256 | 5 | 13101845648.158327 | 0.000092 | 0.666458 | 0.333229 | 3.332288 |
| 1024 | 1024 | 3 | 4 | 256 | 5 | 26203780400.305031 | 0.000092 | 1.309933 | 0.327483 | 6.549664 |
| 1024 | 1024 | 3 | 8 | 256 | 5 | 52407367705.564903 | 0.000092 | 2.614022 | 0.326753 | 13.070112 |
| 1024 | 1024 | 3 | 16 | 256 | 5 | 104815167996.359741 | 0.000092 | 5.180909 | 0.323807 | 25.904545 |

## Discussion And Comparison

- Compare P1 scalar, P2 RVV reduction, and P3 SIMD-like RVV under gem5 simulated time.
- Discuss how vector reductions and SIMD-like grouping change arithmetic intensity and memory access behavior.
- Compare P4 global-memory and shared-memory CUDA modes across block shapes.
- Discuss P5 scaling with increasing pattern count using both `avg_kernel_ms` and `avg_ms_per_pattern`; include `total_kernel_ms` to show the full repeated-kernel measurement used to derive averages.
- Keep gem5 simulated timing separate from host `chrono` timing and CUDA event timing.

## Problems And Solutions

- Tool availability: RVV/gem5 tools may be unavailable on a bare host, so use the course Docker image.
- CUDA portability: V100 runs should use `ARCH=sm_70`; other GPUs may need a different architecture flag.
- Correctness validation: use deterministic input, checksum, selected pixel values, and `max_abs_diff` when a reference exists.
- Timing interpretation: distinguish host smoke timing, gem5 simulated time, and CUDA event timing.

## Conclusion

- Summarize the architectural tradeoffs from scalar CPU, RVV vectorization, CUDA SIMT, and multi-pattern CUDA batching.
- Highlight which implementation provides the best speedup for the tested image sizes and parameters.
- State correctness evidence and any remaining environment or profiling limitations.
