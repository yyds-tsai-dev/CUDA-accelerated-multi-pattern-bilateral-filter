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

Arithmetic bilateral filtering computes each output pixel from a weighted neighborhood:

```text
I_out(p) =
  sum_{q in N(p)} I(q) * exp(-||p - q||^2 / (2 * sigma_s^2))
                       * exp(-(I(p) - I(q))^2 / (2 * sigma_r^2))
  ----------------------------------------------------------------
  sum_{q in N(p)}      exp(-||p - q||^2 / (2 * sigma_s^2))
                       * exp(-(I(p) - I(q))^2 / (2 * sigma_r^2))
```

The spatial term preserves locality, while the range term preserves edges by reducing the weight of neighboring pixels with very different intensities.

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

| Part | Image | Parameters | Simulated time | Simulated ticks | Notes |
| --- | --- | --- | --- | --- | --- |
| P1 scalar | TBD | radius TBD | TBD | TBD | Baseline scalar result. |
| P2 RVV reduction | TBD | radius TBD | TBD | TBD | Compare speedup and `max_abs_diff` against P1. |
| P3 SIMD-like RVV | TBD | radius TBD, k TBD | TBD | TBD | Compare vector grouping effect and `max_abs_diff` against P1. |

### P4 CUDA

| Width | Height | Radius | Block | Mode | Repeats | Checksum | max_abs_diff | avg_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 3 | 8x8 | global | 5 | TBD | TBD | TBD |
| 512 | 512 | 3 | 8x8 | shared | 5 | TBD | TBD | TBD |
| 512 | 512 | 3 | 16x16 | global | 5 | TBD | TBD | TBD |
| 512 | 512 | 3 | 16x16 | shared | 5 | TBD | TBD | TBD |
| 1024 | 1024 | 5 | 16x16 | global | 5 | TBD | TBD | TBD |
| 1024 | 1024 | 5 | 16x16 | shared | 5 | TBD | TBD | TBD |
| 1024 | 1024 | 5 | 32x8 | global | 5 | TBD | TBD | TBD |
| 1024 | 1024 | 5 | 32x8 | shared | 5 | TBD | TBD | TBD |

### P5 CUDA

| Width | Height | Radius | Patterns | Threads/block | Repeats | Checksum | max_abs_diff | avg_kernel_ms | avg_ms_per_pattern | Total kernel ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 3 | 1 | 256 | 5 | TBD | TBD | TBD | TBD | TBD |
| 1024 | 1024 | 3 | 2 | 256 | 5 | TBD | TBD | TBD | TBD | TBD |
| 1024 | 1024 | 3 | 4 | 256 | 5 | TBD | TBD | TBD | TBD | TBD |
| 1024 | 1024 | 3 | 8 | 256 | 5 | TBD | TBD | TBD | TBD | TBD |
| 1024 | 1024 | 3 | 16 | 256 | 5 | TBD | TBD | TBD | TBD | TBD |

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
