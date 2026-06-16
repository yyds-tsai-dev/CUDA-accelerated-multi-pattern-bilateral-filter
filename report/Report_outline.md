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

Docker Engine 29.5.3 was installed on this host, but container execution is blocked by kernel namespace restrictions (`unshare: operation not permitted`). As a fallback, gem5 was built locally from source at `/home/u5977862/gem5/build/RISCV/gem5.opt`. The local gem5 run reports RVV enabled with `VLEN = 256 bits` and `ELEN = 64 bits`.

| Part | gem5 command options | Checksum | max_abs_diff | simSeconds | simInsts | cycles | CPI | D-cache miss rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 scalar | `32 32 3` | 6238116.195794 | N/A | 0.005621 | 3231709 | 11241402 | 3.478470 | 0.009360 |
| P1 scalar | `64 64 3` | 25461956.426892 | N/A | 0.016185 | 9648134 | 32369022 | 3.354952 | 0.004001 |
| P2 RVV reduction | `32 32` | 6238115.943961 | 0.000122 | 0.007834 | 4972814 | 15668938 | 3.150920 | 0.007982 |
| P2 RVV reduction | `64 64` | 25461955.495364 | 0.000122 | 0.025063 | 16594926 | 50125770 | 3.020548 | 0.003477 |
| P3 SIMD-like RVV | `32 32 2` | 6238116.195794 | 0.000000 | 0.005850 | 3440167 | 11699858 | 3.400956 | 0.014585 |
| P3 SIMD-like RVV | `32 32 4` | 6238116.195794 | 0.000000 | 0.005770 | 3338247 | 11540228 | 3.456972 | 0.014675 |
| P3 SIMD-like RVV | `32 32 8` | 6238116.195794 | 0.000000 | 0.005682 | 3262607 | 11364402 | 3.483227 | 0.014732 |
| P3 SIMD-like RVV | `64 64 4` | 25461956.426892 | 0.000000 | 0.015334 | 9160653 | 30668808 | 3.347884 | 0.010826 |

### P4 CUDA

| Width | Height | Radius | Block | Mode | Repeats | Checksum | max_abs_diff | avg_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 3 | 8x8 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.107213 |
| 512 | 512 | 3 | 8x8 | shared | 5 | 1637512415.620656 | 0.000061 | 0.103456 |
| 512 | 512 | 3 | 16x16 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.107546 |
| 512 | 512 | 3 | 16x16 | shared | 5 | 1637512415.620656 | 0.000061 | 0.103776 |
| 1024 | 1024 | 5 | 16x16 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.706176 |
| 1024 | 1024 | 5 | 16x16 | shared | 5 | 6550815804.927626 | 0.000076 | 0.654355 |
| 1024 | 1024 | 5 | 32x8 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.704563 |
| 1024 | 1024 | 5 | 32x8 | shared | 5 | 6550815804.927626 | 0.000076 | 0.656627 |

### P5 CUDA

| Width | Height | Radius | Patterns | Threads/block | Repeats | Checksum | max_abs_diff | avg_kernel_ms | avg_ms_per_pattern | Total kernel ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 3 | 1 | 256 | 5 | 6550831709.315681 | 0.000061 | 0.315187 | 0.315187 | 1.575936 |
| 1024 | 1024 | 3 | 2 | 256 | 5 | 13101845648.158327 | 0.000092 | 0.606144 | 0.303072 | 3.030720 |
| 1024 | 1024 | 3 | 4 | 256 | 5 | 26203780400.305031 | 0.000092 | 1.189171 | 0.297293 | 5.945856 |
| 1024 | 1024 | 3 | 8 | 256 | 5 | 52407367705.564903 | 0.000092 | 2.357427 | 0.294678 | 11.787136 |
| 1024 | 1024 | 3 | 16 | 256 | 5 | 104815167996.359741 | 0.000092 | 4.695072 | 0.293442 | 23.475361 |

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
