# Experiment Results 2026-06-16

## Environment

- Host: Linux `q9evckeehemt-jccn8`, kernel `4.18.0-305.131.1.el8_4.x86_64`
- CUDA: `nvcc` build `cuda_12.9.r12.9/compiler.35813241_0`
- GPU: 2 x Tesla V100-SXM2-32GB
- Nsight Compute: 2025.2.0.0
- RISC-V compiler: `riscv64-linux-gnu-g++` 13.3.0
- gem5: `/home/u5977862/gem5/build/RISCV/gem5.opt`, version 25.1.0.1
- gem5 mode: `TimingSimpleCPU`, classic caches, `l1d_size=32kB`, `l1i_size=32kB`, `cacheline_size=64`
- gem5 RVV: `VLEN = 256 bits`, `ELEN = 64 bits`
- Docker: Engine 29.5.3 installed, but container execution is blocked by `unshare: operation not permitted`

## Commands

```sh
./scripts/check_environment.sh
./scripts/collect_gem5_results.sh
./scripts/collect_cuda_results.sh
make -C P4 ptx
make -C P5 ptx
ncu --set basic ./P4/main 1024 1024 3 16 16 1 5
ncu --set basic ./P5/main 1024 1024 16 256 5
```

## P1-P3 Host Smoke

| Part | Size | k | Checksum | max_abs_diff | chrono ms |
| --- | --- | --- | --- | --- | --- |
| P1 scalar | 32x32 | N/A | 6238116.209287 | N/A | 0.176672 |
| P1 scalar | 64x64 | N/A | 25461956.465892 | N/A | 2.181829 |
| P2 RVV reduction host fallback | 32x32 | N/A | 6238116.209287 | 0.000000 | 0.274514 |
| P2 RVV reduction host fallback | 64x64 | N/A | 25461956.465892 | 0.000000 | 1.121965 |
| P3 SIMD-like host fallback | 32x32 | 2 | 6238116.209287 | 0.000000 | 0.546490 |
| P3 SIMD-like host fallback | 32x32 | 4 | 6238116.209287 | 0.000000 | 0.178255 |
| P3 SIMD-like host fallback | 32x32 | 8 | 6238116.209287 | 0.000000 | 0.246257 |
| P3 SIMD-like host fallback | 64x64 | 4 | 25461956.465892 | 0.000000 | 2.202352 |

## P1-P3 gem5

| Part | gem5 options | Checksum | max_abs_diff | simSeconds | simInsts | cycles | CPI | D-cache miss rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 scalar | `32 32 3` | 6238116.195794 | N/A | 0.005621 | 3231709 | 11241402 | 3.478470 | 0.009360 |
| P1 scalar | `64 64 3` | 25461956.426892 | N/A | 0.016185 | 9648134 | 32369022 | 3.354952 | 0.004001 |
| P2 RVV reduction | `32 32` | 6238115.943961 | 0.000122 | 0.007834 | 4972814 | 15668938 | 3.150920 | 0.007982 |
| P2 RVV reduction | `64 64` | 25461955.495364 | 0.000122 | 0.025063 | 16594926 | 50125770 | 3.020548 | 0.003477 |
| P3 SIMD-like RVV | `32 32 2` | 6238116.195794 | 0.000000 | 0.005850 | 3440167 | 11699858 | 3.400956 | 0.014585 |
| P3 SIMD-like RVV | `32 32 4` | 6238116.195794 | 0.000000 | 0.005770 | 3338247 | 11540228 | 3.456972 | 0.014675 |
| P3 SIMD-like RVV | `32 32 8` | 6238116.195794 | 0.000000 | 0.005682 | 3262607 | 11364402 | 3.483227 | 0.014732 |
| P3 SIMD-like RVV | `64 64 4` | 25461956.426892 | 0.000000 | 0.015334 | 9160653 | 30668808 | 3.347884 | 0.010826 |

## P4 CUDA

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

## P5 CUDA

| Width | Height | Radius | Patterns | Threads/block | Repeats | Checksum | max_abs_diff | avg_kernel_ms | avg_ms_per_pattern | total_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 3 | 1 | 256 | 5 | 6550831709.315681 | 0.000061 | 0.315187 | 0.315187 | 1.575936 |
| 1024 | 1024 | 3 | 2 | 256 | 5 | 13101845648.158327 | 0.000092 | 0.606144 | 0.303072 | 3.030720 |
| 1024 | 1024 | 3 | 4 | 256 | 5 | 26203780400.305031 | 0.000092 | 1.189171 | 0.297293 | 5.945856 |
| 1024 | 1024 | 3 | 8 | 256 | 5 | 52407367705.564903 | 0.000092 | 2.357427 | 0.294678 | 11.787136 |
| 1024 | 1024 | 3 | 16 | 256 | 5 | 104815167996.359741 | 0.000092 | 4.695072 | 0.293442 | 23.475361 |

## Nsight Compute

| Target | Theoretical occupancy | Achieved occupancy | Achieved active warps/SM | Notes |
| --- | --- | --- | --- | --- |
| P4 shared, 1024x1024, radius 3, 16x16 | 75% | about 70.4% | about 45.1 | Occupancy limited by register count. |
| P5 patterns=16, 1024x1024, radius 3 | 50% | about 48.5% | about 31.0 | Occupancy limited by register count. |

## Generated Evidence

- `results/p123_host_smoke.log`
- `results/p123_riscv_build.log`
- `results/p*_gem5_stdout.txt`
- `results/p*_stats.txt`
- `results/cuda_collection.log`
- `results/p4_cuda.csv`
- `results/p5_multi_pattern.csv`
- `results/ptx_generation.log`
- `results/p4_ncu_basic.txt`
- `results/p5_ncu_basic.txt`
