#!/usr/bin/env bash
set -euo pipefail

GEM5_BIN="${GEM5_BIN:-/home/u5977862/gem5/build/RISCV/gem5.opt}"
GEM5_CFG="${GEM5_CFG:-/home/u5977862/gem5/configs/deprecated/example/se.py}"
COMMON_ARGS=(--cpu-type=TimingSimpleCPU --caches --l1d_size=32kB --l1i_size=32kB --cacheline_size=64)

mkdir -p results
rm -f results/p1_scalar.csv results/p2_rvv_reduction.csv results/p3_simd_like_rvv.csv
rm -f results/p*_stats.txt results/p*_gem5_stdout.txt results/p123_host_smoke.log results/p123_riscv_build.log

{
  echo "== P1 host =="
  make -C P1 clean
  make -C P1 host
  ./P1/main 32 32 3
  ./P1/main 64 64 3

  echo "== P2 host =="
  make -C P2 clean
  make -C P2 host
  ./P2/main 32 32
  ./P2/main 64 64

  echo "== P3 host =="
  make -C P3 clean
  make -C P3 host
  ./P3/main 32 32 2
  ./P3/main 32 32 4
  ./P3/main 32 32 8
  ./P3/main 64 64 4
} 2>&1 | tee results/p123_host_smoke.log

{
  echo "== P1 RISC-V static =="
  make -C P1 clean
  make -C P1 riscv CXXFLAGS='-std=c++17 -O2 -Wall -Wextra -I.. -static'
  file P1/main

  echo "== P2 RISC-V static RVV =="
  make -C P2 clean
  make -C P2 riscv RISCV_FLAGS='-std=c++17 -O2 -Wall -Wextra -I.. -march=rv64gcv -mabi=lp64d -static'
  file P2/main

  echo "== P3 RISC-V static RVV =="
  make -C P3 clean
  make -C P3 riscv RISCV_FLAGS='-std=c++17 -O2 -Wall -Wextra -I.. -march=rv64gcv -mabi=lp64d -static'
  file P3/main
} 2>&1 | tee results/p123_riscv_build.log

run_gem5() {
  local part="$1"
  local label="$2"
  shift 2
  local opts="$*"

  echo "== ${part} ${label} opts=${opts} =="
  "$GEM5_BIN" --outdir="${part}/m5out" "$GEM5_CFG" \
    --cmd="$(pwd)/${part}/main" --options="$opts" "${COMMON_ARGS[@]}" \
    2>&1 | tee "results/${part,,}_${label}_gem5_stdout.txt"
  cp "${part}/m5out/stats.txt" "results/${part,,}_${label}_stats.txt"
}

run_gem5 P1 32 32 32 3
run_gem5 P1 64 64 64 3
run_gem5 P2 32 32 32
run_gem5 P2 64 64 64
run_gem5 P3 32_k2 32 32 2
run_gem5 P3 32_k4 32 32 4
run_gem5 P3 32_k8 32 32 8
run_gem5 P3 64_k4 64 64 4

echo "gem5 result files:"
ls -lh results/p*_stats.txt results/p*_gem5_stdout.txt
