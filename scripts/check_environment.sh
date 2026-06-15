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
