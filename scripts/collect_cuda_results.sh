#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

make -C P4 clean
make -C P4 all
./P4/main 512 512 3 8 8 0 5
./P4/main 512 512 3 8 8 1 5
./P4/main 512 512 3 16 16 0 5
./P4/main 512 512 3 16 16 1 5
./P4/main 1024 1024 5 16 16 0 5
./P4/main 1024 1024 5 16 16 1 5
./P4/main 1024 1024 5 32 8 0 5
./P4/main 1024 1024 5 32 8 1 5

make -C P5 clean
make -C P5 all
./P5/main 1024 1024 1 256 5
./P5/main 1024 1024 2 256 5
./P5/main 1024 1024 4 256 5
./P5/main 1024 1024 8 256 5
./P5/main 1024 1024 16 256 5

echo "CUDA result CSV files:"
ls -lh results/p4_cuda.csv results/p5_multi_pattern.csv
