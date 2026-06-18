P4 Bilateral Filter CUDA SIMT
=============================

Default input:
    ../test/cyberpunk2077_in.txt

Default output:
    output/cuda_out.txt

Build inside CUDA Docker:
    cd /workspace/p4
    make clean
    make CUDA_ARCH=sm_89

Run:
    ./main

Or use Makefile run targets:
    make run-gpu CUDA_ARCH=sm_89
    make run-cpu

Custom run:
    ./main ../test/cyberpunk2077_in.txt output/cuda_out.txt 3 256 shared
    ./main ../test/cyberpunk2077_in.txt output/cuda_out_global.txt 3 256 global

Arguments:
    ./main <input_txt> <output_txt> [iterations] [threads_per_block] [shared|global]

Memory modes:
    shared : row-tile shared memory with halo rows, default
    global : original global-memory-only kernel

Convert output txt to PNG from project root:
    python3 tools/txt_to_png.py p4/output/cuda_out.txt p4/output/cuda_out.png --scale 6
