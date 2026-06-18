P4 Bilateral Filter CUDA SIMT
=============================

Default input:
    ../test/cyberpunk2077_in.txt

Default output:
    output/cuda_out.txt

Build inside CUDA Docker:
    cd /workspace/p4
    make clean
    make ARCH=sm_89

Run:
    ./main

Or:
    make run ARCH=sm_89

Custom run:
    ./main ../test/cyberpunk2077_in.txt output/cuda_out.txt 3 256

Arguments:
    ./main <input_txt> <output_txt> [iterations] [threads_per_block]

Convert output txt to PNG from project root:
    python3 tools/txt_to_png.py p4/output/cuda_out.txt p4/output/cuda_out.png --scale 6

Generate PTX:
    make ptx ARCH=sm_89

Run Nsight Compute:
    make ncu ARCH=sm_89
