Part 5 - Multi-pattern GPU Parallelism for Bilateral Filter
===========================================================

This part implements CUDA multi-pattern parallelism. The input image is first
expanded into multiple deterministic image patterns. The CUDA kernel uses a 2D
grid:

  grid.x = output pixel index blocks
  grid.y = pattern index

Therefore, each CUDA thread computes one output pixel for one specific pattern.
This exploits both pixel-level parallelism and pattern-level parallelism.

Default files
-------------
Input : ../test/cyberpunk2077_in.txt
Output: output/multi_cuda_out.txt

Default parameters
------------------
iterations         = 3
patterns           = 32
threads per block  = 256
filter radius      = 4
window size        = 9 x 9 = 81

Build and run in CUDA Docker
----------------------------
From the CUDA container:

  cd /workspace/p5
  make clean
  make ARCH=sm_89
  ./main

Custom run format
-----------------

  ./main <input_txt> <output_txt> [iterations] [patterns] [threads_per_block]

Example:

  ./main ../test/cyberpunk2077_in.txt output/multi_cuda_out.txt 3 32 256

Generate PTX
------------

  make ptx ARCH=sm_89

Run Nsight Compute basic profiling
----------------------------------

  make ncu ARCH=sm_89

Convert output TXT to PNG
-------------------------
From the project root:

  python3 tools/txt_to_png.py p5/output/multi_cuda_out.txt p5/output/multi_cuda_out.png --scale 6

Notes for report
----------------
- Part 5 processes multiple independent input patterns on the GPU.
- The y dimension of the CUDA grid corresponds to the pattern index.
- This usually improves GPU utilization compared with Part 4 because more
  independent threads/warps are launched.
- The program reports CPU multi-pattern time, GPU multi-pattern kernel time,
  speedup, PTXAS register/spill information, and CPU/GPU correctness metrics.
