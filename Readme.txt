CA Final Project - Bilateral Filter (P1~P5)
===========================================

Directory layout
----------------
314510169/
├── tools/
│   ├── preprocess_image_to_txt.py    # PNG -> grayscale input txt + input preview PNG
│   └── txt_to_png.py                 # output txt -> PNG directly
├── raw/
│   └── cyberpunk2077.png             # original source image used for preprocessing
├── test/
│   ├── cyberpunk2077_in.txt          # common input for P1~P5
│   └── cyberpunk2077_in.png          # preview image for report
├── p1/                               # scalar baseline, gem5
├── p2/                               # RVV vector reduction, gem5
├── p3/                               # SIMD-like RVV across-k, gem5
├── p4/                               # CUDA SIMT single-pattern implementation
└── p5/                               # CUDA multi-pattern 2D-grid implementation

All parts use the same input data by default:
    ../test/cyberpunk2077_in.txt

Each part writes its own output into its local output/ directory.


1. Preprocessing: PNG image -> input txt + preview PNG
------------------------------------------------------
Run this command from the project root directory:

General form:
    python3 tools/preprocess_image_to_txt.py <input_png> --out-dir test --name <name> --width <resize_width>

Example:
    python3 tools/preprocess_image_to_txt.py raw/cyberpunk2077.png --out-dir test --name cyberpunk2077 --width 160

This creates:
    test/cyberpunk2077_in.txt
    test/cyberpunk2077_in.png

The txt file is used by P1~P5.
The png file is only for preview/report.

If you want a larger test image, for example width 320:
    python3 tools/preprocess_image_to_txt.py raw/cyberpunk2077.png --out-dir test --name cyberpunk2077_320 --width 320

Then run the programs with the generated txt file as a custom input argument.

Note:
    preprocess_image_to_txt.py does not require Pillow.
    It supports PNG input only. If your source is JPG, convert it to PNG first.


2. Convert output txt -> PNG directly
-------------------------------------
Run from the project root directory.

General form:
    python3 tools/txt_to_png.py <output_txt> <output_png> --scale <scale_factor>

Examples:
    python3 tools/txt_to_png.py p1/output/scalar_out.txt p1/output/scalar_out.png --scale 6
    python3 tools/txt_to_png.py p2/output/rvv_out.txt p2/output/rvv_out.png --scale 6
    python3 tools/txt_to_png.py p3/output/simd_rvv_out.txt p3/output/simd_rvv_out.png --scale 6
    python3 tools/txt_to_png.py p4/output/cuda_out.txt p4/output/cuda_out.png --scale 6
    python3 tools/txt_to_png.py p5/output/multi_cuda_out.txt p5/output/multi_cuda_out.png --scale 6

The --scale option is only for viewing. It does not affect computation.


3. P1/P2/P3 gem5 execution
--------------------------
Use the gem5 Docker container.

If the container already exists:
    docker start -ai ca-fa-pro

If creating a new one from the project root directory on NVL4:
    docker run -it --name ca-fa-pro -v ${cwd}:/workspace -w /workspace weisheng505/gem5-rvv-image:v1

Inside the container:

P1:
    cd /workspace/p1
    make clean
    make

P2:
    cd /workspace/p2
    make clean
    make

P3:
    cd /workspace/p3
    make clean
    make

Default execution through the Makefile needs no program arguments. Each main.cpp also supports:
    ./main <input_txt> <output_txt> [iterations]

The P1/P2/P3 Makefiles run the default gem5 command and do not forward an ARGS variable.
For custom program arguments, invoke gem5 manually with the same options as the
Makefile run target and add --options, for example:
    /root/gem5/build/RISCV/gem5.opt /root/gem5/configs/deprecated/example/se.py \
        -c /workspace/p1/main \
        --options="../test/cyberpunk2077_320_in.txt output/scalar_320_out.txt 3" \
        --cpu-type=TimingSimpleCPU --mem-size=256MB --caches \
        --l1i_size=32kB --l1i_assoc=8 --l1d_size=32kB --l1d_assoc=8 --cacheline=32

For official/default runs, simply use:
    make clean
    make


4. P4/P5 CUDA execution
-----------------------
Use the CUDA Docker container when the GPU is available.

From the project root directory on NVL4, create the CUDA container:
    docker run -it --gpus all --name ca-fa-cuda -v ${cwd}:/workspace -w /workspace weisheng505/cuda-env:v1

If the container already exists:
    docker start -ai ca-fa-cuda

Inside the CUDA container:
    cd /workspace/p4
    make clean
    make CUDA_ARCH=sm_89
    ./main

Or use the Makefile run targets:
    make run-gpu CUDA_ARCH=sm_89
    make run-cpu

P4 custom execution:
    ./main ../test/cyberpunk2077_in.txt output/cuda_out.txt 3 256
    ./main ../test/cyberpunk2077_in.txt output/cuda_out_global.txt 3 256 global

Arguments:
    ./main <input_txt> <output_txt> [iterations] [threads_per_block] [shared|global]

P5 build and default run:
    cd /workspace/p5
    make clean
    make CUDA_ARCH=sm_89
    ./main

P5 custom execution:
    ./main ../test/cyberpunk2077_in.txt output/multi_cuda_out.txt 3 32 256 shared
    ./main ../test/cyberpunk2077_in.txt output/multi_cuda_out_global.txt 3 32 256 global

Arguments:
    ./main <input_txt> <output_txt> [iterations] [patterns] [threads_per_block] [shared|global]

Memory modes:
    shared : row-tile shared memory with halo rows, default
    global : global-memory-only kernel


5. Notes
--------
- Preprocessing is not included in the measured kernel runtime.
- P1~P3 report gem5 simulated statistics from m5out/stats.txt.
- P4 and P5 report CPU reference runtime and CUDA kernel runtime.
- gem5 simulated time and actual host/GPU runtime are different concepts.
