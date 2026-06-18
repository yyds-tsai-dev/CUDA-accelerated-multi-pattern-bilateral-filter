CA Final Project - Bilateral Filter (P1~P4)
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
│   ├── cyberpunk2077_in.txt          # common input for P1~P4
│   └── cyberpunk2077_in.png          # preview image for report
├── p1/                               # scalar baseline, gem5
├── p2/                               # RVV vector reduction, gem5
├── p3/                               # SIMD-like RVV across-k, gem5
├── p4/                               # CUDA SIMT implementation
└── p5/                               # reserved for Part 5

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

The txt file is used by P1~P4.
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

Default execution needs no arguments. Each main.cpp also supports:
    ./main <input_txt> <output_txt> [iterations]

For gem5 custom arguments, each Makefile supports ARGS:
    make clean
    make ARGS="../test/cyberpunk2077_320_in.txt output/scalar_320_out.txt 3"

For official/default runs, simply use:
    make clean
    make


4. P4 CUDA execution
--------------------
Use the CUDA Docker container when the GPU is available.

From the project root directory on NVL4, create the CUDA container:
    docker run -it --gpus all --name ca-fa-cuda -v ${cwd}:/workspace -w /workspace weisheng505/cuda-env:v1

If the container already exists:
    docker start -ai ca-fa-cuda

Inside the CUDA container:
    cd /workspace/p4
    make clean
    make ARCH=sm_89
    ./main

Or use the Makefile run target:
    make run ARCH=sm_89

Custom execution:
    ./main ../test/cyberpunk2077_in.txt output/cuda_out.txt 3 256

Arguments:
    ./main <input_txt> <output_txt> [iterations] [threads_per_block]

Generate PTX:
    make ptx ARCH=sm_89

Run Nsight Compute:
    make ncu ARCH=sm_89


5. Notes
--------
- Preprocessing is not included in the measured kernel runtime.
- P1~P3 report gem5 simulated statistics from m5out/stats.txt.
- P4 reports CPU reference runtime and CUDA kernel runtime.
- gem5 simulated time and actual host/GPU runtime are different concepts.
