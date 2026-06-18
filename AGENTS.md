# Repository Guidelines

## Project Structure & Module Organization

This repository implements bilateral filtering across five parts. `p1/` is the scalar RISC-V baseline, `p2/` and `p3/` are RVV variants for gem5, `p4/` is the single-image CUDA implementation, and `p5/` is the multi-pattern CUDA implementation. Shared sample inputs live in `test/`; each part writes results under its own `output/` directory. `tools/` contains PNG/TXT conversion helpers. `docs/` and `report/` hold assignment and reporting material.

## Build, Test, and Development Commands

Run gem5 parts inside the project-mounted gem5 container:

```sh
cd /workspace/p1 && make clean && make
cd /workspace/p2 && make clean && make
cd /workspace/p3 && make clean && make
```

Build CUDA parts inside the CUDA container:

```sh
cd /workspace/p4 && make clean && make CUDA_ARCH=sm_89
cd /workspace/p5 && make clean && make CUDA_ARCH=sm_89
```

Run CUDA binaries with defaults using `./main`, or pass custom arguments such as `./main ../test/cyberpunk2077_in.txt output/cuda_out.txt 3 256`. Convert outputs for visual checks with `python3 tools/txt_to_png.py p5/output/multi_cuda_out.txt --scale 6`.

## Coding Style & Naming Conventions

Keep C/CUDA code in the existing compact style: 4-space indentation, K&R braces, lowercase helper functions, uppercase constants/macros, and explicit error messages. Match existing filenames (`main.cpp`, `main.cu`, `main_cpu.cpp`) unless adding a clearly separate utility. Python tools should stay dependency-light and use `pathlib` where practical.

## Testing Guidelines

There is no formal unit-test harness. Verify changes by rebuilding the affected part, running it on `test/cyberpunk2077_in.txt`, and comparing output behavior against the relevant baseline. For CUDA changes, run both GPU and CPU targets when available (`make run-gpu`, `make run-cpu`) and inspect reported correctness metrics. Keep generated `m5out/` and `output/` changes only when they are intentional experiment artifacts.

## Commit & Pull Request Guidelines

Recent history uses short, imperative messages with optional prefixes such as `feat:`, `fix:`, `docs:`, `report:`, and `init:`. Prefer that format and keep each commit scoped to one part or documentation update. Pull requests should describe the affected part, commands run, input size, CUDA architecture or gem5 environment, and any output/stat files intentionally updated.

