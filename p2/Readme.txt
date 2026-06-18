P2 Bilateral Filter
=======================

Default input:
    ../test/cyberpunk2077_in.txt

Default output:
    output/rvv_out.txt

Run inside gem5 Docker:
    cd /workspace/p2
    make clean
    make

The main program also supports command-line arguments:
    ./main <input_txt> <output_txt> [iterations]

For custom gem5 arguments, build the binary and invoke gem5 manually:
    make clean
    make main
    /root/gem5/build/RISCV/gem5.opt /root/gem5/configs/deprecated/example/se.py \
        -c /workspace/p2/main \
        --options="../test/cyberpunk2077_in.txt output/custom_out.txt 3" \
        --cpu-type=TimingSimpleCPU --mem-size=256MB --caches \
        --l1i_size=32kB --l1i_assoc=8 --l1d_size=32kB --l1d_assoc=8 --cacheline=32

Convert output txt to PNG from project root:
    python3 tools/txt_to_png.py p2/output/rvv_out.txt p2/output/rvv_out.png --scale 6
