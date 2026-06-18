P1 Bilateral Filter
=======================

Default input:
    ../test/cyberpunk2077_in.txt

Default output:
    output/scalar_out.txt

Run inside gem5 Docker:
    cd /workspace/p1
    make clean
    make

The main program also supports command-line arguments:
    ./main <input_txt> <output_txt> [iterations]

For gem5 custom arguments through the Makefile:
    make clean
    make ARGS="../test/cyberpunk2077_in.txt output/custom_out.txt 3"

Convert output txt to PNG from project root:
    python3 tools/txt_to_png.py p1/output/scalar_out.txt p1/output/scalar_out.png --scale 6
