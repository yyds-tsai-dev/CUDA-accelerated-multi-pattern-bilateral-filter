#!/usr/bin/env python3
import argparse
import struct
import zlib
from pathlib import Path

DEFAULT_IO = {
    "p1": ("p1/output/scalar_out.txt", "p1/output/scalar_out.png"),
    "p2": ("p2/output/rvv_out.txt", "p2/output/rvv_out.png"),
    "p3": ("p3/output/simd_rvv_out.txt", "p3/output/simd_rvv_out.png"),
    "p4": ("p4/output/cuda_out.txt", "p4/output/cuda_out.png"),
    "p5": ("p5/output/multi_cuda_out.txt", "p5/output/multi_cuda_out.png")
}

def detect_part_from_cwd():
    cur = Path.cwd().name
    if cur in DEFAULT_IO:
        return cur
    return "p1"

def read_txt_image(path):
    path = Path(path)
    with path.open("r") as f:
        first = f.readline().strip().split()
        if len(first) != 2:
            raise ValueError("First line must be: <width> <height>")
        width, height = int(first[0]), int(first[1])

        pixels = []
        for line in f:
            if line.strip():
                pixels.extend([int(round(float(x))) for x in line.split()])

    if len(pixels) != width * height:
        raise ValueError(f"Pixel count mismatch: expect {width * height}, got {len(pixels)}")

    pixels = bytes(max(0, min(255, v)) for v in pixels)
    return width, height, pixels

def scale_nearest(width, height, pixels, scale):
    if scale == 1:
        return width, height, pixels

    new_w = width * scale
    new_h = height * scale
    out = bytearray()

    for y in range(new_h):
        src_y = y // scale
        src_row = pixels[src_y * width:(src_y + 1) * width]

        scaled_row = bytearray()
        for v in src_row:
            scaled_row.extend([v] * scale)

        out.extend(scaled_row)

    return new_w, new_h, bytes(out)

def png_chunk(chunk_type, payload):
    return (
        struct.pack(">I", len(payload)) +
        chunk_type +
        payload +
        struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff)
    )

def write_png_gray(path, width, height, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * width:(y + 1) * width])

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
    png.extend(png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(png_chunk(b"IEND", b""))

    path.write_bytes(png)

def default_output_from_input(input_path):
    input_path = Path(input_path)
    stem = input_path.stem

    if stem.endswith("_out"):
        out_stem = stem
    else:
        out_stem = stem + "_out"

    return input_path.with_name(out_stem + ".png")

def main():
    parser = argparse.ArgumentParser(description="Convert bilateral-filter txt output to PNG.")
    parser.add_argument("input", nargs="?", help="Input txt file. Optional if using default part output.")
    parser.add_argument("output", nargs="?", help="Output png file. Optional.")
    parser.add_argument("--part", choices=["p1", "p2", "p3", "p4", "p5"], help="Use default output path of selected part.")
    parser.add_argument("--scale", type=int, default=1, help="Nearest-neighbor scale factor for viewing.")
    args = parser.parse_args()

    if args.input is None:
        part = args.part if args.part else detect_part_from_cwd()

        if Path.cwd().name in DEFAULT_IO:
            input_path = Path("output") / Path(DEFAULT_IO[part][0]).name
            output_path = Path("output") / Path(DEFAULT_IO[part][1]).name
        else:
            input_path = Path(DEFAULT_IO[part][0])
            output_path = Path(DEFAULT_IO[part][1])
    else:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else default_output_from_input(input_path)

    width, height, pixels = read_txt_image(input_path)
    width, height, pixels = scale_nearest(width, height, pixels, args.scale)
    write_png_gray(output_path, width, height, pixels)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Size  : {width} x {height}")

if __name__ == "__main__":
    main()
