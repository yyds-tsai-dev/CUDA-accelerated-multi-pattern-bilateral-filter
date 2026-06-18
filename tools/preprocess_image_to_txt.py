#!/usr/bin/env python3
"""
PNG -> grayscale txt + preview PNG, without Pillow.

Default output names:
  <out-dir>/<name>_in.txt
  <out-dir>/<name>_in.png

The txt format is:
  width height
  p00 p01 p02 ...
  ...
"""
import argparse
import struct
import zlib
from pathlib import Path


def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("Only PNG input is supported by this no-Pillow script. Convert JPG to PNG first.")

    pos = 8
    width = height = None
    bit_depth = color_type = interlace = None
    idat = bytearray()

    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        chunk_type = data[pos:pos + 4]
        pos += 4
        chunk_data = data[pos:pos + length]
        pos += length
        pos += 4  # CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8:
        raise SystemExit("Only 8-bit PNG is supported.")
    if interlace != 0:
        raise SystemExit("Interlaced PNG is not supported.")
    if color_type not in (0, 2, 4, 6):
        raise SystemExit("Only grayscale, RGB, or RGBA PNG is supported.")

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    bpp = channels
    stride = width * channels

    raw = zlib.decompress(bytes(idat))
    rows = []
    idx = 0
    prev = [0] * stride

    for _ in range(height):
        filter_type = raw[idx]
        idx += 1
        cur = list(raw[idx:idx + stride])
        idx += stride
        recon = [0] * stride

        for i in range(stride):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0

            if filter_type == 0:
                val = cur[i]
            elif filter_type == 1:
                val = (cur[i] + left) & 255
            elif filter_type == 2:
                val = (cur[i] + up) & 255
            elif filter_type == 3:
                val = (cur[i] + ((left + up) >> 1)) & 255
            elif filter_type == 4:
                val = (cur[i] + paeth(left, up, up_left)) & 255
            else:
                raise SystemExit("Unsupported PNG filter type.")
            recon[i] = val

        rows.append(recon)
        prev = recon

    gray = []
    for row in rows:
        out_row = []
        for x in range(width):
            base = x * channels
            if color_type in (0, 4):
                g = row[base]
            else:
                r, g0, b = row[base], row[base + 1], row[base + 2]
                g = int(round(0.299 * r + 0.587 * g0 + 0.114 * b))
            out_row.append(max(0, min(255, g)))
        gray.append(out_row)

    return width, height, gray


def resize_nearest(src, src_w, src_h, dst_w, dst_h):
    out = []
    for y in range(dst_h):
        sy = min(src_h - 1, int(y * src_h / dst_h))
        row = []
        for x in range(dst_w):
            sx = min(src_w - 1, int(x * src_w / dst_w))
            row.append(src[sy][sx])
        out.append(row)
    return out


def write_txt(path, pixels, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{width} {height}\n")
        for y in range(height):
            f.write(" ".join(str(v) for v in pixels[y]))
            f.write("\n")


def png_chunk(chunk_type, payload):
    return (
        struct.pack(">I", len(payload)) +
        chunk_type +
        payload +
        struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff)
    )


def write_png_gray(path, pixels, width, height, scale=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if scale < 1:
        scale = 1

    out_w = width * scale
    out_h = height * scale
    raw = bytearray()
    for y in range(out_h):
        src_y = y // scale
        raw.append(0)
        for x in range(out_w):
            src_x = x // scale
            raw.append(pixels[src_y][src_x])

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", out_w, out_h, 8, 0, 0, 0, 0)))
    png.extend(png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(png_chunk(b"IEND", b""))
    path.write_bytes(png)


def main():
    parser = argparse.ArgumentParser(description="Convert PNG to grayscale txt input and preview PNG without Pillow.")
    parser.add_argument("input_image", help="input PNG image")
    parser.add_argument("--out-dir", default="test", help="output directory, default: test")
    parser.add_argument("--name", default=None, help="base output name; default: input file stem")
    parser.add_argument("--width", type=int, default=160, help="resize width, default: 160")
    parser.add_argument("--height", type=int, default=None, help="resize height; default preserves aspect ratio")
    parser.add_argument("--keep-original-size", action="store_true", help="do not resize")
    parser.add_argument("--preview-scale", type=int, default=1, help="scale factor for preview PNG")
    args = parser.parse_args()

    src_w, src_h, pixels = read_png(args.input_image)

    if args.keep_original_size:
        dst_w, dst_h = src_w, src_h
    else:
        dst_w = args.width
        dst_h = args.height if args.height is not None else max(1, round(src_h * dst_w / src_w))

    if dst_w != src_w or dst_h != src_h:
        pixels = resize_nearest(pixels, src_w, src_h, dst_w, dst_h)

    base = args.name if args.name else Path(args.input_image).stem
    out_dir = Path(args.out_dir)
    txt_path = out_dir / f"{base}_in.txt"
    png_path = out_dir / f"{base}_in.png"

    write_txt(txt_path, pixels, dst_w, dst_h)
    write_png_gray(png_path, pixels, dst_w, dst_h, scale=args.preview_scale)

    print(f"Saved txt: {txt_path}")
    print(f"Saved png: {png_path}")
    print(f"Size: {dst_w} x {dst_h}")


if __name__ == "__main__":
    main()
