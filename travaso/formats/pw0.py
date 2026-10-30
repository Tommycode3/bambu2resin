"""PW0 encoding: the layer RLE in Anycubic files.

Each pixel is reduced to 4 bits (16 grey levels) and runs of equal values are
compressed. Full black and full white, which make up the vast majority of the
image, get a 12-bit counter over two bytes; intermediate greys only 4 bits in
one byte, because they appear at the part's edge in short runs.

On decoding an intermediate code `c` maps back to `(c << 4) | c`, so 0 -> 0
and 15 -> 255: the reduction to 4 bits is a rounding, not a loss of the
extremes.
"""

from __future__ import annotations

import numpy as np

WIDE_RUN_LIMIT = 0xFFF  # 4095, 12-bit counter for black and white
NARROW_RUN_LIMIT = 0xF  # 15, 4-bit counter for greys


def encode_pw0(image: np.ndarray) -> bytes:
    """Compress a layer image (uint8, row by row) to PW0."""
    flat = np.ascontiguousarray(image, dtype=np.uint8).ravel() >> 4
    if flat.size == 0:
        return b""

    # Boundaries of the constant-value runs
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [flat.size]))).astype(np.int64)
    values = flat[starts].astype(np.int64)

    wide = (values == 0) | (values == 15)
    limits = np.where(wide, WIDE_RUN_LIMIT, NARROW_RUN_LIMIT)

    # A run longer than the counter must be split into several pieces
    pieces = -(-lengths // limits)
    total = int(pieces.sum())
    run_of = np.repeat(np.arange(len(values)), pieces)
    begins = np.concatenate(([0], np.cumsum(pieces)))[:-1]
    position = np.arange(total) - begins[run_of]

    limit_of = limits[run_of]
    counts = np.minimum(limit_of, lengths[run_of] - position * limit_of)
    colors = values[run_of]
    is_wide = wide[run_of]

    sizes = np.where(is_wide, 2, 1)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    out = np.zeros(int(offsets[-1]), dtype=np.uint8)
    offsets = offsets[:-1]

    if is_wide.any():
        at = offsets[is_wide]
        packed = (colors[is_wide] << 12) | counts[is_wide]
        out[at] = (packed >> 8) & 0xFF
        out[at + 1] = packed & 0xFF

    narrow = ~is_wide
    if narrow.any():
        out[offsets[narrow]] = ((colors[narrow] << 4) | counts[narrow]) & 0xFF

    return out.tobytes()


def decode_pw0(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode PW0. Used to verify the encoder, not for printing."""
    out = np.zeros(width * height, dtype=np.uint8)
    position = 0
    i = 0
    n = len(data)
    while i < n and position < out.size:
        byte = data[i]
        code = byte >> 4
        repeat = byte & 0xF
        if code in (0x0, 0xF):
            color = 0 if code == 0 else 255
            i += 1
            if i >= n:
                repeat = out.size - position
            else:
                repeat = (repeat << 8) + data[i]
        else:
            color = (code << 4) | code
        i += 1
        end = min(position + repeat, out.size)
        out[position:end] = color
        position = end
    return out.reshape(height, width)
