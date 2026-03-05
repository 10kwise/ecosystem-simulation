import numpy as np
from PIL import Image
import math


# ==============================================================================
# WHAT IS YCbCr AND WHY ISN'T IT A FORMAT ITSELF?
# ==============================================================================
#
# YCbCr is not a file format — it is a COORDINATE SYSTEM for colour.
# RGB and YCbCr describe the exact same colours, just from different angles,
# the same way you can describe a location as (x,y) or (distance, angle).
# Converting between them loses nothing — it is perfectly reversible.
#
# Y  = Luma      — how bright the pixel is (white ↔ black)
# Cb = Blue diff — how much bluer/yellower the pixel is vs its brightness
# Cr = Red diff  — how much redder/greener the pixel is vs its brightness
#
# Why this helps compression:
#   In RGB, brightness is split across all 3 channels. A bright white pixel
#   is (255, 255, 255) — high values everywhere. A dark pixel is (10, 10, 10).
#   The channels move together, meaning you're storing the same information
#   (brightness) three times independently.
#
#   In YCbCr, brightness lives entirely in Y. Cb and Cr only carry the colour
#   tint — and for most natural images, especially monochromatic ones, Cb and Cr
#   are nearly flat (close to 128 everywhere). A nearly flat channel has almost
#   no variation, which means very few groups form, which means very few bits
#   are needed to store it.
#
#   Example — a grey pixel (120, 120, 120) in both systems:
#     RGB:   R=120  G=120  B=120   (3 different channels all carrying brightness)
#     YCbCr: Y=120  Cb=128 Cr=128  (brightness in Y, colour channels are neutral)
#
# WHY YCbCr IS NOT A STANDALONE FORMAT:
#   YCbCr is a transformation step, not a complete compression solution.
#   JPEG uses YCbCr internally — it converts to YCbCr first, then runs DCT,
#   then Huffman. PNG doesn't use it because PNG targets lossless compression
#   and stores raw RGB so you always get back the exact original values.
#
#   A "YCbCr file" would just be 3 raw channel arrays with no spatial
#   compression, no entropy coding, nothing. It would likely be LARGER than
#   raw RGB because the floating point conversion adds precision overhead.
#   The value of YCbCr only shows up when combined with a full compression
#   pipeline — which is exactly what we're building here.
#
#   PNG is lossless and widely supported → used when exact pixel accuracy matters
#   JPEG uses YCbCr + DCT + Huffman internally → used when file size matters
#   Our pipeline uses YCbCr + grouping + bit packing + RLE → educational version
#   of the same idea JPEG uses, built from scratch
#
# ==============================================================================


# ==============================================================================
# PART 1 — YCbCr TRANSFORM
#
# Each pixel's (R, G, B) values are multiplied by a 3×3 matrix to produce
# (Y, Cb, Cr). This is a rotation of the colour space — no information is lost.
#
# The matrix values come from the BT.601 standard (the same one JPEG uses).
# Cb and Cr are offset by 128 so they sit in the middle of the 0–255 range
# rather than going negative (makes uint8 storage straightforward).
#
# After transform:
#   Y  channel: full range 0–255, contains all brightness detail
#   Cb channel: clustered near 128 for natural/monochromatic images
#   Cr channel: clustered near 128 for natural/monochromatic images
# → Cb and Cr will form far fewer groups than R or G or B would
# ==============================================================================

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Converts an (H, W, 3) uint8 RGB image to YCbCr.
    Output is float32 to preserve precision during the matrix multiply.
    """
    img = image.astype(np.float32)
    transform = np.array([
        [ 0.299,   0.587,   0.114 ],   # Y  row
        [-0.169,  -0.331,   0.500 ],   # Cb row
        [ 0.500,  -0.419,  -0.081 ],   # Cr row
    ])
    # @ is NumPy's matrix multiply. We transpose so the channels are on axis 0.
    # Each pixel (r,g,b) is treated as a column vector and multiplied by the matrix.
    ycbcr = img @ transform.T
    ycbcr[:, :, 1] += 128   # shift Cb to centre around 128
    ycbcr[:, :, 2] += 128   # shift Cr to centre around 128
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Reverses rgb_to_ycbcr exactly. Applies the inverse matrix and removes
    the 128 offset to recover the original R, G, B values.
    """
    img = image.astype(np.float32)
    img[:, :, 1] -= 128
    img[:, :, 2] -= 128
    inverse = np.array([
        [1.000,  0.000,  1.402],   # R row
        [1.000, -0.344, -0.714],   # G row
        [1.000,  1.772,  0.000],   # B row
    ])
    rgb = img @ inverse.T
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ==============================================================================
# PART 2 — GROUPING
#
# Scans a single channel of the (now YCbCr) image and assigns every pixel
# a small integer group ID. Pixels within `tolerance` of each other share
# a group and will all be reconstructed as the same value.
#
# Returns:
#   label_map — (H, W) array where each cell holds a group ID
#   palette   — 1D array where palette[group_id] = representative value
#
# After the YCbCr transform, Cb and Cr channels cluster near 128 so they
# form far fewer groups than R/G/B would — fewer groups = fewer bits needed.
# ==============================================================================

def channel_to_label_map(
    image: np.ndarray,
    channel: int,
    tolerance: int = 10
) -> tuple[np.ndarray, np.ndarray]:

    if image.ndim != 3:
        raise ValueError(f"Expected (H, W, C), got {image.shape}")
    if not (0 <= channel < image.shape[2]):
        raise ValueError(f"Channel {channel} out of range")

    # int16 prevents underflow when subtracting uint8 values
    # (e.g. 5 - 200 = -195 correctly in int16, wraps to 61 in uint8)
    channel_data = image[:, :, channel].astype(np.int16)
    H, W = channel_data.shape

    label_list = []
    palette    = []
    next_id    = 0

    for r in range(H):
        for c in range(W):
            pixel_val  = channel_data[r, c]
            matched_id = None

            for gid, rep_val in enumerate(palette):
                if abs(pixel_val - rep_val) <= tolerance:
                    matched_id = gid
                    break

            if matched_id is not None:
                label_list.append(matched_id)
            else:
                palette.append(int(pixel_val))
                label_list.append(next_id)
                next_id += 1

    n_groups = len(palette)
    if n_groups <= 256:
        dtype = np.uint8
    elif n_groups <= 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32

    label_map = np.array(label_list, dtype=dtype).reshape(H, W)
    palette   = np.array(palette, dtype=np.uint8)
    return label_map, palette


# ==============================================================================
# PART 3 — BIT PACKING
#
# uint8 stores 8 bits per value. If we only have ≤16 groups we only need
# 4 bits per ID — the top 4 bits of every byte are wasted zeros.
# We fix this by cramming 2 IDs into 1 byte using the high and low nibble:
#
#   byte layout:  [high nibble | low nibble]  =  [bits 7-4 | bits 3-0]
#   pack:         byte = (odd_pixel_id << 4) | even_pixel_id
#   unpack:       even = byte & 0x0F          odd = (byte >> 4) & 0x0F
# ==============================================================================

def pack_label_map(label_map: np.ndarray, n_groups: int) -> np.ndarray:
    bits_needed = max(1, math.ceil(math.log2(n_groups + 1)))
    if bits_needed <= 4:
        flat = label_map.flatten().astype(np.uint8)
        if len(flat) % 2 != 0:
            flat = np.append(flat, 0)
        packed = (flat[1::2] << 4) | flat[0::2]
        return packed.astype(np.uint8)
    return label_map.astype(np.uint8)


def unpack_label_map(packed: np.ndarray, original_shape: tuple, n_groups: int) -> np.ndarray:
    bits_needed = max(1, math.ceil(math.log2(n_groups + 1)))
    H, W = original_shape
    if bits_needed <= 4:
        low  = (packed & 0x0F).astype(np.uint8)
        high = ((packed >> 4) & 0x0F).astype(np.uint8)
        flat = np.empty(len(low) + len(high), dtype=np.uint8)
        flat[0::2] = low
        flat[1::2] = high
        return flat[:H * W].reshape(H, W)
    return packed.reshape(H, W)


# ==============================================================================
# PART 4 — RUN-LENGTH ENCODING (RLE)
#
# Scans the flat packed array for consecutive identical values (runs).
# Instead of storing every value, stores (value, count) pairs.
# Only saves space when runs are long — which they are after grouping,
# since neighbouring pixels usually belong to the same group.
#
# Stored as a flat uint16 array: [v0, c0, v1, c1, v2, c2, ...]
# uint16 for counts because a run can exceed 255 pixels (max uint8 = 255).
# ==============================================================================

def rle_encode(packed: np.ndarray) -> np.ndarray:
    flat = packed.flatten()
    if len(flat) == 0:
        return np.array([], dtype=np.uint16)

    # Find where values change — those indices are run boundaries
    change_points = np.concatenate(([0], np.where(np.diff(flat) != 0)[0] + 1, [len(flat)]))
    values = flat[change_points[:-1]]
    counts = np.diff(change_points)   # keep as int64 — do not cast to uint16 yet

    # uint16 max is 65535. A nearly flat Cb/Cr channel on a large image can
    # produce runs of millions of identical values, silently overflowing uint16
    # and causing the decoded array to come back shorter than expected.
    # Fix: split any run longer than 65535 into multiple (value, count) pairs.
    MAX_COUNT = 65535
    result = []
    for val, count in zip(values, counts):
        while count > MAX_COUNT:
            result.append(int(val))
            result.append(MAX_COUNT)
            count -= MAX_COUNT
        result.append(int(val))
        result.append(int(count))

    return np.array(result, dtype=np.uint16)


def rle_decode(encoded: np.ndarray, original_length: int) -> np.ndarray:
    # np.repeat([0,1,3], [4,2,1]) → [0,0,0,0,1,1,3]
    # rebuilds the full flat array from (value, count) pairs
    values = encoded[0::2].astype(np.uint8)
    counts = encoded[1::2]
    return np.repeat(values, counts)[:original_length]


# ==============================================================================
# PART 5 — RECONSTRUCTION
#
# palette[label_map] is NumPy fancy indexing:
# every group ID in the 2D label map is replaced by its value in the palette.
# This single operation reconstructs the entire channel instantly.
#
# Example:
#   palette   = [200, 50, 10]
#   label_map = [[0, 1], [2, 0]]
#   result    = [[200, 50], [10, 200]]
# ==============================================================================

def reconstruct_channel(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[label_map].astype(np.uint8)


# ==============================================================================
# PART 6 — SIZE REPORTING WITH COMPRESSION RATIOS
#
# .nbytes gives the exact byte count of a NumPy array (no Python overhead).
# Ratio < 1.0 means compression. Ratio = 0.5 means half the original size.
# ==============================================================================

def print_size_report(channel_name, raw_channel, label_map, palette, packed, encoded):
    raw     = raw_channel.nbytes
    label   = label_map.nbytes  + palette.nbytes
    bpacked = packed.nbytes     + palette.nbytes
    rle     = encoded.nbytes    + palette.nbytes

    def ratio(n): return f"{n/raw:.3f}x  ({(1 - n/raw)*100:+.1f}%)"

    print(f"\n  [{channel_name}]  groups: {len(palette)}")
    print(f"    Raw channel        : {raw    /1024:.2f} KB  (baseline)")
    print(f"    After grouping     : {label  /1024:.2f} KB  {ratio(label)}")
    print(f"    After bit packing  : {bpacked/1024:.2f} KB  {ratio(bpacked)}")
    print(f"    After RLE          : {rle    /1024:.2f} KB  {ratio(rle)}")


# ==============================================================================
# PART 7 — SIDE BY SIDE COMPARISON
#
# Places the original and reconstructed images next to each other
# in a single wider canvas with a white divider so you can compare quality.
# ==============================================================================

def side_by_side(original: np.ndarray, reconstructed: np.ndarray) -> Image.Image:
    H, W, _ = original.shape
    canvas = np.ones((H, W * 2 + 4, 3), dtype=np.uint8) * 255
    canvas[:, :W,     :] = original
    canvas[:, W + 4:, :] = reconstructed
    return Image.fromarray(canvas)


# ==============================================================================
# MAIN — FULL PIPELINE
#
# Compress:
#   RGB → YCbCr → group each channel → bit pack → RLE encode
#
# Decompress:
#   RLE decode → unpack bits → palette lookup → stack channels → YCbCr → RGB
# ==============================================================================

if __name__ == "__main__":
    img       = np.array(Image.open("compression example/picel world wallppar.png"))
    tolerance = 15

    channel_names = ["Y (luma)", "Cb (blue diff)", "Cr (red diff)"]

    print(f"\n{'='*55}")
    print(f"  Image size : {img.shape[1]} × {img.shape[0]} px")
    print(f"  Raw total  : {img.nbytes / 1024:.2f} KB")
    print(f"  Tolerance  : {tolerance}")
    print(f"{'='*55}")

    # ── Step 1: transform RGB → YCbCr ────────────────────────────────────────
    # Decorrelates the 3 channels so brightness (Y) is isolated and the colour
    # difference channels (Cb, Cr) become nearly flat for natural images
    ycbcr = rgb_to_ycbcr(img)

    compressed_channels = []
    reconstructed_ycbcr = np.zeros_like(ycbcr)
    total_rle_bytes     = 0

    for i in range(3):
        # ── Step 2: group similar values → label map + palette ────────────────
        label_map, palette = channel_to_label_map(ycbcr, channel=i, tolerance=tolerance)

        # ── Step 3: bit pack — 2 IDs per byte if groups ≤ 16 ─────────────────
        packed = pack_label_map(label_map, n_groups=len(palette))

        # ── Step 4: RLE — collapse runs of identical IDs ──────────────────────
        encoded = rle_encode(packed)

        total_rle_bytes += encoded.nbytes + palette.nbytes
        print_size_report(channel_names[i], ycbcr[:, :, i], label_map, palette, packed, encoded)

        # store what we need for reconstruction
        compressed_channels.append((encoded, palette, label_map.shape, packed.shape, len(palette)))

        # ── Step 5 (decompress): RLE decode ───────────────────────────────────
        decoded_flat = rle_decode(encoded, original_length=packed.size)

        # ── Step 6 (decompress): unpack bits → label map ──────────────────────
        recovered_labels = unpack_label_map(
            decoded_flat.reshape(packed.shape), label_map.shape, len(palette)
        )

        # ── Step 7 (decompress): palette lookup → pixel values ────────────────
        reconstructed_ycbcr[:, :, i] = reconstruct_channel(recovered_labels, palette)

    # ── Step 8 (decompress): YCbCr → RGB ─────────────────────────────────────
    reconstructed_rgb = ycbcr_to_rgb(reconstructed_ycbcr)

    print(f"\n{'='*55}")
    print(f"  Total compressed size : {total_rle_bytes / 1024:.2f} KB")
    print(f"  Total original size   : {img.nbytes / 1024:.2f} KB")
    print(f"  Overall ratio         : {total_rle_bytes / img.nbytes:.3f}x  "
          f"({(1 - total_rle_bytes / img.nbytes) * 100:+.1f}%)")
    print(f"{'='*55}\n")

    side_by_side(img, reconstructed_rgb).show()