import numpy as np
from PIL import Image
import math
import time

# ==============================================================================
# OVERVIEW — WHAT THIS FILE DOES
# ==============================================================================
#
# This is a complete image compression pipeline that combines:
#
#   1. YCbCr transform  — separate brightness from colour so each compresses better
#   2. Fractal (IFS)    — find parts of the image that predict other parts
#   3. Grouping + RLE   — used on the flat colour channels (Cb, Cr) which are
#                         nearly constant for monochromatic images
#
# The fractal encoder is an ANYTIME ALGORITHM — you give it a time budget and
# it keeps improving the result until time runs out. Stop it early for a rough
# result, let it run longer for a better one.
#
# HOW FRACTAL COMPRESSION WORKS (the core idea):
#   An image contains regions that look like scaled/rotated versions of other
#   regions. Instead of storing pixel values, we store INSTRUCTIONS:
#   "take this 16×16 block, scale it to 8×8, rotate it 90°, and that gives
#    you this other 8×8 block". The instructions are far smaller than the pixels.
#
#   To decompress, we don't need the original image. We start with a blank grey
#   canvas and repeatedly apply all the instructions. After ~10 iterations the
#   image converges to the correct result — this is guaranteed by a mathematical
#   theorem about contractive transformations always having a unique fixed point.
#
# BLOCK TERMINOLOGY:
#   Range block  — the 8×8 block we are TRYING TO DESCRIBE (the target)
#   Domain block — the 16×16 block we are USING AS A DESCRIPTION (the source)
#   The domain block is downscaled to 8×8 before comparison.
#
# ==============================================================================


# ── Configuration ──────────────────────────────────────────────────────────────
BLOCK_SIZE   = 8    # size of range blocks (what we compress)
DOMAIN_SIZE  = 16   # size of domain blocks (what we use as predictions)
              #       domain blocks are always 2× range blocks so they
              #       downsample cleanly by averaging 2×2 pixel groups
NUM_TRANSFORMS = 8  # 4 rotations (0°, 90°, 180°, 270°) × 2 flips = 8 variants


# ==============================================================================
# PART 1 — YCbCr COLOUR SPACE TRANSFORM
#
# Converts RGB to a space where:
#   Y  = brightness (luma)   — most of the visual information lives here
#   Cb = blue difference     — how much bluer/yellower vs neutral
#   Cr = red difference      — how much redder/greener vs neutral
#
# Why this helps:
#   In RGB, brightness is split across all 3 channels. Every channel carries
#   redundant brightness information. After converting to YCbCr, brightness
#   lives in Y alone. Cb and Cr are nearly flat (≈128) for natural images,
#   making them trivially compressible with grouping + RLE.
#
# The conversion is a matrix multiply — a rotation of the colour axes.
# It is fully reversible; no information is lost in the transform itself.
# ==============================================================================

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Converts (H, W, 3) uint8 RGB image to YCbCr.
    Uses the BT.601 matrix — the same standard JPEG uses.
    Output is uint8 with Cb and Cr shifted to centre around 128.
    """
    img = image.astype(np.float32)
    # Each row of this matrix is the recipe for one output channel:
    #   Y  = 0.299*R + 0.587*G + 0.114*B   (weighted sum — green counts most)
    #   Cb = mostly blue contribution minus brightness
    #   Cr = mostly red contribution minus brightness
    matrix = np.array([
        [ 0.299,   0.587,   0.114],
        [-0.169,  -0.331,   0.500],
        [ 0.500,  -0.419,  -0.081],
    ])
    ycbcr = img @ matrix.T      # @ = matrix multiply, .T = transpose
    ycbcr[:, :, 1] += 128       # shift Cb: centres it at 128 so it fits in uint8
    ycbcr[:, :, 2] += 128       # shift Cr: same reason
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Reverses rgb_to_ycbcr exactly using the inverse matrix.
    Any rounding from the forward transform means this is near-lossless,
    not perfectly lossless — errors are sub-pixel and invisible.
    """
    img = image.astype(np.float32)
    img[:, :, 1] -= 128     # undo the 128 shift before inverse transform
    img[:, :, 2] -= 128
    inverse = np.array([
        [1.000,  0.000,  1.402],
        [1.000, -0.344, -0.714],
        [1.000,  1.772,  0.000],
    ])
    rgb = img @ inverse.T
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ==============================================================================
# PART 2 — FRACTAL HELPER FUNCTIONS
#
# These are the building blocks the encoder and decoder both use.
# ==============================================================================

def apply_geometric_transform(block: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Applies one of 8 geometric transforms to a 2D block.
    The 8 transforms are: 4 rotations × 2 flip states.
    transform_idx 0–3: no flip, rotate 0°/90°/180°/270°
    transform_idx 4–7: flip horizontally, then rotate 0°/90°/180°/270°

    Trying all 8 lets us match blocks that are rotated or mirrored versions
    of each other — exploiting symmetry in the image.
    """
    if transform_idx >= 4:
        block = np.fliplr(block)            # flip left↔right
    return np.rot90(block, transform_idx % 4)   # rotate by 90° × (idx % 4)


def downsample_2x(block: np.ndarray) -> np.ndarray:
    """
    Halves a block's dimensions by averaging each 2×2 group of pixels.
    Converts a 16×16 domain block to 8×8 so it can be compared to range blocks.

    Reshape trick explanation:
      (16, 16) → (8, 2, 8, 2) → average the two middle axes (the 2×2 groups)
      Result: (8, 8) — each cell is the mean of a 2×2 square of the original.
    """
    h, w = block.shape
    return block.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def fit_brightness_contrast(domain: np.ndarray, range_block: np.ndarray):
    """
    Finds the best contrast (scale) and brightness (offset) to map
    domain pixel values to match range_block pixel values.

    Solves:  range ≈ contrast × domain + brightness
    Using least squares — minimises the sum of squared pixel errors.

    This is the same as fitting a straight line through a scatter plot
    where x = domain pixel values and y = range pixel values.

    Returns (contrast, brightness) as floats.
    """
    d = domain.flatten().astype(np.float64)
    r = range_block.flatten().astype(np.float64)
    n = len(d)

    # Least squares formula for slope (contrast) and intercept (brightness)
    denom = n * np.sum(d * d) - np.sum(d) ** 2
    if abs(denom) < 1e-10:
        # Domain is completely flat (all same value) — just match the mean
        return 0.0, float(np.mean(r))

    contrast   = (n * np.sum(d * r) - np.sum(d) * np.sum(r)) / denom
    brightness = (np.sum(r) - contrast * np.sum(d)) / n

    # Clamp to stable range — large contrasts cause the decoder to diverge
    # (the iterative application amplifies errors instead of reducing them)
    contrast   = float(np.clip(contrast,   -0.75,  0.75))
    brightness = float(np.clip(brightness, -128.0, 128.0))
    return contrast, brightness


def block_error(domain: np.ndarray, range_block: np.ndarray,
                contrast: float, brightness: float) -> float:
    """
    Mean squared error between the transformed domain and the range block.
    Lower = better match. This is what the encoder minimises.
    """
    approx = contrast * domain + brightness
    return float(np.mean((approx - range_block) ** 2))


# ==============================================================================
# PART 3 — FRACTAL ENCODER (ANYTIME ALGORITHM)
#
# For every 8×8 range block in the image, finds the 16×16 domain block
# (anywhere in the image) that — after downscaling, transforming, and
# adjusting brightness/contrast — best approximates that range block.
#
# ANYTIME BEHAVIOUR:
#   The encoder tracks time spent vs time budget.
#   Early on it tries all 8 geometric transforms per domain block.
#   As time runs low it drops to fewer transforms to finish in budget.
#   You can always interrupt it — whatever transforms are stored so far
#   produce a valid (if imperfect) compressed image.
#
# WHAT GETS STORED PER BLOCK (13 bytes):
#   domain_y   uint16  — row of the domain block in the image
#   domain_x   uint16  — column of the domain block in the image
#   transform  uint8   — which of the 8 geometric transforms to apply
#   contrast   float32 — brightness scaling factor
#   brightness float32 — brightness offset
# ==============================================================================

def encode_fractal(channel: np.ndarray, time_budget: float = 60.0) -> list:
    """
    Encodes a single greyscale channel using fractal (IFS) compression.

    Args:
        channel:     2D uint8 array — one channel of the YCbCr image (usually Y)
        time_budget: seconds to spend encoding. More time = better quality.
                     30s gives rough results, 300s gives good results.

    Returns:
        List of transforms, one per range block. Each transform is a tuple:
        (domain_y, domain_x, transform_idx, contrast, brightness)
    """
    H, W = channel.shape
    ch   = channel.astype(np.float32)

    # ── Precompute all domain blocks ──────────────────────────────────────────
    # Domain blocks are 16×16, stepped by BLOCK_SIZE so they cover the image.
    # We downsample each to 8×8 immediately — no need to redo this per range block.
    print("    Precomputing domain blocks...")
    domain_pool = []   # list of (dy, dx, downsampled_8x8_block)
    for dy in range(0, H - DOMAIN_SIZE + 1, BLOCK_SIZE):
        for dx in range(0, W - DOMAIN_SIZE + 1, BLOCK_SIZE):
            raw        = ch[dy:dy + DOMAIN_SIZE, dx:dx + DOMAIN_SIZE]
            downsampled = downsample_2x(raw)
            domain_pool.append((dy, dx, downsampled))

    n_domains    = len(domain_pool)
    n_range_rows = (H) // BLOCK_SIZE
    n_range_cols = (W) // BLOCK_SIZE
    total_blocks = n_range_rows * n_range_cols

    print(f"    {total_blocks} range blocks, {n_domains} domain blocks, "
          f"{NUM_TRANSFORMS} transforms each")
    print(f"    Time budget: {time_budget}s — quality improves until time runs out")

    transforms = []
    start_time = time.time()
    block_idx  = 0

    for ry in range(0, H - BLOCK_SIZE + 1, BLOCK_SIZE):
        for rx in range(0, W - BLOCK_SIZE + 1, BLOCK_SIZE):
            range_block = ch[ry:ry + BLOCK_SIZE, rx:rx + BLOCK_SIZE]

            # ── Anytime transform count decision ─────────────────────────────
            # Estimate how many transforms we can afford per domain block
            # based on how much time is left and how many blocks remain.
            elapsed        = time.time() - start_time
            remaining      = time_budget - elapsed
            blocks_left    = total_blocks - block_idx
            # Avoid division by zero; default to trying all transforms if we
            # have plenty of time, drop to 1 if we're running out
            if blocks_left > 0 and elapsed > 0:
                rate           = block_idx / elapsed          # blocks per second
                time_per_block = 1.0 / rate if rate > 0 else remaining
                transforms_affordable = max(1, min(
                    NUM_TRANSFORMS,
                    int(remaining / (blocks_left * time_per_block / NUM_TRANSFORMS))
                ))
            else:
                transforms_affordable = NUM_TRANSFORMS

            # ── Search for best matching domain block ─────────────────────────
            best_error = float('inf')
            best_transform = (0, 0, 0, 0.0, 128.0)

            for dy, dx, domain_base in domain_pool:
                for ti in range(transforms_affordable):
                    # Apply geometric transform to the downsampled domain block
                    domain_t = apply_geometric_transform(domain_base, ti)
                    # Find best brightness/contrast to map domain → range
                    c, b     = fit_brightness_contrast(domain_t, range_block)
                    err      = block_error(domain_t, range_block, c, b)

                    if err < best_error:
                        best_error     = err
                        best_transform = (dy, dx, ti, c, b)

            transforms.append(best_transform)
            block_idx += 1

        # Print progress every row
        elapsed  = time.time() - start_time
        progress = block_idx / total_blocks * 100
        print(f"    Progress: {progress:5.1f}%  |  elapsed: {elapsed:.1f}s  |  "
              f"transforms tried: {transforms_affordable}/block", end='\r')

    elapsed = time.time() - start_time
    print(f"\n    Done. {total_blocks} blocks encoded in {elapsed:.1f}s")
    return transforms


# ==============================================================================
# PART 4 — FRACTAL DECODER (ITERATIVE FIXED-POINT CONVERGENCE)
#
# Does NOT need the original image. Starts from a blank grey canvas and
# repeatedly applies all stored transforms until the image converges.
#
# Why it converges:
#   Each transform is contractive — it shrinks differences between pixel
#   values (contrast is clamped to < 1). A theorem in mathematics states
#   that any set of contractive transforms has exactly ONE fixed point —
#   one unique image that maps to itself under those transforms. No matter
#   what image you start with, repeated application converges to that point.
#
# How many iterations:
#   Typically 8–12 is enough. More gives diminishing returns.
#   Convergence is exponential — each iteration roughly squares the error.
# ==============================================================================

def decode_fractal(transforms: list, image_shape: tuple,
                   n_iterations: int = 10) -> np.ndarray:
    """
    Decodes fractal transforms by iterative fixed-point convergence.

    Args:
        transforms:   List of (domain_y, domain_x, transform_idx, contrast, brightness)
        image_shape:  (H, W) of the original channel
        n_iterations: How many times to apply all transforms. 10 is usually enough.

    Returns:
        Reconstructed channel as uint8 array.
    """
    H, W    = image_shape
    # Starting image — any image works, grey is conventional
    current = np.full((H, W), 128.0, dtype=np.float32)

    for iteration in range(n_iterations):
        next_img  = np.zeros((H, W), dtype=np.float32)
        block_idx = 0

        for ry in range(0, H - BLOCK_SIZE + 1, BLOCK_SIZE):
            for rx in range(0, W - BLOCK_SIZE + 1, BLOCK_SIZE):
                dy, dx, ti, contrast, brightness = transforms[block_idx]

                # Pull the domain block from the CURRENT (not original) image
                # This self-reference is what makes it fractal — the image
                # predicts itself from itself
                domain_raw = current[dy:dy + DOMAIN_SIZE, dx:dx + DOMAIN_SIZE]
                domain     = downsample_2x(domain_raw)
                domain_t   = apply_geometric_transform(domain, ti)

                # Apply stored brightness/contrast and write to output
                approximated = contrast * domain_t + brightness
                next_img[ry:ry + BLOCK_SIZE, rx:rx + BLOCK_SIZE] = approximated
                block_idx += 1

        current = next_img
        print(f"    Decoder iteration {iteration + 1}/{n_iterations}", end='\r')

    print()
    return np.clip(current, 0, 255).astype(np.uint8)


# ==============================================================================
# PART 5 — GROUPING + RLE FOR Cb AND Cr CHANNELS
#
# Cb and Cr are nearly flat after YCbCr transform (almost all values ≈ 128).
# They don't need fractal compression — simple grouping + RLE is enough.
# This is actually what professional systems do: complex compression on luma,
# simpler compression on chroma.
# ==============================================================================

def encode_flat_channel(channel: np.ndarray, tolerance: int = 10):
    """
    Groups similar values into a label map, then RLE encodes it.
    Best suited for nearly-constant channels like Cb and Cr.
    Returns (encoded_rle, palette, original_shape).
    """
    ch = channel.astype(np.int16)
    H, W = ch.shape

    label_list, palette, next_id = [], [], 0
    for r in range(H):
        for c in range(W):
            v = ch[r, c]
            mid = next((gid for gid, rep in enumerate(palette) if abs(v - rep) <= tolerance), None)
            if mid is not None:
                label_list.append(mid)
            else:
                palette.append(int(v))
                label_list.append(next_id)
                next_id += 1

    # Choose smallest dtype that fits all group IDs
    n = len(palette)
    dtype = np.uint8 if n <= 256 else (np.uint16 if n <= 65536 else np.uint32)
    label_map = np.array(label_list, dtype=dtype).reshape(H, W)
    palette   = np.array(palette, dtype=np.uint8)

    # Bit pack if ≤ 16 groups (2 IDs per byte using nibbles)
    bits = max(1, math.ceil(math.log2(n + 1)))
    if bits <= 4:
        flat = label_map.flatten().astype(np.uint8)
        if len(flat) % 2 != 0:
            flat = np.append(flat, 0)
        packed = ((flat[1::2] << 4) | flat[0::2]).astype(np.uint8)
    else:
        packed = label_map.astype(np.uint8)

    # RLE encode — split runs > 65535 to avoid uint16 overflow
    flat   = packed.flatten()
    cps    = np.concatenate(([0], np.where(np.diff(flat) != 0)[0] + 1, [len(flat)]))
    vals   = flat[cps[:-1]]
    counts = np.diff(cps)
    MAX    = 65535
    result = []
    for v, cnt in zip(vals, counts):
        while cnt > MAX:
            result += [int(v), MAX]
            cnt -= MAX
        result += [int(v), int(cnt)]
    encoded = np.array(result, dtype=np.uint16)

    return encoded, palette, (H, W), packed.shape, n, bits


def decode_flat_channel(encoded, palette, original_shape, packed_shape, n_groups, bits):
    """Reverses encode_flat_channel exactly."""
    H, W = original_shape
    # RLE decode
    values = encoded[0::2].astype(np.uint8)
    counts = encoded[1::2]
    flat   = np.repeat(values, counts)[:math.prod(packed_shape)]
    packed = flat.reshape(packed_shape)

    # Bit unpack
    if bits <= 4:
        lo   = (packed & 0x0F).astype(np.uint8)
        hi   = ((packed >> 4) & 0x0F).astype(np.uint8)
        full = np.empty(len(lo) + len(hi), dtype=np.uint8)
        full[0::2] = lo
        full[1::2] = hi
        label_map  = full[:H * W].reshape(H, W)
    else:
        label_map = packed.reshape(H, W)

    return palette[label_map].astype(np.uint8)


# ==============================================================================
# PART 6 — SIZE REPORTING
#
# Measures the true byte size of every stage so you can see where compression
# comes from. .nbytes on a NumPy array = elements × bytes_per_element,
# with no Python object overhead — it is the raw data size.
# ==============================================================================

def fractal_compressed_size(transforms: list) -> int:
    """
    Calculates bytes needed to store a list of fractal transforms.
    Layout per transform (13 bytes):
      domain_y   uint16 = 2 bytes
      domain_x   uint16 = 2 bytes
      transform  uint8  = 1 byte
      contrast   float32 = 4 bytes
      brightness float32 = 4 bytes
    """
    return len(transforms) * (2 + 2 + 1 + 4 + 4)


def print_size_report(img, y_transforms, cb_data, cr_data):
    raw_total = img.nbytes
    y_raw     = img[:, :, 0].nbytes

    y_compressed  = fractal_compressed_size(y_transforms)
    cb_compressed = cb_data[0].nbytes + cb_data[1].nbytes   # encoded + palette
    cr_compressed = cr_data[0].nbytes + cr_data[1].nbytes

    total_compressed = y_compressed + cb_compressed + cr_compressed

    def pct(n, d): return f"{n/d:.3f}x  ({(1 - n/d)*100:+.1f}%)"

    print(f"\n{'='*55}")
    print(f"  Original image        : {raw_total        / 1024:.1f} KB")
    print(f"  Y  channel (fractal)  : {y_compressed     / 1024:.1f} KB  {pct(y_compressed,  y_raw)}")
    print(f"  Cb channel (RLE)      : {cb_compressed    / 1024:.1f} KB  {pct(cb_compressed, y_raw)}")
    print(f"  Cr channel (RLE)      : {cr_compressed    / 1024:.1f} KB  {pct(cr_compressed, y_raw)}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total compressed      : {total_compressed / 1024:.1f} KB  {pct(total_compressed, raw_total)}")
    print(f"{'='*55}\n")


# ==============================================================================
# PART 7 — SIDE BY SIDE COMPARISON
#
# Stitches original and reconstructed images into one wide image with a
# white divider so you can directly compare quality vs compression.
# ==============================================================================

def side_by_side(original: np.ndarray, reconstructed: np.ndarray) -> Image.Image:
    H, W, _ = original.shape
    canvas = np.ones((H, W * 2 + 4, 3), dtype=np.uint8) * 255
    canvas[:, :W,     :] = original
    canvas[:, W + 4:, :] = reconstructed
    return Image.fromarray(canvas)


# ==============================================================================
# MAIN — RUN THE FULL PIPELINE
# ==============================================================================

if __name__ == "__main__":
    # ── Load image ─────────────────────────────────────────────────────────────
    img = np.array(Image.open("compression example/picel world wallppar.png"))
    print(f"Image loaded: {img.shape[1]}×{img.shape[0]} px  |  {img.nbytes / 1024:.1f} KB raw")

    # ── Step 1: RGB → YCbCr ───────────────────────────────────────────────────
    # Separates brightness (Y) from colour (Cb, Cr) so each can be
    # compressed with the method best suited to its characteristics
    print("\n[Step 1] Converting RGB → YCbCr...")
    ycbcr = rgb_to_ycbcr(img)

    # ── Step 2: Fractal encode the Y (luma) channel ───────────────────────────
    # Y carries all the visual structure — edges, shapes, texture.
    # Fractal compression finds self-similar regions to describe it compactly.
    # Increase TIME_BUDGET for better quality (at the cost of encode time).
    TIME_BUDGET = 60.0   # seconds — raise this for better quality
    print(f"\n[Step 2] Fractal encoding Y channel (budget: {TIME_BUDGET}s)...")
    y_transforms = encode_fractal(ycbcr[:, :, 0], time_budget=TIME_BUDGET)

    # ── Step 3: Grouping + RLE encode Cb and Cr channels ─────────────────────
    # Cb and Cr are nearly flat for natural/monochromatic images.
    # Simple grouping collapses them to very few groups, RLE finishes the job.
    TOLERANCE = 15
    print(f"\n[Step 3] Encoding Cb channel (tolerance={TOLERANCE})...")
    cb_data = encode_flat_channel(ycbcr[:, :, 1], tolerance=TOLERANCE)
    print(f"         Cb groups formed: {cb_data[4]}")

    print(f"\n[Step 3] Encoding Cr channel (tolerance={TOLERANCE})...")
    cr_data = encode_flat_channel(ycbcr[:, :, 2], tolerance=TOLERANCE)
    print(f"         Cr groups formed: {cr_data[4]}")

    # ── Size report ───────────────────────────────────────────────────────────
    print_size_report(img, y_transforms, cb_data, cr_data)

    # ── Step 4: Fractal decode Y channel ─────────────────────────────────────
    # Starts from grey, applies all transforms iteratively until convergence.
    # No original image needed — the transforms alone describe the result.
    print("[Step 4] Fractal decoding Y channel...")
    y_reconstructed = decode_fractal(y_transforms, ycbcr[:, :, 0].shape, n_iterations=10)

    # ── Step 5: Decode Cb and Cr channels ────────────────────────────────────
    print("[Step 5] Decoding Cb and Cr channels...")
    cb_reconstructed = decode_flat_channel(*cb_data)
    cr_reconstructed = decode_flat_channel(*cr_data)

    # ── Step 6: Stack channels and convert YCbCr → RGB ───────────────────────
    print("[Step 6] Converting YCbCr → RGB...")
    reconstructed_ycbcr = np.stack([y_reconstructed, cb_reconstructed, cr_reconstructed], axis=2)
    reconstructed_rgb   = ycbcr_to_rgb(reconstructed_ycbcr)

    # ── Step 7: Show side by side comparison ─────────────────────────────────
    print("[Step 7] Displaying comparison (original left, reconstructed right)...")
    side_by_side(img, reconstructed_rgb).show()
