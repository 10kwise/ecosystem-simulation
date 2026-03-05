import numpy as np
from PIL import Image
import math


# ==============================================================================
# COMPRESSION TECHNIQUES — HOW THEY WORK
# ==============================================================================
#
# Every compression technique exploits a different kind of REDUNDANCY in data.
# Redundancy means "information that is repeated or predictable and doesn't
# need to be stored explicitly".
#
# ── 1. GROUPING / QUANTISATION (what we do in Part 1) ─────────────────────────
#   Redundancy exploited: similar values that don't need to be distinct
#   How: replace a range of values (e.g. 198–212) with a single representative
#        (e.g. 200). You trade precision for fewer unique values.
#   Lossy: yes — you cannot recover the exact original values
#   Real world: JPEG uses this on frequency coefficients
#
# ── 2. BIT PACKING (what we do in Part 2) ─────────────────────────────────────
#   Redundancy exploited: unused bits inside each byte
#   How: if you only have 14 groups, a uint8 wastes its top 4 bits on zeros.
#        Pack two 4-bit IDs into one byte instead.
#   Lossy: no — all information is preserved, just rearranged
#
# ── 3. RUN-LENGTH ENCODING / RLE (what we add in Part 3) ──────────────────────
#   Redundancy exploited: long runs of the same value in a row
#   How: instead of storing [0,0,0,0,0,1,1,1] (8 values), store [(0,5),(1,3)]
#        — a (value, count) pair per run. Only helps when runs are long.
#   Lossy: no
#   Real world: BMP, fax machines, old game sprites
#
# ── 4. HUFFMAN ENCODING ────────────────────────────────────────────────────────
#   Redundancy exploited: some values appear far more often than others
#   How: build a frequency table, then assign shorter bit codes to common
#        values and longer codes to rare ones. Like morse code — common letters
#        get short codes (E = "."), rare ones get long codes (Z = "--..").
#        Requires storing the code table alongside the data so it can be reversed.
#   Lossy: no
#   Real world: ZIP, PNG, JPEG all use Huffman as a final stage
#
# ── 5. LZ77 / LZ78 / LZW (used by ZIP and PNG) ────────────────────────────────
#   Redundancy exploited: repeated sequences anywhere in the data (not just runs)
#   How: maintain a sliding window of recent data. When a sequence is seen again,
#        replace it with a (distance back, length) pointer instead of the data.
#        Example: "ABCABC" → "ABC" + (go back 3, copy 3)
#   Lossy: no
#   Real world: ZIP, PNG's DEFLATE, GIF
#
# ── 6. DELTA ENCODING ──────────────────────────────────────────────────────────
#   Redundancy exploited: values that change slowly (small differences)
#   How: instead of storing [200, 202, 198, 201], store the first value then
#        only the differences: [200, +2, -4, +3]. Differences are small numbers
#        which Huffman can then compress very efficiently.
#   Lossy: no
#   Real world: PNG uses delta encoding (called "filtering") before DEFLATE
#
# ==============================================================================
# RECONSTRUCTION — HOW WE GET BACK TO AN IMAGE ARRAY
# ==============================================================================
#
# At every stage we store exactly what we need to reverse the step.
# Think of it like a set of instructions that can be played forwards or backwards:
#
#   COMPRESS                          DECOMPRESS
#   ────────────────────────────────────────────
#   channel_data                      packed_rle
#       ↓ grouping                        ↓ RLE decode   → label_list (flat)
#   label_map (H×W IDs)                   ↓ reshape      → label_map (H×W)
#       ↓ bit packing                     ↓ unpack bits  → label_map (uint8)
#   packed (half the bytes)               ↓ palette lookup
#       ↓ RLE encode                  reconstructed channel (H×W uint8)
#   [(id, count), (id, count)...]
#
# The palette is the KEY that connects group IDs back to pixel values.
# Without it, you'd know WHICH group each pixel belongs to but not WHAT
# colour that group represents.
#
# palette lookup (reconstruct_channel) works like this:
#   palette = [200, 50, 10, 180, ...]   ← index = group ID, value = pixel value
#   label_map = [[0, 0, 1, 2],          ← every cell holds a group ID
#                [0, 3, 1, 0], ...]
#   palette[label_map] →                ← NumPy replaces each ID with its value
#            [[200, 200, 50,  10],
#             [200, 180, 50, 200], ...]  ← fully reconstructed channel
#
# ==============================================================================


# ==============================================================================
# PART 1 — GROUPING
# Scan a channel and assign every pixel a small integer group ID.
# Pixels whose values are within `tolerance` of each other share a group.
# ==============================================================================

def channel_to_label_map(
    image: np.ndarray,
    channel: int,
    tolerance: int = 10
) -> tuple[np.ndarray, np.ndarray]:

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array (H, W, C), got shape {image.shape}")
    if not (0 <= channel < image.shape[2]):
        raise ValueError(f"Channel {channel} out of range")

    # Cast to int16 so that subtraction between uint8 values can't underflow
    # (e.g. 5 - 200 = -195 in int16, but wraps to 61 in uint8 — wrong!)
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
# PART 2 — BIT PACKING
# If we have ≤16 groups we only need 4 bits per ID, not 8.
# Pack two IDs per byte using the high and low nibble of each byte.
# ==============================================================================

def pack_label_map(label_map: np.ndarray, n_groups: int) -> np.ndarray:
    bits_needed = max(1, math.ceil(math.log2(n_groups + 1)))

    if bits_needed <= 4:
        flat = label_map.flatten().astype(np.uint8)
        if len(flat) % 2 != 0:
            flat = np.append(flat, 0)   # pad so every pixel has a pair
        # Even pixel → low nibble (bits 0-3)
        # Odd pixel  → high nibble (bits 4-7), shifted left by 4
        packed = (flat[1::2] << 4) | flat[0::2]
        return packed.astype(np.uint8)

    return label_map.astype(np.uint8)


def unpack_label_map(
    packed: np.ndarray,
    original_shape: tuple,
    n_groups: int
) -> np.ndarray:
    bits_needed = max(1, math.ceil(math.log2(n_groups + 1)))
    H, W = original_shape

    if bits_needed <= 4:
        low  = (packed & 0x0F)          # mask top 4 bits → even pixels
        high = (packed >> 4) & 0x0F     # shift down top 4 bits → odd pixels
        flat = np.empty(len(low) + len(high), dtype=np.uint8)
        flat[0::2] = low
        flat[1::2] = high
        return flat[:H * W].reshape(H, W)

    return packed.reshape(H, W)


# ==============================================================================
# PART 3 — RUN-LENGTH ENCODING (RLE)
#
# The label map is scanned row by row as a flat sequence of group IDs.
# Consecutive identical IDs are collapsed into (id, count) pairs.
#
# Example flat label map row:  [0, 0, 0, 0, 1, 1, 3, 0, 0]
# RLE encoded:                 [(0,4), (1,2), (3,1), (0,2)]
#
# To reconstruct, each (id, count) pair is expanded back:
#   (0,4) → [0, 0, 0, 0]
#   (1,2) → [1, 1]
#   etc.
# Then the flat list is reshaped back to (H, W).
#
# We store the RLE as a flat NumPy array of alternating [id, count, id, count...]
# so it's a single contiguous block of memory rather than a list of tuples.
# ==============================================================================

def rle_encode(packed: np.ndarray) -> np.ndarray:
    """
    Encodes a 1D array of uint8 values using RLE.
    Output is a flat uint16 array: [val, count, val, count, ...]
    uint16 for counts because a run could be longer than 255 pixels.
    """
    flat = packed.flatten()
    if len(flat) == 0:
        return np.array([], dtype=np.uint16)

    # Find where the value changes — those are the run boundaries
    # np.diff returns the difference between consecutive elements
    # nonzero() gives us the indices where the difference is not zero
    change_points = np.concatenate(([0], np.where(np.diff(flat) != 0)[0] + 1, [len(flat)]))

    values = flat[change_points[:-1]]                       # value at start of each run
    counts = np.diff(change_points).astype(np.uint16)       # length of each run

    # Interleave values and counts: [v0, c0, v1, c1, v2, c2, ...]
    encoded = np.empty(len(values) * 2, dtype=np.uint16)
    encoded[0::2] = values
    encoded[1::2] = counts
    return encoded


def rle_decode(encoded: np.ndarray, original_length: int) -> np.ndarray:
    """
    Reverses rle_encode. Expands (value, count) pairs back into a flat array,
    then the caller reshapes it to (H, W).
    """
    values = encoded[0::2].astype(np.uint8)   # every even index is a value
    counts = encoded[1::2]                    # every odd index is a count

    # np.repeat(values, counts) expands each value by its count:
    # np.repeat([0, 1, 3], [4, 2, 1]) → [0, 0, 0, 0, 1, 1, 3]
    return np.repeat(values, counts)[:original_length]


# ==============================================================================
# PART 4 — RECONSTRUCTION
# palette[label_map] uses NumPy fancy indexing:
# every group ID in label_map is replaced with its corresponding pixel value
# from the palette array in a single vectorised operation.
# ==============================================================================

def reconstruct_channel(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[label_map].astype(np.uint8)


# ==============================================================================
# PART 5 — SIZE REPORTING
# ==============================================================================

def print_size_report(img, channel, label_map, palette, packed, encoded):
    raw_bytes    = img[:, :, channel].nbytes
    label_bytes  = label_map.nbytes + palette.nbytes
    packed_bytes = packed.nbytes    + palette.nbytes
    rle_bytes    = encoded.nbytes   + palette.nbytes

    print(f"\n{'='*50}")
    print(f"  Groups formed            : {len(palette)}")
    print(f"  Raw channel              : {raw_bytes   / 1024:.2f} KB  (1.00x)")
    print(f"  Label map (uint8)        : {label_bytes  / 1024:.2f} KB  ({label_bytes / raw_bytes:.2f}x)")
    print(f"  Bit-packed               : {packed_bytes / 1024:.2f} KB  ({packed_bytes / raw_bytes:.2f}x)")
    print(f"  Bit-packed + RLE         : {rle_bytes    / 1024:.2f} KB  ({rle_bytes / raw_bytes:.2f}x)")
    print(f"{'='*50}\n")


# ==============================================================================
# PART 6 — SIDE-BY-SIDE COMPARISON
# ==============================================================================

def side_by_side(original: np.ndarray, reconstructed: np.ndarray) -> Image.Image:
    H, W, _ = original.shape
    canvas = np.ones((H, W * 2 + 4, 3), dtype=np.uint8) * 255
    canvas[:, :W,     :] = original
    canvas[:, W + 4:, :] = reconstructed
    return Image.fromarray(canvas)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    img       = np.array(Image.open("compression example/picel world wallppar.png"))
    tolerance = 15
    new_img   = np.zeros_like(img)

    for i in range(3):
        # Step 1 — group similar values into a label map
        label_map, palette = channel_to_label_map(img, channel=i, tolerance=tolerance)

        # Step 2 — bit pack: 2 group IDs per byte (only helps when groups ≤ 16)
        packed = pack_label_map(label_map, n_groups=len(palette))

        # Step 3 — RLE: collapse runs of identical IDs into (value, count) pairs
        encoded = rle_encode(packed)

        if i == 0:
            print_size_report(img, i, label_map, palette, packed, encoded)

        # ── Reconstruct ───────────────────────────────────────────────────────
        # Step 3 reversed: expand (value, count) pairs back into a flat array
        decoded_flat = rle_decode(encoded, original_length=packed.size)

        # Step 2 reversed: unpack nibbles back into one-ID-per-element
        recovered_labels = unpack_label_map(decoded_flat.reshape(packed.shape),
                                            label_map.shape, len(palette))

        # Step 1 reversed: replace every group ID with its palette value
        new_img[:, :, i] = reconstruct_channel(recovered_labels, palette)

    # Show original (left) and reconstructed (right) side by side
    side_by_side(img, new_img).show()