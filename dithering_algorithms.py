import numpy as np

INPUT_BIT_DEPTH: int = 10
MAX_VAL: int = (1 << INPUT_BIT_DEPTH) - 1
DTYPE_IMG = np.uint16

class LFSR:
    def __init__(self, seed: int, taps: tuple[int, ...]):
        self.state = seed or 1
        self.taps = taps
        self.nbits = max(self.state.bit_length(), *(taps or (8,)))

    def step(self) -> int:
        if not self.taps:
            self.state >>= 1
            return self.state
        feedback = 0
        for tap in self.taps:
            if 0 < tap <= self.nbits:
                feedback ^= (self.state >> (tap - 1)) & 1
        self.state = ((self.state >> 1) | (feedback << (self.nbits - 1))) & (
            (1 << self.nbits) - 1
        )
        return self.state

    def get_random_bits(self, n: int) -> int:
        return self.step() & ((1 << n) - 1)

BAYER_2X2 = np.array([[0, 2], [3, 1]], dtype=np.int32)
BAYER_4X4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.int32
)

def create_input_gradient_image(height: int, width: int, *, is_rgb: bool = False) -> np.ndarray:
    if is_rgb:
        img = np.zeros((height, width, 3), dtype=DTYPE_IMG)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = (x * MAX_VAL) // (width - 1) if width > 1 else 0
                img[y, x, 1] = (y * MAX_VAL) // (height - 1) if height > 1 else 0
                diag_den = width + height - 2
                img[y, x, 2] = ((x + y) * MAX_VAL) // diag_den if diag_den else 0
    else:
        img = np.zeros((height, width), dtype=DTYPE_IMG)
        for y in range(height):
            line_val = (y * 0)
            for x in range(width):
                img[y, x] = (x * MAX_VAL) // (width - 1) if width > 1 else 0
    return img

def quantize_value(value: int, reduction_bits: int) -> int:
    if not (0 <= reduction_bits <= INPUT_BIT_DEPTH):
        raise ValueError("reduction_bits must be within the input bit-depth range")
    if reduction_bits == 0:
        return np.clip(value, 0, MAX_VAL).astype(DTYPE_IMG)
    output_bits = INPUT_BIT_DEPTH - reduction_bits
    levels = 1 << output_bits
    level_idx = (value * (levels - 1) * 2 + MAX_VAL) // (MAX_VAL * 2)
    level_idx = np.clip(level_idx, 0, levels - 1)
    quantized = (level_idx * MAX_VAL * 2 + (levels - 1)) // ((levels - 1) * 2)
    return int(quantized)

def truncation_dither(img: np.ndarray, reduction_bits: int) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()
    img_int = img.astype(np.int32)
    rounding_offset = 1 << (reduction_bits - 1)
    img_int = np.clip(img_int + rounding_offset, 0, MAX_VAL)
    mask = (~((1 << reduction_bits) - 1)) & ((1 << INPUT_BIT_DEPTH) - 1)
    img_int &= mask
    return img_int.astype(DTYPE_IMG)

# Blue noise texture (4x4) - carefully designed values
BLUE_NOISE_4X4 = np.array([
    [1, -1, 1, -1],
    [-2, 2, -2, 2],
    [1, -1, 1, -1],
    [0, 0, 0, 0]
], dtype=np.int32)

# ... existing functions ...

def ased_dither(
    img: np.ndarray,
    reduction_bits: int,
    noise_lfsr: LFSR,  # Keep for interface compatibility
    dist_lfsr: LFSR,
    *,
    noise_strength: int = 1,
    lookahead_k: int = 3,
) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()
    h, w = img.shape
    out = np.zeros_like(img, dtype=DTYPE_IMG)

    coeff_k2 = [(4, 0), (3, 1), (2, 2), (1, 3)]
    coeff_k3 = [(8, 0, 0), (4, 4, 0), (4, 2, 2), (2, 4, 2)]
    coeff_k4 = [(16, 0, 0, 0), (8, 4, 2, 2), (8, 8, 0, 0), (4, 4, 4, 4)]

    if lookahead_k == 2:
        coeff_sets = coeff_k2
        denom_shift = 2
    elif lookahead_k == 3:
        coeff_sets = coeff_k3
        denom_shift = 3
    elif lookahead_k == 4:
        coeff_sets = coeff_k4
        denom_shift = 4
    else:
        raise ValueError("lookahead_k must be 3 or 4")

    future_err = np.zeros(lookahead_k, dtype=np.int32)

    for y in range(h):
        dist_lfsr.state ^= (y + 1)  # Only update distribution LFSR
        future_err.fill(0)
        for x in range(w):
            val = int(img[y, x]) + future_err[0]
            
            # Use blue noise texture for noise injection
            added_noise = BLUE_NOISE_4X4[y % 4, x % 4] * noise_strength
            
            val_n = np.clip(val + added_noise, 0, MAX_VAL)
            q_val = quantize_value(val_n, reduction_bits)
            out[y, x] = q_val
            q_err = val_n - q_val
            future_err[:-1] = future_err[1:]
            future_err[-1] = 0
            coeff = coeff_sets[dist_lfsr.get_random_bits(2)]
            for k in range(lookahead_k):
                if x + 1 + k < w:
                    future_err[k] += (q_err * coeff[k]) >> denom_shift
    return out