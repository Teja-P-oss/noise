from __future__ import annotations
import os
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, List, Optional, Union

from dithering_algorithms import (
    INPUT_BIT_DEPTH,
    MAX_VAL,
    DTYPE_IMG,
    LFSR,
    create_input_gradient_image,
    srled_dither,
    random_dither,
    ordered_dither,
    truncate,
    rounding
)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def analyze_patterns(img: np.ndarray) -> Dict[str, int]:
    """Re‑use helper from the original runner (unchanged)."""
    data = img.astype(np.int64)
    h, w = data.shape
    res: Dict[str, int] = {}
    col_sums = data.sum(axis=0) if w else np.array([])
    if col_sums.size:
        mean = int(col_sums.mean())
        res["vert_line_metric"] = int(np.sqrt(((col_sums - mean) ** 2).mean()))
    else:
        res["vert_line_metric"] = 0
    row_sums = data.sum(axis=1) if h else np.array([])
    if row_sums.size:
        mean_r = int(row_sums.mean())
        res["horiz_line_metric"] = int(np.sqrt(((row_sums - mean_r) ** 2).mean()))
    else:
        res["horiz_line_metric"] = 0
    res["texture_col_diff_metric"] = int(np.abs(np.diff(col_sums)).mean()) if col_sums.size > 1 else 0
    res["texture_row_diff_metric"] = int(np.abs(np.diff(row_sums)).mean()) if row_sums.size > 1 else 0
    return res


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean‑square error computed in float32 for precision."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))

def run_dithering_comparison(
    *,
    image_height: int = 512,
    image_width: int = 512,
    reduction_bits_list: Optional[List[int]] = None,
    dump_dir: Union[str, Path] = "outputs_bmp",
) -> None:

    reduction_bits_list = reduction_bits_list or [2,4]
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    lfsr_seed = 0xD3ADBEEF
    lfsr_taps = (32, 22, 2, 1)
    noise_lfsr = LFSR(lfsr_seed, lfsr_taps)
    dist_lfsr = LFSR(lfsr_seed + 1, lfsr_taps)

    for is_rgb in [True]:  # grayscale only for now – mirror original
        input_image, image_width, image_height = create_input_gradient_image(image_height, image_width, is_rgb=is_rgb)
        img_type = "RGB" if is_rgb else "Grayscale"
        channels = 3 if is_rgb else 1
        print(f"\n=== {img_type} ({image_height}×{image_width}) - 10-bit input ===")

        # also save the *source* once (avoid repeats across reductions)
        src_fname = dump_dir / f"Original_{img_type.lower()}_{image_height}x{image_width}.bmp"
        if not src_fname.exists():
            _write_bmp(src_fname, input_image)

        for rbits in reduction_bits_list:
            out_bits = INPUT_BIT_DEPTH - rbits
            print(f"\n-- Reduction: {rbits} bits  ⇒  {out_bits}-bit output --")

            algos: Dict[str, np.ndarray | None] = {
                "Original": input_image,
                "Trunc": None,
                #"Rounding": None,
                "Ordered": None,
                "Random": None,
                "srled K3 S0": None,
                #"srled K3 S1": None,
            }
            psnr_res: Dict[str, float] = {}
            mse_res: Dict[str, float] = {}
            patt_res: Dict[str, Dict[str, int]] = {}

            # ---------------------------------------------------
            # Run all algorithms
            # ---------------------------------------------------
            for name in algos:
                if name == "Original":
                    continue  # already have the input

                # allocate output container per‑algo
                out_full = np.zeros_like(input_image, dtype=DTYPE_IMG)

                for ch in range(channels):
                    src = input_image[..., ch] if is_rgb else input_image
                    if name == "Trunc":
                        out = truncate(src, rbits)
                    elif name == "srled K3 S0":
                        out = srled_dither(src, rbits, noise_lfsr, dist_lfsr, noise_strength=0)
                    elif name == "srled K3 S1":
                        out = srled_dither(src, rbits, noise_lfsr, dist_lfsr, noise_strength=1)
                    elif name == "Ordered":
                        out = ordered_dither(src, rbits)
                    elif name == "Random":
                        out = random_dither(src, rbits)
                    else:
                        raise RuntimeError("Unknown algorithm label – keep list in sync")

                    if is_rgb:
                        out_full[..., ch] = out
                    else:
                        out_full = out

                algos[name] = out_full

                # --------------------------- metrics per algorithm
                if is_rgb:
                    psnr_ch = [psnr(input_image[..., c], out_full[..., c], data_range=MAX_VAL) for c in range(3)]
                    mse_ch = [mse(input_image[..., c], out_full[..., c]) for c in range(3)]
                    psnr_res[name] = sum(psnr_ch) / 3
                    mse_res[name] = sum(mse_ch) / 3

                    patt_aggr = {k: 0 for k in analyze_patterns(out_full[..., 0])}
                    for c in range(3):
                        p = analyze_patterns(out_full[..., c])
                        for k in patt_aggr:
                            patt_aggr[k] += p[k]
                    for k in patt_aggr:
                        patt_aggr[k] //= 3
                    patt_res[name] = patt_aggr
                else:
                    psnr_res[name] = psnr(input_image, out_full, data_range=MAX_VAL)
                    mse_res[name] = mse(input_image, out_full)
                    patt_res[name] = analyze_patterns(out_full)

                print(
                    f"  {name:<10}  PSNR={psnr_res[name]:6.2f} dB  "
                    f"MSE={mse_res[name]:8.2f}  "
                    f"V:{patt_res[name]['vert_line_metric']:<4} "
                    f"H:{patt_res[name]['horiz_line_metric']:<4} "
                )

                # -------------- dump BMP (including 16‑bit → 8‑bit scale)
                fname = dump_dir / f"{name.replace(' ', '_')}_RB{rbits}_{img_type.lower()}.bmp"
                _write_bmp(fname, out_full)

            # ---------------------------------------------------
            # Visual comparison (unchanged)
            # ---------------------------------------------------
            _show_comparison_fig(algos, img_type, image_height, image_width, MAX_VAL)


# -----------------------------------------------------------------------------
# Helper: write BMP, scaling ≥9‑bit sources down to 8‑bit first
# -----------------------------------------------------------------------------

def _write_bmp(path: Path, img: np.ndarray) -> None:
    """Save *img* as an 8‑bit BMP to *path* (RGB or grayscale)."""
    if img.dtype != np.uint8 or MAX_VAL > 255:
        # scale to full 0‑255 range; works for both 1‑ & 3‑channel images
        img8 = (img.astype(np.float32) * (255.0 / MAX_VAL)).round().astype(np.uint8)
    else:
        img8 = img

    iio.imwrite(path, img8, format="bmp")


def _show_comparison_fig(algos: Dict[str, np.ndarray], img_type: str, h: int, w: int, vmax: int) -> None:
    n_alg = len(algos)
    fig, axes = plt.subplots(2, n_alg, figsize=(n_alg * 3, 6))
    fig.suptitle(f"{img_type} – output comparison", fontsize=14)

    zoom = 4
    h_z, w_z = h // zoom, w // zoom
    y0, x0 = h // 2 - h_z // 2, w // 2 - w_z // 2
    sl = (slice(y0, y0 + h_z), slice(x0, x0 + w_z))

    is_rgb = algos["Original"].ndim == 3
    cmap = None if is_rgb else "gray"

    for idx, (name, im) in enumerate(algos.items()):
        display_im = im.astype(np.float32) / vmax

        axes[0, idx].imshow(display_im, cmap=cmap, vmin=0, vmax=1.0)
        axes[0, idx].set_title(name, fontsize=8)
        axes[0, idx].axis("off")

        zoom_im = display_im[sl] if not is_rgb else display_im[sl[0], sl[1], :]
        axes[1, idx].imshow(zoom_im, cmap=cmap, vmin=0, vmax=1.0, interpolation="nearest")
        axes[1, idx].set_title("zoom×4", fontsize=8)
        axes[1, idx].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_dithering_comparison()
