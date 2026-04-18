import numpy as np
import torch
import matplotlib.cm as cm
from PIL import Image

from config import LABEL_NAMES


def compute_spectrum(image):
    """Compute the 2D FFT log power spectrum of an image.

    AI-generated images often exhibit periodic grid artifacts from
    upsampling or unnatural frequency energy distributions that are
    invisible in pixel space but visible in the frequency domain.

    Args:
        image: PIL Image

    Returns:
        PIL Image of the colorized log power spectrum
    """
    gray = np.array(image.convert("L"), dtype=np.float64)

    # Hanning window reduces spectral leakage from image edges
    h_win = np.hanning(gray.shape[0])
    w_win = np.hanning(gray.shape[1])
    windowed = gray * np.outer(h_win, w_win)

    # 2D FFT, center zero-frequency, log scale
    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    spectrum = np.log1p(np.abs(f_shift))

    # Normalize to [0, 1]
    lo, hi = spectrum.min(), spectrum.max()
    spectrum = (spectrum - lo) / (hi - lo + 1e-8)

    # Colorize for visual clarity
    colored = cm.inferno(spectrum)[:, :, :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def multi_crop_predict(image, model, preprocess_fn, device="cpu"):
    """Classify the full image and 5 spatial crops independently.

    A real photo should classify consistently across regions.
    Disagreement between crops may indicate localized editing,
    inpainting, or splicing that whole-image classification misses.

    Args:
        image: PIL Image (RGB)
        model: CLIPImageClassifier
        preprocess_fn: callable(PIL Image) -> pixel_values tensor
        device: torch device

    Returns:
        list of dicts with keys Region, Real, Fake, Prediction
    """
    w, h = image.size
    if min(w, h) < 32:
        raise ValueError("Image too small for multi-crop analysis")
    crop_size = min(w, h) // 2

    regions = [("Full Image", image)]

    cx, cy = w // 2, h // 2
    half = crop_size // 2
    crops = [
        ("Center", (cx - half, cy - half, cx + half, cy + half)),
        ("Top-Left", (0, 0, crop_size, crop_size)),
        ("Top-Right", (w - crop_size, 0, w, crop_size)),
        ("Bottom-Left", (0, h - crop_size, crop_size, h)),
        ("Bottom-Right", (w - crop_size, h - crop_size, w, h)),
    ]
    for name, box in crops:
        regions.append((name, image.crop(box)))

    results = []
    for name, region in regions:
        pixel_values = preprocess_fn(region.convert("RGB")).to(device)
        with torch.inference_mode():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)
        pred = LABEL_NAMES[probs.argmax(dim=1).item()]
        results.append({
            "Region": name,
            "Real": probs[0, 0].item(),
            "Fake": probs[0, 1].item(),
            "Prediction": pred,
        })

    return results
