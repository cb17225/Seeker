import numpy as np
import pytest
from PIL import Image

from forensics import compute_spectrum, multi_crop_predict
from gradcam import overlay_heatmap


# ---- Smoke tests ----

def test_smoke_imports():
    """All modules load without errors."""
    import config
    import model
    import dataset
    import forensics
    import gradcam
    import evaluate
    import train
    import app


# ---- FFT Spectrum ----

class TestComputeSpectrum:
    def test_output_type_and_mode(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = compute_spectrum(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_preserves_dimensions(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        result = compute_spectrum(img)
        assert result.size == (200, 100)

    def test_grayscale_input(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
        result = compute_spectrum(img)
        assert isinstance(result, Image.Image)


# ---- GradCAM Overlay ----

class TestOverlayHeatmap:
    def test_output_type_and_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        heatmap = np.random.rand(7, 7).astype(np.float32)
        result = overlay_heatmap(img, heatmap)
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)

    def test_zero_heatmap_preserves_image(self):
        original = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(original)
        heatmap = np.zeros((7, 7), dtype=np.float32)
        result = overlay_heatmap(img, heatmap)
        np.testing.assert_allclose(np.array(result), original, atol=1)

    def test_non_square_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        heatmap = np.random.rand(7, 7).astype(np.float32)
        result = overlay_heatmap(img, heatmap)
        assert result.size == (200, 100)


# ---- Multi-Crop Consistency ----

class TestMultiCropPredict:
    def test_rejects_tiny_image(self):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="too small"):
            multi_crop_predict(img, model=None, preprocess_fn=None)
