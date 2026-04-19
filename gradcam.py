import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from transformers import CLIPImageProcessor

from config import MODEL_NAME, LABEL_NAMES


class CLIPGradCAM:
    """GradCAM for CLIP's vision transformer.

    Hooks into the last encoder layer to capture activations and gradients,
    then projects them onto the spatial patch grid to produce a heatmap
    showing which image regions influenced the prediction.

    CLIP ViT-B/32 with 224x224 input has patch_size=32, giving a 7x7 grid
    of 49 patches + 1 CLS token = 50 tokens total.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hook into the last vision encoder layer
        target_layer = model.clip.vision_model.encoder.layers[-1]
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def remove_hooks(self):
        """Remove registered hooks to prevent accumulation."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def _save_activations(self, module, input, output):
        # CLIPEncoderLayer returns a tuple (hidden_states, ...) in older
        # transformers and the tensor directly in newer versions — handle both.
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output  # (batch, num_tokens, hidden_dim)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (batch, num_tokens, hidden_dim)

    def generate(self, pixel_values, target_class=None):
        """Generate a GradCAM heatmap for the given image.

        Args:
            pixel_values: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class to explain (0=Real, 1=Fake). If None, uses predicted class.

        Returns:
            heatmap: Normalized heatmap as numpy array (7, 7)
            predicted_class: The model's prediction
            confidence: Softmax probability of the predicted class
        """
        self.model.eval()
        pixel_values = pixel_values.requires_grad_(True)

        # Forward pass
        logits = self.model(pixel_values)
        probs = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass on target class score
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Compute GradCAM: weight each token's features by their gradient
        # Skip CLS token (index 0), keep only patch tokens
        gradients = self.gradients[0, 1:]   # (49, hidden_dim)
        activations = self.activations[0, 1:]  # (49, hidden_dim)

        # Global average pooling of gradients gives importance weights per feature
        weights = gradients.mean(dim=0)  # (hidden_dim,)

        # Weighted combination of activations
        cam = (activations * weights).sum(dim=1)  # (49,)

        # ReLU — only keep positive contributions
        cam = torch.relu(cam)

        # Reshape to spatial grid (7x7 for ViT-B/32 with 224x224 input)
        grid_size = int(cam.shape[0] ** 0.5)
        assert grid_size * grid_size == cam.shape[0], (
            f"Patch count {cam.shape[0]} is not a perfect square"
        )
        heatmap = cam.reshape(grid_size, grid_size).detach().cpu().numpy()

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, predicted_class, confidence


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay a GradCAM heatmap on the original image.

    Args:
        image: PIL Image
        heatmap: Numpy array (7, 7) with values in [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)

    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BICUBIC
        )
    ) / 255.0

    # Apply colormap
    colormap = cm.jet(heatmap_resized)[:, :, :3]  # drop alpha channel

    # Blend with original image — scale alpha by heatmap intensity so
    # low-activation areas stay clear and only hot regions get colored
    image = image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    alpha_mask = heatmap_resized[..., np.newaxis] * alpha
    blended = (1 - alpha_mask) * image_np + alpha_mask * colormap
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))


_processor = None


def _get_processor():
    """Lazy-load and cache the CLIP processor."""
    global _processor
    if _processor is None:
        _processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    return _processor


def explain_image(model, image, device="cpu"):
    """Full pipeline: take a PIL image, return prediction + heatmap overlay.

    Args:
        model: Trained CLIPImageClassifier
        image: PIL Image
        device: torch device

    Returns:
        overlay: PIL Image with GradCAM heatmap
        label: Predicted label string ("Real" or "Fake")
        confidence: Prediction confidence (0-1)
    """
    processor = _get_processor()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    gradcam = CLIPGradCAM(model)
    try:
        heatmap, pred_class, confidence = gradcam.generate(pixel_values)
    finally:
        gradcam.remove_hooks()

    overlay = overlay_heatmap(image.convert("RGB"), heatmap)
    label = LABEL_NAMES[pred_class]

    return overlay, label, confidence
