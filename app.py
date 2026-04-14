import os
import threading

import torch
import gradio as gr
from PIL import Image
from transformers import CLIPImageProcessor

from config import MODEL_NAME, LABEL_NAMES, SAVE_DIR, DEVICE
from model import CLIPImageClassifier
from gradcam import explain_image

MAX_DIM = 4096
Image.MAX_IMAGE_PIXELS = 100 * 1024 * 1024  # 100 MP — blocks decompression bombs, allows modern phone photos
_explain_lock = threading.Lock()
GENERIC_ERROR = "Unable to process image. Please try a different file."


# ---------- Load model ----------

def load_model():
    """Load trained model from checkpoint."""
    checkpoint_path = os.path.join(SAVE_DIR, "model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py first."
        )
    model = CLIPImageClassifier()
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


try:
    model = load_model()
except FileNotFoundError as e:
    print(f"[Seeker] {e}")
    model = None

_processor = None


def _get_processor():
    """Lazy-load the CLIP processor so a transient HF Hub outage at startup
    doesn't prevent the Space from booting."""
    global _processor
    if _processor is None:
        _processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    return _processor


MODEL_MISSING_MSG = "Model not loaded. No checkpoint found at seeker-model/model.pth."


def _prepare_image(image):
    """Cap dimensions then convert to RGB. Order matters — thumbnail first
    so decompression-bomb images get downsampled before the full decode."""
    if max(image.size) > MAX_DIM:
        image.thumbnail((MAX_DIM, MAX_DIM))
    return image.convert("RGB")


# ---------- Inference functions ----------

def predict(image):
    """Classify an image as Real or Fake."""
    if model is None:
        raise gr.Error(MODEL_MISSING_MSG)
    try:
        image = _prepare_image(image)
        inputs = _get_processor()(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)

        return {name: probs[0, i].item() for i, name in enumerate(LABEL_NAMES)}
    except Exception as e:
        print(f"[Seeker] predict error: {e!r}")
        raise gr.Error(GENERIC_ERROR)


def explain(image):
    """Generate GradCAM heatmap showing which regions influenced the prediction."""
    if model is None:
        raise gr.Error(MODEL_MISSING_MSG)
    try:
        image = _prepare_image(image)
        with _explain_lock:
            overlay, label, confidence = explain_image(model, image, device=DEVICE)
        caption = f"{label} ({confidence:.1%})"
        return overlay, caption
    except Exception as e:
        print(f"[Seeker] explain error: {e!r}")
        raise gr.Error(GENERIC_ERROR)


# ---------- Gradio UI ----------

with gr.Blocks(title="Seeker — AI Image Detector") as demo:
    gr.Markdown("# Seeker — AI Image Detector")
    gr.Markdown(
        "Upload an image to check if it's **real** or **AI-generated**, "
        "powered by a fine-tuned CLIP ViT-B/32 model."
    )

    with gr.Tabs():
        with gr.TabItem("Predict"):
            with gr.Row():
                predict_input = gr.Image(type="pil", label="Upload Image")
                predict_output = gr.Label(num_top_classes=2, label="Prediction")
            predict_btn = gr.Button("Predict", variant="primary")
            predict_btn.click(
                fn=predict,
                inputs=predict_input,
                outputs=predict_output,
                concurrency_limit=4,
            )

        with gr.TabItem("Explain"):
            gr.Markdown(
                "GradCAM highlights which regions of the image the model "
                "focused on when making its prediction."
            )
            with gr.Row():
                explain_input = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    explain_output = gr.Image(type="pil", label="GradCAM Heatmap")
                    explain_caption = gr.Textbox(label="Prediction", interactive=False)
            explain_btn = gr.Button("Explain", variant="primary")
            explain_btn.click(
                fn=explain,
                inputs=explain_input,
                outputs=[explain_output, explain_caption],
                concurrency_limit=2,
            )

if __name__ == "__main__":
    demo.launch()
