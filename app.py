import os
import threading

import torch
import gradio as gr
from PIL import Image
from transformers import CLIPImageProcessor

from config import MODEL_NAME, LABEL_NAMES, SAVE_DIR, DEVICE
from model import CLIPImageClassifier
from gradcam import explain_image
from forensics import compute_spectrum, multi_crop_predict

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
        image = image.copy()
        image.thumbnail((MAX_DIM, MAX_DIM))
    return image.convert("RGB")


# ---------- Inference functions ----------

def predict(image):
    """Classify an image as Real or Fake."""
    if image is None:
        raise gr.Error("Please upload an image first.")
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
    if image is None:
        raise gr.Error("Please upload an image first.")
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


def analyze_frequency(image):
    """Compute FFT power spectrum."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        image = _prepare_image(image)
        return compute_spectrum(image)
    except Exception as e:
        print(f"[Seeker] frequency error: {e!r}")
        raise gr.Error(GENERIC_ERROR)


def check_consistency(image):
    """Multi-crop consistency check."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    if model is None:
        raise gr.Error(MODEL_MISSING_MSG)
    try:
        image = _prepare_image(image)

        def preprocess(img):
            inputs = _get_processor()(images=img, return_tensors="pt")
            return inputs["pixel_values"]

        results = multi_crop_predict(image, model, preprocess, device=DEVICE)

        rows = [
            [r["Region"], f'{r["Real"]:.1%}', f'{r["Fake"]:.1%}', r["Prediction"]]
            for r in results
        ]

        full_pred = results[0]["Prediction"]
        crop_preds = [r["Prediction"] for r in results[1:]]
        real_count = crop_preds.count("Real")
        fake_count = crop_preds.count("Fake")
        summary = (
            f"Full image: {full_pred}. "
            f"Crops: {real_count} Real, {fake_count} Fake."
        )

        return rows, summary
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        print(f"[Seeker] consistency error: {e!r}")
        raise gr.Error(GENERIC_ERROR)


# ---------- Gradio UI ----------

with gr.Blocks(title="Seeker — AI Image Detector") as demo:
    gr.Markdown("# Seeker — AI Image Detector")
    gr.Markdown(
        "Upload an image to check if it's **real** or **AI-generated**, "
        "powered by a fine-tuned CLIP ViT-B/32 model."
    )
    gr.Markdown(
        "> **Limitations:** Trained on 32×32 CIFAKE images (upscaled to 224×224) — "
        "performance may vary on high-resolution inputs. "
        "AI-generated images in the training set are from **Stable Diffusion v1.4** only; "
        "newer generators (DALL·E 3, Midjourney, Flux) may not be detected as reliably. "
        "GradCAM heatmaps are coarse (7×7 grid) due to ViT-B/32's patch size."
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

        with gr.TabItem("Frequency"):
            gr.Markdown(
                "FFT power spectrum reveals frequency-domain artifacts. "
                "AI-generated images often show periodic patterns or unnatural "
                "energy distributions invisible in pixel space."
            )
            with gr.Row():
                freq_input = gr.Image(type="pil", label="Upload Image")
                freq_output = gr.Image(type="pil", label="Power Spectrum")
            freq_btn = gr.Button("Analyze", variant="primary")
            freq_btn.click(
                fn=analyze_frequency,
                inputs=freq_input,
                outputs=freq_output,
                concurrency_limit=4,
            )

        with gr.TabItem("Consistency"):
            gr.Markdown(
                "Classifies the full image and 5 spatial crops independently. "
                "Consistent predictions suggest uniform authenticity; disagreement "
                "may indicate localized editing or splicing."
            )
            with gr.Row():
                consist_input = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    consist_table = gr.Dataframe(
                        headers=["Region", "Real", "Fake", "Prediction"],
                        label="Per-Region Results",
                    )
                    consist_summary = gr.Textbox(
                        label="Agreement", interactive=False
                    )
            consist_btn = gr.Button("Check", variant="primary")
            consist_btn.click(
                fn=check_consistency,
                inputs=consist_input,
                outputs=[consist_table, consist_summary],
                concurrency_limit=2,
            )

if __name__ == "__main__":
    demo.launch()
