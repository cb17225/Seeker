import torch
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor

from config import MODEL_NAME, LABEL_NAMES, SAVE_DIR, DEVICE
from model import CLIPImageClassifier
from gradcam import explain_image


# ---------- Load model ----------

def load_model():
    """Load trained model from checkpoint."""
    model = CLIPImageClassifier()
    checkpoint_path = f"{SAVE_DIR}/model.pth"
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


# ---------- Inference functions ----------

def predict(image):
    """Classify an image as Real or Fake."""
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1)

    return {name: probs[0, i].item() for i, name in enumerate(LABEL_NAMES)}


def explain(image):
    """Generate GradCAM heatmap showing which regions influenced the prediction."""
    overlay, label, confidence = explain_image(model, image, device=DEVICE)
    caption = f"{label} ({confidence:.1%})"
    return overlay, caption


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
            predict_btn.click(fn=predict, inputs=predict_input, outputs=predict_output)

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
            )

if __name__ == "__main__":
    demo.launch()
