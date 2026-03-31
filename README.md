---
title: Seeker - AI Image Detector
emoji: "\U0001F50D"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
license: mit
---

# Seeker - AI Image Detector

An end-to-end machine learning tool that distinguishes between **real** and **AI-generated** images using a fine-tuned [CLIP ViT-B/32](https://openai.com/research/clip) model.

## How It Works

1. **Upload** any image via drag-and-drop or file picker
2. **Seeker** runs the image through a fine-tuned CLIP vision encoder
3. Get a **Real / Fake prediction** with a confidence score

## Model Details

| Component | Details |
|-----------|---------|
| **Architecture** | CLIP ViT-B/32 + classification head (512 → 256 → 2) |
| **Training Data** | [CIFAKE](https://huggingface.co/datasets/CIFAKE) — 60K real (CIFAR-10) + 60K AI-generated (Stable Diffusion) images |
| **Training Strategy** | Two-phase: frozen backbone (5 epochs) → partial fine-tuning of last 2 layers (3 epochs) |
| **Framework** | PyTorch + Hugging Face Transformers |

## Project Structure

```
Seeker/
├── README.md              # This file (HF Spaces metadata + docs)
├── app.py                 # Gradio inference app
├── requirements.txt       # HF Spaces dependencies
├── config.py              # Hyperparameters and paths
├── model.py               # CLIPImageClassifier architecture
├── dataset.py             # CIFAKE dataset loading
├── train.py               # Training entry point
├── evaluate.py            # Metrics, confusion matrix, error analysis
└── notebooks/
    └── explore.ipynb      # EDA, embedding visualization, training curves
```

## Training

Train on Google Colab with a free T4 GPU:

```bash
git clone https://github.com/YOUR_USERNAME/Seeker.git
cd Seeker
pip install -r requirements.txt
python train.py
```

## Limitations

- Trained on 32x32 CIFAKE images (resized to 224x224 for CLIP) — performance may vary on high-resolution images
- AI-generated images in the training set come from **Stable Diffusion v1.4** only — newer generators (DALL-E 3, Midjourney v5, Flux) may not be detected as reliably
- Binary classification only (Real vs Fake) — does not identify the specific generator

## References

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://arxiv.org/abs/2303.14126) (Bird & Lotfi, 2024)
