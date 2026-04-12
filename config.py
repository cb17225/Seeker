import os
import torch

# ---------- Model ----------
MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_LABELS = 2
LABEL_NAMES = ["Real", "Fake"]

# ---------- Dataset ----------
KAGGLE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"

# ---------- Training ----------
BATCH_SIZE = 64
NUM_EPOCHS_FROZEN = 5
NUM_EPOCHS_UNFROZEN = 3
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 42

# ---------- Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Hugging Face Hub ----------
HUB_MODEL_ID = "cb17225/seeker-clip-cifake"

# ---------- Paths ----------
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeker-model")
