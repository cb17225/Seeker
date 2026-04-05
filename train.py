import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import (
    DEVICE, SEED, SAVE_DIR, HUB_MODEL_ID, MODEL_NAME, NUM_LABELS, LABEL_NAMES,
    NUM_EPOCHS_FROZEN, NUM_EPOCHS_UNFROZEN,
    LR_HEAD, LR_BACKBONE, WEIGHT_DECAY,
)
from model import CLIPImageClassifier
from dataset import get_dataloaders


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, labels in tqdm(loader, desc="Training", leave=False):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for pixel_values, labels in tqdm(loader, desc="Evaluating", leave=False):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    config_path = os.path.join(os.path.dirname(path), "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Checkpoint saved to {path}")


def main():
    set_seed()
    print(f"Using device: {DEVICE}")

    # ---- Data ----
    print("Loading CIFAKE dataset...")
    train_loader, test_loader = get_dataloaders()

    # ---- Model ----
    print("Loading CLIP model...")
    model = CLIPImageClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    model_config = {
        "base_model": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
    }

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # ---- Phase 1: Frozen backbone ----
    print("\n--- Phase 1: Training classifier head (backbone frozen) ---")
    for param in model.clip.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(NUM_EPOCHS_FROZEN):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_FROZEN} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    save_checkpoint(model, os.path.join(SAVE_DIR, "phase1.pth"), model_config)

    # ---- Phase 2: Unfreeze last 2 vision encoder layers ----
    print("\n--- Phase 2: Fine-tuning last 2 encoder layers ---")
    for layer in model.clip.vision_model.encoder.layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.clip.visual_projection.parameters():
        param.requires_grad = True
    for param in model.clip.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
        {"params": model.clip.vision_model.encoder.layers[-2].parameters(), "lr": LR_BACKBONE},
        {"params": model.clip.vision_model.encoder.layers[-1].parameters(), "lr": LR_BACKBONE},
        {"params": model.clip.visual_projection.parameters(), "lr": LR_BACKBONE},
        {"params": model.clip.vision_model.post_layernorm.parameters(), "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS_UNFROZEN):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_UNFROZEN} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # ---- Save final model ----
    save_checkpoint(model, os.path.join(SAVE_DIR, "model.pth"), model_config)

    # ---- Save training history ----
    history_path = os.path.join(SAVE_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
