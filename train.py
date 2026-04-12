import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import (
    DEVICE, SEED, SAVE_DIR, MODEL_NAME, NUM_LABELS, LABEL_NAMES,
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


def save_resume_checkpoint(path, model, optimizer, epoch, phase, history):
    """Atomically save full training state for interruption recovery."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "history": history,
    }, tmp_path)
    os.replace(tmp_path, path)


def main():
    set_seed()
    print(f"Using device: {DEVICE}")

    # ---- Data ----
    print("Loading CIFAKE dataset...")
    train_loader, val_loader, _ = get_dataloaders()

    # ---- Model ----
    print("Loading CLIP model...")
    model = CLIPImageClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    model_config = {
        "base_model": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
    }

    # ---- Resume state ----
    resume_path = os.path.join(SAVE_DIR, "resume.pth")
    resume_data = None
    if os.path.exists(resume_path):
        print(f"Found resume checkpoint at {resume_path}, loading...")
        resume_data = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(resume_data["model_state"])
        history = resume_data["history"]
        start_phase = resume_data["phase"]
        start_epoch = resume_data["epoch"] + 1
        print(f"Resuming from Phase {start_phase}, Epoch {start_epoch + 1}")
    else:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_phase = 1
        start_epoch = 0

    # ---- Phase 1: Frozen backbone ----
    if start_phase == 1:
        print("\n--- Phase 1: Training classifier head (backbone frozen) ---")
        for param in model.clip.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
        )
        if resume_data and resume_data["phase"] == 1:
            optimizer.load_state_dict(resume_data["optimizer_state"])

        for epoch in range(start_epoch, NUM_EPOCHS_FROZEN):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS_FROZEN} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            save_resume_checkpoint(resume_path, model, optimizer, epoch, 1, history)

        save_checkpoint(model, os.path.join(SAVE_DIR, "phase1.pth"), model_config)
        start_epoch = 0  # reset for phase 2

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

    if resume_data and resume_data["phase"] == 2:
        optimizer.load_state_dict(resume_data["optimizer_state"])

    for epoch in range(start_epoch, NUM_EPOCHS_UNFROZEN):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_UNFROZEN} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        save_resume_checkpoint(resume_path, model, optimizer, epoch, 2, history)

    # ---- Save final model ----
    save_checkpoint(model, os.path.join(SAVE_DIR, "model.pth"), model_config)

    # ---- Clean up resume checkpoint on successful completion ----
    if os.path.exists(resume_path):
        os.remove(resume_path)

    # ---- Save training history ----
    history_path = os.path.join(SAVE_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
