import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from config import DEVICE, LABEL_NAMES, SAVE_DIR
from model import CLIPImageClassifier
from dataset import get_dataloaders


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference on a DataLoader and return all labels, predictions, and probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for pixel_values, labels in tqdm(loader, desc="Evaluating"):
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_classification_report(labels, preds):
    """Print precision, recall, F1, and accuracy."""
    print(classification_report(labels, preds, target_names=LABEL_NAMES))


def plot_confusion_matrix(labels, preds, save_path=None):
    """Display a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def show_misclassified(dataset, labels, preds, probs, num_samples=10):
    """Display misclassified images with their predicted and actual labels."""
    wrong_indices = np.where(labels != preds)[0]
    if len(wrong_indices) == 0:
        print("No misclassified samples found.")
        return

    # Sort by confidence (most confidently wrong first)
    wrong_confs = probs[wrong_indices].max(axis=1)
    sorted_idx = wrong_indices[np.argsort(-wrong_confs)][:num_samples]

    cols = min(5, len(sorted_idx))
    rows = (len(sorted_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for i, idx in enumerate(sorted_idx):
        # Load the original PIL image from the file path
        path, _ = dataset.samples[int(idx)]
        image = Image.open(path).convert("RGB")
        actual = LABEL_NAMES[labels[idx]]
        predicted = LABEL_NAMES[preds[idx]]
        confidence = probs[idx].max() * 100

        axes[i].imshow(image)
        axes[i].set_title(f"True: {actual}\nPred: {predicted} ({confidence:.1f}%)", fontsize=9)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(sorted_idx), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Misclassified Samples (sorted by confidence)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    print("Loading model and data...")
    model = CLIPImageClassifier()
    model.load_state_dict(torch.load(f"{SAVE_DIR}/model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    _, _, test_loader = get_dataloaders()

    print("Running evaluation on test set...\n")
    labels, preds, probs = get_predictions(model, test_loader, DEVICE)

    print_classification_report(labels, preds)
    plot_confusion_matrix(labels, preds)
    show_misclassified(test_loader.dataset, labels, preds, probs)


if __name__ == "__main__":
    main()
