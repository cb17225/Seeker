import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPProcessor
from PIL import Image
import kagglehub

from config import MODEL_NAME, KAGGLE_DATASET, BATCH_SIZE, SEED


class CIFAKEDataset(Dataset):
    """Loads CIFAKE images from a directory with REAL/ and FAKE/ subdirectories.

    Each item returns preprocessed pixel values and a label (0=Real, 1=Fake).
    """

    def __init__(self, root_dir, processor):
        self.processor = processor
        self.samples = []

        for label, class_name in enumerate(["REAL", "FAKE"]):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise ValueError(f"Expected directory not found: {class_dir}")
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, SyntaxError) as e:
            raise RuntimeError(f"Corrupted image: {path}") from e
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=BATCH_SIZE, val_split=0.1):
    """Download CIFAKE from Kaggle and return train/val/test DataLoaders.

    Splits the training set into train (90%) and val (10%) so the test set
    stays fully held out for final evaluation.
    """
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    train_full = CIFAKEDataset(os.path.join(path, "train"), processor)
    test_set = CIFAKEDataset(os.path.join(path, "test"), processor)

    # Split training data into train + validation
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
