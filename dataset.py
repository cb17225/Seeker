import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset

from config import MODEL_NAME, DATASET_NAME, BATCH_SIZE, SEED


class CIFAKEDataset(Dataset):
    """Wraps the CIFAKE HuggingFace dataset for use with a CLIP model.

    Each item returns preprocessed pixel values and a label (0=Real, 1=Fake).
    """

    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=BATCH_SIZE, val_split=0.1):
    """Load CIFAKE from HuggingFace and return train/val/test DataLoaders.

    Splits the training set into train (90%) and val (10%) so the test set
    stays fully held out for final evaluation.
    """
    dataset = load_dataset(DATASET_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Split training data into train + validation
    split = dataset["train"].train_test_split(test_size=val_split, seed=SEED)

    train_set = CIFAKEDataset(split["train"], processor)
    val_set = CIFAKEDataset(split["test"], processor)
    test_set = CIFAKEDataset(dataset["test"], processor)

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
