import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset

from config import MODEL_NAME, DATASET_NAME, BATCH_SIZE


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


def get_dataloaders(batch_size=BATCH_SIZE):
    """Load CIFAKE from HuggingFace and return train/test DataLoaders."""
    dataset = load_dataset(DATASET_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    train_set = CIFAKEDataset(dataset["train"], processor)
    test_set = CIFAKEDataset(dataset["test"], processor)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, test_loader
