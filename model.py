import torch.nn as nn
from transformers import CLIPModel

from config import MODEL_NAME, NUM_LABELS


class CLIPImageClassifier(nn.Module):
    """CLIP ViT-B/32 vision encoder with a binary classification head.

    Uses the 512-dim projected embedding space (via visual_projection)
    rather than the raw 768-dim pooler output, since CLIP was trained
    to be discriminative in the projected space.
    """

    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output              # (batch, 768)
        image_embeds = self.clip.visual_projection(pooled_output)  # (batch, 512)
        logits = self.classifier(image_embeds)
        return logits
