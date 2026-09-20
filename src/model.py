import torch
from torch import nn
from transformers.models.dinov3_vit import DINOv3ViTModel


class DinoV3Classifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        self.backbone = DINOv3ViTModel.from_pretrained(model_id)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x.last_hidden_state[:, 0, :])
