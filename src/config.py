from dataclasses import dataclass

import yaml


@dataclass
class Config:
    dataset_id: str
    model_id: str
    epochs: int
    scheduler_step_size: int
    scheduler_gamma: float
    learning_rate: float
    device: str
    checkpoint_steps: int
    samples_size: int
    batch_size: int
    test_size: float
    seed: int
    num_classes: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    use_wandb: bool
    wandb_project: str
    output_path: str
    colab_output_path: str

    @classmethod
    def from_yaml(cls, path):
        with open(path) as file:
            return cls(**yaml.safe_load(file))
