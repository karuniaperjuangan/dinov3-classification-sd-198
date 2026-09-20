import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch


def get_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_output_paths(config):
    output_path = config.output_path
    try:
        from google.colab import drive
    except ImportError:
        pass
    else:
        drive.mount("/content/drive")
        output_path = config.colab_output_path

    checkpoint_path = os.path.join(output_path, "checkpoints")
    sample_path = os.path.join(output_path, "samples")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    return checkpoint_path, sample_path


def save_sample_predictions(model, dataset, transform, device, samples_size, sample_path, run=None):
    sample_images = dataset[:samples_size]["image"]
    sample_labels = dataset[:samples_size]["label"]
    inputs = torch.stack([transform(image.convert("RGB")) for image in sample_images]).to(device)
    model.eval()
    fig, axes = plt.subplots(1, samples_size, figsize=(20, 10))
    plt.tight_layout()

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)

        for i in range(samples_size):
            print(f"Sample {i+1}:")
            print(f"  Predicted: {predictions[i].item()}")
            print(f"  True: {sample_labels[i]}")
            print(f"  Input shape: {inputs[i].shape}")
            axes[i].imshow(sample_images[i])
            axes[i].set_title(f"Predicted: {predictions[i].item()}, True: {sample_labels[i]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(sample_path, f"sample_predictions_{timestamp}.png"))
    if run:
        import wandb
        run.log({"sample_predictions": wandb.Image(fig)})
    plt.show()
    plt.close()
