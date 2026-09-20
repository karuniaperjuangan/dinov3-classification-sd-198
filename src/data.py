from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import torch
from datasets import ClassLabel, Dataset, Features, Image
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from torchvision.transforms import v2 as transforms


train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.RandomAffine(degrees=(-45, 45), scale=(0.8, 1.2), translate=(0.2, 0.2)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_sd198(dataset_id):
    archive = hf_hub_download(dataset_id, "sd-198.zip", repo_type="dataset")
    extract_dir = Path(archive).parent / "sd-198-extracted"
    marker = extract_dir / ".complete"
    with ZipFile(archive) as zip_file:
        classes = [line.split(maxsplit=1)[1] for line in zip_file.read("sd-198/classes.txt").decode().splitlines()]
        images = [line.split(maxsplit=1)[1] for line in zip_file.read("sd-198/images.txt").decode().splitlines()]
        if not marker.exists():
            for path in tqdm(images, desc="Extracting SD-198", unit="image"):
                zip_file.extract(f"sd-198/images/{path}", extract_dir)
            marker.touch()

    assert len(classes) == 198 and len(images) == 6584
    image_dir = extract_dir / "sd-198" / "images"
    return Dataset.from_dict(
        {
            "image": [str(image_dir / path) for path in images],
            "label": [PurePosixPath(path).parent.name for path in images],
        },
        features=Features({"image": Image(), "label": ClassLabel(names=classes)}),
    )


def apply_transform(batch, transform):
    batch["image"] = [transform(image.convert("RGB")) for image in batch["image"]]
    return batch
