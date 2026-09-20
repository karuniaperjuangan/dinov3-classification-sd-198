import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from src.config import Config
from src.data import apply_transform, load_sd198, test_transform, train_transforms
from src.model import DinoV3Classifier
from src.train import EarlyStopping, run_one_epoch
from src.utils import get_device, prepare_output_paths, save_sample_predictions


def main(config_path="config.yaml"):
    config = Config.from_yaml(config_path)
    device = get_device(config.device)
    checkpoint_path, sample_path = prepare_output_paths(config)
    print("Using device:", device)

    run = None
    if config.use_wandb:
        import wandb
        run = wandb.init(
            project=config.wandb_project,
            config={
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "scheduler_step_size": config.scheduler_step_size,
            },
        )

    ds = load_sd198(config.dataset_id).train_test_split(
        test_size=config.test_size,
        seed=config.seed,
        stratify_by_column="label",
    )
    ds_train_raw = ds["train"]
    ds_test_raw = ds["test"]
    ds_train = ds_train_raw.with_transform(lambda batch: apply_transform(batch, train_transforms))
    ds_test = ds_test_raw.with_transform(lambda batch: apply_transform(batch, test_transform))

    print(len(ds_train))
    print(len(ds_test))

    train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)

    model = DinoV3Classifier(config.model_id, config.num_classes)
    summary(model, (3, 224, 224), depth=4)
    model = model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)

    for epoch in range(config.epochs):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss, val_acc = run_one_epoch(model, test_loader, optimizer, loss_fn, device, False)
        scheduler.step()
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if run:
            run.log({"epoch": epoch + 1, "train/loss": train_loss, "train/accuracy": train_acc, "validation/loss": val_loss, "validation/accuracy": val_acc, "learning_rate": scheduler.get_last_lr()[0]})

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": val_loss,
            "accuracy": val_acc,
        }

        if (epoch + 1) % config.checkpoint_steps == 0 and epoch > 0:
            torch.save(checkpoint, os.path.join(checkpoint_path, f"checkpoint_{epoch+1}.pth"))

        if val_loss < early_stopping.best_loss:
            print("Validation loss improved. Saving best model...")
            torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint_best.pth"))

        if early_stopping.step(val_loss):
            print("Early stopping triggered")
            break

    save_sample_predictions(
        model,
        ds_test_raw,
        test_transform,
        device,
        config.samples_size,
        sample_path,
        run,
    )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
