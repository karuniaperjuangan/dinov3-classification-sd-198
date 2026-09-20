import torch
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_one_epoch(model, loader, optimizer, loss_fn, device, is_training):
    model.train(is_training)
    model.backbone.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0

    for batch in tqdm(
        loader,
        desc="Training" if is_training else "Validation",
        leave=False,
        position=1,
    ):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        num_correct += (outputs.argmax(dim=1) == labels).sum().item()
        num_samples += labels.size(0)

    return total_loss / num_samples, num_correct / num_samples


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler
