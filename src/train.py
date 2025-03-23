# https://zhuanlan.zhihu.com/p/408610877
import argparse
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast  # 新增 AMP 支援
from torch.amp.grad_scaler import GradScaler  # 新增 AMP 支援

from .utils import dice_score
from .oxford_pet import load_dataset
from .util.logging import log, job_progress
from .models.unet import Unet
from .models.resnet34_unet import Resnet34Unet


def train(
    model: nn.Module,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    dataloader_train: torch.utils.data.DataLoader,
    checkpoint_read_file: str | None = None,
    checkpoint_save_file: str = "UNet_{epoch}.pt",
    checkpoint_interval: int = 1,
):
    # Setup optimizer, loss function, learning rate scheduler...
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 1
    scaler = GradScaler()  # 使用 AMP 的梯度縮放器

    # If a checkpoint is provided, load the model and optimizer states
    if checkpoint_read_file:
        checkpoint = torch.load(
            checkpoint_read_file, map_location=device, weights_only=True
        )
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 確保 optimizer 的狀態也在正確的裝置上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    log.info(
        f"""Starting training:
        epochs:           {epochs}
        batch_size:       {batch_size}
        learning_rate:    {learning_rate}
        device:           {device}
        train start from: {start_epoch} {"" if checkpoint_read_file is None else f"(checkpoint: {checkpoint_read_file})"}
        save interval:    {checkpoint_interval}
    """
    )

    model.train()
    model.to(device)
    progress = job_progress()
    prog_epoch = progress.add_task("Epoch", total=epochs)
    with progress:
        for epoch in range(start_epoch, start_epoch + epochs):
            batch_epoch = progress.add_task("Batch", total=len(dataloader_train))
            for batch_idx, batch in enumerate(dataloader_train):
                with torch.set_grad_enabled(True), autocast(device.type):  # 啟用 AMP
                    images = batch[:, :-1, ...].to(device)
                    masks = batch[:, -1:, ...].to(device)
                    pred = model(images)
                    loss = criterion(pred, masks)
                    loss += dice_score(torch.sigmoid(pred), masks)  # 使用 torch.sigmoid

                optimizer.zero_grad()
                scaler.scale(loss).backward()  # 使用 AMP 梯度縮放
                scaler.step(optimizer)
                scaler.update()
                if batch_idx % 10 == 0:
                    log.info(
                        f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(dataloader_train.dataset)} ({100. * batch_idx / len(dataloader_train):.0f}%)]\tLoss: {loss.item():.6f}"
                    )
                progress.advance(batch_epoch)
            progress.remove_task(batch_epoch)
            scheduler.step()  # not yet used
            if (
                len(checkpoint_save_file) > 0
                and checkpoint_save_file.endswith(".pt")
                and epoch % checkpoint_interval == 0
            ):
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{path.dirname(__file__)}/../saved_models/"
                    + checkpoint_save_file.format(epoch=epoch),
                )
            progress.advance(prog_epoch)
        progress.remove_task(prog_epoch)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--data_path", type=str, help="path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=1e-5, help="learning rate"
    )
    # custom options below
    parser.add_argument(
        "--model",
        choices=["unet", "resnet34_unet"],
        default="unet",
        help="model to train",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint file to resume training from",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="interval for saving checkpoints",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device {device}")

    if args.model == "unet":
        model = Unet(num_classes=1)
    elif args.model == "resnet34_unet":
        model = Resnet34Unet(num_classes=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    dataset_train = load_dataset(args.data_path, "train")
    dataset_val = load_dataset(args.data_path, "valid")
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    train(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        dataloader_train=dataloader_train,
        checkpoint_read_file=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )
