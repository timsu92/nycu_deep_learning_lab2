import argparse
import os

import cv2
import torch
import numpy as np

from .models.unet import Unet
from .models.resnet34_unet import Resnet34Unet
from .oxford_pet import load_dataset
from .util.logging import log, job_progress
from .utils import dice_score
from .util.visualize import plot


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        choices=["unet", "resnet_unet"],
        default="unet",
        help="Whether to use unet or resnet_unet",
    )
    parser.add_argument(
        "--checkpoint", default="MODEL.pth", help="path to the stored model weight"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"{os.path.dirname(__file__)}/../dataset/oxford-iiit-pet",
        help="path to the input data",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not os.path.isfile(args.checkpoint):
        print("Model not found at", args.checkpoint)
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    ### checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = Unet(num_classes=1) if args.model == "unet" else Resnet34Unet(in_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])

    log.info(
        f"""Inference
        model: {args.model}
        epoch: {checkpoint["epoch"]}
        device: {device}
        dataset: {args.data_path}
        """
    )

    model.to(device)
    model.eval()

    ### data
    dataset = load_dataset(args.data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=6
    )

    ### inference & dice score
    gallery = []
    dice = 0
    progress = job_progress()
    prog_epoch = progress.add_task("Test", total=len(dataloader))
    with progress:
        for i, data in enumerate(dataloader):
            imgs = data[:, :-1, ...].to(device)
            masks = data[:, -1:, ...].to(device)
            with torch.no_grad():
                pred = model(imgs)
                dice += dice_score(torch.round(torch.sigmoid(pred)), masks)
            gallery.append(
                [(img, {"boxes": None, "masks": mask}) for img, mask in zip(imgs.cpu(), pred.cpu())]
            )
            progress.advance(prog_epoch)
        progress.remove_task(prog_epoch)
    gallery = plot(
        gallery,
        row_title=np.array(
            [
                (f"{i}-{i+batch_size-1} image", f"{i}-{i+batch_size-1} mask")
                for i in range(1, len(dataset) + 1, batch_size)
            ]
        ).flatten(),
    )
    dice /= len(dataloader)
    print("dice score:", float(dice))
    cv2.imshow("", gallery)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
