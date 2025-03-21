### start copy from https://github.com/pytorch/vision/blob/8ea4772e97bc11b2cfee48a415e7df8cd17fa682/gallery/transforms/helpers.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    """
    Visualizes images and their corresponding annotations (e.g., bounding boxes, masks).

    Args:
        imgs (list or list of lists): A list of images or a 2D list of images.
            Each image can be:
                - A single image tensor (C, H, W).
                - A tuple (image_tensor, target), where:
                    - image_tensor: The image tensor (C, H, W).
                    - target: A dictionary containing annotations, such as:
                        - "boxes": Bounding boxes (Tensor[N, 4]) or None.
                        - "masks": Segmentation masks (Tensor[N, H, W]) or None.
        row_title (list, optional): Titles for each row in the grid.
        **imshow_kwargs: Additional keyword arguments for `imshow`.

    Returns:
        numpy.ndarray: The plotted image in BGR format.
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(
                    img,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=0.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    ### end copy from https://github.com/pytorch/vision/blob/8ea4772e97bc11b2cfee48a415e7df8cd17fa682/gallery/transforms/helpers.py
    fig = plt.gcf()
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_from_plot = image_from_plot.reshape((h, w, 4))

    # ARGB -> BGR
    image_bgr = image_from_plot[:, :, [3, 2, 1]]
    return image_bgr
