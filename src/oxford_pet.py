import os
import shutil
from typing import Literal

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = tv_tensors.Image(Image.open(image_path))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        mask = tv_tensors.Mask(mask)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(sample)

        return torch.cat([sample["image"], sample["mask"][None, ...]], dim=0)

    @staticmethod
    def _preprocess_mask(mask):
        # Original: 1: Foreground 2:Background 3: Not classified xmls/ Head bounding box
        # Modified: 0: Background 1: Foreground & bounding box
        mask[mask == 2] = 0
        mask[(mask == 1) | (mask == 3)] = 1
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        # train:validation = 8:1
        if self.mode == "train":
            filenames = [x for i, x in enumerate(filenames) if i % 9 != 0]
        elif self.mode == "valid":
            filenames = [x for i, x in enumerate(filenames) if i % 9 == 0]
        # return all for tests?
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(
    data_path="./dataset/oxford-iiit-pet",
    mode: Literal["train", "valid", "test"] = "train",
):
    transform_train = [
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(
            (572, 572), scale=(0.64, 1)
        ),  # TODO: what's the input size for ResNet34?
        transforms.RandomPerspective(0.35, 0.7),
        transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.4, hue=0.1),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    transform_test = [
        transforms.ToImage(),
        transforms.Resize((572, 572)),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    if mode != "test":
        transform = transforms.Compose(transform_train)
    else:
        transform = transforms.Compose(transform_test)

    dataset = OxfordPetDataset(data_path, mode=mode, transform=transform)

    return dataset
