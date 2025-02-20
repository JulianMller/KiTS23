import logging

import monai
import torch
from monai.data import CacheDataset, DataLoader, Dataset, ImageDataset
from monai.transforms import CastToTyped, RandFlipd, RandGaussianNoised, ScaleIntensityRanged, \
    Spacingd, Orientationd, LoadImaged, Compose, EnsureChannelFirstd, CropForegroundd, \
    Resized, EnsureTyped,ToTensord, RandFlipd, RandRotate90d, RandShiftIntensityd
from monai.data.utils import pad_list_data_collate
from torch.utils.data import random_split
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from kits23.configuration.paths import TRAINING_DIR
from kits23 import TRAINING_CASE_NUMBERS


class KiTS23DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.train_data_dir = TRAINING_DIR     

    def setup(self, stage: str) -> None:
        images = [(TRAINING_DIR / f"case_{n:05d}" / "imaging.nii.gz").resolve(strict=True) for n in TRAINING_CASE_NUMBERS]
        segs = [(TRAINING_DIR / f"case_{n:05d}" / "segmentation.nii.gz").resolve(strict=True) for n in TRAINING_CASE_NUMBERS]
        
        train_images, val_images, train_labels, val_labels = train_test_split(images, segs, test_size=0.1)
        train_files = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
        val_files = [{"image": i, "label": l} for i, l in zip(val_images, val_labels)]
        
        with open(r'./models/unets/validation_cases.txt', 'w') as fp:
            for item in val_files:
                fp.write("%s\n" % item)
            print('Validation File written.')
        
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.62, 1.62, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-80,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=(160, 160, 128)),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ])

        val_transform = Compose([LoadImaged(keys=["image", "label"]),
                                 EnsureChannelFirstd(keys=["image", "label"]),
                                 Orientationd(keys=["image", "label"], axcodes="RAS"),
                                 Spacingd(keys=["image", "label"], pixdim=(1.62, 1.62, 2)),
                                 ScaleIntensityRanged(keys="image", a_min=-80, a_max=305, b_min=0.0, b_max=1.0, clip=True,),
                                 Resized(keys=["image", "label"], spatial_size=(160, 160, 128)),
                                 ToTensord(keys=["image", "label"]),
                                 EnsureTyped(keys=["image", "label"]),
                             ])

        self.train_ds = Dataset(data=train_files, transform=train_transform)
        self.val_ds = Dataset(data=val_files, transform=val_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=2, collate_fn=pad_list_data_collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, collate_fn=pad_list_data_collate)

    def test_dataloader(self):
        # no test data
        return DataLoader(self.val_ds, batch_size=1)
        

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
