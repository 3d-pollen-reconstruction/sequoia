import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from .utils import make_splits_from_json

load_dotenv()


class PollenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        n_images: int = 2,
        image_transforms=None,
        include_augmentations: bool = True,
    ):
        super().__init__()
        self.data_dir = os.getenv("DATA_DIR_PATH")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_transforms = image_transforms
        self.include_augmentations = include_augmentations
        self.n_images = n_images

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Called by Lightning:
          - before .fit():   stage="fit"
          - before .test():  stage="test" (or "validate")
        """
        split_json_path = os.path.join(self.data_dir, "splits.json")

        train_ds, val_ds, test_ds = make_splits_from_json(
            split_json_path=split_json_path,
            image_transforms=self.image_transforms,
            n_images=self.n_images,
            include_augmentations=self.include_augmentations,
        )

        if stage in (None, "fit"):
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage in (None, "test", "fit"):
            self.test_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
