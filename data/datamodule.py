# data/datamodule.py
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from .pollen_dataset import PollenDataset


class PollenDataModule(pl.LightningDataModule):
    """
    LightningDataModule that performs a one-off random split into
    train / val / test subsets.

    Parameters
    ----------
    val_fraction : float
        Fraction **of the full dataset** that should become the validation set.
    test_fraction : float
        Fraction **of the full dataset** that should become the test set.
        (train_fraction is inferred as 1 − val_fraction − test_fraction)
    seed : int
        RNG seed to make the split reproducible.
    """

    def __init__(
        self,
        image_transforms: Optional[object] = None,
        mesh_transforms: Optional[object] = None,
        batch_size: int = 32,
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        assert 0 < val_fraction < 1 and 0 < test_fraction < 1, "fractions in (0, 1)"
        assert (val_fraction + test_fraction) < 1, \
            "val_fraction + test_fraction must be < 1"

        self.image_transforms = image_transforms
        self.mesh_transforms = mesh_transforms
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.num_workers = num_workers
        self.seed = seed

        # Will be filled in `setup`
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ---------------------------------------------------------------------
    # No downloading required, but keep the hook for completeness
    # ---------------------------------------------------------------------
    def prepare_data(self):
        pass

    # ---------------------------------------------------------------------
    # Split once, reuse the result for every stage Lightning asks for
    # ---------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            # Already split → nothing to do
            return

        full_ds = PollenDataset(
            image_transforms=self.image_transforms,
            mesh_transforms=self.mesh_transforms,
        )

        n_total = len(full_ds)
        n_test = int(n_total * self.test_fraction)
        n_val = int(n_total * self.val_fraction)
        n_train = n_total - n_val - n_test

        g = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_ds, lengths=[n_train, n_val, n_test], generator=g
        )

        # Lightning will call setup("fit") and later setup("test").
        # The three attributes above are already prepared, so we don't need
        # to branch on `stage`.

    # ---------------------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
