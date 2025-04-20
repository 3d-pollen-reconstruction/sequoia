# data/datamodule.py
from torch.utils.data import DataLoader, Subset
import lightning.pytorch as pl  # Use consistent import here
from sklearn.model_selection import KFold
from .pollen_dataset import PollenDataset

class PollenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_transforms=None,
        mesh_transforms=None,
        batch_size=32,
        n_splits=5,
        fold_idx=0,
        num_workers=4,
        seed=42
    ):
        super().__init__()
        self.image_transforms = image_transforms
        self.mesh_transforms = mesh_transforms
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.full_dataset = PollenDataset(
            image_transforms=self.image_transforms,
            mesh_transforms=self.mesh_transforms
        )
        
        indices = list(range(len(self.full_dataset)))
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        splits = list(kfold.split(indices))
        train_idx, val_idx = splits[self.fold_idx]
        
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
