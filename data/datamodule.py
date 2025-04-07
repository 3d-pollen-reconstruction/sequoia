from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl

from .pollen_dataset import PollenDataset

class PollenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_transforms=None,
        mesh_transforms=None,
        batch_size=8,
        n_splits=5,
        fold_idx=0,
        num_workers=4,
        seed=42
    ):
        """
        Args:
            image_transforms: Transformations to apply on the images.
            mesh_transforms: Transformations to apply on the meshes.
            batch_size: Batch size for the DataLoader.
            n_splits: Number of folds for cross-validation.
            fold_idx: Which fold to use as the validation set.
            num_workers: Number of workers for data loading.
            seed: Seed for reproducibility.
        """
        super().__init__()
        self.image_transforms = image_transforms
        self.mesh_transforms = mesh_transforms
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        # Instantiate the full dataset using your preprocessing output.
        self.full_dataset = PollenDataset(
            image_transforms=self.image_transforms,
            mesh_transforms=self.mesh_transforms
        )
        
        # Create a list of indices for the full dataset.
        indices = list(range(len(self.full_dataset)))
        
        # Use KFold to split indices.
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        splits = list(kfold.split(indices))
        
        # Extract the training and validation indices for the given fold.
        train_idx, val_idx = splits[self.fold_idx]
        
        # Create Subset objects for train and validation splits.
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
