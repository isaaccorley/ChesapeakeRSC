import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule
from torch.utils.data import random_split
from .datasets import ChesapeakRSC
import torch


class ChesapeakRSCDataModule(NonGeoDataModule):

    mean = 0.0
    std = 255.0

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        differentiate_tree_canopy_over_roads: bool = False,
        **kwargs
    ) -> None:
        """Initialize a new DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            differentiate_tree_canopy_over_roads: Whether to separate out the different
                road classes.
            **kwargs: Additional keyword arguments passed to the
                `NonGeoDataModule` constructor.
        """
        super().__init__(ChesapeakRSC, batch_size, num_workers, **kwargs)
        self.differentiate_tree_canopy_over_roads = differentiate_tree_canopy_over_roads

        augmentations = [
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        ]

        self.train_aug = AugmentationSequential(
            *augmentations,
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = ChesapeakRSC(split="train", differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads, **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = ChesapeakRSC(split="val", differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads, **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = ChesapeakRSC(split="test", differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads, **self.kwargs)
