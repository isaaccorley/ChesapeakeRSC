from typing import List, Optional

import kornia.augmentation as K
from torchgeo.datamodules.geo import NonGeoDataModule

from .datasets import ChesapeakeRSC

CUTOUT_CONFIGS = {
    "small": [
        K.RandomErasing(scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0.0, p=0.5),
    ],
    "medium": [
        K.RandomErasing(scale=(0.10, 0.25), ratio=(0.3, 3.3), value=0.0, p=0.5),
    ],
    "large": [
        K.RandomErasing(scale=(0.25, 0.50), ratio=(0.3, 3.3), value=0.0, p=0.75),
    ],
    "multi": [
        K.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0, p=0.7),
        K.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0, p=0.7),
        K.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0, p=0.7),
    ],
}


class ChesapeakeRSCDataModule(NonGeoDataModule):
    mean = 0.0
    std = 255.0

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        differentiate_tree_canopy_over_roads: bool = False,
        cutout: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize a new DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            differentiate_tree_canopy_over_roads: Whether to separate out the different
                road classes.
            cutout: Spatial cutout strategy for training augmentation. One of
                'small', 'medium', 'large', 'multi', or None (no cutout).
            **kwargs: Additional keyword arguments passed to the
                `NonGeoDataModule` constructor.
        """
        super().__init__(ChesapeakeRSC, batch_size, num_workers, **kwargs)
        self.differentiate_tree_canopy_over_roads = differentiate_tree_canopy_over_roads

        cutout_layers: List[K.RandomErasing] = []
        if cutout is not None:
            if cutout not in CUTOUT_CONFIGS:
                raise ValueError(
                    f"Unknown cutout strategy '{cutout}'. "
                    f"Choose from: {list(CUTOUT_CONFIGS.keys())}"
                )
            cutout_layers = CUTOUT_CONFIGS[cutout]

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            *cutout_layers,
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            keepdim=True,
            data_keys=None,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), keepdim=True, data_keys=None
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = ChesapeakeRSC(
                split="train",
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = ChesapeakeRSC(
                split="val",
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = ChesapeakeRSC(
                split="test",
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
