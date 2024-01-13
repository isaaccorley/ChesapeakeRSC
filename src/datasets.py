import torch
import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torchgeo.datasets import NonGeoDataset
import numpy as np
import fiona


class ChesapeakRSC(NonGeoDataset):

    def __init__(self, root, split="train", differentiate_tree_canopy_over_roads=False, transforms=None):
        self.root = root
        self.differentiate_tree_canopy_over_roads = differentiate_tree_canopy_over_roads
        self.transforms = transforms
        assert split in ["train", "val", "test"]
        self.image_fns = []
        self.mask_fns = []

        colors = [
            (0, 0, 0),
            (1, 1, 1),
        ]
        if differentiate_tree_canopy_over_roads:
            colors.append((1, 0, 0))
        self._cmap = ListedColormap(colors)

        with open(os.path.join(root, f"{split}_idxs.txt")) as f:
            idxs = set(map(int, f.read().strip().split("\n")))

        with fiona.open(os.path.join(root, "patches.gpkg")) as f:
            for row in f:
                idx = row["properties"]["idx"]
                if idx not in idxs:
                    continue
                image_fn = os.path.join(root, "data", f"{idx}_image.tif")
                mask_fn = os.path.join(root, "data", f"{idx}_mask.tif")
                if os.path.exists(image_fn) and os.path.exists(mask_fn):
                    self.image_fns.append(image_fn)
                    self.mask_fns.append(mask_fn)

    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        mask_fn = self.mask_fns[idx]

        with rasterio.open(image_fn) as f:
            image = f.read()
            image = torch.from_numpy(image).float()

        with rasterio.open(mask_fn) as f:
            mask = f.read().squeeze()
            road_mask = mask == 9
            road_canopy_mask = mask == 12
            mask = np.zeros_like(mask)
            mask[road_mask] = 1
            if self.differentiate_tree_canopy_over_roads:
                mask[road_canopy_mask] = 2
            else:
                mask[road_canopy_mask] = 1

            mask = torch.from_numpy(mask).long()

        sample = {
            "image": image,
            "mask": mask
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_fns)

    def plot(self, sample, show_titles=True, suptitle=None):

        img = sample["image"].numpy().transpose(1, 2, 0)
        mask = sample["mask"].numpy()

        if "prediction" in sample:
            n_cols = 3
            width = 15
            prediction = sample["prediction"].numpy()
        else:
            n_cols = 2
            width = 10

        fig, axs = plt.subplots(1, n_cols, figsize=(width, 5))
        axs[0].imshow(img[:, :, :3])
        axs[1].imshow(mask, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap, interpolation="none")
        if "prediction" in sample:
            axs[2].imshow(prediction, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap, interpolation="none")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Labels")
            if "prediction" in sample:
                axs[2].set_title("Predictions")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()

        return fig
