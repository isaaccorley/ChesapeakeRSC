import argparse
import os
from src.datasets import ChesapeakRSC
from src.modules import CustomSemanticSegmentationTask
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_fn", required=True, type=str, help="Model checkpoint to load"
    )
    parser.add_argument(
        "--three_class", action="store_true", help="Whether to use three classes metrics"
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU to use for inference (default: 0)"
    )
    parser.add_argument(
        "--eval_set", default="test", type=str, choices=["test", "val"], help="Which set to run over"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Whether to use TQDM progress bar"
    )
    return parser


def preprocess(sample):
    sample["image"] = sample["image"].float() / 255.0
    return sample


def main(args):
    model_fn = os.path.realpath(args.model_fn)
    assert os.path.exists(model_fn)

    device = torch.device(f"cuda:{args.gpu}")

    ds = ChesapeakRSC("data/ChesapeakRSC/", split=args.eval_set, differentiate_tree_canopy_over_roads=True, transforms=preprocess)
    dl = DataLoader(ds, batch_size=8, num_workers=6)
    if not args.quiet:
        dl = tqdm(dl)

    task = CustomSemanticSegmentationTask.load_from_checkpoint(model_fn, map_location="cpu")
    model = task.model.eval().to(device)

    if args.three_class:
        cnf = np.zeros((3, 3), dtype=np.int64)

        for batch in dl:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.inference_mode():
                preds = model(images).argmax(dim=1)

            for true_class_idx in [0,1,2]:
                true_mask = masks == true_class_idx
                for pred_class_idx in [0,1,2]:
                    pred_mask = preds == pred_class_idx
                    cnf[true_class_idx, pred_class_idx] += (true_mask & pred_mask).sum().item()

        # compute per class precision and recall from cnf
        recall_background = cnf[0,0] / (cnf[0,0] + cnf[0,1] + cnf[0,2])
        recall_road = cnf[1,1] / (cnf[1,0] + cnf[1,1] + cnf[1,2])
        recall_tree_canopy_over_road = cnf[2,2] / (cnf[2,0] + cnf[2,1] + cnf[2,2])

        precision_background = cnf[0,0] / (cnf[0,0] + cnf[1,0] + cnf[2,0])
        precision_road = cnf[1,1] / (cnf[0,1] + cnf[1,1] + cnf[2,1])
        precision_tree_canopy_over_road = cnf[2,2] / (cnf[0,2] + cnf[1,2] + cnf[2,2])

        print(f"{recall_background},{precision_background},{recall_road},{precision_road},{recall_tree_canopy_over_road},{precision_tree_canopy_over_road}")
    else:
        cnf = np.zeros((3, 2), dtype=np.int64)
        for batch in dl:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.inference_mode():
                preds = model(images).argmax(dim=1)

            for true_class_idx in [0,1,2]:
                true_mask = masks == true_class_idx
                for pred_class_idx in [0,1]:
                    pred_mask = preds == pred_class_idx
                    cnf[true_class_idx, pred_class_idx] += (true_mask & pred_mask).sum().item()

        recall_background = cnf[0,0] / (cnf[0,0] + cnf[0,1])
        recall_road = cnf[1,1] / (cnf[1,0] + cnf[1,1])
        recall_tree_canopy_over_road = cnf[2,1] / (cnf[2,0] + cnf[2,1])

        precision_background = cnf[0,0] / (cnf[0,0] + cnf[1,0] + cnf[2,0])
        precision_road = cnf[1,1] / (cnf[0,1] + cnf[1,1] + cnf[2,1])

        print(f"{recall_background},{precision_background},{recall_road},{precision_road},{recall_tree_canopy_over_road}")


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
