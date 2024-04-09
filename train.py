import argparse
import os
os.environ['PROJ_NETWORK'] = 'OFF'
import warnings

import lightning.pytorch as pl

from src.datamodules import ChesapeakeRSCDataModule
from src.modules import CustomSemanticSegmentationTask

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of each mini-batch."
    )
    parser.add_argument(
        "--model",
        choices=["deeplabv3+", "fcn", "custom_fcn", "unet", "unet++"],
        default="unet",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=150,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=64,
        help="Number of filters to use with FCN models.",
    )
    parser.add_argument(
        "--backbone",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
        ],
        default="resnet50",
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=50,
        help="Cycle size for cosine lr scheudler.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="Name of the experiment to run.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=False,
        help="GPU ID to use (defaults to all GPUs if none).",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data/ChesapeakeRSC/",
        help="Root directory of the dataset.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Main training routine."""
    # torch.set_float32_matmul_precision('medium')

    dm = ChesapeakeRSCDataModule(
        root=args.root_dir,
        batch_size=args.batch_size,
        num_workers=8,
        differentiate_tree_canopy_over_roads=False,
    )

    task = CustomSemanticSegmentationTask(
        model=args.model,
        backbone=args.backbone,
        weights=True,
        in_channels=4,
        num_classes=2,
        num_filters=args.num_filters,
        loss="ce",
        lr=args.lr,
        tmax=args.tmax,
        class_weights=None,
    )

    experiment_name = None
    if args.experiment_name is not None:
        experiment_name = os.path.join("logs", args.experiment_name)

    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = [args.gpu_id]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpu_id,
        min_epochs=args.num_epochs,
        max_epochs=args.num_epochs,
        log_every_n_steps=15,
        default_root_dir=experiment_name,
    )

    trainer.fit(task, dm)


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
