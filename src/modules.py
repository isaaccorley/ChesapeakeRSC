import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, FBetaScore, Precision, Recall
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from lightning.pytorch.callbacks import ModelCheckpoint
from .models import CustomFCN

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, tmax=50, **kwargs) -> None:
        super().__init__()

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        tmax: Optional[int] = self.hparams.get("tmax", 50)

        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                ),
                "OverallPrecision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                ),
                "OverallRecall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    multidim_average="global",
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                ),
                "Precision": ClasswiseWrapper(
                    Precision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                ),
                "Recall": ClasswiseWrapper(
                    Recall(
                        task="multiclass",
                        num_classes=num_classes,
                        average="none",
                        multidim_average="global",
                    ),
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        task="multiclass",
                        num_classes=num_classes,
                        beta=1.0,
                        average="none",
                        multidim_average="global",
                    ),
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")


    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        num_filters: int = self.hparams["num_filters"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        elif model == "custom_fcn":
            self.model = CustomFCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.train_metrics(y_hat_hard, y)
        self.log_dict({f"{k}": v for k, v in self.train_metrics.compute().items()})
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.val_metrics(y_hat_hard, y)
        self.log_dict(
            {f"{k}": v for k, v in self.val_metrics.compute().items()}, on_epoch=True
        )

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Optional[Figure] = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(
            {f"{k}": v for k, v in self.test_metrics.compute().items()}, on_epoch=True
        )

    def configure_callbacks(self):
        """Initialize model-specific callbacks.

        Returns:
            List of callbacks to apply.
        """
        print("Using our callbacks")
        return [
            ModelCheckpoint(every_n_epochs=50, save_top_k=-1),
            ModelCheckpoint(monitor=self.monitor, mode=self.mode, save_top_k=5),
        ]

    def on_train_epoch_start(self) -> None:
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar("lr", lr, self.current_epoch)
