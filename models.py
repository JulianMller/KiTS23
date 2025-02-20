import torch

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureType, AsDiscrete

import pytorch_lightning as pl
from lion_pytorch import Lion

from kits23.configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_SD_TOLERANCES_MM
from kits23.evaluation.metrics import compute_metrics_for_label
import numpy as np


class LightningUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        # self.best_val_dice = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        # self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.post_output = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=4)])

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = Lion(
                    self._model.parameters(),
                    lr=1e-4,
                    weight_decay=1e-2,
#                     use_triton=True # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
                )
#         optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self.forward(batch["image"])
        loss = self.loss_function(output, batch["label"])
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["image"])
        loss = self.loss_function(outputs, batch["label"])
        outputs = [self.post_output(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(batch["label"])]
        
        self.dice_metric(y_pred=outputs, y=labels)
        
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
       
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice}"
            f"current loss:' {mean_val_loss}"
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}
    

class LightningUNet2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.BATCH,
            dropout=0.1
        )
        
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        # self.best_val_dice = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        # self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.post_output = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=4)])

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = Lion(
                    self._model.parameters(),
                    lr=1e-4,
                    weight_decay=1e-2,
#                     use_triton=True # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
                )
#         optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self.forward(batch["image"])
        loss = self.loss_function(output, batch["label"])
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["image"])
        loss = self.loss_function(outputs, batch["label"])
        outputs = [self.post_output(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(batch["label"])]
        
        self.dice_metric(y_pred=outputs, y=labels)
        
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
       
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice}"
            f"current loss:' {mean_val_loss}"
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}

    
