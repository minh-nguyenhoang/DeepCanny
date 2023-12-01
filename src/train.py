from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import argparse
import lightning
from .loss import TracingLoss

class LightningModel(lightning.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: torch.nn.Module = model
        self.loss_fn = TracingLoss(tex_factor=.2, bdr_factor= .2)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, prog_bar= True, logger= True)
        # self.training_step_outputs.append({"loss": loss.cpu().detach(), "output": {k:v.cpu().detach() for k, v in output_dict.items()}})
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.training_step_outputs = []
        return super().on_train_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 5, eta_min= 1e-4),
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss, prog_bar= True, logger= True)
        # self.validation_step_outputs.append({"loss": loss.cpu().detach(), "output": {k:v.cpu().detach() for k, v in output_dict.items()}})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss, prog_bar= True, logger= True)
        # self.validation_step_outputs.append({"loss": loss.cpu().detach(), "output": {k:v.cpu().detach() for k, v in output_dict.items()}})
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.validation_step_outputs = []
        return super().on_validation_epoch_end()