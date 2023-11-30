from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import argparse
import lightning
from loss import TracingLoss

class LightningModel(lightning.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: torch.nn.Module = model
        self.loss_fn = TracingLoss(tex_factor=.7, bdr_factor= .5)
        self.training_step_outputs = []
        self.validation_step_outputs = []


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
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss, prog_bar= True, logger= True)
        # self.validation_step_outputs.append({"loss": loss.cpu().detach(), "output": {k:v.cpu().detach() for k, v in output_dict.items()}})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        output_dict = self.model(x)
        loss = self.loss_fn(output_dict, y)
        self.log("val_loss", loss, prog_bar= True, logger= True)
        # self.validation_step_outputs.append({"loss": loss.cpu().detach(), "output": {k:v.cpu().detach() for k, v in output_dict.items()}})
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.validation_step_outputs = []
        return super().on_validation_epoch_end()