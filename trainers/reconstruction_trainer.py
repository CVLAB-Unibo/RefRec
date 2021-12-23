import torch
import pytorch_lightning as pl
from networks.factory import get_model
from hesiod import hcfg, get_cfg_copy
from utils.optimizers import get_optimizer
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PyTorchEMD.emd import earth_mover_distance
from utils.losses import ChamferLoss

class Reconstruction_trainer(pl.LightningModule):
    def __init__(self, dm, device):
        super().__init__()

        self.net = get_model(device, hcfg("net.name"), hcfg("restore_weights"))
        self.dm = dm
        self.save_hyperparameters(get_cfg_copy())
        self.log_prediction = True
        self.writer = SummaryWriter(wandb.run.dir)
        self.chamfer = ChamferLoss()
        self.emd = earth_mover_distance

    def forward(self, x):
        feature, output = self.net(x)
        return feature, output

    def loss(self, xb, yb):
        feature, output = self(xb)
        loss_cd = self.chamfer(output, yb)
        loss_emd = torch.sum(self.emd(output, yb, transpose=False))
        return feature, output, loss_cd, loss_emd 

    def training_step(self, batch, batch_idx):
        coords = batch["coordinates"]
        _, _, loss_cd, loss_emd = self.loss(coords, coords)

        if self.global_step % 500 == 0 and self.global_step != 0:
            self.logger.experiment.log(
                {"train/loss_cd": loss_cd.item(),
                "train/loss_emd": loss_emd.item()*hcfg("weight_emd")
                }, commit=False, step=self.global_step
            )
        return loss_cd + loss_emd*hcfg("weight_emd")

    def validation_step(self, batch, batch_idx, dataloader_idx):
        coords = batch["coordinates"]
        feature, _ = self(coords)
        return {"feature": feature, "labels": batch["labels"]}

    def validation_epoch_end(self, validation_step_outputs):
        if (self.current_epoch%200 == 0) or self.current_epoch==hcfg("epochs")-1:
            source_data = validation_step_outputs[0]
            source_embeddings = np.concatenate(
                [x["feature"].squeeze().cpu().numpy() for x in source_data]
            )
            source_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in source_data]
            )
            self.writer.add_embedding(source_embeddings, metadata=source_labels, global_step=self.global_step, tag="source")
        
            target_data = validation_step_outputs[1]
            target_embeddings = np.concatenate(
                [x["feature"].squeeze().cpu().numpy() for x in target_data]
            )
            target_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in target_data]
            )
            self.writer.add_embedding(target_embeddings, metadata=target_labels, global_step=self.global_step, tag="target")

    def on_train_end(self) -> None:
        self.writer.close()
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)

    def configure_optimizers(self):
        opt = get_optimizer(self.net)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.hparams.epochs)
        return [opt], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]
