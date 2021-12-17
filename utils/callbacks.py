import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from hesiod import hcfg

class PCPredictionLogger(pl.Callback):
    def __init__(self, dataModule):
        super().__init__()
        self.dataModule = dataModule

    def log_point_clouds(self, trainer, pl_module, batch, split):
        
        
        pcs = batch["coordinates"].squeeze().cpu().numpy()[:6, ...]
        trainer.logger.experiment.log(
                {"point_clouds_" + split: [wandb.Object3D(pc) for i, pc in enumerate(pcs)]}
        )

        if hasattr(pl_module, 'log_prediction'):
            _, pcs = pl_module(batch["coordinates"].to(pl_module.device))
            pcs = pcs[:6, ...].cpu().numpy()
            trainer.logger.experiment.log(
                {"point_clouds_REC" + split: [wandb.Object3D(pc) for i, pc in enumerate(pcs)]})


    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 25 == 0:
            samples_train, samples_source, samples_target = self.dataModule.get_val_samples()
            self.log_point_clouds(trainer, pl_module, samples_train, "train")
            self.log_point_clouds(trainer, pl_module, samples_source, "source")
            self.log_point_clouds(trainer, pl_module, samples_target, "target")
