from hesiod import hmain, get_cfg_copy, get_out_dir, get_run_name
from pathlib import Path
import pytorch_lightning as pl
import wandb
import os, sys
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from hesiod import hcfg

from utils.callbacks import PCPredictionLogger
from trainers.reconstruction_trainer import Reconstruction_trainer
from datamodules.reconstruction_datamodule import DataModule

# @hmain(base_cfg_dir=Path("cfg"), template_cfg_file=Path("cfg/template.yaml"))
@hmain(base_cfg_dir=Path("cfg"), run_cfg_file=Path(sys.argv[1]), create_out_dir=False, parse_cmd_line=False)
def main():
    
    device = "cuda"
    cfg = get_cfg_copy()
    print(f"***Summary*** :\n{cfg}")
    print(os.getcwd())
    
    run_name = hcfg("net.name")+"_"+hcfg("project_name")
    print(hcfg("project_name"), run_name, get_out_dir())
 
    # fit the model
    run = wandb.init(
        job_type="train",
        project=hcfg("project_name"),
        name=get_run_name(),
        entity=hcfg("entity"),
        dir=get_out_dir(),
    )

    dm = DataModule()
    wandb_logger = WandbLogger(
        project=hcfg("project_name"), name=run_name, save_dir=get_out_dir()
    )

    checkpoint_dir = os.path.join(get_out_dir(), "checkpoint")
    print(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model",
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model = Reconstruction_trainer(dm=dm, device=device)

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        log_every_n_steps=100,  # set the logging frequency,
        gpus=hcfg("gpu"),
        max_epochs=hcfg("epochs"),
        benchmark=True,
        progress_bar_refresh_rate=200,
        callbacks=[
            PCPredictionLogger(dm),
            checkpoint_callback,
            lr_monitor
        ], 
        num_sanity_val_steps=2,
    )

    trainer.fit(model, datamodule=dm)
    
    model_artifact = wandb.Artifact(
        get_run_name(), 
        type="model",
        description=hcfg("net.name"),
        metadata=cfg)

    model_artifact.add_file(checkpoint_callback.last_model_path)
    run.log_artifact(model_artifact)
    wandb.finish()

if __name__ == "__main__":
    main()