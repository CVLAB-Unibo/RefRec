#%%
import os, sys
os.environ['WANDB_SILENT']="true"
import numpy as np
import open3d as o3d
import open3d as o3d
import wandb
from pathlib import Path
from hesiod import hmain, get_cfg_copy

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

#%%

root = Path("logs/m_2_sc_rec")
ckpt_path = root / "checkpoint/last.ckpt"
run_file_path = root / "run.yaml"
name = "m_2_sc_rec"
project_name = "m2sc"
entity = "3dv"
# 

@hmain(base_cfg_dir=Path("cfg"), run_cfg_file=Path(run_file_path), create_out_dir=False, parse_cmd_line=False)
def main():
    cfg = get_cfg_copy()
    print(cfg)
    run = wandb.init(
                job_type="model",
                project=project_name,
                name=name,
                entity=entity,
                save_code=False,
    )
    model_artifact = wandb.Artifact(
      name, 
      type="model",
      description=name,
      metadata=cfg
      )

    model_artifact.add_file(ckpt_path)
    run.log_artifact(model_artifact)

main()