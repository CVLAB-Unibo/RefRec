import torch
from torch._C import dtype
import torch.utils.data as data
import os
import numpy as np
import glob
from .transformations import *
from hesiod import hcfg, get_cfg_copy, get_out_dir, get_run_name
import open3d as o3d
import wandb

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class Dataset(data.Dataset):
    def __init__(self, name, split):
        super().__init__()

        self.split = split
        self.pc_list = []
        self.lbl_list = []
        self.pc_path = []
        self.pc_input_num = hcfg("pc_input_num")
        self.aug = hcfg("aug")
        self.name = name
      
        print("using", self.name + ":latest")
        data_artifact = wandb.run.use_artifact(self.name + ":latest")
        dataset = data_artifact.download()
        
        filename = split + ".npz"
        data = np.load(os.path.join(dataset, filename))
        self.pc_list = data["x"]
        self.lbl_list = data["y"]
        self.pc_path = data["path"]
        
        if len(self.pc_path)>1:
            self.categories = [c.split(os.path.sep)[-3] for c in self.pc_path]
            self.categories = sorted(set(self.categories))
            
        print(f"{split} data num: {len(self.pc_list)}")

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc_path = self.pc_path[idx]
        pc_original = pc.copy()

        if self.aug and self.split == "train":
            pc = random_rotate_one_axis(pc, axis="z")
            pc = jitter_point_cloud(pc)
            if hcfg("occlusions") and ("modelnet.ply" in pc_path or "shapenet.ply" in pc_path) :
                pc = remove(pc)

        if pc.shape[0] > self.pc_input_num:
            if hcfg("sampling") == "fps":
                # apply Further Point Sampling
                pc = np.swapaxes(np.expand_dims(pc, 0), 1, 2)
                _, pc = farthest_point_sample_np(pc, self.pc_input_num)
                pc = np.swapaxes(pc.squeeze(), 1, 0).astype(np.float32)
            elif hcfg("sampling") == "uniform":
                ids = np.random.choice(
                    pc.shape[0], size=self.pc_input_num, replace=False
                )
                pc = pc[ids]
            else:
                pc = pc[: self.pc_input_num]

        return {
            "coordinates": pc,
            "original_coordinates": pc_original,
            "labels": lbl,
            "paths": pc_path,
        }

    def __len__(self):
        return len(self.pc_list)