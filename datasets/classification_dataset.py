import torch.utils.data as data
import os
import numpy as np
from .transformations import *
from hesiod import hcfg
import open3d as o3d
import wandb

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class ClassificationDataset(data.Dataset):
    def __init__(self, name, split, short_val_split=False, occlusions=False):
        super().__init__()

        self.split = split
        self.pc_list = []
        self.lbl_list = []
        self.pc_path = []
        self.pc_input_num = hcfg("pc_input_num")
        self.aug = hcfg("aug")
        self.name = name
        self.short_val_split = short_val_split
        self.occlusions = occlusions
        
        print("using", self.name + ":latest")
        data_artifact = wandb.run.use_artifact(self.name + ":latest")
        dataset = data_artifact.download()
        
        filename = split + ".npz"
        data = np.load(os.path.join(dataset, filename))
        self.pc_list = data["x"]
        self.lbl_list = data["y"]
        self.pc_path = data["path"]
        
        if self.short_val_split:
            print("shortening test set for SOURCE domain #################")
            selected = np.zeros_like(self.lbl_list)
            for c in range(hcfg("num_classes")):
                elems_class_c = np.where(self.lbl_list==c)[0][:200]
                selected[elems_class_c] = True
            selected = selected.astype(np.bool)
            self.pc_list = self.pc_list[selected]
            self.lbl_list = self.lbl_list[selected]
            self.pc_path = self.pc_path[selected]

        if len(self.pc_path)>1:
            self.categories = [c.split(os.path.sep)[-3] for c in self.pc_path]
            self.categories = sorted(set(self.categories))
            n_samples = []
            for c in range(hcfg("num_classes")):
                n_samples.append(np.count_nonzero(self.lbl_list==c))

        print(f"{split} data num: {len(self.pc_list)}")

    def __getitem__(self, idx):

        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc_original = pc.copy()
        pc_path = self.pc_path[idx]

        if self.aug and self.split == "train":
            pc = random_rotate_one_axis(pc, axis="z")
            pc = jitter_point_cloud(pc)
            if hcfg("occlusions") and ("modelnet.ply" in pc_path or "shapenet.ply" in pc_path) :
                pc = remove(pc)

        if pc.shape[0] > self.pc_input_num:
            if hcfg("sampling") == "uniform":
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