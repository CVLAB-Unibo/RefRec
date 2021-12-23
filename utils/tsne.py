#%%
import sys
sys.path.append(".")
import os
os.environ['WANDB_SILENT']="true"

import torch
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
from hesiod import hcfg, hmain, set_cfg, get_cfg_copy
from networks.factory import get_model
from torchmetrics import Accuracy
import torch.nn.functional as F
from datamodules.classification_datamodule import DataModule

pl.seed_everything(42)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def tsne(embeddings, embedding_labels, classes, num, perp=10, n_iter=4000, proj="test"):
    tsne = TSNE(2, verbose=1, perplexity=perp, n_iter=n_iter)
    tsne_proj = tsne.fit_transform(embeddings)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(12,12))
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    num_categories = 10
    clip = 300
    for lab in range(num_categories):
        indices = embedding_labels==lab
        # if np.count_nonzero(indices)>clip:
        #     indices = np.where(embedding_labels==lab)[0][:clip]
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(labels=classes, fontsize='large', markerscale=2)

    fig.savefig(f"utils/tsne/{proj}/{perp}_{num}_{n_iter}.pdf", dpi=180)

@hmain(base_cfg_dir=Path("cfg"), create_out_dir=False, parse_cmd_line=False)
def main():
    split = "train"
    entity = "3dv"
    restore_weights = "m_2_sc_rec"
    project_name = "m2sc"
    s = "modelnet"
    t = "scannet"
    device = "cuda:0"
    
    run = wandb.init(
                job_type="pseudo_labels",
                project=project_name,
                name="tsne",
                entity=entity,
                save_code=False,
    )

    model_artifact = wandb.run.use_artifact(restore_weights+ ":latest", type='model')
    cfg = model_artifact.metadata
    proj = cfg["project_name"]
    cfg["aug"] = False
    cfg["dataset_hard"] = 'null'
    cfg["restore_weights"] = restore_weights
    cfg["dataset_source"]["name"] = s
    cfg["dataset_target"]["name"] = t
    cfg["dataset_target"]["name"] = t
    cfg["short_val_split"] = False

    for k, v in cfg.items():
        set_cfg(k, v)
    
    model = get_model(device, hcfg("net.name"), hcfg("restore_weights"))
    model.to(device).eval()

    cfg = get_cfg_copy()
    dm = DataModule(cfg)
    # valid_acc_target = Accuracy(compute_on_step=False)

    if split=="train":
        dataloader = dm.train_dataloader()
    else:
        _, dataloader = dm.val_dataloader()

    categories = dm.train_ds.categories
    embeddings_cls = torch.zeros((0, 1024))

    labels = []
    for batch in tqdm(dataloader):
        original_coords_b = batch["original_coordinates"].to(device)
        labels_b = batch["labels"]

        labels.extend(labels_b.numpy().tolist())
        with torch.no_grad():
            feature_cls, out = model(original_coords_b[:1024])
            # valid_acc_target(F.softmax(out.cpu(), dim=1), labels_b)
            embeddings_cls = torch.cat([embeddings_cls.cpu(), feature_cls.squeeze(dim=-1).cpu()], dim=0)

    # print(valid_acc_target.compute().item())
    labels = np.array(labels)

    print("cls")
    tsne(embeddings_cls, labels, categories, 1, perp=20, n_iter=10000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=30, n_iter=10000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=40, n_iter=1000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=50, n_iter=1000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=20, n_iter=2000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=30, n_iter=2000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=40, n_iter=2000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=50, n_iter=2000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=20, n_iter=4000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=30, n_iter=4000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=40, n_iter=4000, proj=proj)
    tsne(embeddings_cls, labels, categories, 1, perp=50, n_iter=4000, proj=proj)

main()



# %%
# %%
