import sys
from unicodedata import name

from hesiod.core import get_cfg_copy
sys.path.append(".")
import os
os.environ['WANDB_SILENT']="true"

from networks.factory import get_model

import torch
from hesiod import hcfg, hmain, set_cfg
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import open3d as o3d
import wandb
import glob
from datamodules.classification_datamodule import DataModule
from torchmetrics import Accuracy

pl.seed_everything(42)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def create_dataset(root, pts_list, categories):
    pc_list = []
    lbl_list = []
    pc_path = []

    if len(pts_list)>0:
        for elem in pts_list:
            pc = o3d.io.read_point_cloud(elem)
            pc = np.array(pc.points).astype(np.float32)
            pc_list.append(pc)
            pc_path.append(elem.replace(str(root), ""))
            lbl_list.append(categories.index(elem.split("/")[-3]))
    
    return np.stack(pc_list), np.stack(lbl_list), pc_path
    
def filter_data(predictions, probs, p, num_classes):
    thres = []
    for i in range(num_classes):
        x = probs[predictions==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x, _ = torch.sort(x)
        thres.append(x[int(round(len(x)*p))])
    thres = torch.tensor(thres)
    print("class confidence treshold", thres)
    selected = torch.ones_like(probs, dtype=torch.bool)

    for i in range(len(predictions)):
        for c in range(num_classes):
            if probs[i]<thres[c]*(predictions[i]==c):
                selected[i] = False
    return selected

def write_pointclouds(coords_class_c, dirname, ext):
    for j, pc in enumerate(coords_class_c):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.io.write_point_cloud(dirname + f"/{j}_{ext}", pcd)

def closest_embeddings(unknown, pl, k):
    dist = torch.linalg.norm(pl - unknown, dim=1)    
    knn = torch.topk(dist, k, largest=False)
    return knn.indices.cpu()

def biKNN(pl_embeddings, unknown_embeddings, pl_labels, k=1):
    pl_embeddings_normalized = F.normalize(pl_embeddings, dim=1)
    unknown_embeddings_normalized = F.normalize(unknown_embeddings, dim=1)
    unknown_labels = [] 
    unknown_indeces = []
    counter = 0

    for i in tqdm(range(len(unknown_embeddings_normalized))):
        idxs = closest_embeddings(unknown_embeddings_normalized[i], pl_embeddings_normalized, k)
        idxs_inverse = closest_embeddings(pl_embeddings_normalized[idxs[0]], unknown_embeddings_normalized, k)
        if idxs_inverse[0]==i:
            unknown_labels.append(pl_labels[idxs])
            unknown_indeces.append(i)
            counter += 1

    return torch.tensor(unknown_labels), torch.tensor(unknown_indeces)

def top_KNN(pl_embeddings, unknown_embeddings, pl_labels, k=3):
    pl_embeddings_normalized = F.normalize(pl_embeddings, dim=1)
    unknown_embeddings_normalized = F.normalize(unknown_embeddings, dim=1)
    unknown_labels = [] 
    for i in tqdm(range(len(unknown_embeddings_normalized))):
        idxs = closest_embeddings(unknown_embeddings_normalized[i], pl_embeddings_normalized, k)
        nn_labels = pl_labels[idxs]
        if len(np.unique(nn_labels))==k:
            l = nn_labels[0]
        else:   
            l = torch.argmax(torch.bincount(nn_labels.long()))
        unknown_labels.append(l)
    return torch.tensor(unknown_labels)

def load_data_WandB(new_dataset_root, categories, dataset_name, restore_weights, cfg, source_domain, target_domain, wandb_run, p):
    pts_list_train = glob.glob(os.path.join(new_dataset_root, "*", "train", "*.ply"))
    pts_list_test = glob.glob(os.path.join(new_dataset_root, "*", "test", "*.ply"))

    train = create_dataset(new_dataset_root, pts_list_train, categories)
    test = create_dataset(new_dataset_root, pts_list_test, categories)

    datasets = [train, test]
    names = ["train", "test"]
    
    raw_data = wandb.Artifact(
        dataset_name, 
        type="dataset",
        description=dataset_name + " PL",
        metadata={
                "mdoel": restore_weights,
                "ckpt": cfg["run_name"],
                "percentile": p,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "sizes": [len(dataset[0]) for dataset in datasets]})

    for name, data in zip(names, datasets):
        with raw_data.new_file(name + ".npz", mode="wb") as file:
            np.savez(file, x=data[0], y=data[1], path=data[2])

    wandb_run.log_artifact(raw_data)

@hmain(base_cfg_dir=Path("cfg"), create_out_dir=False, parse_cmd_line=True)
def main():

    log_artifacts = True
    save = True
    save_source_data = True
    p = 0.9
    top_k=1
    entity = hcfg("entity")
    source_domain = hcfg("dataset_source")
    target_domain = hcfg("dataset_target")
    restore_weights = hcfg("restore_weights")

    domains = {"modelnet":"m", "shapenet":"s", "scannet":"sc"}
    s = domains[source_domain]
    t = domains[target_domain]

    project_name = s+"2"+t
    dataset_name_easy_split = hcfg("easy_split")
    dataset_name_hard_split = hcfg("hard_split")

    device = "cuda" 

    run = wandb.init(
                job_type="pseudo_labels",
                project=project_name,
                name=dataset_name_easy_split + "_generation",
                entity=entity,
                save_code=False,
    )
    
    model_artifact = wandb.run.use_artifact(restore_weights+ ":latest", type='model')
    cfg = model_artifact.metadata
    cfg["aug"] = False
    cfg["short_val_split"] = False
    cfg["restore_weights"] = restore_weights

    for k, v in cfg.items():
        set_cfg(k, v)

    # get target train data for pseudo-labels
    set_cfg("dataset_source", target_domain) 
    dm = DataModule()
    dataloader_target = dm.train_dataloader()

    # get source datasets
    set_cfg("dataset_source", source_domain) 
    dm = DataModule()
    dataloader_source = dm.train_dataloader()
    dataloader_source_val, _ = dm.val_dataloader()

    categories = dm.train_ds.categories

    model_rec = get_model(device, "reconstruction", hcfg("restore_weights_rec"))
    model_rec.to(device).eval()
    model = get_model(device, hcfg("net.name"), hcfg("restore_weights"))
    model.to(device).eval()

    valid_acc_target = Accuracy(compute_on_step=False)

    coords = torch.zeros((0, 2048, 3))
    labels = torch.zeros((0))
    predictions = torch.zeros((0))
    probs = torch.zeros((0))
    embeddings = torch.zeros((0, 1024))
    prototypes = [[] for i in range(cfg["num_classes"] )]
    mean_protoypes = torch.zeros((cfg["num_classes"] , 1024))

    paths = []

    for batch in tqdm(dataloader_target):
        coords_b = batch["coordinates"].to(device)
        original_coords_b = batch["original_coordinates"]
        labels_b = batch["labels"]
        paths_b = batch["paths"]

        with torch.no_grad():
            _, out_t = model(coords_b.to(device)[:, :1024], embeddings=True)
            features, _ = model_rec(coords_b.to(device)[:, :1024])
            embeddings = torch.cat([embeddings, features.squeeze().cpu()], dim=0)

            logits = F.softmax(out_t, dim=1)
            probs_b, predictions_b = torch.max(logits, dim=1)

            coords = torch.cat([coords, original_coords_b.cpu()], dim=0)
            predictions = torch.cat([predictions, predictions_b.cpu()], dim=0).long()
            probs = torch.cat([probs, probs_b.cpu()], dim=0)
            labels = torch.cat([labels, labels_b], dim=0)
            valid_acc_target(F.softmax(logits.cpu(), dim=1), labels_b.long())
            paths.extend(paths_b)
            
            for e, pl in enumerate(predictions_b):
                prototypes[pl].append(features[e].squeeze())

    # print(valid_acc_target.compute())
    
    #save prototypes for self training
    for c, protos_class_c in enumerate(prototypes):
        pro_class_c = torch.stack(protos_class_c, dim=0)
        mean_protoypes[c] = pro_class_c.mean(dim=0)
    np.save(hcfg("prototypes_path"), mean_protoypes.cpu().numpy())

    selected = filter_data(predictions.clone(), probs.clone(), p, cfg["num_classes"])
    
    # predictions_uncertain = predictions[~selected]
    coords_uncertain = coords[~selected]
    # labels_uncertain = labels[~selected]
    embeddings_uncertain = embeddings[~selected]

    coords = coords[selected]
    labels_pl = labels[selected]
    predictions = predictions[selected]
    embeddings = embeddings[selected]

    print("Selected", len(predictions), "confident samples")

    avg_accuracy = 0 
    for c in range(cfg["num_classes"]):
        avg_accuracy += np.sum(predictions[labels_pl==c].numpy()==labels_pl[labels_pl==c].numpy())/len(labels_pl[labels_pl==c].numpy())

    print("avg accuracy confident samples", avg_accuracy/cfg["num_classes"])
    print("accuracy confident samples:", np.sum(predictions.numpy()==labels_pl.numpy())/len(labels_pl.numpy()))
    
    BiKNN_predictions, BiKNN_indeces = biKNN(embeddings, embeddings_uncertain, predictions, top_k)
    BiKNN_matches_coordinates = coords_uncertain[BiKNN_indeces]
    BiKNN_embeddings = embeddings_uncertain[BiKNN_indeces]

    # extract unmatched embeddings
    mask = torch.ones(len(embeddings_uncertain), dtype=torch.bool)
    mask[BiKNN_indeces] = 0
    remaining_embeddings = embeddings_uncertain[mask]
    remaining_coordinates = coords_uncertain[mask]

    print("filtered by BiKNN", len(BiKNN_predictions))

    #merge confident ones with bidirectional 1knn to get easy split
    coords_easy_split = torch.cat([coords, BiKNN_matches_coordinates], dim=0)
    predictions_easy_split = torch.cat([predictions, BiKNN_predictions], dim=0)
    embeddings_easy_split = torch.cat([embeddings, BiKNN_embeddings], dim=0)

    # knn for each sample in the remainining set to obtain hard split
    top_KNN_predictions = top_KNN(embeddings_easy_split, remaining_embeddings, predictions_easy_split)
    coords_hard_split = torch.cat([coords_easy_split, remaining_coordinates], dim=0)
    predictions_hard_split = torch.cat([predictions_easy_split, top_KNN_predictions], dim=0)

    if save:
        new_dataset_root_easy_split = Path("data") / Path(dataset_name_easy_split)
        os.makedirs(new_dataset_root_easy_split, exist_ok=True)

        new_dataset_root_hard_split = Path("data") / Path(dataset_name_hard_split)
        os.makedirs(new_dataset_root_hard_split, exist_ok=True)

        for i, c in tqdm(enumerate(categories)):
            # write easy split on disk
            dirname = f"{new_dataset_root_easy_split}/{c}/test"
            os.makedirs(dirname, exist_ok=True)
            dirname = f"{new_dataset_root_easy_split}/{c}/train"
            os.makedirs(dirname, exist_ok=True)
            coords_class_c = coords_easy_split[predictions_easy_split==i]
            write_pointclouds(coords_class_c, dirname, "st.ply")

            # write hard split on disk
            dirname = f"{new_dataset_root_hard_split}/{c}/test"
            os.makedirs(dirname, exist_ok=True)
            dirname = f"{new_dataset_root_hard_split}/{c}/train"
            os.makedirs(dirname, exist_ok=True)
            coords_class_c = coords_hard_split[predictions_hard_split==i]
            write_pointclouds(coords_class_c, dirname, "st.ply")

        if save_source_data:
            coords_source = torch.zeros((0, 2048, 3))
            labels_source = torch.zeros((0))
            phase = "train"
            
            for batch in tqdm(dataloader_source):
                original_coords_b = batch["original_coordinates"]
                labels_b = batch["labels"]
                coords_source = torch.cat([coords_source, original_coords_b.cpu()], dim=0)
                labels_source = torch.cat([labels_source, labels_b], dim=0)

            for j, c in enumerate(categories):
                coords_class_c = coords_source[labels_source==j]
                dirname = f"{new_dataset_root_easy_split}/{c}/{phase}"
                write_pointclouds(coords_class_c, dirname, f"{source_domain}.ply")

            coords_source = torch.zeros((0, 2048, 3))
            labels_source = torch.zeros((0))
            phase = "test"
            
            for batch in tqdm(dataloader_source_val):
                original_coords_b = batch["original_coordinates"]
                labels_b = batch["labels"]
                coords_source = torch.cat([coords_source, original_coords_b.cpu()], dim=0)
                labels_source = torch.cat([labels_source, labels_b], dim=0)

            for j, c in enumerate(categories):
                coords_class_c = coords_source[labels_source==j]
                dirname = f"{new_dataset_root_easy_split}/{c}/{phase}"
                write_pointclouds(coords_class_c, dirname, f"{source_domain}.ply")
                dirname = f"{new_dataset_root_hard_split}/{c}/{phase}"
                write_pointclouds(coords_class_c, dirname, f"{source_domain}.ply")

    if log_artifacts:
        load_data_WandB(new_dataset_root_easy_split, categories, dataset_name_easy_split, restore_weights, cfg, source_domain, target_domain, run, p)
        load_data_WandB(new_dataset_root_hard_split, categories, dataset_name_hard_split, restore_weights, cfg, source_domain, target_domain, run, p)
    
main()
