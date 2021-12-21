import sys

from hesiod.core import get_cfg_copy
sys.path.append(".")
import os
os.environ['WANDB_SILENT']="true"

from pathlib import Path
import numpy as np
import open3d as o3d
from networks.factory import get_model
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import open3d as o3d
from hesiod import set_cfg, hmain, hcfg
from tqdm import tqdm
import wandb
from datamodules.classification_datamodule import DataModule

pl.seed_everything(42)

def closest_embeddings(unknown, pl, k):
    dist = torch.linalg.norm(pl - unknown, dim=1)    
    knn = torch.topk(dist, k, largest=False)
    return knn.indices.cpu()

def assign_labels(pl_embeddings, unknown_embeddings, pl_labels):
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
    print(len(unknown_labels))
    print(counter)
    return np.array(unknown_labels), np.array(unknown_indeces)

def create_dataset(root, pts_list, categories, true_labels=None):
    pc_list = []
    lbl_list = []
    pc_path = []
    true_lbl_list = []
    mask_list = []

    for elem in pts_list:
        pc = o3d.io.read_point_cloud(elem)
        pc = np.array(pc.points).astype(np.float32)
        pc_list.append(pc)
        pc_path.append(elem.replace(str(root), ""))
        lbl_list.append(categories.index(elem.split("/")[-3]))
        if true_labels is not None:
            true_lbl_list.append(true_labels[elem][0])
        mask_list.append(true_labels[elem][3])

    if true_labels is None:
        true_lbl_list.append("none")
        if target_only:
            pc_list.append(0)
            lbl_list.append(0)
            pc_path.append("none")
            mask_list.append(np.zeros((1, 2048),dtype=np.bool))

    return np.stack(pc_list),\
                np.stack(lbl_list), \
                pc_path, np.stack(true_lbl_list),\
                 np.stack(mask_list)

#%%

@hmain(base_cfg_dir=Path("cfg"), create_out_dir=False, parse_cmd_line=True)
def main():

    log_artifacts = False
    save = False
    # split = "val"
    split = "train"
    entity = hcfg("entity")
    source_domain = hcfg("dataset_source")
    target_domain = hcfg("dataset_target")
    restore_weights = hcfg("restore_weights")
    if source_domain=="modelnet":
        s = "m"
    elif source_domain=="shapenet":
        s = "s"
    elif source_domain=="scannet":
        s = "sc"

    if target_domain=="modelnet":
        t = "m"
    elif target_domain=="shapenet":
        t = "s"
    elif target_domain=="scannet":
        t = "sc"
    project_name = s+"2"+t

    dataset_name_target_only = f"easy_split_{project_name}_target_only"
    dataset_name_target = f"easy_split_{project_name}"

    device = "cuda" 
    k = 1

    run = wandb.init(
                job_type="pseudo_labels",
                project=project_name,
                name=dataset_name_target + "_generation",
                entity=entity,
                save_code=False,
    )

    model_artifact = wandb.run.use_artifact(restore_weights+ ":latest", type='model')
    cfg = model_artifact.metadata
    cfg["aug"] = False
    cfg["dataset_source"] = source_domain
    cfg["short_val_split"] = False

    for k, v in cfg.items():
        set_cfg(k, v)

    dm = DataModule()
    dataloader = dm.train_dataloader()
    classes = dataloader.dataset.categories

    embeddings = torch.zeros((0, 1024))
    predictions = torch.zeros((0))
    coords = torch.zeros((0 ,2048, 3))

    model = get_model(device)
    model.to(device).eval()
    paths = []

    for j, batch in enumerate(tqdm(dataloader)):
        coords_b = batch["coordinates"].to(device)
        original_coords_b = batch["original_coordinates"]
        paths_b = batch["paths"]
        predictions_b = batch["labels"]

        with torch.no_grad():
            feature_t, out_t = model(coords_b[:, :1024])
        
        paths.extend(paths_b)
        coords = torch.cat([coords, original_coords_b.cpu()], dim=0)
        embeddings = torch.cat([embeddings, feature_t.squeeze(2).cpu()], dim=0)
        predictions = torch.cat([predictions, predictions_b.cpu()], dim=0)

#     idx = np.zeros(len(paths), dtype=np.bool)
#     for i, p in enumerate(paths):
#         if "st.ply" in p:
#             idx[i] = 1

#     pl_pc = coords[idx]
#     unknown_pc = coords[~idx]
    
#     pl_embeddings = embeddings[idx]
#     unknown_embeddings = embeddings[~idx]
    
#     pl_predictions=predictions[idx]
#     pl_true_label=true_labels[idx]
    
#     old_unknown_predictions=predictions[~idx]
#     unknown_true_labels=true_labels[~idx]

#     paths = np.array(paths)
#     pl_path = paths[idx]
#     unknown_path = paths[~idx]
#     pl_masks = masks[idx]
#     unknow_masks = masks[~idx]

#     print(len(pl_path), len(unknown_path), len(pl_path)+len(unknown_path))
  
#     new_unknown_predictions, new_unknown_indeces = assign_labels(pl_embeddings, unknown_embeddings, pl_predictions)
#     filtered_old_unknown_predictions = old_unknown_predictions[new_unknown_indeces]

#     print("filtered by BiKNN", len(new_unknown_predictions))

#     filtered_unknown_pc = unknown_pc[new_unknown_indeces]
#     filtered_unknown_path = unknown_path[new_unknown_indeces]
#     filtered_unknown_true_labels = unknown_true_labels[new_unknown_indeces]
#     filtered_unknown_masks = unknow_masks[new_unknown_indeces]

#     mask = np.ones(len(old_unknown_predictions), np.bool)
#     mask[new_unknown_indeces] = 0

#     filtered_OUT_unknown_pc = unknown_pc[mask]
#     filtered_OUT_unknown_path = unknown_path[mask]
#     filtered_OUT_old_unknown_predictions = old_unknown_predictions[mask]
#     filtered_OUT_unknown_true_labels = unknown_true_labels[mask]
#     filtered_OUT_unknown_masks = unknow_masks[mask]
    
#     print("filtered OUT by BiKNN", len(filtered_OUT_old_unknown_predictions))
#     print("unique:", np.unique(new_unknown_predictions))

#     new_unknown_predictions = torch.tensor(new_unknown_predictions, dtype=torch.float32)
#     print("overlap knn/PL:", np.count_nonzero(filtered_old_unknown_predictions==new_unknown_predictions)/len(new_unknown_predictions)*100)

#     #failure cases visualization #################################################################################
#     different_mask = filtered_old_unknown_predictions != new_unknown_predictions
#     correct_refine_mask = new_unknown_predictions != filtered_unknown_true_labels
#     wrong_refines = torch.logical_and(different_mask, correct_refine_mask)
#     for p, r, c in zip(filtered_old_unknown_predictions[wrong_refines] ,new_unknown_predictions[wrong_refines],filtered_unknown_path[wrong_refines]):
#         print(p, r, c)
#     #failure cases visualization #################################################################################

#     print("total accuracy:", np.count_nonzero(predictions==true_labels)/len(true_labels)*100)
    
#     print("kept confidence set accuracy:", np.count_nonzero(pl_predictions==pl_true_label)/len(pl_true_label)*100)
#     print("discarded confidence set accuracy:", np.count_nonzero(old_unknown_predictions==unknown_true_labels)/len(unknown_true_labels)*100)
    
#     cm = confusion_matrix(filtered_unknown_true_labels, filtered_old_unknown_predictions)
#     cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm_scores = cmn.diagonal()
#     for c in range(cfg["num_classes"] ):
#             print(f"class {c}: {cm_scores[c]*100:5.2f}, elem: {len(filtered_unknown_true_labels[filtered_unknown_true_labels==c])}, Pred: {len(filtered_old_unknown_predictions[filtered_old_unknown_predictions==c])}")
#     print("old accuracy:", np.count_nonzero(filtered_old_unknown_predictions==filtered_unknown_true_labels)/len(filtered_unknown_true_labels)*100)
#     print("old AVG accuracy", (cm_scores*100).mean())
    
#     print("-------")

#     cm = confusion_matrix(filtered_unknown_true_labels, new_unknown_predictions)
#     cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm_scores = cmn.diagonal()
#     for c in range(cfg["num_classes"] ):
#             print(f"class {c}: {cm_scores[c]*100:5.2f}, elem: {len(filtered_unknown_true_labels[filtered_unknown_true_labels==c])}, Pred: {len(new_unknown_predictions[new_unknown_predictions==c])}")
#     print("KNN accuracy:", np.count_nonzero(new_unknown_predictions==filtered_unknown_true_labels)/len(filtered_unknown_true_labels)*100)
#     print("KNN AVG accuracy", (cm_scores*100).mean())

#     print("Filtered OUT PL:", np.count_nonzero(filtered_OUT_old_unknown_predictions==filtered_OUT_unknown_true_labels)/len(filtered_OUT_unknown_true_labels)*100)

#     true_labels_paths = {}
#     if save:
#         new_dataset_root = Path("data/PointDA_data_ply") / Path(dataset_name_target)
#         os.makedirs(new_dataset_root, exist_ok=True)

#         for i in range(len(pl_predictions)):
#             current_path = new_dataset_root / classes[int(pl_predictions[i])] / Path("train")
#             os.makedirs(current_path, exist_ok=True)
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(pl_pc[i])
#             o3d.io.write_point_cloud(f"{current_path}/{i}_st.ply" , pcd)
#             true_labels_paths[f"{current_path}/{i}_st.ply"] = (pl_true_label[i], int(pl_predictions[i]), pl_path[i], pl_masks[i])


#         for i in range(len(new_unknown_predictions)):
#             current_path = new_dataset_root / classes[int(new_unknown_predictions[i])] / Path("train")
#             os.makedirs(current_path, exist_ok=True)
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(filtered_unknown_pc[i])
#             o3d.io.write_point_cloud(f"{current_path}/{i}_biknn_st.ply" , pcd)
#             true_labels_paths[f"{current_path}/{i}_biknn_st.ply" ] = (filtered_unknown_true_labels[i], int(new_unknown_predictions[i]), filtered_unknown_path[i], filtered_unknown_masks[i])

#         if target_only:

#             for i in range(len(filtered_OUT_old_unknown_predictions)):
#                 current_path = new_dataset_root / classes[int(filtered_OUT_old_unknown_predictions[i])] / Path("train")
#                 os.makedirs(current_path, exist_ok=True)
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(filtered_OUT_unknown_pc[i])
#                 o3d.io.write_point_cloud(f"{current_path}/{i}_uncertain.ply" , pcd)
#                 true_labels_paths[f"{current_path}/{i}_uncertain.ply" ] = (filtered_OUT_unknown_true_labels[i], int(filtered_OUT_old_unknown_predictions[i]), filtered_OUT_unknown_path[i], filtered_OUT_unknown_masks[i])

#         else:
#             # copy data from source domain
#             for j, c in enumerate(classes):
#                 for phase in ["test", "train"]:
#                     os.makedirs(f"{new_dataset_root}/{c}/{phase}", exist_ok=True)
#                     for f in (new_dataset_root.parent / Path(source_domain) / Path(c) / Path(phase)).glob("*.ply"):
#                         pc = o3d.io.read_point_cloud(str(f))
#                         pc = np.array(pc.points)
#                         pc = normal_pc(pc)

#                         if source_domain == "scannet" or source_domain == "shapenet":
#                             if not (source_domain == "shapenet" and c == "plant"):
#                                 pc = rotate_shape(pc, "x", -np.pi / 2)

#                         name = f.name.replace(".ply", f"_{source_domain}.ply")

#                         pcd = o3d.geometry.PointCloud()
#                         pcd.points = o3d.utility.Vector3dVector(pc)
#                         o3d.io.write_point_cloud(f"{new_dataset_root}/{c}/{phase}/{name}", pcd)
#                         true_labels_paths[f"{new_dataset_root}/{c}/{phase}/{name}"] = (j, j, f, np.zeros((2048)))

#     if log_artifacts:
#         pts_list_train = glob.glob(os.path.join(new_dataset_root, "*", "train", "*.ply"))
#         pts_list_test = glob.glob(os.path.join(new_dataset_root, "*", "test", "*.ply"))

#         train = create_dataset(new_dataset_root, pts_list_train, classes, true_labels_paths)
#         if target_only:
#             test = create_dataset(new_dataset_root, pts_list_test, classes)
#         else:
#             test = create_dataset(new_dataset_root, pts_list_test, classes, true_labels_paths)
#         datasets = [train, test]
#         names = ["train", "test"]
        
#         raw_data = wandb.Artifact(
#             dataset_name_target, type="dataset",
#             description=dataset_name_target + " PL",
#             metadata={
#                     "source dataset": dataset_name_source,
#                     "ckpt": cfg["run_name"],
#                     "source_domain": source_domain,
#                     "target_domain": target_domain,
#                     "sizes": [len(dataset[0]) for dataset in datasets]})

#         for name, data in zip(names, datasets):
#             with raw_data.new_file(name + ".npz", mode="wb") as file:
#                 np.savez(file, x=data[0], y=data[1], path=data[2], truey=data[3], mask=data[4])

#         run.log_artifact(raw_data)

main()
