from hesiod import hcfg
from networks.pointnet import Pointnet
from networks.reconstruction_net import ReconstructionNet
import torch
import wandb
import glob

def get_model(device):

    if hcfg("net.name") == "reconstruction":
        model = ReconstructionNet(device, feat_dims=hcfg("net.feat_dims"))
    else:
        model = Pointnet(num_class=hcfg("num_classes"), device=device, feat_dims=hcfg("net.feat_dims"), target_cls=hcfg("target_cls"))

    if hcfg("restore_weights") != "null":
        model_artifact = wandb.run.use_artifact(hcfg("restore_weights")+ ":latest", type='model')
        model_dir = model_artifact.download()
        model_paths = [path for path in glob.glob(model_dir+"/*.ckpt")] 
        saved_state_dict = torch.load(model_paths[0])
        
        if "state_dict" in saved_state_dict:
            saved_state_dict = saved_state_dict["state_dict"]
            new_params = model.state_dict().copy()
            start_from = 1
            for it, i in enumerate(saved_state_dict):
                i_parts = i.split('.')
                if '.'.join(i_parts[start_from:]) in new_params.keys():
                    new_params['.'.join(i_parts[start_from:])] = saved_state_dict[i]
                    if it ==0:
                        print("####################### loading from" + model_paths[0] + " #######################")
            model.load_state_dict(new_params)
    return model