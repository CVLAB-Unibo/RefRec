import torch
from torch import nn
import pytorch_lightning as pl
from networks.factory import get_model
from utils.losses import get_loss_fn
from hesiod import hcfg
from hesiod import get_cfg_copy
from utils.optimizers import get_optimizer
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
if "minkowski" in hcfg("net.name"):
    import MinkowskiEngine as ME


class Classifier(pl.LightningModule):
    def __init__(self, dm, device):
        super().__init__()

        self.net = get_model(device)
        self.ema = get_model(device)
        self.dm = dm
        self.best_accuracy_source = 0
        self.best_accuracy_target = 0
        self.alpha_teacher = 0.99
        self.loss_fn = get_loss_fn()
        self.train_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_source = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_target = pl.metrics.Accuracy(compute_on_step=False)
        self.save_hyperparameters(get_cfg_copy())
        self.dl_target_pl = iter(self.dm.train_dataloader_pl())
        self.loss_fn_target = nn.CrossEntropyLoss(reduction="none")

    def create_ema_model(self):
        for param in self.ema.parameters():
            param.detach_()
        mp = list(self.net.parameters())
        mcp = list(self.ema.parameters())
        n = len(mp)
        for i in range(0, n):
            mcp[i].data[:] = mp[i].data[:].clone()

    def update_ema_variables(self):
        # Use the "true" average until the exponential average is more correct
        alpha_teacher = min(1 - 1 / (self.global_step + 1), self.alpha_teacher)
        for ema_param, param in zip(self.ema.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)

        for t, s in zip(self.ema.buffers(), self.net.buffers()):
            if not t.dtype == torch.int64:
                t.data.mul_(alpha_teacher).add_(s.data, alpha=1 - alpha_teacher)
    
    def on_train_start(self):
        self.writer = SummaryWriter(wandb.run.dir)

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        self.net.apply(set_bn_eval)
        if hcfg("use_proto"):
            project_name = hcfg("project_name")
            # self.protoypes = np.load(f"prototypes/{project_name}_proto_perc.npy")
            self.protoypes = np.load("prototypes/scnn2m_proto.npy")
            self.protoypes = torch.from_numpy(self.protoypes).to(device=self.device)
            self.protoypes = F.normalize(self.protoypes, dim=1)
        self.create_ema_model()

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, weight, length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return weight * self.sigmoid_rampup(self.current_epoch, length)

    def forward(self, x):
        embeddings, output = self.net(x, embeddings=True, target=True)
        return embeddings, output

    def loss(self, xb, yb, target=False):
        if target:
            embeddings, output = self.net(xb, embeddings=True, target=True)
        else:
            embeddings, output = self.net(xb, embeddings=True)
        loss = self.loss_fn(output, yb)
        return embeddings, output, loss

    def training_step(self, batch, batch_idx):
        weight = self.get_current_consistency_weight(hcfg("max_weight"), length=hcfg("warm_up"))
        coords = batch["coordinates"]
        feats = batch["features"]
        labels = batch["labels"]

        _, logits, loss_source = self.loss(coords, labels)

        loss = loss_source
        try:
            batch_target = next(self.dl_target_pl)
        except:
            self.dl_target_pl = iter(self.dm.train_dataloader_pl())
            batch_target = next(self.dl_target_pl)
        
        coords_target = batch_target["coordinates"].to(self.device)
        labels_target = batch_target["labels"].to(self.device)
        self.train_acc(F.softmax(logits, dim=1), labels)

        with torch.no_grad():
            embeddings_t, output_t = self.ema(coords_target, embeddings=True, target=True)
            _, output_source_branch_ema = self.ema(coords_target, embeddings=True, target=False)
        predictions_teacher = torch.argmax(output_t+output_source_branch_ema, dim=1)
        embeddings_s, output_s = self.net(coords_target, embeddings=True, target=True)
        embeddings_s = F.normalize(embeddings_s.squeeze(2), dim=1)
        diff_proto = embeddings_s.unsqueeze(1).repeat(1,hcfg("num_classes"),1).detach() - self.protoypes
        norms = torch.linalg.norm(diff_proto, dim=2)
        target_weights = F.softmax(-norms, dim=1)
        target_weights = [w[predictions_teacher[i]] for i, w in enumerate(target_weights)]
        target_weights = torch.tensor(target_weights, device=self.device)

        target_loss = (self.loss_fn_target(output_s, labels_target)*(target_weights)).mean()*(1-weight) + (self.loss_fn_target(output_s, predictions_teacher)*(target_weights)).mean()*weight
        loss += target_loss

        if self.global_step % 500 == 0 and self.global_step != 0:
            self.logger.experiment.log(
                {
                    "train/loss": loss_source.item(),
                    "train/target_loss": target_loss.item(),
                    # "train/centroid_loss_target": centroid_loss_target.item(),
                    "train/weight": weight
                }, commit=False, step=self.global_step
            )

        self.update_ema_variables()
        return loss

    def training_epoch_end(self, outputs):

        self.logger.experiment.log(
            {
                "train/accuracy": self.train_acc.compute(),
            },
            commit=True,
            step=self.global_step,
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        coords = batch["coordinates"]
        feats = batch["features"]
        labels = batch["labels"]

        if "minkowski" in hcfg("net.name"):
            # coords = ME.SparseTensor(coordinates=coords, features=feats)
            coords = ME.TensorField(coordinates=coords, features=feats)

        if dataloader_idx == 0:
            embeddings, predictions, loss = self.loss(coords, labels)
            self.valid_acc_source(F.softmax(predictions, dim=1), labels)
        if dataloader_idx == 1:
            embeddings, predictions, loss = self.loss(coords, labels, target=True)
            self.valid_acc_target(F.softmax(predictions, dim=1), labels)

        predictions = (predictions.argmax(dim=1), labels)
        return {"loss": loss, "predictions": predictions,
                "embeddings": embeddings, "labels": labels
        }

    def validation_epoch_end(self, validation_step_outputs):
        # print("\n")

        valid_acc_source = self.valid_acc_source.compute().item()
        print("SOURCE:", valid_acc_source)
        source_data = validation_step_outputs[0]
        predictions_source = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in source_data]
        )
        label_source = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in source_data]
        )
        avg_loss_source = np.mean([x["loss"].item() for x in source_data])

        valid_acc_target = self.valid_acc_target.compute().item()
        print("TARGET:", valid_acc_target)
        target_data = validation_step_outputs[1]
        predictions_target = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in target_data]
        )
        label_target = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in target_data]
        )
        avg_loss = np.mean([x["loss"].item() for x in target_data])

        if valid_acc_target >= self.best_accuracy_target and self.global_step != 0:
            self.logger.log_metrics({"best_accuracy": valid_acc_target})
            self.best_accuracy_target = valid_acc_target

        # take best model according to source test set
        if valid_acc_source >= self.best_accuracy_source and self.global_step != 0:
            self.logger.log_metrics({
                "final_accuracy": valid_acc_target,
                "best_accuracy_source": valid_acc_source
            }
            )
            self.best_accuracy_source = valid_acc_source

        self.logger.experiment.log(
            {
                "valid/source_loss": avg_loss_source,
                "valid/target_loss": avg_loss,
                "valid/target_accuracy": valid_acc_target,
            },
            commit=False,
        )
        self.log("valid/source_accuracy", valid_acc_source)
        self.valid_acc_target.reset()
        self.valid_acc_source.reset()

        # write embeddings on last epochs for source and target validation data
        if self.current_epoch==hcfg("epochs")-1:
            self.logger.experiment.log(
            {
                "confusion_source": wandb.plot.confusion_matrix(
                    preds=predictions_source,
                    y_true=label_source,
                    class_names=self.dm.train_ds.categories,
                ),
                "confusion_target": wandb.plot.confusion_matrix(
                    preds=predictions_target,
                    y_true=label_target,
                    class_names=self.dm.train_ds.categories,
                ),
            },
            commit=False,
            )

            source_data = validation_step_outputs[0]
            source_embeddings = np.concatenate(
                [x["embeddings"].squeeze().cpu().numpy() for x in source_data]
            )
            source_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in source_data]
            )
            self.writer.add_embedding(source_embeddings, metadata=source_labels, global_step=self.global_step, tag="source")

            target_data = validation_step_outputs[1]
            target_embeddings = np.concatenate(
                [x["embeddings"].squeeze().cpu().numpy() for x in target_data]
            )
            target_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in target_data]
            )
            # self.writer.add_embedding(target_embeddings, metadata=target_labels, global_step=self.global_step, tag="target")

    def on_train_end(self) -> None:
        self.writer.close()
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        print("starting final evaluation")
        self.validation_step(batch, batch_idx, 1)
        print("end final evaluation")

    def test_epoch_end(self, outputs):
        target_accuracy = self.valid_acc_target.compute().item()
        print("TARGET:", target_accuracy)
        self.log("final_accuracy", target_accuracy)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_accuracy_target"] = self.best_accuracy_target
        checkpoint["best_accuracy_source"] = self.best_accuracy_source

    def on_load_checkpoint(self, checkpointed_state):
        self.best_accuracy_target = checkpointed_state["best_accuracy_target"]
        self.best_accuracy_source = checkpointed_state["best_accuracy_source"]

    def configure_optimizers(self):
        opt = get_optimizer(self.net)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.hparams.epochs)
        return [opt], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]
