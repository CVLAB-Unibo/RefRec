import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from networks.factory import get_model
from utils.losses import get_loss_fn
from hesiod import hcfg
from hesiod import get_cfg_copy
from utils.optimizers import get_optimizer
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Classifier(pl.LightningModule):
    def __init__(self, dm, device):
        super().__init__()

        self.net = get_model(device)
        self.dm = dm
        self.best_accuracy_source = 0
        self.best_accuracy_target = 0
        self.loss_fn = get_loss_fn()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc_source = torchmetrics.Accuracy()
        self.valid_acc_target = torchmetrics.Accuracy()
        self.save_hyperparameters(get_cfg_copy())

    def on_train_start(self):
        self.writer = SummaryWriter(wandb.run.dir)
        self.net.apply(self.set_bn_eval)

    def set_bn_eval(self, m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if hasattr(m, 'weight'):
                m.weight.requires_grad_(False)
            if hasattr(m, 'bias'):
                m.bias.requires_grad_(False)
            m.eval()
            # print(m.training)

    def forward(self, x):
        embeddings, output = self.net(x, embeddings=True)
        return embeddings, output

    def loss(self, xb, yb):
        embeddings, output = self(xb)
        loss = self.loss_fn(output, yb)
        return embeddings, output, loss

    def training_step(self, batch, batch_idx):
        coords = batch["coordinates"]
        labels = batch["labels"]

        _, predictions, loss = self.loss(coords, labels)

        self.train_acc(F.softmax(predictions, dim=1), labels)

        if self.global_step % 500 == 0 and self.global_step != 0:
            self.logger.experiment.log(
                {"train/loss": loss.item()}, commit=False, step=self.global_step
            )
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
        labels = batch["labels"]
        embeddings, predictions, loss = self.loss(coords, labels)

        if dataloader_idx == 0:
            self.valid_acc_source(F.softmax(predictions, dim=1), labels)
        if dataloader_idx == 1:
            self.valid_acc_target(F.softmax(predictions, dim=1), labels)

        predictions = (predictions.argmax(dim=1), labels)
        return {"loss": loss, "predictions": predictions,
                "embeddings": embeddings, "labels": labels
        }

    def validation_epoch_end(self, validation_step_outputs):

        valid_acc_source = self.valid_acc_source.compute().item()
        source_data = validation_step_outputs[0]
        predictions_source = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in source_data]
        )
        label_source = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in source_data]
        )
        avg_loss_source = np.mean([x["loss"].item() for x in source_data])

        valid_acc_target = self.valid_acc_target.compute().item()
        target_data = validation_step_outputs[1]
        predictions_target = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in target_data]
        )
        label_target = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in target_data]
        )
        avg_loss = np.mean([x["loss"].item() for x in target_data])

        if valid_acc_target > self.best_accuracy_target and self.global_step != 0:
            self.logger.log_metrics({"best_accuracy": valid_acc_target})
            self.best_accuracy_target = valid_acc_target

        # take best model according to source test set
        if valid_acc_source > self.best_accuracy_source and self.global_step != 0:
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
            self.writer.add_embedding(target_embeddings, metadata=target_labels, global_step=self.global_step, tag="target")

    def on_train_end(self) -> None:
        self.writer.close()
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)

    def test_epoch_end(self, outputs):
        target_accuracy = self.valid_acc_target.compute().item()
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
