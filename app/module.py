import pytorch_lightning as pl
import torch
from timm.loss import AsymmetricLossSingleLabel
from timm.optim import create_optimizer_v2
from torchmetrics import Accuracy, F1Score, MetricCollection
from transformers import ViltForQuestionAnswering, ViltProcessor

NUM_CLASSES = 4507


class ViltVQAModule(pl.LightningModule):
    def __init__(
        self,
        model: ViltForQuestionAnswering,
        processor: ViltProcessor,
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.processor = processor
        self.optimizer = optimizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # self.loss = AsymmetricLossSingleLabel()
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        accuracy = Accuracy(num_classes=NUM_CLASSES, task="multiclass", top_k=1)
        f1 = F1Score(num_classes=NUM_CLASSES, average="macro")
        metrics = MetricCollection([accuracy, f1])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.model(**batch)
        loss = self.loss(output.logits, labels)

        # label_argmax = torch.argmax(labels, dim=-1)
        metric = self.train_metrics(output.logits, labels)
        metric["train/loss"] = loss

        self.log_dict(metric, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.model(**batch)
        loss = self.loss(output.logits, labels)

        # label_argmax = torch.argmax(labels, dim=-1)
        metric = self.train_metrics(output.logits, labels)
        metric["val/loss"] = loss

        self.log_dict(metric, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.model.named_parameters()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = create_optimizer_v2(
            optimizer_grouped_parameters,
            opt=self.optimizer,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
