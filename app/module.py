import pytorch_lightning as pl
import torch
from colossalai.nn.optimizer import HybridAdam
from timm.loss import AsymmetricLossSingleLabel
from torchmetrics import Accuracy, F1Score, MetricCollection
from transformers import ViltForQuestionAnswering, ViltProcessor

NUM_CLASSES = 4507


class ViltVQAModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        processor: ViltProcessor,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.processor = processor

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.loss = AsymmetricLossSingleLabel()

        accuracy = Accuracy(num_classes=NUM_CLASSES, task="multiclass", top_k=1)
        f1 = F1Score(num_classes=NUM_CLASSES, average="macro")
        metrics = MetricCollection([accuracy, f1])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def configure_sharded_model(self) -> None:
        self.model = ViltForQuestionAnswering.from_pretrained(self.model_name)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        logits = output.logits
        label_oh = torch.nn.functional.one_hot(batch["labels"], num_classes=NUM_CLASSES)
        loss = self.loss(logits, label_oh)

        metric = self.train_metrics(logits, batch["labels"])
        metric["train/loss"] = loss

        self.log_dict(metric, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        logits = output.logits
        label_oh = torch.nn.functional.one_hot(batch["labels"], num_classes=NUM_CLASSES)
        loss = self.loss(logits, label_oh)

        metric = self.val_metrics(logits, batch["labels"])
        metric["val/loss"] = loss

        self.log_dict(metric, on_epoch=True)
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

        optimizer = HybridAdam(
            optimizer_grouped_parameters,
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