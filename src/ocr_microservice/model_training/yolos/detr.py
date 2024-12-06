from transformers import AutoModelForObjectDetection
import pytorch_lightning as pl
import torch

class Detr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small", num_labels=1, ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896

    def init_for_training(self, train_dataloader, test_dataloader, id2label, batch_size, lr, weight_decay):
        self.batch_size = batch_size
        self.trai_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small", 
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay
        return self

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs
     
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, batch_size=self.batch_size)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=self.batch_size)
        #self.log("training_loss", loss, on_epoch=True, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, batch_size=self.batch_size)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        
        return optimizer

    def train_dataloader(self):
        return self.trai_dataloader

    def val_dataloader(self):
        return self.test_dataloader
