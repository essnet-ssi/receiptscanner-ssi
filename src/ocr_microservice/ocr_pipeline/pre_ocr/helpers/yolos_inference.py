# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from io import BytesIO
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection
import torch
from pytorch_lightning import Trainer
import PIL
import torchvision.transforms as transforms 
from pathlib import Path

feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_result(pil_img, id2label, prob, box):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    p, (xmin, ymin, xmax, ymax), c = prob,box.tolist(),colors
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color="yellow", linewidth=3))
    cl = p.argmax()
    text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
    ax.text(xmin, ymin, text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_results(pil_img, id2label, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf).convert('RGB')

def visualize_predictions(image, id2label, outputs, threshold):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0,keep].cpu(), image.size)

    bboxes_scaled_list = bboxes_scaled.tolist()

    contour_image = plot_results(image, id2label, probas[keep], bboxes_scaled_list)

    # crop
    if len(bboxes_scaled_list):
        cropped_image = image.crop(bboxes_scaled_list[0])
    else:
        cropped_image = image
    
    return contour_image, cropped_image
     
#class CocoDetection(torchvision.datasets.CocoDetection):
    #def __init__(self, img_folder, feature_extractor, train=True):
        #ann_file = os.path.join(img_folder, "result.json")
        #super(CocoDetection, self).__init__(img_folder, ann_file)
        #self.feature_extractor = feature_extractor

    #def __getitem__(self, idx):
        ## read in PIL image and target in COCO format
        #img, target = super(CocoDetection, self).__getitem__(idx)
        
        ## preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        #image_id = self.ids[idx]
        #target = {'image_id': image_id, 'annotations': target}
        #encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        #pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        #target = encoding["labels"][0] # remove batch dimension

        #return pixel_values, target

class Detr(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small", num_labels=1, ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896

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


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['labels'] = labels
    return batch

def process_file(image, model, device):
    id2label = {0: 'receipt'}
    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device) # remove batch dimension

    outputs = model(pixel_values=pixel_values)

    return visualize_predictions(image, id2label, outputs, 0.9)

def load_model(checkpoint: str, device):
    model = Detr.load_from_checkpoint(checkpoint, map_location=device, strict=False)
    model.eval()
    return model
