# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb

import torchvision
import os
import argparse
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from ocr_microservice.model_training.yolos.detr import Detr

feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)
base_dir = "step3-coco-formatted"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "result.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

def show_example(train_dataset):
    import numpy as np
    import os
    from PIL import Image, ImageDraw

    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = train_dataset.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image n°{}'.format(image_id))
    image = train_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(f'{base_dir}/train', image['file_name']))

    annotations = train_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')

    image.show()

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['labels'] = labels
    return batch

def main(args):
    # todo: put feature extractor here

    train_dataset = CocoDetection(img_folder=f'{base_dir}/train', feature_extractor=feature_extractor)
    test_dataset = CocoDetection(img_folder=f'{base_dir}/test', feature_extractor=feature_extractor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of test examples:", len(test_dataset))

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    #show_example(train_dataset)
    batch_size = 3

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=15)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=15)
    batch = next(iter(train_dataloader))

    model = Detr().init_for_training(train_dataloader, test_dataloader, id2label, batch_size=batch_size, lr=2.5e-5, weight_decay=1e-4)

    trainer = Trainer(max_steps=1000, gradient_clip_val=0.1, accumulate_grad_batches=4)
    trainer.fit(model)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())

    main(args)

