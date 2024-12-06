# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb

import torch
import matplotlib.pyplot as plt
import os
import argparse
from transformers import AutoFeatureExtractor
import torch
from PIL import Image
from pathlib import Path
from ocr_microservice.model_training.yolos.detr import Detr

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

def plot_results(filename, pil_img, id2label, prob, boxes, show):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    #plt.savefig('tmp/' + filename)
    if show:
        plt.show()
    plt.close()

def visualize_predictions(filename, image, id2label, outputs, threshold, show):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
  
    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0,keep].cpu(), image.size)

    # plot results
    plot_results(filename, image, id2label, probas[keep], bboxes_scaled, show)
     
def process_file(image_path, id2label, model, device, show):
    image = Image.open(image_path)
    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device) # remove batch dimension

    outputs = model(pixel_values=pixel_values)

    visualize_predictions(Path(image_path).name, image, id2label, outputs, 0.9, show)

def main(model_path, image_path, show):
    id2label = {0: 'receipt'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Detr.load_from_checkpoint(model_path,map_location=device,strict=False)
    model.eval()


    if os.path.isfile(image_path):
        process_file(image_path, id2label, model, device, show=show)
    else:
        images = [f for f in os.listdir(args["image"]) if os.path.isfile(os.path.join(args["image"], f))]
        for image in images:
            process_file(os.path.join(image_path,image), id2label, model, device, show=show)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="model")
    ap.add_argument("image", type=str, help="image path or directory")
    args = vars(ap.parse_args())

    main(args['model'], args['image'], show=True)
