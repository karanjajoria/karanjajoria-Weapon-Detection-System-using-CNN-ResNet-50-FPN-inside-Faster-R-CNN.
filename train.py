import os
import torch
from tqdm import tqdm
import torchvision
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import math
import sys
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
# python train.py --config configs/train.json --epochs 40 --batch-size 1 --device cuda

# --------- CPU/GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# If you also want to show GPU name when available:
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------- Utils ----------
def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def reduce_dict(input_dict, average=True):
    """
    Reduces the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as input_dict.
    """
    world_size = 1  # single GPU only
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# --------- Training & Evaluation ----------
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")

    for i, (images, targets) in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    torch.cuda.empty_cache()


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    print("Evaluating...")

    all_preds = []
    all_labels = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            true_labels = target["labels"].cpu().numpy()

            if len(output["scores"]) > 0:  # if model predicted something
                best_idx = output["scores"].argmax().item()
                pred_label = output["labels"][best_idx].cpu().item()
            else:
                pred_label = 0  # background / no detection

            # use majority GT label (if multiple objects, just pick first)
            true_label = true_labels[0] if len(true_labels) > 0 else 0

            all_preds.append(pred_label)
            all_labels.append(true_label)

    # ✅ Now lengths will match
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print("Confusion Matrix:\n", cm)



# --------- Transform wrapper ----------
class ComposeWithTarget:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            # Only apply ToTensor if the image is not already a Tensor
            if isinstance(image, torch.Tensor) and isinstance(t, T.ToTensor):
                continue
            image = t(image)
        return image, target


def get_transform(train):
    transforms = []
    # Make sure ToTensor is always first, but only applied if needed
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return ComposeWithTarget(transforms)



# --------- Custom Dataset ----------
class CocoDetectionWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for obj in anns:
            xmin, ymin, w, h = obj["bbox"]
            xmax, ymax = xmin + w, ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # clamp labels to avoid "Target out of bounds"
        labels = torch.clamp(labels, min=1, max=len(self.coco.getCatIds()))

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms is not None:
            img, target_out = self.transforms(img, target_out)

        return img, target_out


# --------- Main training ----------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dataset paths
    data_root = os.path.join("data", "coco")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = CocoDetectionWrapper(
        train_dir, os.path.join(train_dir, "_annotations.coco.json"),
        transforms=get_transform(train=True)
    )
    val_dataset = CocoDetectionWrapper(
        val_dir, os.path.join(val_dir, "_annotations.coco.json"),
        transforms=get_transform(train=False)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    # number of classes = categories + background
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    print(f"Detected {num_classes-1} categories, using {num_classes} classes including background.")

    # model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # training loop
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, val_loader, device=device)

    torch.save(model.state_dict(), "fasterrcnn_weapon.pth")


# --------- Entry point ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    main(args)
