import torch
import config
from torchvision.utils import save_image
import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [x1, y1, x2, y2, prob_score,  class_pred, class_name]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # assert type(bboxes) == list

    

    bboxes = [box for box in bboxes if box[:4] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[:4], reverse=True)
    #conver midpoints to corners
    boxes = xywh2xyxy(bboxes[:,:4])
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[:5] != chosen_box[:5]
            or intersection_over_union(
                torch.tensor(chosen_box[:4]),
                torch.tensor(box[:4]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# def yolov5_nms(detections, iou_threshold=0.5):
#     """
#     Apply non-maximum suppression to YOLOv5 detections to remove overlapping boxes
#     Args:
#         detections (tensor): YOLOv5 detections tensor of size (N, 7)
#         iou_threshold (float): The overlap thresh for suppressing unnecessary boxes
#     Returns:
#         A list of filtered detections after non-maximum suppression, size (M, 7)
#     """
#     # Extract boxes, scores, and classes from the detections tensor
#     for xi, x in enumerate(detections):
#         # print(x.shape)
#         boxes = x[:, :4]
#         # print(boxes.shape)
#         boxes = 
#         scores = x[:, 4]
#         # print(scores.shape)
#         keep_indices = ops.nms(boxes, scores, iou_threshold)
#         print(keep_indices.shape)
#     return detections[: keep_indices]

def save_generated_image(y_fake, folder):
    save_image(y_fake, folder + f"yolo.png")


def save_some_examples(conv_block, cross_attn, gen, val_loader, epoch, folder):
    for batch in val_loader:
        # x = batch['image_vis']
        # y = batch['image_ir']
        # a = batch['target_vis']
        # b = batch['target_ir']
        x, y = batch
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        x_feat = conv_block(x)
        y_feat = conv_block(y)

        attn1, attn2 = cross_attn(x_feat, y_feat)

        gen.eval()
        with torch.no_grad():
            y_fake = gen(x, y, attn1, attn2)
            # y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            # y_fake = y_fake  
            save_image(y_fake, folder + f"Fused/FusedL1_{epoch}.png")
            # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
            save_image(x, folder + f"VIS/VISL1_{epoch}.png")
            save_image(y, folder + f"IR/IRL1_{epoch}.png")
            save_image(attn1, folder + f"attention_map/Attn1_{epoch}.png")
            save_image(attn2, folder + f"attention_map/Attn2_{epoch}.png")
            # if epoch == 0:
            #     save_image(y, folder + f"/label_{epoch}.png")
                # save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        gen.train()

def num_detections(results):
    # Apply non-maximum suppression to remove overlapping detections
    results = non_max_suppression(results, 0.45, 0.7, "corners")

    # Count the number of detections
    num_detections = 0
    for res in results:
        if res is not None:
            num_detections += len(res)
    return num_detections


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
