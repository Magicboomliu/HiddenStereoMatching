import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


def get_bounding_boxes(segmentation_masks):
    """
    Convert instance segmentation masks to bounding boxes.

    Args:
    segmentation_masks (torch.Tensor): A tensor of shape [N, H, W] where N is the number of instances,
                                       H is the height, and W is the width of the masks.

    Returns:
    torch.Tensor: A tensor of shape [N, 4] representing bounding boxes for each instance.
                  Each bounding box is in the format [x_min, y_min, x_max, y_max].
    """
    N, H, W = segmentation_masks.shape
    bounding_boxes = torch.zeros((N, 4), dtype=torch.int64)

    for i in range(N):
        mask = segmentation_masks[i]
        y_indices, x_indices = torch.where(mask)

        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = torch.min(x_indices), torch.max(x_indices)
            y_min, y_max = torch.min(y_indices), torch.max(y_indices)
            
            if x_min>=x_max:
                x_min = x_min -1
                if x_min<0:
                    x_min = 0
            
            if y_min>=y_max:
                y_min = y_min-1
                if y_min<0:
                    y_min = 0
                
            bounding_boxes[i] = torch.tensor([x_min, y_min, x_max, y_max])

    return bounding_boxes