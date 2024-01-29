import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
# import cv2
import random
import numpy as np
from stereosdf.datasets.visualization.vis_info import numpy_function

def draw_2d_bounding_boxes(image, detections):
    """
    Draw bounding boxes on the image.

    Args:
    image (np.ndarray): An image array of shape (H, W, 3).
    detections (np.ndarray): A numpy array of shape (N, 4), where each row is [x_min, y_min, x_max, y_max].

    Returns:
    np.ndarray: The image with bounding boxes drawn.
    """
    for (x_min, y_min, x_max, y_max) in detections:
        start_point = (x_min, y_min)
        end_point = (x_max, y_max)
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2  # Line thickness

        # Draw the bounding box
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image




def draw_segmentation_masks(image, masks, alpha=0.5):
    """
    Draw segmentation masks on the image.

    Args:
    image (np.ndarray): An image array of shape (H, W, 3).
    masks (np.ndarray): A segmentation mask array of shape (H, W, N).
    alpha (float): Transparency for the masks.

    Returns:
    np.ndarray: The image with segmentation masks drawn.
    """
    H, W, N = masks.shape
    colored_image = image.copy()
    


    # Create a color for each instance
    colors = [tuple(random.choices(range(256), k=3)) for _ in range(N)]

    for i in range(N):
        mask = masks[:, :, i]
        color = colors[i]

        # Create a colored mask
        colored_mask = np.zeros((H, W, 3)).astype(np.float32)
        # colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        colored_mask[mask == 1] = color
        

        # Blend the colored mask with the image
        colored_image = cv2.addWeighted(colored_image, 1, colored_mask, alpha, 0)

    return colored_image





LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

def clip_lines_to_front(lines, epsilon=1e-6):

    points_1, points_2 = torch.unbind(lines, dim=-2)
    depths_1, depths_2 = points_1[..., -1:], points_2[..., -1:]

    points_1, points_2 = (
        torch.where(depths_1 > depths_2, points_1, points_2),
        torch.where(depths_1 > depths_2, points_2, points_1),
    )
    depths_1, depths_2 = (
        torch.where(depths_1 > depths_2, depths_1, depths_2),
        torch.where(depths_1 > depths_2, depths_2, depths_1),
    )

    weights = depths_1 / torch.clamp(depths_1 - depths_2, min=epsilon)
    weights = torch.clamp(weights, max=1.0)

    points_2 = points_1 + (points_2 - points_1) * weights
    lines = torch.stack([points_1, points_2], dim=-2)

    masks = points_1[..., -1] > 0

    return lines, masks


def draw_boxes_3d(image, boxes_3d, intrinsic_matrix, 
                    color=None,thickness=None,lineType=None):
    
    # boxes_3d = boxes_3d.flatten(-2, -1)
    
    line_indices=LINE_INDICES + [[0, 5], [1, 4]]
    
    intrinsic_matrix = intrinsic_matrix[:3,:3]
    
    # print(line_indices)
    

    # is_float = image.dtype.kind == "f"

    # if is_float:
    #     image = skimage.img_as_ubyte(image)

    # image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # NOTE: the KITTI-360 "annotation" format is different from the KITTI-360 "evaluation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552

    for box_3d in boxes_3d:

        lines = box_3d[line_indices, ...]
        lines, masks = numpy_function(clip_lines_to_front)(lines)
        
        lines = lines @ intrinsic_matrix.T
        
        lines = lines[..., :-1] / np.clip(lines[..., -1:], 1e-3, None)

        for (point_1, point_2) in lines[masks, ...]:

            image = cv2.line(
                img=image,
                pt1=tuple(map(int, point_1)),
                pt2=tuple(map(int, point_2)),
                color = color,
                thickness= thickness,
                lineType= lineType,
            )


    return image