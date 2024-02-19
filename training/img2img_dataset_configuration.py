import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from dataloader.kitti_image_enhance_loader import KITTIImageEnhanceDataset
from torch.utils.data import DataLoader

from dataloader import transforms
import os
import random


# Get Dataset Here
def prepare_dataset(data_name,
                    datapath=None,
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):
    
    # set the config parameters
    dataset_config_dict = dict()
    
    if data_name == 'kitti':
        train_transform_list = [
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        train_transform = transforms.Compose(train_transform_list)

        val_transform_list = [transforms.ToTensor(),
                        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        val_transform = transforms.Compose(val_transform_list)


        train_dataset = KITTIImageEnhanceDataset(root_dir=os.path.join(datapath, 'train'), transform=train_transform)
        test_dataset = KITTIImageEnhanceDataset(root_dir=os.path.join(datapath, 'val'), transform=val_transform)
    else:
        raise NotImplementedError


    img_height, img_width = train_dataset.get_img_size()


    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader,test_loader),dataset_config_dict

def Disparity_Normalization(disparity):
    min_value = torch.min(disparity)
    max_value = torch.max(disparity)
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity

def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768, base=32):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    # Calculate new size
    new_height, new_width = int(original_H * downscale_factor), int(original_W * downscale_factor)

    # Adjust to make divisible by base
    new_height = base * round(new_height / base)
    new_width = base * round(new_width / base)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         size=(new_height, new_width),mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor
    
def resize_small_res_tensor(input_tensor,is_disp=False, recom_resolution=512):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = max(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    # Calculate new size
    new_height, new_width = int(original_H * downscale_factor), int(original_W * downscale_factor)

    # Adjust to make divisible by base
    # Adjust to make divisible by base
    base =32
    new_height = base * round(new_height / base)
    new_width = base * round(new_width / base)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         size=(new_height, new_width),mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor
    
def random_crop_batch(batch_tensor, target_height, target_width):
    """
    Randomly crops a batch of image tensors to the specified size.
    
    Args:
    batch_tensor (Tensor): The input batch of images with shape [B, C, H, W].
    target_height (int): The target height of the crop.
    target_width (int): The target width of the crop.

    Returns:
    Tensor: The batch of randomly cropped images.
    """
    _, _, height, width = batch_tensor.shape

    # Ensure the target size is smaller than the original size
    if target_height > height or target_width > width:
        raise ValueError("Target size must be smaller than the original size")

    # Randomly choose the top-left corner of the cropping area
    top = random.randint(0, height - target_height)
    left = random.randint(0, width - target_width)

    cropped_tensors = batch_tensor[:, :, top:top + target_height, left:left + target_width]

    # Concatenate all the cropped images along the batch dimension
    return cropped_tensors