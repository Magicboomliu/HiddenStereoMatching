from __future__ import division
from genericpath import samefile
from typing import Any
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        
        if 'left_img' in sample.keys():
            left = np.transpose(sample['left_img'], (2, 0, 1))  # [3, H, W]
            sample['left_img'] = torch.from_numpy(left) / 255.
        if 'right_img' in sample.keys():
            right = np.transpose(sample['right_img'], (2, 0, 1))
            sample['right_img'] = torch.from_numpy(right) / 255.
        if 'left_depth' in sample.keys():
            sample['left_depth'] = torch.from_numpy(sample['left_depth'])
        
        if 'left_calib' in sample.keys():
            sample['left_calib'] = torch.from_numpy(sample['left_calib'])
        
        if 'right_calib'  in sample.keys():
            sample['right_calib'] = torch.from_numpy(sample['right_calib'])
        
        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left_img', 'right_img']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


def cut_or_pad_img(img, targetHW,depth=None,seg=None):

    bbox_shift = np.array([[0, 0, 0, 0]])

    t_H, t_W = targetHW
    H, W = img.shape[0], img.shape[1]

    padW = np.abs(t_W - W)
    half_padW = int(padW//2)
    

    # crop
    if W > t_W:
        img = img[:, half_padW:half_padW+t_W]
        bbox_shift[0, [0, 2]] -= half_padW
        if seg is not None:
            seg = seg[:,:, half_padW:half_padW+t_W]
        if depth is not None:
            
            depth = depth[:, half_padW:half_padW+t_W]
        
    # pad
    elif W < t_W:
        img = np.pad(img, [(0, 0), (half_padW, padW-half_padW), (0, 0)], 'constant')
        bbox_shift[0, [0, 2]] += half_padW

    # crop
    padH = np.abs(t_H - H)
    if H > t_H:
        img = img[padH:, :]
        bbox_shift[0, [1, 3]] -= padH
        if seg is not None:
            seg = seg[:,padH:, :]
        if depth is not None:
            depth = depth[padH:, :]
            
    # pad
    elif H < t_H:
        padH = t_H - H
        img = np.pad(img, [(padH, 0), (0, 0), (0, 0)], 'constant')
        bbox_shift[0, [1, 3]] += padH
    
    if depth is not None:
        return depth
    if seg is not None:
        return seg
    
    return img, bbox_shift


def adjust_intrinsics_after_crop(K, original_size, new_size):
    H, W = original_size
    H_new, W_new = new_size
    
    cx, cy = K[0, 2], K[1, 2]

    # Adjusting the principal point
    cx_new = cx - (W - W_new) / 2
    cy_new = cy - (H - H_new) / 2
    

    K_new = K.copy()
    K_new[0, 2] = cx_new
    K_new[1, 2] = cy_new

    return K_new



class CenterCrop(object):
    def __init__(self,targetHW):
        self.targetHW = targetHW

    def __call__(self,sample):
        
        # left image and right images
        original_left_img = sample['left_img']
        original_right_img = sample['right_img']
    
        
        if 'left_img' in sample.keys():
            sample['left_img'],bbox_shift_left = cut_or_pad_img(sample['left_img'],self.targetHW)
        
        if 'right_img' in sample.keys():
            sample['right_img'],bbox_shift_right = cut_or_pad_img(sample['right_img'],self.targetHW)
        
        if 'left_2d_box' in sample.keys():
            
            if sample['left_2d_box'] is not None:
                sample['left_2d_box'] = sample['left_2d_box'] + torch.from_numpy(bbox_shift_left)

        if 'right_2d_box' in sample.keys():
            if sample['right_2d_box'] is not None:
                sample['right_2d_box'] = sample['right_2d_box'] + torch.from_numpy(bbox_shift_right)


        if 'left_depth' in sample.keys():            
            sample['left_depth'] = cut_or_pad_img(original_left_img,self.targetHW,
                                                depth=sample['left_depth'],seg=None)
            
        if 'left_seg' in sample.keys():
            if sample['left_seg'] is not None:
                sample['left_seg'] = cut_or_pad_img(original_left_img,self.targetHW,
                                                depth=None,seg=sample['left_seg'])
        
        if 'right_seg' in sample.keys():
            if sample['right_seg'] is not None:
                sample['right_seg'] = cut_or_pad_img(original_right_img,self.targetHW,
                                                depth=None,seg=sample['right_seg'])
        
        
        if 'left_calib' in sample.keys():
            sample['left_calib'] = adjust_intrinsics_after_crop(K=sample['left_calib'],
                                                                original_size=original_left_img.shape[:2],
                                                                new_size=self.targetHW)
        
        if 'right_calib' in sample.keys():
            sample['right_calib'] = adjust_intrinsics_after_crop(K=sample['right_calib'],
                                                                original_size=original_right_img.shape[:2],
                                                                new_size=self.targetHW)
            
        
           
        return sample
        
        
    


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['left_img'] = F.adjust_contrast(sample['left_img'], contrast_factor)
            sample['right_img'] = F.adjust_contrast(sample['right_img'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left_img'] = F.adjust_gamma(sample['left_img'], gamma)
            sample['right_img'] = F.adjust_gamma(sample['right_img'], gamma)

        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['left_img'] = F.adjust_brightness(sample['left_img'], brightness)
            sample['right_img'] = F.adjust_brightness(sample['right_img'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left_img'] = F.adjust_hue(sample['left_img'], hue)
            sample['right_img'] = F.adjust_hue(sample['right_img'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['left_img'] = F.adjust_saturation(sample['left_img'], saturation)
            sample['right_img'] = F.adjust_saturation(sample['right_img'], saturation)
        
        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample

class ToPILImage(object):

    def __call__(self, sample):
        sample['left_img'] = Image.fromarray(sample['left_img'].astype('uint8'))
        sample['right_img'] = Image.fromarray(sample['right_img'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left_img'] = np.array(sample['left_img']).astype(np.float32)
        sample['right_img'] = np.array(sample['right_img']).astype(np.float32)

        return sample