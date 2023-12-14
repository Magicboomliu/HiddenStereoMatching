import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Get_the_PickleFile(filepath):
    with open(filepath,'rb') as f:
        data = pickle.load(f)
    return data


def draw2d_bbox(image,det_2d):
    nums_instances = det_2d.shape[0]
    base_image = np.copy(image)
    for i in range(nums_instances):
            bbox_2d = det_2d[i]
            cv2.rectangle(
            base_image,
            (int(bbox_2d[0]), int(bbox_2d[1])),
            (int(bbox_2d[2]), int(bbox_2d[3])),
            [0,255,0],
            2)
    return base_image

def get_seg_mask(seg_mask):
    nums_instances = seg_mask.shape[0]

    total_seg = np.zeros((seg_mask.shape[-2],seg_mask.shape[-1]))
    for i in range(nums_instances):
         cur_seg_mask = seg_mask[i]
         total_seg = total_seg + cur_seg_mask
    
    total_seg = total_seg.astype(np.float16)
    total_seg = np.clip(total_seg,a_min=0,a_max=1.0)

    return total_seg




if __name__=="__main__":

    left_image = "/home/liuzihua/InternImage/detection/example2.png"

    left_image = np.array(Image.open(left_image))

    path = "/home/liuzihua/InternImage/detection/demo/result.pkl"
    data = Get_the_PickleFile(path)
    detection2d = data['det']['car'][2:3,:]
    segmenation = data['seg']['car'][2:3,:]

    det_2d_image = draw2d_bbox(left_image,detection2d)

    total_seg = get_seg_mask(segmenation)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(det_2d_image/255)

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(total_seg)
    plt.show()
    pass