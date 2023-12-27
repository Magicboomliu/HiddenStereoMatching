import sys
import numpy as np
import torch

from stereosdf.datasets.visualization.semantic_visualization import draw_2d_bounding_boxes,draw_segmentation_masks,draw_boxes_3d
from stereosdf.datasets.kitti_360_dataset import KITTI360_Dataset
from stereosdf.datasets import transforms
from config import cfg
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]






if __name__=="__main__":
    
    train_transform_list = [transforms.CenterCrop([300,900]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ]
    train_transform = transforms.Compose(train_transform_list)
    
    
    # explore the kitti-360 dataloader
    dataset = KITTI360_Dataset(cfg=cfg,
                               mode='train',
                               transform=train_transform)
    

    for idx, sample in enumerate(dataset):
        
        left_image = sample['left_img']
        right_image = sample['right_img']
        left_depth = sample['left_depth']
        left_calib = sample['left_calib']
        right_calib = sample['right_calib']
        left_pose = sample['left_pose']
        right_pose = sample['right_pose']
        left_seg_mask = sample['left_seg']
        right_seg_mask = sample['right_seg']
        left_det_2d = sample['left_2d_box']
        right_det_2d = sample['right_2d_box']
        left_det_3d = sample["left_boxes_3d"]
        right_det_3d = sample['right_boxes_3d']
        left_labels = sample['left_labels']
        right_labels = sample['right_labels']
        
        
        print("{}/{}".format(idx,len(dataset)))
        
        
        # left_mask = torch.where(left_labels==0,True,False)
        
        # print(left_mask)
        
        # images_projeceted3d = draw_boxes_3d(image=right_image,boxes_3d=right_det_3d,
        #               intrinsic_matrix=right_calib,
        #               color=[0,255,0],
        #               thickness=2)
        
        
        # images_with_det2d = draw_2d_bounding_boxes(image=left_image.permute(1,2,0).cpu().numpy(),
        #                                            detections=left_det_2d.cpu().numpy())
        # images_with_det2d_seg = draw_segmentation_masks(image=right_image.permute(1,2,0).cpu().numpy(),
        #                                                 masks=right_seg_mask.permute(1,2,0).cpu().numpy(),alpha=0.9)

        
        # plt.figure(figsize=(15,7))
        # plt.axis("off")
        # plt.imshow(images_with_det2d_seg)
        # plt.show()



        

        # break
        