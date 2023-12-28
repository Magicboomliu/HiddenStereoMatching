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


# ["left_img","right_img","left_depth","right_depth","left_calib","right_calib",
#                          "left_pose","right_pose","left_seg","right_seg",
#                          "left_2d_box","right_2d_box","left_3d_box","right_3d_box"]


def filter_out_without_interets(input_list):
    return [True if element is not None else False for element in input_list]


def my_collate_fn(batch):
    
    preception_data_left = False
    preception_data_right = False
    
    batched_sample = dict()
    
    if 'left_img' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['left_img'] = torch.stack([item['left_img'] for item in batch])
         
    if 'left_calib' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['left_calib'] = torch.stack([item['left_calib'] for item in batch])
    if 'left_pose' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['left_pose'] = torch.stack([item['left_pose'] for item in batch])
        
    if 'right_img' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['right_img'] = torch.stack([item['right_img'] for item in batch])
    
    if 'right_calib' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['right_calib'] = torch.stack([item['right_calib'] for item in batch])
    
    if 'right_pose' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['right_pose'] = torch.stack([item['right_pose'] for item in batch])
        
    if 'left_depth' in cfg.DATA.VISIBLE_LISTS:
        batched_sample['left_depth'] = torch.stack([item['left_depth'] for item in batch])
    
    
    
    
    if 'left_seg' in cfg.DATA.VISIBLE_LISTS:
        # contains None
        batched_sample['left_seg'] = [item['left_seg'] for item in batch]
        preception_data_left = True
    
    if 'right_seg' in cfg.DATA.VISIBLE_LISTS:
        # contains None
        batched_sample['right_seg'] = [item['right_seg'] for item in batch]
        preception_data_right = True
    
    if 'left_2d_box' in cfg.DATA.VISIBLE_LISTS:
        # contains None
        batched_sample['left_2d_box'] = [item['left_2d_box'] for item in batch]
        preception_data_left = True

    if 'right_2d_box' in cfg.DATA.VISIBLE_LISTS:
        # contains None
        batched_sample['right_2d_box'] = [item['right_2d_box'] for item in batch]
        preception_data_right = True
    
    if "left_3d_box" in cfg.DATA.VISIBLE_LISTS:
        batched_sample['left_boxes_3d'] = [item['left_boxes_3d'] for item in batch]
        preception_data_left = True
    
    if "right_3d_box" in cfg.DATA.VISIBLE_LISTS:
        batched_sample['right_boxes_3d'] = [item['right_boxes_3d'] for item in batch] 
        preception_data_right = True
    
    if preception_data_left == True:
        batched_sample['left_labels'] = [item['left_labels'] for item in batch] 
    if preception_data_right ==True:
        batched_sample['right_labels'] = [item['right_labels'] for item in batch]
    
    

    
    return batched_sample
        



if __name__=="__main__":
    
    train_transform_list = [
                    # transforms.CenterCrop([300,900]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ]
    train_transform = transforms.Compose(train_transform_list)
    
    
    # explore the kitti-360 dataloader
    dataset = KITTI360_Dataset(cfg=cfg,
                               mode='train',
                               transform=train_transform)
    

    train_loader = DataLoader(dataset, batch_size = 4, \
                                shuffle = True, num_workers = 2, \
                                pin_memory = True,
                                collate_fn=my_collate_fn)

    for i_batch, sample_batched in enumerate(train_loader):
        left_images = sample_batched['left_img'] # (B,3,H,W)
        left_pose = sample_batched['left_pose']  #(B,4,4)
        left_calib = sample_batched['left_calib'] #(B,3,4)
        
        
        right_images = sample_batched['right_img']
        right_pose = sample_batched['right_pose']
        right_calib = sample_batched['right_calib']
        
        
        left_det2d = sample_batched['left_2d_box'] #(N,)
        left_seg = sample_batched['left_seg']      #(N,)
        left_det3d = sample_batched['left_boxes_3d'] #(N,)

        right_det2d = sample_batched['right_2d_box'] #(N,)
        right_seg = sample_batched['right_seg']      #(N,)
        right_det3d = sample_batched['right_boxes_3d'] #(N,)
        
        left_depth = sample_batched['left_depth']
        
        left_labels = sample_batched['left_labels']
        
        

        
        

        
        if i_batch>100:
            break        
        
        
    # for idx, sample in enumerate(dataset):
        
    #     left_image = sample['left_img']
    #     right_image = sample['right_img']
    #     left_depth = sample['left_depth']
    #     left_calib = sample['left_calib']
    #     right_calib = sample['right_calib']
    #     left_pose = sample['left_pose']
    #     right_pose = sample['right_pose']
    #     left_seg_mask = sample['left_seg']
    #     right_seg_mask = sample['right_seg']
    #     left_det_2d = sample['left_2d_box']
    #     right_det_2d = sample['right_2d_box']
    #     left_det_3d = sample["left_boxes_3d"]
    #     right_det_3d = sample['right_boxes_3d']
    #     left_labels = sample['left_labels']
    #     right_labels = sample['right_labels']
        
        
    #     print("{}/{}".format(idx,len(dataset)))
        
        
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
        