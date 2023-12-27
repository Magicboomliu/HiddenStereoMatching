from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys

from stereosdf.datasets.utils import file_io
from data_prepration.make_annotations import get_bounding_boxes
from stereosdf.datasets.utils.kitti_io import read_disp,read_img

from skimage import io, transform
import numpy as np
from PIL import Image



class KITTI360_Dataset(Dataset):
    def __init__(self,cfg,
                 mode='train',
                 transform=None) -> None:
        super().__init__()
        
        self.mode= mode
        
        # only loaded the visible data
        self.visible_data = cfg.DATA.VISIBLE_LISTS
        self.classes_list = cfg.DATA.USED_CLAESSES 
        
    
        self.datapath_kitti360 = cfg.DATA.ROOT_360_PATH
        self.dataset_name = cfg.DATA.MODE
        self.img_size= cfg.DATA.IMAGE_SIZE
        self.scale_size =cfg.DATA.SCALE_SIZE
        
        self.trainlist = cfg.DATA.TRAINLIST
        self.vallist = cfg.DATA.VALLIST
        self.testlist = cfg.DATA.TESTLIST
        
        
        self.save_fname = False
        self.transform = transform
        
        
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            'test': self.vallist
        }
        
        
        self.samples = []
        data_filenames = dataset_dict[mode]
        
        contents = file_io.read_text_lines(data_filenames)
        
        
        for line in contents:
            sample = dict()
            splits = line.split()
            
            if self.save_fname:
                sample['saved_fname'] = splits[0].replace("/","_")[:-4]
            if "left_img" in self.visible_data:
                
                left_img = splits[0]
                sample['left_img'] = os.path.join(self.datapath_kitti360,left_img)
            if "right_img" in self.visible_data:
                right_img = left_img.replace("image_00","image_01")
                sample['right_img'] = os.path.join(self.datapath_kitti360,right_img)
                
            if "left_depth" in self.visible_data:
                left_depth = left_img.replace("image_data/data_2d_raw","sparse_lidar")
                left_depth = left_depth.replace("image_00/data_rect","projected_lidar/data")
                left_depth = left_depth.replace(".png",".npy")
                sample['left_depth'] = os.path.join(self.datapath_kitti360,left_depth)
        
            if len(self.visible_data)>3:
                # Left Annotations
                left_annotations = left_img.replace("image_data/data_2d_raw","annotations")
                left_annotations = left_annotations.replace(".png",'.json')
                left_annotations_fname = os.path.join(self.datapath_kitti360,left_annotations)
                # Right Annotions
                right_annotations_fname = left_annotations_fname.replace("image_00","image_01")
                sample['left_annotations'] = left_annotations_fname
                sample['right_annotations'] = right_annotations_fname
                
            
            file_io.check_file_existence(sample)
            self.samples.append(sample)
        
            
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        
        if self.save_fname:
            sample['saved_fname'] = sample_path['saved_fname']


        # Training Set
        if self.mode =='train' or self.mode=='val':
            if "left_img" in self.visible_data:
                sample['left_img'] = read_img(sample_path['left_img'])
            if "right_img" in self.visible_data:
                sample['right_img'] = read_img(sample_path['right_img'])
            if "left_depth" in self.visible_data:
                sample['left_depth'] = np.load(sample_path['left_depth'])
            if ('left_calib' in self.visible_data) or ('left_pose' in self.visible_data):
                camera_info_left = file_io.read_annotation(sample_path['left_annotations'],class_names=self.classes_list,
                                                               calib_only=True)
                if 'left_calib' in self.visible_data:
                    sample['left_calib'] = np.array([552.554261, 0.000000, 682.049453,
                                                        0.000000, 0.000000, 552.554261,
                                                    238.769549, 0.000000, 0.000000, 0.000000, 
                                                    1.000000, 0.000000])
                    sample['left_calib'] = sample['left_calib'].reshape(3,4)
                    
                if 'left_pose' in self.visible_data: 
                    sample['left_pose'] = camera_info_left['extrinsic_matrix'] 
                     
            if ('right_calib' in self.visible_data) or ('right_pose' in self.visible_data):
                
                camera_info_right = file_io.read_annotation(sample_path['right_annotations'],class_names=self.classes_list,
                                                               calib_only=True)
                if 'right_calib' in self.visible_data:
                    sample['right_calib'] = np.array([552.554261, 0.000000, 682.049453, -328.318735, 
                                                        0.000000, 552.554261, 238.769549, 0.000000,
                                                    0.000000,0.000000, 1.000000, 0.000000])
                    sample['right_calib'] = sample['right_calib'].reshape(3,4)
                if 'right_pose' in self.visible_data:
                    sample['right_pose'] = camera_info_right['extrinsic_matrix'] 

  
            if ("left_seg" in self.visible_data) or ("left_2d_box" in self.visible_data) or ("left_3d_box" in self.visible_data):
                semantic_annotations_left = file_io.read_annotation(sample_path['left_annotations'],class_names=self.classes_list)        
                if 'left_seg' in self.visible_data:
                    sample['left_seg'] = semantic_annotations_left["masks"]
                if 'left_2d_box' in self.visible_data:
                    sample['left_2d_box'] = get_bounding_boxes(segmentation_masks=sample['left_seg'])
                if "left_3d_box" in self.visible_data:
                    sample['left_boxes_3d'] = semantic_annotations_left['boxes_3d']
                sample['left_labels'] = semantic_annotations_left['labels']
                    
                    
            if ('right_seg' in self.visible_data) or ('right_2d_box' in self.visible_data) or ('right_3d_box' in self.visible_data):
                semantic_annotations_right = file_io.read_annotation(sample_path['right_annotations'],class_names=self.classes_list) 
                if "right_seg" in self.visible_data:
                    sample['right_seg'] = semantic_annotations_right['masks']
                if "right_2d_box" in self.visible_data:
                    sample['right_2d_box'] = get_bounding_boxes(sample['right_seg'])
                if "right_3d_box" in self.visible_data:
                    sample['right_boxes_3d'] = semantic_annotations_right['boxes_3d']
                sample['right_labels'] = semantic_annotations_right['labels']
                

        # Testing Set
        elif self.mode =='test':
            if "left_img" in self.visible_data:
                sample['left_img'] = read_img(sample_path['left_img'])
            if "right_img" in self.visible_data:
                sample['right_img'] = read_img(sample_path['right_img'])
            if "left_depth" in self.visible_data:
                sample['left_depth'] = np.load(sample_path['left_depth'])
            if ('left_calib' in self.visible_data) or ('left_pose' in self.visible_data):
                camera_info_left = file_io.read_annotation(sample_path['left_annotations'],class_names=self.classes_list,
                                                               calib_only=True)
                if 'left_calib' in self.visible_data:
                    sample['left_calib'] = np.array([552.554261, 0.000000, 682.049453,
                                                        0.000000, 0.000000, 552.554261,
                                                    238.769549, 0.000000, 0.000000, 0.000000, 
                                                    1.000000, 0.000000])
                    sample['left_calib'] = sample['left_calib'].reshape(3,4)
                    
                if 'left_pose' in self.visible_data: 
                    sample['left_pose'] = camera_info_left['extrinsic_matrix'] 
                     
            if ('right_calib' in self.visible_data) or ('right_pose' in self.visible_data):
                
                camera_info_right = file_io.read_annotation(sample_path['right_annotations'],class_names=self.classes_list,
                                                               calib_only=True)
                if 'right_calib' in self.visible_data:
                    sample['right_calib'] = np.array([552.554261, 0.000000, 682.049453, -328.318735, 
                                                        0.000000, 552.554261, 238.769549, 0.000000,
                                                    0.000000,0.000000, 1.000000, 0.000000])
                    sample['right_calib'] = sample['right_calib'].reshape(3,4)
                if 'right_pose' in self.visible_data:
                    sample['right_pose'] = camera_info_right['extrinsic_matrix'] 

  
            if ("left_seg" in self.visible_data) or ("left_2d_box" in self.visible_data) or ("left_3d_box" in self.visible_data):
                semantic_annotations_left = file_io.read_annotation(sample_path['left_annotations'],class_names=self.classes_list)        
                if 'left_seg' in self.visible_data:
                    sample['left_seg'] = semantic_annotations_left["masks"]
                if 'left_2d_box' in self.visible_data:
                    sample['left_2d_box'] = get_bounding_boxes(segmentation_masks=sample['left_seg'])
                if "left_3d_box" in self.visible_data:
                    sample['left_boxes_3d'] = semantic_annotations_left['boxes_3d']
                sample['left_labels'] = semantic_annotations_left['labels']
                    
                    
            if ('right_seg' in self.visible_data) or ('right_2d_box' in self.visible_data) or ('right_3d_box' in self.visible_data):
                semantic_annotations_right = file_io.read_annotation(sample_path['right_annotations'],class_names=self.classes_list) 
                if "right_seg" in self.visible_data:
                    sample['right_seg'] = semantic_annotations_right['masks']
                if "right_2d_box" in self.visible_data:
                    sample['right_2d_box'] = get_bounding_boxes(sample['right_seg'])
                if "right_3d_box" in self.visible_data:
                    sample['right_boxes_3d'] = semantic_annotations_right['boxes_3d']
                sample['right_labels'] = semantic_annotations_right['labels']
        
        
        # data processing
        if self.transform is not None:
            sample = self.transform(sample)
        
        
        
        return sample
            
        
    


    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size




