import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")
from stereosdf.datasets.utils.file_io import read_text_lines,list2txt,read_annotation
from data_prepration.get_train_val_test  import split_the_train_val_and_test_fnames
from config import cfg
import os
import json
from tqdm import tqdm





            
    



if __name__=="__main__":
    
    kitti_all = cfg.DATA.TRAINLIST
    kitti_360_path = cfg.DATA.ROOT_360_PATH
    kitti_used_classes = cfg.DATA.USED_CLAESSES
    
    
    
    contents = read_text_lines(kitti_all)
    
    samples_with_annotations = [] # with gt segmentation masks
    
    
    classes_names  = []
    for fname in tqdm(contents):
        
        # /media/zliu/data1/dataset/KITTI/KITTI360/annotations/2013_05_28_drive_0000_sync/image_00/data_rect/
        left_im = fname
        right_im = fname.replace("image_00","image_01")

        left_annotations = left_im.replace("image_data/data_2d_raw","annotations")
        left_annotations = left_annotations.replace(".png",'.json')
        left_annotations = os.path.join(kitti_360_path,left_annotations)
        

        annotations = read_annotation(left_annotations,class_names=kitti_used_classes)
        
        
        
        


        
        

    



