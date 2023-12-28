import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append("../..")
from config import cfg
import matplotlib.pyplot as plt

from stereosdf.datasets.utils import file_io



if __name__ =="__main__":
    
    
    kitti_annotated_files = "/home/zliu/Desktop/ECCV2024/code/StereoSDF/HiddenStereoMatching/filenames/kitti360_annotated/kitti360_with_anno_all.txt"
    contents = file_io.read_text_lines(kitti_annotated_files)
    
    selected_classes = cfg.DATA.USED_CLAESSES
    datapath = cfg.DATA.ROOT_360_PATH
    classes_list = cfg.DATA.USED_CLAESSES
    
    
    for idx, line in enumerate(contents):
        
        left_annotations = line.replace("image_data/data_2d_raw","annotations")
        left_annotations = left_annotations.replace(".png",'.json')
        left_annotations_fname = os.path.join(datapath,left_annotations)
        
        
        
        print(line)
    
    


