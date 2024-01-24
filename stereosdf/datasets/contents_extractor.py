import os
import  sys
sys.path.append("..")
from stereosdf.datasets.utils import file_io
from data_prepration.make_annotations import get_bounding_boxes
from stereosdf.datasets.utils.kitti_io import read_disp,read_img
import numpy as np


CONTENTS_TYPE=["left_img","right_img","left_depth","left_calib","right_calib",
                         "left_pose","right_pose","left_seg","right_seg",
                         "left_2d_box","right_2d_box","left_3d_box","right_3d_box"]

def extract_contents_from_name(left_image_path,contents_type=None):
    # extract the filename from the give fname
    results = dict()

    
    if 'left_img' in contents_type:
        results['left_img'] =read_img(left_image_path)
    if 'right_img' in contents_type:
        right_img = left_image_path.replace("image_00","image_01")
        results["right_img"] = read_img(right_img)
    if 'left_depth' in contents_type:
        left_depth = left_image_path.replace("image_data/data_2d_raw","sparse_lidar")
        left_depth = left_depth.replace("image_00/data_rect","projected_lidar/data")
        left_depth = left_depth.replace(".png",".npy")
        results["left_depth"] = np.load(left_depth)
    
    if len(contents_type)>=3:
        left_annotations = left_image_path.replace("image_data/data_2d_raw","annotations")
        left_annotations = left_annotations.replace(".png",'.json')
        left_annotations_fname = left_annotations
        right_annotations_fname = left_annotations_fname.replace("image_00","image_01")
        
        
        if 'left_calib' in contents_type:
            results['left_calib'] = np.array([552.554261, 0.000000, 682.049453,
                                                        0.000000, 0.000000, 552.554261,
                                                    238.769549, 0.000000, 0.000000, 0.000000, 
                                                    1.000000, 0.000000]).reshape(3,4)
        if 'right_calib' in contents_type:
            results["right_calib"] = np.array([552.554261, 0.000000, 682.049453, -328.318735, 
                                                        0.000000, 552.554261, 238.769549, 0.000000,
                                                    0.000000,0.000000, 1.000000, 0.000000]).reshape(3,4)
        
    
        if 'left_pose' in contents_type:
            pass
        
        if "right_pose" in contents_type:
            pass
        
        if "left_seg" in contents_type:
            pass
        
        if 'right_seg' in contents_type:
            pass
        
        if 'left_2d_box' in contents_type:
            pass
        
        if 'right_2d_box' in contents_type:
            pass
        
        if "left_3d_box" in contents_type:
            pass
        
        if "right_2d_box" in contents_type:
            pass
        
        if "left_3d_box" in contents_type:
            pass
        
        if "right_3d_box" in contents_type:
            pass
    
    
    
    
    return results

