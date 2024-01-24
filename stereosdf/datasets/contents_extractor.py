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
        
        
        if ('left_pose' in contents_type) or ('left_seg' in contents_type) or ('left_2d_box' in contents_type) or ('left_3d_box' in contents_type):
            semantic_annotations_left = file_io.read_annotation(left_annotations_fname,class_names=['car'])
            
        
        if ('right_pose' in contents_type) or ('right_seg' in contents_type) or ('right_2d_box' in contents_type) or ('right_3d_box' in contents_type):
            semantic_annotations_right = file_io.read_annotation(right_annotations_fname,class_names=['car'])
            
        if 'left_pose' in contents_type:
            results['left_pose'] = semantic_annotations_left['extrinsic_matrix']
        if "right_pose" in contents_type:
            results['right_pose'] = semantic_annotations_right['extrinsic_matrix']
        
        if "left_seg" in contents_type:
            if len(semantic_annotations_left.keys())==2:
                results['left_seg'] = None
            else:
                results['left_seg'] = semantic_annotations_left["masks"]
            
        if 'right_seg' in contents_type:
            if len(semantic_annotations_right.keys())==2:
                results['right_seg'] = None
            else:
                results['right_seg'] = semantic_annotations_right["masks"]
        
        if 'left_2d_box' in contents_type:
            if len(semantic_annotations_left.keys())==2:
                results['left_2d_box'] = None
            else:
                results['left_2d_box'] = get_bounding_boxes(segmentation_masks=results['left_seg'])
        
        if 'right_2d_box' in contents_type:
            if len(semantic_annotations_right.keys())==2:
                results['right_2d_box'] = None
            else:
                results['right_2d_box'] = get_bounding_boxes(segmentation_masks=results['right_seg'])
        
        if "left_3d_box" in contents_type:
            if len(semantic_annotations_left.keys())==2:
                results['left_3d_box'] = None
            else:
                results['left_3d_box'] = semantic_annotations_left['boxes_3d']
        
        
        if "right_3d_box" in contents_type:
            if len(semantic_annotations_right.keys())==2:
                results['right_3d_box'] = None
            else:
                results['right_3d_box'] = semantic_annotations_right['boxes_3d']
        

    
    
    
    return results

