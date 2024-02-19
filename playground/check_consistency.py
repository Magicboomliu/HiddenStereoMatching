import os
import numpy  as np
import sys



if __name__=="__main__":
    
    saved_path_gt = "/media/zliu/data12/dataset/KITTI/KITTI_Rendered_GT/GT"
    saved_path_render = "/media/zliu/data12/dataset/KITTI/KITTI_Rendered_GT/Input"
    
    gt_file = os.listdir(saved_path_gt)
    renderd_file = os.listdir(saved_path_render)
    
    gt_file = sorted(gt_file)
    renderd_file = sorted(renderd_file)
    
    for idx, fname in enumerate(gt_file):
        gt_file_cur = gt_file[idx]
        render_file_cur = renderd_file[idx]
        assert gt_file_cur == render_file_cur
    
    print("all is oK!")
        
    
    
