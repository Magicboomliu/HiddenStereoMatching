import os
import numpy as np
import sys
from tqdm import tqdm


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



if __name__=="__main__":
    
    small_kitti_dataset_filename = "/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/total_kitti_set.txt"
    lines = read_text_lines(small_kitti_dataset_filename)
    
    gt_path_root = "/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
    render_path_root = "/media/zliu/data12/dataset/KITTI/rendered_data_kitti_train/"
    
    saved_path_render = "/media/zliu/data12/dataset/KITTI/KITTI_Rendered_NewView/"
    
    
    os.makedirs(saved_path_render,exist_ok=True)

    
    for line in tqdm(lines):
        splits = line.split()
        left = splits[0]
        right = left.replace("image_02","image_03")

        left_gt_path = os.path.join(gt_path_root,left)
        right_gt_path = os.path.join(gt_path_root,right)
        
        basename = os.path.basename(left)
        
        render_right_right = right.replace(basename,"right_right_from_right_"+basename)
        render_left_left = left.replace(basename,"left_left_from_left_"+basename)
        render_left_left_path = os.path.join(render_path_root,render_left_left)
        render_right_right_path = os.path.join(render_path_root,render_right_right)
        
        assert os.path.exists(render_left_left_path)
        assert os.path.exists(render_right_right_path)
        assert os.path.exists(left_gt_path)
        assert os.path.exists(right_gt_path)
        
        
        saved_rendered_left_left = render_left_left.replace("/","_")
        saved_rendered_right_right = render_right_right.replace("/","_")
        
        saved_rendered_left_left = os.path.join(saved_path_render,saved_rendered_left_left)
        saved_rendered_right_right = os.path.join(saved_path_render,saved_rendered_right_right)

        
        os.system("cp {} {}".format(render_left_left_path,saved_rendered_left_left))
        os.system("cp {} {}".format(render_right_right_path,saved_rendered_right_right))
        
        
        # # saved the ground truth images.
        # saved_left_path = left_gt_path[len(gt_path_root)+1:]
        # saved_right_path = right_gt_path[len(gt_path_root)+1:]
        # saved_left_path = saved_left_path.replace(os.path.basename(saved_left_path),"left_"+os.path.basename(saved_left_path))
        # saved_right_path = saved_right_path.replace(os.path.basename(saved_right_path),"right_"+os.path.basename(saved_right_path))
        # saved_left_path = saved_left_path.replace("/","_")
        # saved_right_path = saved_right_path.replace("/","_")
        # saved_left_path = os.path.join(saved_path_gt,saved_left_path)
        # saved_right_path = os.path.join(saved_path_gt,saved_right_path)
        
        
        # # saved the render images
        # saved_left_rendered_path = render_left_path[len(render_path_root)+1:]
        # saved_right_rendered_path = render_right_path[len(render_path_root)+1:]
        
        # saved_left_rendered_path = saved_left_rendered_path.replace("left_from_right_","left_")
        # saved_right_rendered_path = saved_right_rendered_path.replace("right_from_left_","right_")
        # saved_left_rendered_path = saved_left_rendered_path.replace("image_03","image_02")
        # saved_right_rendered_path = saved_right_rendered_path.replace("image_02","image_03")

        # saved_left_rendered_path = saved_left_rendered_path.replace("/","_")
        # saved_right_rendered_path = saved_right_rendered_path.replace("/","_")
        # saved_left_rendered_path  = os.path.join(saved_path_render,saved_left_rendered_path)
        # saved_right_rendered_path = os.path.join(saved_path_render,saved_right_rendered_path)

        
        
        # os.system("cp {} {}".format(left_gt_path,saved_left_path))
        # os.system("cp {} {}".format(right_gt_path,saved_right_path))


        # os.system("cp {} {}".format(render_left_path,saved_left_rendered_path))
        # os.system("cp {} {}".format(render_right_path,saved_right_rendered_path))


