import os
import  sys
sys.path.append("..")
from stereosdf.datasets.utils import file_io
from stereosdf.datasets.contents_extractor import extract_contents_from_name

if __name__=="__main__":
    
    root_path = "/data1/liu/KITTI360/KITTI360"
    filename_path = "//home/zliu/ECCV2024/HiddenStereoMatching/filenames/kitti360_annotated/kitti360_with_anno_all.txt"
    
    contents_type = ["left_img","right_img","left_depth","right_depth","left_calib","right_calib",
                         "left_pose","right_pose","left_seg","right_seg",
                         "left_2d_box","right_2d_box","left_3d_box","right_3d_box"]
    
    
    lines = file_io.read_text_lines(filename_path)
    
    
    for idx, fname in enumerate(lines):
        splits = fname.split()
        left_image_fname = splits[0]
        left_image_fname_abs = os.path.join(root_path,left_image_fname)
        
        extract_contents_from_name(left_image_fname_abs,contents_type=contents_type)
        
        break    
