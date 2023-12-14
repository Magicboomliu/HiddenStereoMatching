# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import numpy as np
import json
import pickle
import  os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--filelist', help='Image file')
    parser.add_argument('--datapath', help='Image file')

    args = parser.parse_args()
    return args

# Draw 2D bbox
def draw2Dbbox(image,bbox2d,color=[0,255,0]):
    base_image = image.copy()
    nums_samples = len(bbox2d)

    for idx, bbox in enumerate(bbox2d):
        cv2.rectangle(
            base_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
           color=color,thickness=2)
    return base_image

def FilterOut_WithThreshold(threshold,detection_cars):

    biggest_pro = np.max(detection_cars[:,-1])
    detection_nums = detection_cars.shape[0]
    valid_ind = np.zeros((detection_nums,))
    

    if biggest_pro>=threshold:
        for instance_id in range(detection_nums):
            detection_confidence = detection_cars[instance_id,-1]
            if detection_confidence>=threshold:
                valid_ind[instance_id] = 1.0
    valid_ind  = valid_ind.astype(np.bool_)

    detection_cars = detection_cars[valid_ind]

    return detection_cars



def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)


    DATAPATH = "/data1/liu/KITTI360/KITTI360"
    image_txt = "/home/zliu/ECCV2024/HiddenStereoMatching/filenames/kitti_360_all.txt"
    lines = read_text_lines(image_txt)

    threshold1 = 0.3
    threshold2 = 0.5
    car_existence = False

    for idx, line in enumerate(lines):
        line = os.path.join(DATAPATH,line)
        result = inference_detector(model, line) # results is a tuple
        classes_categories = model.CLASSES
        detection_results = result[0]
        segmentation_results = result[1]
        detection_results = detection_results[2] # car
        segmentation_results = segmentation_results[2]
        print(segmentation_results[0].shape)
        
        # print(classes_categories)
        
        break
        
        # if detection_results.shape[0]>0:
        #     detection_results_filtered =FilterOut_WithThreshold(threshold=threshold1,detection_cars=detection_results)
        #     filter_out_nums = detection_results_filtered.shape[0]
        #     if filter_out_nums>0:
        #         car_existence = True
        #     else:
        #         car_existence = False


        #     if car_existence:
        #         savd_det2d = np.ones((filter_out_nums,6)).astype(np.str_)
        #         savd_det2d[:,0] = np.array(['Car']*filter_out_nums)
        #         savd_det2d[:,1:] = detection_results_filtered
        #         line = line.replace("image_data","det2d")
        #         line = line.replace("data_rect","threshold30")
        #         line = line.replace(".png",".txt")
        #         saved_folder_name = line[:-len(os.path.basename(line))]
        #         if not os.path.exists(saved_folder_name):
        #             os.makedirs(saved_folder_name)
        #         np.savetxt(line,savd_det2d,fmt = '%s')
        # if idx%10==0:
        #     print("Processed {}/{}".format(idx,len(lines)))
 




    
        




if __name__=="__main__":

    args = parse_args()
    main(args)