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


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    left_image1 = "/home/liuzihua/FlowFormer-Official/image2/0000000000.png"
    left_image2 = "/home/liuzihua/FlowFormer-Official/image2/0000000001.png"
    left_image3 = "/home/liuzihua/FlowFormer-Official/image2/0000000005.png"

    right_image1 = "/home/liuzihua/FlowFormer-Official/image3/0000000000.png"
    right_image2 = "/home/liuzihua/FlowFormer-Official/image3/0000000001.png"
    right_image3 = "/home/liuzihua/FlowFormer-Official/image3/0000000005.png"

    filelist = [left_image1,left_image2,left_image3,right_image1,right_image2,right_image3]
    save_list = [f.replace(".png",".pkl") for f in filelist]

    for all_idx, file_name in enumerate(filelist):
        # test a single image
        result = inference_detector(model, file_name) # results is a tuple
        classes_categories = model.CLASSES
        detection_results = result[0]
        segmenatation_results = result[1]

        saved_dict_images = dict()
        saved_dict_images['det'] = dict()
        saved_dict_images['seg'] = dict()
        threshold = 0.50
        # Detetection Branch
        for idx, det in enumerate(detection_results):
            # inner cls
            cur_ls = classes_categories[idx]
            nums = det.shape[0]
            segmenation = segmenatation_results[idx]
            assert len(segmenation) == nums
            if nums!=0:
                biggest_pro = np.max(det[:,-1])
                if biggest_pro>=threshold:
                    two_detection_valid_list = []
                    instance_mask_valid_list = []
                    for instance_id in range(nums):
                        detection_confidence = det[instance_id,-1]
                        if detection_confidence>=threshold:
                            two_detection_valid_list.append(det[instance_id])
                            instance_mask_valid_list.append(segmenation[instance_id])
                    
                    saved_dict_images['det'][cur_ls] = two_detection_valid_list
                    saved_dict_images['seg'][cur_ls] =  instance_mask_valid_list
                    
                    saved_dict_images['det'][cur_ls] = np.stack(two_detection_valid_list,axis=0)
                    saved_dict_images['seg'][cur_ls] = np.stack(instance_mask_valid_list,axis=0)

        with open(save_list[all_idx],'wb') as f:
            pickle.dump(saved_dict_images,f)



if __name__ == '__main__':
    args = parse_args()
    main(args)
