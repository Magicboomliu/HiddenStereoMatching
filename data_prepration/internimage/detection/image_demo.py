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
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img) # results is a tuple
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




    with open("/home/liuzihua/InternImage/detection/demo/result.pkl",'wb') as f:
        pickle.dump(saved_dict_images,f)

    # with open("/home/liuzihua/InternImage/detection/demo/result.json",'wb') as json_file:
    #     json.dump(saved_dict_images,json_file)
    # print(saved_dict_images['det']['car'].shape)
    # print(saved_dict_images['seg']['car'].shape)

    # # segmenation branch
    # for idx, seg in enumerate(segmenatation_results):
    #     cur_ls = classes_categories[idx]

    #     # nums = seg.shape[0]
    #     # if nums!=0:
    #     #     print(cur_ls)






    
    mmcv.mkdir_or_exist(args.out)
    out_file = osp.join(args.out, osp.basename(args.img))
    # show the results
    model.show_result(
        args.img,
        result,
        score_thr=args.score_thr,
        show=False,
        bbox_color=args.palette,
        text_color=(200, 200, 200),
        mask_color=args.palette,
        out_file=out_file
    )
    print(f"Result is save at {out_file}")



if __name__ == '__main__':
    args = parse_args()
    main(args)