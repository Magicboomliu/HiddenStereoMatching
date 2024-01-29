from yacs.config import CfgNode as CN

_C = CN()
_C.TRAIN = CN()
_C.VAL = CN()
_C.INFER = CN()
_C.DATA = CN()

_C.EXP_NAME = "default"
_C.RESTORE_PATH = None
_C.RESTORE_EPOCH = None

_C.LOG_DIR = './log'
_C.CHECKPOINTS_DIR ='checkpoints'




_C.DATA.CLS_LIST = ['Car']
_C.DATA.MODE = 'KITTI Raw'
_C.DATA.ROOT_360_PATH = '/data3/yyang/estimated-depth/KITTI360/'
_C.DATA.IMAGE_SIZE = ""
_C.DATA.SCALE_SIZE=""
_C.DATA.KITTI_RAW_PATH = ''
_C.DATA.TRAINLIST="/home/yyang/Desktop/yyang/HiddenStereoMatching/filenames/kitti360_annotated/kitti360_with_anno_all.txt"
_C.DATA.VALLIST="/home/yyang/Desktop/yyang/HiddenStereoMatching/filenames/kitti360_annotated/kitti360_with_anno_val.txt"
_C.DATA.TESTLIST="/home/yyang/Desktop/yyang/HiddenStereoMatching/filenames/kitti360_annotated/kitti360_with_anno_test.txt"
_C.DATA.VISIBLE_LISTS = ["left_img","right_img","left_depth","right_depth","left_calib","right_calib",
                         "left_pose","right_pose","left_seg","right_seg",
                         "left_2d_box","right_2d_box","left_3d_box","right_3d_box"]
# _C.DATA.VISIBLE_LISTS = ["left_img","right_img"]

_C.DATA.USED_CLAESSES = ['car','person','bus','train','truck']
#_C.DATA.USED_CLAESSES = ['car']

# ['traffic light', 'stop', 'bicycle', 'building', 'vegetation', 'motorcycle', 
# 'trash bin', 'sidewalk', 'ground', 'bus', 'train', 'guard rail', 'terrain', 'wall', 
# 'bridge', 'person', 'truck', 'rail track', 'parking', 'trailer', 'lamp', 'rider', 'traffic sign', 
# 'box', 'unknown object', 'tunnel', 'gate', 'unknown vehicle', 'pole', 'caravan', 'fence', 
# 'sky', 'unknown construction', 'car', 'garage', 'smallpole', 'road', 'vending machine']


_C.DATA.TYPE = ['Car', 'Cyclist', 'Pesdstrain']
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]

# _C.DATA.DIM_PRIOR = [[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]