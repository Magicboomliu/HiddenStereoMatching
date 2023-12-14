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
_C.DATA.ROOT_360_PATH = ''
_C.DATA.IMAGE_SIZE = ""
_C.DATA.SCALE_SIZE=""
_C.DATA.KITTI_RAW_PATH = ''
_C.DATA.TRAINLIST=""
_C.DATA.VALLIST=""
_C.DATA.TESTLIST=""


_C.DATA.TYPE = ['Car', 'Cyclist', 'Pesdstrain']
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]

# _C.DATA.DIM_PRIOR = [[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]