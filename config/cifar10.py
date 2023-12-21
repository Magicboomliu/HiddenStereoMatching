from yacs.config import CfgNode as CN



_C = CN()
_C.STATE='train'
_C.T=1000 # How many diffuison iterations.
_C.BETA_1 = 1e-4
_C.BETA_T=0.02
_C.IMG_SIZE=32
_C.DEVICES=[0]


_C.TRAIN = CN()
_C.TRAIN.EPOCH=200
_C.TRAIN.BATCH_SIZE=80
_C.TRAIN.LR = 1e-4
_C.TRAIN.MULTIPLIER=2
_C.TRAIN.GRAD_CLIP=1
_C.TRAIN.LOADED_WEIGHT = None
_C.TRAIN.SAVED_DIR='./checkpoints'
_C.TRAIN_LOGS = "Logs"
_C.TRAIN.INTERMEDIATE="Processing_Results"



_C.MODEL = CN()
_C.MODEL.CHANNEL=128
_C.MODEL.CHANNEL_MULT=[1, 2, 3, 4]
_C.MODEL.ATTN = [2]
_C.MODEL.NUM_RES_BLOCKS=2
_C.MODEL.DROPOUT_RATE=0.15


_C.TEST=CN()
_C.TEST.LOADED_WEIGHT="/home/zliu/Desktop/ECCV2024/code/Diffusion/DenoisingDiffusionProbabilityModel-ddpm-/Checkpoints/DiffusionWeight.pt"
_C.TEST.SAMPLE_DIR="sample_images"
_C.TEST.SAMPLE_NOISE_IMAGE="GuassinNoise.png"
_C.TEST.SAMPLE_IMG_NAME="GeneratedImages.png"
_C.TEST.NROW=8
