from yacs.config import CfgNode as CN


_C = CN()
_C.STATE='train'
_C.T=500 # How many diffuison iterations.
_C.BETA_1 = 1e-4
_C.BETA_T=0.028
_C.IMG_SIZE=32
_C.DEVICES=[0]
_C.T = 500



_C.TRAIN = CN()
_C.TRAIN.EPOCH=70
_C.TRAIN.BATCH_SIZE=80
_C.TRAIN.LR=1e-4
_C.TRAIN.IMG_SIZE=32
_C.TRAIN.GRAD_CLIP=1
_C.TRAIN.W = 1.8   # Control and Guidance, Maybe the same effect of the S? in the BLOGS
_C.TRAIN.SAVED_DIR="checkpoints_conditions"
_C.TRAIN.MULTIPLIER=2.5
_C.TRAIN.RESUME_WEIGHT=None

_C.TRAIN_LOGS = "logs_conditioned"
_C.TRAIN.INTERMEDIATE="Processing_Results_conditioned"



_C.TEST = CN()
_C.TEST.TEST_PRETRAINED_WEIGHT="/home/zliu/Desktop/ECCV2024/code/Diffusion/HiddenStereoMatching/pretrained/DiffusionConditionWeight.pt"
_C.TEST.SAMPLED_DIR="TEST_Diffusion_Conditioned"
_C.TEST.SAMPLE_NOISE_IMGNAME="NoiseInput.png"
_C.TEST.SAMPLE_IMG_NAME="Rendered_Image.png"
_C.TEST.NROW=8




_C.MODEL = CN()
_C.MODEL.CHANNEL=128
_C.MODEL.CHANNEL_MULT=[1, 2, 2, 2]
_C.MODEL.NUM_RES_BLOCKS=2
_C.MODEL.DROPOUT_RATE=0.15