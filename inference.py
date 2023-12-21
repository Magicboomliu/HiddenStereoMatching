import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from dataset.cifar10_dataset import Loaded_CIFAR19
from trainer.simple_diffusion.Scheduler import GradualWarmupScheduler
from trainer.simple_diffusion.diffusion_trainer import GaussianDiffusionTrainer,GaussianDiffusionSampler
from trainer.simple_diffusion.unet import UNet

from config import cfg_cifar10
from utils.logger import logger
from torchvision.utils import save_image



def inference(cfg):
    # load the model and evaluation
    
    with torch.no_grad():
        # network setup
        model = UNet(T=cfg.T,ch=cfg.MODEL.CHANNEL,
                    ch_mult=cfg.MODEL.CHANNEL_MULT,attn=cfg.MODEL.ATTN,
                    num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                    dropout=cfg.MODEL.DROPOUT_RATE).cuda()
        # model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES).cuda()
        
        ckpt = torch.load(cfg.TEST.LOADED_WEIGHT)
        model.load_state_dict(ckpt)
        logger.info("Successfully Loaded the Model")
        # define a sample
        model.eval()
        
        sampler = GaussianDiffusionSampler(
            model=model,beta_1=cfg.BETA_1,
            beta_T=cfg.BETA_T,
            T=cfg.T
        ).cuda()
        
        noisyImage = torch.randn(
            size=[cfg.TRAIN.BATCH_SIZE, 3, 32, 32]).cuda()
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        
        if not os.path.exists(cfg.TEST.SAMPLE_DIR):
            os.makedirs(cfg.TEST.SAMPLE_DIR)
        
        save_image(saveNoisy, os.path.join(
            cfg.TEST.SAMPLE_DIR, cfg.TEST.SAMPLE_NOISE_IMAGE), nrow=cfg.TEST.NROW)
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            cfg.TEST.SAMPLE_DIR,  cfg.TEST.SAMPLE_IMG_NAME), nrow=cfg.TEST.NROW)


if __name__=="__main__":
    
    inference(cfg=cfg_cifar10)
