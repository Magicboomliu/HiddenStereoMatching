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
from trainer.simple_diffusion.diffusion_trainer import GaussianDiffusionSampler,GaussianDiffusionTrainer
from trainer.simple_diffusion.unet import UNet


from config import cfg_cifar10
from utils.logger import logger
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter



def train(cfg):
    
    noisyImage_test = torch.randn(
            size=[cfg.TRAIN.BATCH_SIZE, 3, 32, 32]).cuda()
    
    summary_writer = SummaryWriter(cfg.TRAIN_LOGS)
    
    if not os.path.exists(cfg.TRAIN_LOGS):
        os.makedirs(cfg.TRAIN_LOGS)
    
    
    total_epochs = cfg.TRAIN.EPOCH
    # load the dataset 
    cifar10_dataloader = Loaded_CIFAR19(cfg)

    # network setup
    model = UNet(T=cfg.T,ch=cfg.MODEL.CHANNEL,
                 ch_mult=cfg.MODEL.CHANNEL_MULT,attn=cfg.MODEL.ATTN,
                 num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                 dropout=cfg.MODEL.DROPOUT_RATE)
    model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES).cuda()
    if cfg.TRAIN.LOADED_WEIGHT is not None:
        model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.SAVED_DIR, cfg.TRAIN.LOADED_WEIGHT)))
        logger.info("Loaded Pre-Trained Weight from {}/{}".format(cfg.TRAIN.SAVED_DIR, cfg.TRAIN.LOADED_WEIGHT))
        
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=total_epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=cfg.TRAIN.MULTIPLIER, warm_epoch=total_epochs // 10, after_scheduler=cosineScheduler)
    
        
    diffusion_trainer = GaussianDiffusionTrainer(model=model,
                                                 beta_1=cfg.BETA_1,
                                                 beta_T=cfg.BETA_T,
                                                 T = cfg.T).cuda()
    logger.info("Begin Training the Model.")
    
    total_iterations = 0
    for cur_epoch in range(total_epochs):
        inside_iter = 0
        with tqdm(cifar10_dataloader,dynamic_ncols=True) as tqdmDataLoader:
            total_iter= len(tqdmDataLoader)
            for images, labels in tqdmDataLoader:
                inside_iter = inside_iter +1
                optimizer.zero_grad()
                x_0 = torch.autograd.Variable(images.cuda(), requires_grad=False)
                
                loss = diffusion_trainer(x_0).sum()/1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": "{}/{}".format(cur_epoch,total_epochs),
                    "iterations:": "{}/{}".format(inside_iter,total_iter),
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                
                summary_writer.add_scalar("Loss",loss.item(),total_iterations+1)
                summary_writer.add_scalar("learning_rate",optimizer.state_dict()['param_groups'][0]["lr"],total_iterations+1)
                total_iterations = total_iterations+1
        
        warmUpScheduler.step()
        if not os.path.exists(cfg.TRAIN.SAVED_DIR):
            os.makedirs(cfg.TRAIN.SAVED_DIR)
        
        if not os.path.exists(cfg.TRAIN.INTERMEDIATE):
            os.makedirs(cfg.TRAIN.INTERMEDIATE)
            
            
        torch.save(model.state_dict(), os.path.join(
            cfg.TRAIN.SAVED_DIR, 'ckpt_' + str(cur_epoch) + "_.pt"))
        
        validation(model=model,test_image=noisyImage_test,
                   cfg=cfg,cur_epoch=cur_epoch)
        
        
def validation(model,test_image,cfg,cur_epoch):
    
    model.eval()
    with torch.no_grad():
        sampler = GaussianDiffusionSampler(
            model=model,beta_1=cfg.BETA_1,
            beta_T=cfg.BETA_T,
            T=cfg.T
        ).cuda()
        
        visualiation_test_images = torch.clamp(test_image * 0.5 + 0.5, 0, 1)
        sampledImgs = sampler(test_image)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        
        save_image(visualiation_test_images, os.path.join(
            cfg.TRAIN.INTERMEDIATE, "original_noise.png"), nrow=cfg.TEST.NROW)
        
        save_image(sampledImgs, os.path.join(
            cfg.TRAIN.INTERMEDIATE,  "Epoch_{}.png".format(cur_epoch)), nrow=cfg.TEST.NROW)
        
        logger.info("Images generated at {}".format(cfg.TRAIN.INTERMEDIATE))


        
    


def inference(cfg):
    # load the model and evaluation
    
    with torch.no_grad():
        # network setup
        model = UNet(T=cfg.T,ch=cfg.MODEL.CHANNEL,
                    ch_mult=cfg.MODEL.CHANNEL_MULT,attn=cfg.MODEL.ATTN,
                    num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                    dropout=cfg.MODEL.DROPOUT_RATE)
        model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES).cuda()
        
        ckpt = torch.load(cfg.TEST.LOADED_WEIGHT)
        model.load_state_dict(ckpt)
        logger.info("Successfully Loaded the Model")
        # define a sample
        model.eval()
        
        sampler = GaussianDiffusionSampler(
            model=model,beta_1=cfg.BETA_1,
            beta_T=cfg.BETA_T,
            T=cfg.T
        )
        
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
    
    if cfg_cifar10.STATE == "train":
        train(cfg=cfg_cifar10)
    elif cfg_cifar10.STATE=='val':
        inference(cfg=cfg_cifar10)
    else:
        raise NotImplementedError