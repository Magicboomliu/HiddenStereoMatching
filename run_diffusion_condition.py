import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from dataset.cifar10_dataset import Loaded_CIFAR19


from config import cfg_condition
from utils.logger import logger
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


from trainer.diffusion_condition_trainer.Scheduler import GradualWarmupScheduler
from trainer.diffusion_condition_trainer.diffusion_condition import GaussianDiffusionSampler,GaussianDiffusionTrainer
from trainer.diffusion_condition_trainer.unet_conditioned import UNet_Classifier_Free


CLASSSES_INFO ={
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}




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
    model = UNet_Classifier_Free(T=cfg.T,ch=cfg.MODEL.CHANNEL,
                                 num_labels=len(CLASSSES_INFO),
                 ch_mult=cfg.MODEL.CHANNEL_MULT,
                 num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                 dropout=cfg.MODEL.DROPOUT_RATE)
    model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES).cuda()
    
    if cfg.TRAIN.RESUME_WEIGHT is not None:
        model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.SAVED_DIR, cfg.TRAIN.RESUME_WEIGHT)))
        logger.info("Loaded Pre-Trained Weight from {}/{}".format(cfg.TRAIN.SAVED_DIR, cfg.TRAIN.RESUME_WEIGHT))
        
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
                
                batch_size = images.shape[0]
                inside_iter = inside_iter +1
                optimizer.zero_grad()
                x_0 = torch.autograd.Variable(images.cuda(), requires_grad=False)
                
                labels = labels.cuda() + 1 # 0 for non-conditioned learning, 1-10 to CIFAR10 categories.
                
                # non-conditioned regions.
                if np.random.rand()<0.1:
                    labels = torch.zeros_like(labels).cuda()
                
                loss = diffusion_trainer(x_0, labels).sum() / (batch_size ** 2.)

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

        # current image generation results.
        validation(model=model,cfg=cfg,noise_input=noisyImage_test,cur_epoch=cur_epoch)

    
        
# intermediate results during the training phase.
def validation(cfg,model,noise_input,cur_epoch):
    # should get the list.
    with torch.no_grad():
        # create the coresponding labels.
        step = int(cfg.TRAIN.BATCH_SIZE//len(CLASSSES_INFO))
        labels_list = []
        rows = 0
        for i in range(1,cfg.TRAIN.BATCH_SIZE+1):
            labels_list.append(torch.ones(size=[1]).long() * rows)
            if i%step==0:
                # skip the final
                if rows< len(CLASSSES_INFO)-1:
                    rows = rows +1
        
        # +1 for conditioned diffusion model.
        labels = torch.cat(labels_list,dim=0).long().cuda() + 1
        print("Current Labels: ",labels)
        model.eval()
        diffsuion_sampler = GaussianDiffusionSampler(model=model,beta_1=cfg.BETA_1,
                                                     beta_T=cfg.BETA_T,
                                                     T=cfg.T,
                                                     w=cfg.TRAIN.W).cuda()
    
        visualiation_test_images = torch.clamp(noise_input * 0.5 + 0.5, 0, 1)
        sampledImgs = diffsuion_sampler(noise_input,labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        
        save_image(visualiation_test_images, os.path.join(
            cfg.TRAIN.INTERMEDIATE, "original_noise.png"), nrow=cfg.TEST.NROW)
        
        save_image(sampledImgs, os.path.join(
            cfg.TRAIN.INTERMEDIATE,  "Epoch_{}.png".format(cur_epoch)), nrow=cfg.TEST.NROW)
        
        logger.info("Images generated at {}".format(cfg.TRAIN.INTERMEDIATE))
        
        
def inference(cfg):
    # load model and evaluate
    
    with torch.no_grad():
        step = int(cfg.TRAIN.BATCH_SIZE//len(CLASSSES_INFO))
        labels_list = []
        rows = 0
        for i in range(1,cfg.TRAIN.BATCH_SIZE+1):
            labels_list.append(torch.ones(size=[1]).long() * rows)
            if i%step==0:
                # skip the final
                if rows< len(CLASSSES_INFO)-1:
                    rows = rows +1
        
        # +1 for conditioned diffusion model.
        labels = torch.cat(labels_list,dim=0).long().cuda() + 1
        print("Current Labels: ",labels)
        
        model = UNet_Classifier_Free(T=cfg.T,
                                    ch=cfg.MODEL.CHANNEL,
                                    num_labels=len(CLASSSES_INFO), 
                                    ch_mult=cfg.MODEL.CHANNEL_MULT,
                                    num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                                    dropout=cfg.MODEL.DROPOUT_RATE).cuda()
        # model = torch.nn.DataParallel(model, device_ids=cfg.DEVICES).cuda()


        ckpt = torch.load(cfg.TEST.TEST_PRETRAINED_WEIGHT)
        model.load_state_dict(ckpt)
        logger.info("Successfully Loaded the Model")
        # define a sample
        model.eval()


        diffsuion_sampler = GaussianDiffusionSampler(model=model,beta_1=cfg.BETA_1,
                                                        beta_T=cfg.BETA_T,
                                                        T=cfg.T,
                                                        w=cfg.TRAIN.W).cuda()
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
                size=[cfg.TRAIN.BATCH_SIZE, 3, 32, 32]).cuda()

        if not os.path.exists(cfg.TEST.SAMPLED_DIR):
            os.makedirs(cfg.TEST.SAMPLED_DIR)
            
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        
        save_image(saveNoisy, os.path.join(
            cfg.TEST.SAMPLED_DIR, cfg.TEST.SAMPLE_NOISE_IMGNAME), nrow=cfg.TEST.NROW)
        sampledImgs = diffsuion_sampler(noisyImage,labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            cfg.TEST.SAMPLED_DIR,  cfg.TEST.SAMPLE_IMG_NAME), nrow=cfg.TEST.NROW)


if __name__=="__main__":
    
    cfg = cfg_condition
    
    if cfg.STATE == "train":
        train(cfg=cfg)
    elif cfg.STATE=='val':
        inference(cfg=cfg)
    else:
        raise NotImplementedError