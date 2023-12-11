import numpy as np
import sys
sys.path.append("..")
from Dataset.dataset import Dataset
from configs import cfg
from utils.logger import logger
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse


from models.embedder import get_embedder
from models.distancec_field import SDFNetwork


class Runner:
    def __init__(self,cfg,mode='train',is_continue=False):
        
        self.devcie = torch.device("cuda")
        
        # create the output dir
        if not os.path.exists(cfg.EXP_NAME):
            os.makedirs(cfg.EXP_NAME)
        
        # create the dataloader
        self.dataset = Dataset(conf=cfg,logger=logger)
        
        self.iter_step = 0
        
        
        # Training Parameters
        self.end_iter = cfg.TRAIN.END_ITERATIONS
        self.save_freq = cfg.TRAIN.SAVE_FREQ
        self.report_freq = cfg.TRAIN.REPORT_FREQ
        
        self.val_freq = cfg.TRAIN.VAL_FREQ
        self.val_mesh_freq = cfg.TRAIN.VAL_MESH_FREQ
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.validate_resolution_level = cfg.VALIDATE_RESOLUTATION_LEVEL
        self.learning_rate = cfg.TRAIN.LEARNING_RATE
        self.learning_rate_alpha = cfg.TRAIN.LEARNING_ALPHA
        self.use_white_bkgd = cfg.TRAIN.USE_WHITE_BKGD
        self.warm_up_end = cfg.TRAIN.WARM_UP_END
        self.anneal_end = cfg.TRAIN.ANNEAL_END


        # Weights 
        self.igr_weight = cfg.TRAIN.IGR_WEIGHT
        self.mask_weight = cfg.TRAIN.MASK_WEIGHT
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        
        # Networks Definition.        
        params_to_train = []
        
        self.sdf_network = SDFNetwork(conf=cfg)
    
    
    
    def train(self):
        pass
    
    
    def validate_image(self):
        pass
    
    def validate_mesh(self):
        pass
    
    

        


if __name__=="__main__":
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    
    args = parser.parse_args()
    
    
    
    runner = Runner(cfg=cfg,mode=args.mode,is_continue=args.is_continue)
    
    
    if args.mode =='train':
        runner.train()

    pass