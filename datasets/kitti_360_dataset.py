from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys

from utils import file_io
from utils.kitti_io import read_img, read_disp
from skimage import io, transform
import numpy as np
from PIL import Image




class KITTI360_Dataset(Dataset):
    def __init__(self,cfg,
                 mode='train',
                 transform=None) -> None:
        super().__init__()
        
        self.mode= mode
        
        
        self.dataset_name = cfg.DATA.MODE
        self.img_size= cfg.DATA.IMAGE_SIZE
        self.scale_size =cfg.DATA.SCALE_SIZE
        
        self.trainlist = cfg.DATA.TRAINLIST
        self.vallist = cfg.DATA.VALLIST
        self.testlist = cfg.DATA.TESTLIST
        
        self.transform = transform
        
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            'test': self.vallist
        }
        
        data_filenames = dataset_dict[mode]
        
        
        contents = file_io.read_text_lines(data_filenames)
        

        
    
    
    def __getitem__(self, index):
        sample = {}
        
        
        return super().__getitem__(index)


    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size




