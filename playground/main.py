import numpy as np
import sys
sys.path.append("..")
from Dataset.dataset import Dataset
from configs import cfg
from utils.logger import logger


if __name__=="__main__":
    
    dataset = Dataset(conf=cfg,logger=logger)
    pass