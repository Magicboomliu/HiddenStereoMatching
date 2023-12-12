import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleVarianceNetwork(nn.Module):
    def __init__(self, conf):
        super(SingleVarianceNetwork,self).__init__()
        
        init_val = conf.MODEL.VARIANCE_NETWORK.INIT_VAL
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
    
    def forward(self,x):
        return torch.ones((len(x),1)) * torch.exp(self.variance * 10.0)