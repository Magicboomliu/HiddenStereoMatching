import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


class NeRF(nn.Module):
    def __init__(self,conf):
        super(NeRF, self).__init__()
        
        self.D = conf.MODEL.NERF.D
        self.W = conf.MODEL.NERF.W
        
        self.d_in = conf.MODEL.NERF.D_IN
        self.d_in_view = conf.MODEL.NERF.D_IN_VIEW
        self.multires = conf.MODEL.NERF.MULTIRES
        self.multires_view = conf.MODEL.NERF.MULTIRES_VIEW
        self.output_ch = conf.MODEL.NERF.OUTPUT_CH
        self.skips = conf.MODEL.NERF.SKIPS
        self.use_viewdirs = conf.MODEL.NERF.USE_VIEWDIRS
        
        

        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if self.multires > 0:
            embed_fn, input_ch = get_embedder(self.multires, input_dims=self.d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if self.multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(self.multires_view, input_dims=self.d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = self.skips
        self.use_viewdirs = self.use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] +
            [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D - 1)])

 
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + self.W, self.W // 2)])

        # use views
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.W, self.W)
            self.alpha_linear = nn.Linear(self.W, 1)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, input_pts, input_views):
        
        
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        
        # for hidden states aggregation.
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha liner
            alpha = self.alpha_linear(h)
            # feature linear
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False