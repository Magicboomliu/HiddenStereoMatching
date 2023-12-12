import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


class SDFNetwork(nn.Module):
    def __init__(self,conf):
        super(SDFNetwork, self).__init__()

        self.d_in = conf.MODEL.SDF_NETWORK.D_IN
        self.d_out = conf.MODEL.SDF_NETWORK.D_OUT
        self.d_hidden = conf.MODEL.SDF_NETWORK.D_HIDDEN
        self.n_layers =conf.MODEL.SDF_NETWORK.N_LAYERS
        self.skip_in = conf.MODEL.SDF_NETWORK.SKIP_IN
        self.multires = conf.MODEL.SDF_NETWORK.MULTIRES
        self.bias = conf.MODEL.SDF_NETWORK.BIAS
        self.scale = conf.MODEL.SDF_NETWORK.SCALE
        self.geometric_init = conf.MODEL.SDF_NETWORK.GEOMETRIC_INIT
        self.weight_norm = conf.MODEL.SDF_NETWORK.WEIGHT_NORM
        self.inside_outside = False
        
        
        # [3, 256, 256, 256, 256, 256, 256, 256, 256, 257]
        dims = [self.d_in] + [self.d_hidden for _ in range(self.n_layers)] + [self.d_out]
        
        self.embed_fn_fine = None
        if self.multires>0:
            embed_fn, input_ch = get_embedder(self.multires, input_dims=self.d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
    

        self.num_layers = len(dims) # default is 10:  [3, 256, 256, 256, 256, 256, 256, 256, 256, 257]

        # network construction
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            
            if self.geometric_init:
                if l == self.num_layers - 2:
                    if not self.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.bias)
                        
                elif self.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100) # a smooth approximation of the ReLU  activation function

        
    def forward(self, inputs):
        
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)
        
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        
        # d_outputs is the weight to compute the gradients, so it is zero.
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
        

            
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 conf):
        super().__init__()
        
        self.mode = conf.MODEL.RENDERING_NETWORK.MODE
        self.squeeze_out = conf.MODEL.RENDERING_NETWORK.SQUEEZE_OUT
        
        d_feature = conf.MODEL.RENDERING_NETWORK.D_FEATURE
        d_in = conf.MODEL.RENDERING_NETWORK.D_IN
        d_out = conf.MODEL.RENDERING_NETWORK.D_OUT
        d_hidden = conf.MODEL.RENDERING_NETWORK.D_HIDDEN
        n_layers = conf.MODEL.RENDERING_NETWORK.N_LAYERS
        weight_norm = conf.MODEL.RENDERING_NETWORK.WEIGHT_NORM
        multires_view = conf.MODEL.RENDERING_NETWORK.MULTIRES_VIEW
        
        # dimenions
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        
        # Get the positional encoding.
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        '''
        Inputs is 9: Means Points+Normals+View Dirs
        
        '''
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
