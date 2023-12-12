import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


# extract the distance fields
def extract_fields(bound_min, bound_max, resolution, query_func):
    # resolution is 512
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32) #512,512,512
    
    # cude SDF
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) #(64**3,3)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() #(64,64,64)
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    
    # vertices, traingles. 
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    
    return vertices, triangles

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # to be the same points #(512,64)
    
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) #(512,16)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    # https://pytorch.org/docs/stable/generated/torch.searchsorted.html
    inds = torch.searchsorted(cdf, u, right=True) #(512,16)
    
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds) # max is 64
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)    
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] #[512,16,64]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) #(512,16,2)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) #512,16,2

    denom = (cdf_g[..., 1] - cdf_g[..., 0]) # pdf--->512,16
   
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom  #512,16
    
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 conf=None):
        
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = conf.MODEL.NEUS_RENDERER.N_SAMPLES
        self.n_importance = conf.MODEL.NEUS_RENDERER.N_IMPORTANCES  
        self.n_outside = conf.MODEL.NEUS_RENDERER.N_OUTSIDE
        self.up_sample_steps = conf.MODEL.NEUS_RENDERER.UP_SAMPLE_STEPS
        self.perturb = conf.MODEL.NEUS_RENDERER.PERTURB
    
    # Using the NERF for rendering the background.
    def render_core_outside(self,rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        '''render the background'''
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5 

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3) #(512,160,3)
 

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0)) #(N*B43)
        dirs = dirs.reshape(-1, 3) #(N*B,3)

        # Nerf's OutPut is the density and the color.
        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color) # using a sigmoid as the finall color.
        
        # density to alpha
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists) 
        alpha = alpha.reshape(batch_size, n_samples) #(512,160)
        
        # alpha to weight
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3) # color before weight sime
        
        # nerf the weight summation is the color.
        color = (weights[:, :, None] * sampled_color).sum(dim=1)

        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }
    
    
    
    # Mix rendering.
    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        

        batch_size, n_samples = z_vals.shape # (512,128)

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1] #(512,127)
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1) # final add the sample dist
        mid_z_vals = z_vals + dists * 0.5  # mid sample.
 
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape) #(512,128,3)


        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # get the SDF output
        sdf_nn_output = sdf_network(pts)
        
        # SDF value
        sdf = sdf_nn_output[:, :1] #(512*128,)
        # SDF features
        feature_vector = sdf_nn_output[:, 1:] #(512*128,256)
        gradients = sdf_network.gradient(pts).squeeze()#(512*128,3)--> (dx,dy,dz)

        # color is using the coloder network for RGB calucaltions.
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)#(512,128,3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter

        inv_s = inv_s.expand(batch_size * n_samples, 1) #(512*128,1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf # cdf

        # alpha culminative
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach() # inside a sphere.
        relax_inside_sphere = (pts_norm < 1.2).float().detach() # relax a sphere.

    
        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        # consider the weight summation
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        
        # render color
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }
    
    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        
        batch_size = len(rays_o) # 512
        # Assuming the region of interest is a unit sphere
        sample_dist = 2.0 / self.n_samples
        
        # near and far's shape is [512,1]
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :] #(N,64)
        
        # Out Side the Sphere, the sample Z_Vals.
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb #1.0

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            # add sampling perturbs
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1]) # 31 dimension: the center point?
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
        
        
        if self.n_outside > 0:
            # outside the far
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None
        
        
        # upsample
        if self.n_importance>0:
            # Foreground SDF  Hierarchical Sample
            with torch.no_grad():
                # rays_o is [512,3]
                # rays_d is [512,3]
                # z_vals is [512,64]
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                # pts shape is [512,64,3]
                # get the sdf(no optimization and leanring)---> [512,64]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                
                # Hierarchical sampling
                for i in range(self.up_sample_steps):
                    # new sample intervals
                    new_z_vals = self.hierarchical_sample_z_val_cat(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)

                    # sample according to new_z_vals: The new SDF and updates the z_vals
                    z_vals, sdf = self.hierarchical_sample_sdf_sample(
                                                    rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    new_z_vals,
                                                    sdf,
                                                    last=(i + 1 == self.up_sample_steps))
                    


            n_samples = self.n_samples + self.n_importance # 128
        
        # Using the NERF to render the outside scenes,
        if self.n_outside>0:
            #(z_vals)---> [512,128]
            #(z_vals_outside)--->[512,32]
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1) #(512,128+32)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)


        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }



           
    def hierarchical_sample_z_val_cat(self,rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        '''
        Unsampling Give a fixed inv_s
        '''
        # batch_size = 512
        # n_sample is 64
        batch_size,n_samples = z_vals.shape
        #(512,64,3)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3(x,y,z)
        
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False) #(512,64)

        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        
        sdf = sdf.reshape(batch_size, n_samples) #(512,64)
        
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5) #gradient on the z direction. #(512,63)
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1) # gradient on the z direction.  #(512,63)
        
        cos_val = torch.stack([prev_cos_val,cos_val],dim=-1) #(512,63,2)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere # only consider the inside_sphere.
        
        
        dist = (next_z_vals - prev_z_vals) # dist-->63
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5 # SDF--> Z is smaller, SDF is bigger
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5 # SDF--> z is bigger, sdf is smaller
    
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s) # cdf
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s) # sdf

        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) # alpha #(512,63)        
        # weight
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        # sample 16's points--> n impartant is 16
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach() #(512,16)
        

        return z_samples


    def hierarchical_sample_sdf_sample(self,rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape #(512,64)
        
        _, n_importance = new_z_vals.shape#(512,16)
        
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None] # new sample idx #(512,16,3)
        
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1) # new sample points
        z_vals, index = torch.sort(z_vals, dim=-1)      

        # why the last iteration does not upsample the SDF Value.
        # TODO
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1) # new sdfs
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        
        return z_vals, sdf
        

        
    
    
    # inner class functions
    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        # get negative sdf
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))



