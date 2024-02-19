
from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    ControlNetModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer



def resize_small_res(img: Image.Image, small_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = max(
        small_edge_resolution / original_width, small_edge_resolution / original_height
    )
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    # Adjust to make divisible by base
    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    return resized_img

import random
import cv2
from empatches import EMPatches


class Img2ImgPipeline(DiffusionPipeline):
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 controlnet: ControlNetModel,
                 scheduler:DDIMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 use_cfg=False,
                 cfg_level=3
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            controlnet=controlnet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.text_embed = None
        self.use_cfg = use_cfg
        self.cfg_level = cfg_level

        self.rgb_latent_scale_factor = self.vae.config.scaling_factor
        self.depth_latent_scale_factor = self.vae.config.scaling_factor
        
        
    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 denosing_steps: int = 10,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 show_progress_bar:bool = True,
                 text_prompt:str = '',
                 strength=0.4,
                 tile_upscale=1,
                 patch_size=512
                 ) -> np.ndarray:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denosing_steps >=1
        
        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res >0:
            # input_image = resize_max_res(
            #     input_image, max_edge_resolution=processing_res
            # ) # resize image: for kitti is 231, 768
            tgt_size = int(512 * tile_upscale)
            input_image = resize_small_res(input_image, tgt_size) # short of H,W to 512
            # input_image = random_crop_batch_pil([input_image], 512, 512)[0]
            
        
        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        input_image = np.array(input_image)

        # into 512x512 patches
        emp = EMPatches()
        img_patches, indices = emp.extract_patches(input_image, patchsize=patch_size, overlap=0.3)

        result_patches = []

        for img in img_patches:
            # Normalize RGB Values.
            rgb = np.transpose(img,(2,0,1))
            rgb_norm = (rgb / 255.0 - 0.5) * 2
            rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
            rgb_norm = rgb_norm.to(device)

            rgb_norm = rgb_norm.half()
            rgb_norm = rgb_norm.unsqueeze(0)
            

            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            
            # ----------------- predicting rgb -----------------
            # already normed to [0, 1]
            rgb_pred = self.single_infer(
                input_rgb=rgb_norm,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
                text_prompt=text_prompt,
                strength=strength
            )
            
            # ----------------- Post processing -----------------
            # Convert to numpy
            # print('Image shape:', rgb_pred.size())
            rgb_pred = rgb_pred.squeeze().permute(1, 2, 0).detach().clone().cpu().numpy().astype(np.float32)

            rgb_pred = rgb_pred.clip(0, 1)

            result_patches.append(rgb_pred)
        
        final_rgb_pred = emp.merge_patches(result_patches, indices, mode='avg')

        return final_rgb_pred[..., ::-1]
    
    def __encode_text(self, textprompt, maxlength=None):
        """
        Encode text embedding for empty prompt
        """
        # print('current prompt:', textprompt)
        prompt = textprompt
        if maxlength is None:
            maxlength = self.tokenizer.model_max_length
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=maxlength,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        # print(text_input_ids.shape)

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.text_encoder.device)
        else:
            attention_mask = None

        text_embed = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0].to(self.dtype) #[1,2,1024]
        text_embed = text_embed.half()

        return text_embed

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    
        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,
                     text_prompt:str,
                     strength:float):
        
        
        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        num_inference_steps = int(num_inference_steps * 1 / strength)
        # print('num_inference_steps:', num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]

        # print('timesteps:', num_inference_steps, timesteps)

        if strength <= 1 and strength >= 0:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        else:
            raise NotImplementedError

            # print('timesteps 2:', timesteps)
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. 
        noise = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
        noise = noise.half()


        if self.text_embed is None:
            self.text_embed = self.__encode_text(text_prompt)
            
        batch_text_embed = self.text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]
        # noise_free_rgb_latent = rgb_latent

        rgb_latent = self.scheduler.add_noise(rgb_latent, noise, timesteps[:1])

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            if self.use_cfg:
                # for condition of controlnet, it should be unnormed image
                zero_image = torch.zeros(input_rgb.shape, device=device, dtype=self.dtype).half()
                unet_input = torch.cat([rgb_latent] * 2)
                controlnet_cond = torch.cat([(input_rgb+1)/2, zero_image])
                batch_text_embed_input = torch.cat([batch_text_embed, batch_text_embed])
            else:
                unet_input = rgb_latent # this order is important: [1,4,H,W]
                controlnet_cond = (input_rgb+1)/2
                batch_text_embed_input = batch_text_embed
                # print(unet_input.shape, batch_text_embed.shape)
            # notice this scale is different from the scaling in encoding
            # both scaling in encoding and here should be done
            # unet_input = self.scheduler.scale_model_input(unet_input, t)
            # print(unet_input.shape)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                unet_input,
                t,
                encoder_hidden_states=batch_text_embed_input,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )            

            weight_dtype = torch.float16

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, 
                t, 
                encoder_hidden_states=batch_text_embed_input,
                down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                return_dict=False
            )[0]  # [B, 4, h, w]

            if self.use_cfg:
                # print('Inference: Using Classifier-free Guidance with value', self.cfg_level)
                noise_pred, noise_pred_uncond = noise_pred.chunk(2)
                guidance_scale = self.cfg_level
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            rgb_latent = self.scheduler.step(noise_pred, t, rgb_latent, return_dict=False)[0]
        
        torch.cuda.empty_cache()
        # depth = self.decode_depth(depth_latent)
        rgb = self.decode_depth(rgb_latent)
        # clip prediction
        rgb = torch.clip(rgb, -1.0, 1.0)
        # shift to [0, 1]
        rgb = (rgb + 1.0) / 2.0
        return rgb
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        
        # encode
        h = self.vae.encode(rgb_in).latent_dist.mode()
        # print('h shape:', h.shape)

        # moments = self.vae.quant_conv(h)
        # mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = h * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    # def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
    #     """
    #     Decode depth latent into depth map.

    #     Args:
    #         depth_latent (`torch.Tensor`):
    #             Depth latent to be decoded.

    #     Returns:
    #         `torch.Tensor`: Decoded depth map.
    #     """
    #     # scale latent
    #     depth_latent = depth_latent / self.depth_latent_scale_factor

    #     depth_latent = depth_latent.half()
    #     # decode
    #     z = self.vae.post_quant_conv(depth_latent)
    #     stacked = self.vae.decoder(z)
    #     # mean of output channels
    #     depth_mean = stacked.mean(dim=1, keepdim=True)
    #     return depth_mean

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor

        depth_latent = depth_latent.half()
        # decode
        # z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decode(depth_latent, return_dict=False)[0]
        # mean of output channels
        # depth_mean = stacked.mean(dim=1, keepdim=True)
        return stacked


