import torch
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, StableDiffusionDiffEditPipeline, StableDiffusionPanoramaPipeline, ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline
from zero_cross_att import CrossFrameAttnProcessor
from diffusers.schedulers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler
import sys
sys.path.append("..")
from PIL import Image
from offline_inference.SDConImg2ImgNoiseInverseMultiDiffusionPipeline import SDConImg2ImgNoiseInversionMultiDiffusionPipeline
from offline_inference.controlnet_tile import ControlNetTileModel

import imageio
import os
import argparse
from tqdm import tqdm

def resize_for_condition_image(input_image: Image, resolution: int=512):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def main():
    
    
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Super Resolution."
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default='None',
        help="pretrained model path from hugging face or local dir",
    )  

    parser.add_argument(
        "--unet_path",
        type=str,
        default="Vhey/a-zovya-photoreal-v2",
        help="pretrained  unet model path from hugging face or local dir",
    )
    
    parser.add_argument(
        "--root_path",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/rendered_data_kitti_train/",
        help="KITTI Folder Path ",
    ) 

    parser.add_argument(
        "--filename_list",
        type=str,
        default="/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/datafiles/KITTI/kitti_raw_train.txt",
        help="pretrained  unet model path from hugging face or local dir",
    )

    parser.add_argument(
        "--requested_steps",
        type=int,
        default=32,
        help="pretrained  unet model path from hugging face or local dir",
    )

    parser.add_argument(
        "--requested_steps_inverse",
        type=int,
        default=10,
        help="pretrained  unet model path from hugging face or local dir",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.2,
        help="pretrained  unet model path from hugging face or local dir",
    )
    parser.add_argument(
        "--saved_folders",
        type=str,
        default="/media/zliu/data12/dataset/KITTI/SR_New_Views",
        help="pretrained  unet model path from hugging face or local dir",
    )



    args = parser.parse_args()
    
    os.makedirs(args.saved_folders,exist_ok=True)
    
    contents = read_text_lines(args.filename_list)

    inverse_scheduler_init = DPMSolverMultistepInverseScheduler()
    controlnet = ControlNetTileModel.from_pretrained(args.controlnet_model_path,subfolder="controlnet").half()
    pipe = SDConImg2ImgNoiseInversionMultiDiffusionPipeline.from_pretrained(
        args.unet_path, controlnet=controlnet, torch_dtype=torch.float16, inverse_scheduler=inverse_scheduler_init
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
    pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, use_karras_sigma=True)
    print(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    requested_steps = args.requested_steps
    requested_steps_inverse = args.requested_steps_inverse
    strength = args.strength
    
    idx = 0
    for fname in tqdm(contents):
        
        idx = idx +1
        basename = os.path.basename(fname)        
        left_left_from_left = fname.replace(basename,"left_left_from_left_"+basename)
        right_fname = fname.replace("image_02","image_03")
        right_right_from_right = right_fname.replace(basename,"right_right_from_right_"+basename)
        
        left_left_from_left_path = os.path.join(args.root_path,left_left_from_left)
        right_right_from_right_path = os.path.join(args.root_path,right_right_from_right)
        
        assert os.path.exists(left_left_from_left_path)
        assert os.path.exists(right_right_from_right_path)
        
        guided_images = []
        
        img_left_left = Image.open(left_left_from_left_path)
        original_size = img_left_left.size
        img_left_left = resize_for_condition_image(img_left_left)
        img_right_right = Image.open(right_right_from_right_path)
        img_right_right = resize_for_condition_image(img_right_right)

        guided_images.append(img_left_left)
        guided_images.append(img_right_right)

        prompt = ["best quality"] * len(guided_images)
        negative_prompt = ["blur, lowres, bad anatomy, bad hands, cropped, worst quality"] * len(guided_images)

        generator = torch.manual_seed(12345)
        view_batch_size = 2

        use_cross_view_att = False

        sub_batch_bmm = None

        if use_cross_view_att:
            pipe.unet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
            pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(video_length=len(guided_images), group_size=5, sub_batch_bmm=sub_batch_bmm))
        else:
            view_batch_size *= 8

        inv_latents = pipe.invert(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=guided_images,
                    strength=strength,
                    generator=generator,
                    num_inference_steps=int(requested_steps_inverse / min(strength, 0.999)) if strength > 0 else 0,
                    width=guided_images[0].size[0],
                    height=guided_images[0].size[1],
                    guidance_scale=7,
                    view_batch_size=view_batch_size,
                    circular_padding=True
                )


        result = pipe(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    # image=guided_images, 
                    image=inv_latents,
                    control_image=guided_images, 
                    width=guided_images[0].size[0],
                    height=guided_images[0].size[1],
                    strength=strength,
                    guidance_scale=7,
                    controlnet_conditioning_scale=1.,
                    generator=generator,
                    num_inference_steps=int(requested_steps / min(strength, 0.999)) if strength > 0 else 0,
                    guess_mode=True,
                    view_batch_size=view_batch_size,
                    circular_padding=True
                    ).images

        left_left_from_left_sub_folder = left_left_from_left[:-len(os.path.basename(left_left_from_left))]
        right_right_from_right_sub_folder = right_right_from_right[:-len(os.path.basename(right_right_from_right))]
        left_left_from_left_sub_folder = os.path.join(args.saved_folders, left_left_from_left_sub_folder)
        right_right_from_right_sub_folder = os.path.join(args.saved_folders, right_right_from_right_sub_folder)
        os.makedirs(left_left_from_left_sub_folder,exist_ok=True)
        os.makedirs(right_right_from_right_sub_folder,exist_ok=True)
        
        
        saved_left_left_path = os.path.join(args.saved_folders,left_left_from_left)
        saved_right_right_path = os.path.join(args.saved_folders,right_right_from_right)
        
        
        left_left_data = result[0]
        right_right_data = result[1]
        
        
        resized_left_left_data = left_left_data.resize(original_size)
        resized_right_right_data = right_right_data.resize(original_size)
  
        
        
        imageio.imsave(saved_left_left_path, resized_left_left_data)
        imageio.imsave(saved_right_right_path, resized_right_right_data)
        
        
        
        print("Finished {}/{}".format(idx,len(contents)))
        


        
        
    




if __name__=="__main__":
    main()



