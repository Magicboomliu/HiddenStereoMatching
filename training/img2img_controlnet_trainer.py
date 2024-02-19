
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import os
import logging
import tqdm

from accelerate import Accelerator
import transformers
import datasets
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

import random

from pylab import plt
import cv2
# import torchvision


import diffusers
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    DDIMInverseScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import accelerate

import sys
sys.path.append("..")
from Inference.img2img_controlnet_direct_pipeline_half import Img2ImgPipeline


from training.img2img_dataset_configuration import prepare_dataset,resize_small_res_tensor,random_crop_batch


from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")



def log_validation(vae,text_encoder,tokenizer,unet,controlnet,args,accelerator,weight_dtype,scheduler,epoch,
                   input_image_path="/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/2011_09_26_2011_09_26_drive_0001_sync_image_02_data_left_0000000013.png",
                   save_path='/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/playground/rendered_sample.png',
                   textprompt='',
                   strength=0.3,
                   use_cfg=False,
                   cfg_level=3,
                   upscale_ratio=1,
                   patch_size=512
                   ):
    
    denoise_steps = 10
    processing_res = 768
    match_input_res = True
    
    logger.info("Running validation ... ")
    pipeline = Img2ImgPipeline(vae=accelerator.unwrap_model(vae),
                               text_encoder=accelerator.unwrap_model(text_encoder),
                               tokenizer=tokenizer,
                               unet = accelerator.unwrap_model(unet),
                               controlnet=accelerator.unwrap_model(controlnet),
                               scheduler = accelerator.unwrap_model(scheduler),
                               use_cfg=use_cfg,
                               cfg_level=cfg_level
                               )
    # pipeline = Img2ImgPipeline.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name_or_path,)

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        input_image_pil = Image.open(input_image_path)

        pipe_out = pipeline(input_image_pil,
             denosing_steps=denoise_steps,
             processing_res = processing_res,
             match_input_res = match_input_res,
             show_progress_bar = True,
             text_prompt=textprompt,
             strength=strength,
             tile_upscale=upscale_ratio,
             patch_size=patch_size
             )

        rgb_pred = pipe_out
        rgb_pred = np.asarray(rgb_pred*255, dtype=np.uint8)
        cv2.imwrite(save_path, rgb_pred)

        del pipeline
        torch.cuda.empty_cache()





def parse_args():
    parser = argparse.ArgumentParser(description="Image to Image Diffusion on KITTI, using low-quality image as condition")
    
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier of controlnet from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kitti",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/yli/NeRF_Reimp/img2img_diff/Datasets/kitti_noviewdir_diff",
        required=True,
        help="The Root Dataset Path.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--recom_resolution",
        type=int,
        default=768,
        help=(
            "The resolution for resizeing the input images and the depth/disparity to make full use of the pre-trained model from \
                from the stable diffusion vae, for common cases, do not change this parameter"
        ),
    )
    #TODO : Data Augmentation
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=70)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--use_cfg",
        action="store_true",
        help="Whether or not to use classifier-free guidance (uncond input with guidance image = 0).",
    )
    parser.add_argument(
        "--cfg_level",
        type=float,
        default=3,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--training_cfg_zero_rate",
        type=float,
        default=0.1,
        help="Possibility of Classifier-free guidance during training (setting input image guidance as 0).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--training_upscale_ratio",
        type=float,
        default=1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    # using EMA for improving the generalization
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--textprompt",
        type=str,
        default="",
        help="textprompt to diffusion model",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--crop_size_input",
        type=int,
        default=512,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # using xformers for efficient training 
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # noise offset?::: #TODO HERE
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # validations every 5 Epochs
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.dataset_path is None:
        raise ValueError("Need either a dataset name or a DataPath.")

    return args
    
    
def main():
    
    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()
    # save  the tensorboard log files
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # tell the gradient_accumulation_steps, mix precison, and tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs
    # set the warning levels
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    logger.info("loading the noise scheduler and the tokenizer from {}".format(args.pretrained_model_name_or_path),main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
        
        # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
        #                                             in_channels=4, sample_size=96,
        #                                             low_cpu_mem_usage=False)
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=4, sample_size=64,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet)
        # ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
        #                                     in_channels=8, sample_size=96,
        #                                     low_cpu_mem_usage=False,
        #                                     ignore_mismatched_sizes=True)

    # Freeze vae and text_encoder and set unet to trainable.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # only make the unet-trainable
    controlnet.train()
    
    # using xformers for efficient attentions.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # using checkpint  for saving the memories
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # optimizer settings
    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    with accelerator.main_process_first():
        (train_loader,test_loader), dataset_config_dict = prepare_dataset(data_name=args.dataset_name,
                                                                      datapath=args.dataset_path,
                                                                      batch_size=args.train_batch_size,
                                                                      test_batch=1,
                                                                      datathread=args.dataloader_num_workers,
                                                                      logger=logger)

    # because the optimizer not optimized every time, so we need to calculate how many steps it optimizes,
    # it is usually optimized by 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    controlnet, unet, optimizer, train_loader, test_loader,lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_loader, test_loader,lr_scheduler
    )

    # scale factor.
    rgb_latent_scale_factor = vae.config.scaling_factor
    depth_latent_scale_factor = vae.config.scaling_factor


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process:
        controlnet.eval()
        log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                scheduler=noise_scheduler,
                epoch=0,
                strength=0.3,
                textprompt=args.textprompt,
                use_cfg=args.use_cfg,
                cfg_level=args.cfg_level,
                upscale_ratio=args.training_upscale_ratio,
                patch_size=args.crop_size_input
            )
    
    
    print_cfg = False
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train() 
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                # # convert the images and the depths into lantent space.
                # left_image_data = batch['img_left']
                # left_disparity = batch['gt_disp']
                
                # left_disp_single = left_disparity.unsqueeze(0)
                # left_disparity_stacked = left_disp_single.repeat(1,3,1,1)
                # left_image_data_resized = resize_max_res_tensor(left_image_data,is_disp=False) #range in (0-1)
                
                # left_disparity_resized = resize_max_res_tensor(left_disparity_stacked,is_disp=True) # not range
                # # depth normalization: [([1, 3, 432, 768])]
                # left_disparity_resized_normalized = Disparity_Normalization(left_disparity_resized)

                input_image = batch['input'] #[1,3,H,W]-->[375,1242]
                gt_image = batch['gt']       #[1,3,H,W]

                input_image_resized = resize_small_res_tensor(input_image, is_disp=False, recom_resolution=512*args.training_upscale_ratio) #range in (0-1)
                gt_image_resized = resize_small_res_tensor(gt_image, is_disp=False, recom_resolution=512*args.training_upscale_ratio)
                input_image_resized, gt_image_resized = random_crop_batch(torch.cat([input_image_resized, gt_image_resized]), args.crop_size_input, args.crop_size_input).chunk(2)

                if args.use_cfg and random.random() <= args.training_cfg_zero_rate:
                    # to simulate classifier-free guidance
                    if not print_cfg:
                        logger.info(f'Set input image to 0 for CFG during inference with possibility of {args.training_cfg_zero_rate}')
                        print_cfg = True
                    input_image_resized[..., :] = 0
                input_image_resized = (input_image_resized - 0.5) * 2
                gt_image_resized = (gt_image_resized - 0.5) * 2
                
                # convert images and the disparity into latent space.
                mean_rgb_gt = vae.encode(gt_image_resized.to(weight_dtype)).latent_dist.sample()
                rgb_gt_latents = mean_rgb_gt * rgb_latent_scale_factor    #torch.Size([1, 4, 54, 96])
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(rgb_gt_latents) # create noise
                # here is the setting batch size, in our settings, it can be 1.0
                bsz = rgb_gt_latents.shape[0]

                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=rgb_gt_latents.device)
                timesteps = timesteps.long()
                
                # add noise to the depth lantents
                noisy_rgb_gt_latents = noise_scheduler.add_noise(rgb_gt_latents, noise, timesteps)
                # Encode text embedding for empty prompt
                prompt = args.textprompt
                # logger.info(f"text prompt during training: {prompt}")
                text_inputs =tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device) #[1,2]
                # print(text_input_ids.shape)
                empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)

                # controlnet_cond should be unnormed image
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_rgb_gt_latents,
                    timesteps,
                    encoder_hidden_states=empty_text_embed,
                    controlnet_cond=(input_image_resized+1)/2,
                    return_dict=False,
                )


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(rgb_gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                batch_empty_text_embed = empty_text_embed.repeat((noisy_rgb_gt_latents.shape[0], 1, 1))  # [B, 2, 1024]
                
                # predict the noise residual
                noise_pred = unet(
                    noisy_rgb_gt_latents,
                    timesteps,
                    encoder_hidden_states=batch_empty_text_embed,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False)[0]  # [B, 4, h, w]
                
                # loss functions
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # currently the EMA is not used.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if accelerator.is_main_process:
                    if global_step % 1000 == 0:
                        log_validation(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            scheduler=noise_scheduler,
                            epoch=epoch,
                            strength=0.3,
                            textprompt=args.textprompt,
                            use_cfg=args.use_cfg,
                            cfg_level=args.cfg_level,
                            upscale_ratio=args.training_upscale_ratio,
                            patch_size=args.crop_size_input
                        )
                
                
                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # save_path = os.path.join(args.output_dir, f"checkpoint")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # os.removedirs(f'{args.output_dir}/checkpoints*')
                        
                        accelerator.save_state(save_path)
                        with open(os.path.join(args.output_dir, f"checkpoint-{global_step}", 'steps.txt'), 'w') as f:
                            f.writelines(f'Current step: {global_step}')
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break
            


        
        

        if accelerator.is_main_process:
            # validation each epoch by calculate the epe and the visualization depth
                
            # validation inference here
            if epoch % 1 == 0:
                controlnet.eval()
                for strength in [0.3]:
                    log_validation(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            scheduler=noise_scheduler,
                            epoch=epoch,
                            strength=strength,
                            textprompt=args.textprompt,
                            use_cfg=args.use_cfg,
                            cfg_level=args.cfg_level,
                            upscale_ratio=args.training_upscale_ratio,
                            patch_size=args.crop_size_input
                        )
                controlnet.train()
                

    
    

        
    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    
    
        
        
    


if __name__=="__main__":
    main()