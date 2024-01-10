from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


if __name__=="__main__":
    
    '''Creating the pipelines'''
    # autoencoder: for encoding the image
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)
    # diffusion model
    unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
    # for inference
    scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    
    vae.cuda()
    text_encoder.cuda()
    unet.cuda()
    
    
    '''Test Embedding'''
    prompt = ["a photograph of an anime girl with red eyes"]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 100  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise: create images 
    # batch size
    batch_size = len(prompt) # only 1 prompt
    
    # embed the test into a encoding space: max length is 77, type is batchencoding: [1,77]
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]

    #You’ll also need to generate the unconditional text embeddings which are the embeddings for the padding token. 
    # These need to have the same shape (batch_size and seq_length) as the conditional text_embeddings:
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(unet.device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    #This is the latent representation of the image, and it’ll be gradually denoised. At this point, the latent image is smaller 
    #than the final image size but that’s okay though because the model will transform it into the final 512x512 image dimensions later.
    
    print(unet.config.in_channels) # 4 channels
    
    # Create random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator).cuda() #[1,4,H/8,W/8]
    
    latents = latents * scheduler.init_noise_sigma # initail sigma is 1.0
    
    
    
    '''Inference'''
    scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    
    # Decode the image
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    
    ''' Visualizations '''
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    # images = (image * 255).round().astype("uint8")
    
    plt.imshow(image)
    plt.show()