from diffusers import DDPMPipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel
import torch





if __name__=="__main__":
    
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
    
    scheduler.set_timesteps(100)

    # sample size: the input image size
    sample_size = model.config.sample_size
    
    print(scheduler.timesteps)
    
    '''Inference the Image Here'''
    # random gussain noise
    noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
    input = noise
    for t in scheduler.timesteps:
        # predict the noise residual
        with torch.no_grad():
            # current noise and timestep.
            noisy_residual = model(input, t).sample
        
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample
    
    
    '''Do Viusalizations'''
    # visualization
    image = (input / 2 + 0.5).clamp(0, 1).squeeze() # transfer from, [-1,1] to [0,1]
    image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
    
    plt.imshow(image)
    plt.show()