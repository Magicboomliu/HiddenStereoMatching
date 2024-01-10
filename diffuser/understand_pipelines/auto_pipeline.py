from diffusers import AutoPipelineForText2Image
import torch


if __name__=="__main__":
    pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

    image = pipeline(prompt, num_inference_steps=25).images[0]
    