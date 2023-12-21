run_diffusion(){
cd ..
CUDA_VISIBLE_DEVICES=0 python run_diffusion.py
}

inference_diffusion(){
cd ..
CUDA_VISIBLE_DEVICES=0 python inference.py
}

inference_diffusion_conditioned(){
cd ..
CUDA_VISIBLE_DEVICES=0 python run_diffusion_condition.py

}


# inference_diffusion
# run_diffusion
inference_diffusion_conditioned