LAUNCH_SD_FINETUNE_MINE(){
cd ..
cd text_to_images

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/pokemon-blip-captions"


accelerate launch --mixed_precision="fp16"  --multi_gpu stable_fine_tune_accelate.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" 
}

NoteBook_LAUNCH(){
cd ..
cd text_to_images

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/pokemon-blip-captions"
python stable_fine_tune_notebook.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" 



}


NoteBook_LAUNCH

# LAUNCH_SD_FINETUNE_MINE
