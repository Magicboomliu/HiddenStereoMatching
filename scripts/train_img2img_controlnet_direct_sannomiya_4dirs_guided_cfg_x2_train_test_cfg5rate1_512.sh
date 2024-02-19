LAUNCH_TRAINING(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'
root_path='/media/zliu/data12/dataset/KITTI/KITTI_Rendered_GT/'
dataset_name='kitti'
output_dir='../outputs/step2_refine'
train_batch_size=1
num_train_epochs=15
gradient_accumulation_steps=16
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='sannomiya_528_12dirs_guided_img2img_pretrain_tracker'
checkpointing_steps=10000
textprompt=""
crop_size_input=512
controlnet_model_name_or_path='lllyasviel/control_v11f1e_sd15_tile'
cfg_level=3
training_upscale_ratio=1
training_cfg_zero_rate=0.1


CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" img2img_controlnet_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name\
                  --dataset_path $root_path\
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --checkpointing_steps $checkpointing_steps \
                  --textprompt "$textprompt" \
                  --use_cfg \
                  --cfg_level $cfg_level \
                  --training_upscale_ratio $training_upscale_ratio \
                  --training_cfg_zero_rate $training_cfg_zero_rate \
                  --crop_size_input $crop_size_input \
                  --controlnet_model_name_or_path $controlnet_model_name_or_path

}



LAUNCH_TRAINING