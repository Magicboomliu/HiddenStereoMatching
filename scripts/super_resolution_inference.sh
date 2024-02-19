SuperResolution_Inference(){
cd ..
cd offline_inference

controlnet_model_path="/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/outputs/step2_refine/checkpoint-70000"
unet_path="Vhey/a-zovya-photoreal-v2"
root_path="/media/zliu/data12/dataset/KITTI/rendered_data_kitti_train/"
filename_list="/home/zliu/Desktop/ECCV2024/Ablations/ImageEnhancement/datafiles/KITTI/kitti_raw_train.txt"


python super_resolution_run.py --controlnet_model_path $controlnet_model_path \
                               --unet_path $unet_path \
                               --root_path $root_path \
                               --filename_list $filename_list

}

SuperResolution_Inference