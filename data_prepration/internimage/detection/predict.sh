
InternImage_Inference(){

image=/home/liuzihua/InternImage/detection/example2.png
config=configs/coco/cascade_internimage_xl_fpn_3x_coco.py
checkpoint=pretrained_InternImage_Models/cascade_internimage_xl_fpn_3x_coco.pth


python detection_generation.py --img $image \
                    --config $config \
                    --checkpoint $checkpoint


}

InternImage_Inference