Train_Neus()
{
cd ..
mode="train"
mcube_threshold=0

CUDA_VISIVLE_DEVICES=0 python main.py --mode $mode --mcube_threshold $mcube_threshold 


}

Train_Neus