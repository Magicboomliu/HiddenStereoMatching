Train_Neus()
{
cd ..
mode="train"
mcube_threshold=0
config='configs/neus.py'
CUDA_VISIVLE_DEVICES=0 python main.py --mode $mode --mcube_threshold $mcube_threshold  --conf $config

}

Train_Neus