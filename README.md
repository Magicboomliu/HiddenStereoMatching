# Diffusion Modedls  

## Dependencies (Creat Env) 

Install **Diffusers** and the **Transformers** and **Pytorch** 
```
conda create -n diffusion python=3.9 
conda activate diffusion
pip install -r requirements.txt
```
Install **xformer** properly to enable efficient transformers.(Options)   
**Note** that install from the source may cost a lot of time depend on your device.
```
conda install xformers -c xformers
# from source
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```  

#### Examples 
[Stable Diffsuion Fine-Tune: Pokemone Generation(Text2Image)](https://magicboomliu.github.io//posts/2012/08/blog-post-4/)
