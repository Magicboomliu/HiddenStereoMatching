### Dependencies (Creat Env) 

Install **Diffusers** and the **Transformers** and **Pytorch** 

* Pytorch=1.X
```
conda create -n diffusion python=3.10
conda activate diffusion
pip install -r requriements_torch1x.txt
```
* Pytorch=2.0.1

```
conda create -n diffusion python=3.10
conda activate diffusion
pip install -r requriements_torch20.txt
```

Install **xformers**  
* Pytorch=1.X
```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```  
* Pytorch=2.0.1
```
pip install xformers==0.0.20
```  


