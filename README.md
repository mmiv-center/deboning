## Remove Bones from X-Ray images


This project attempts to remove bone structures from plane film XRay images to make the soft tissue structure easier to appreciate. 

<p align="center">
   <img width="300" src="https://github.com/mmiv-center/deboning/blob/master/img/orig.png?raw=true">
   <img width="300" src="https://github.com/mmiv-center/deboning/blob/master/img/logo.png?raw=true">
   <img width="300" src="https://github.com/mmiv-center/deboning/blob/master/img/db.png?raw=true">
</p>

### Setup

A general pytorch environment:
```
conda create --name fastai
conda activate fastai
conda install -c pytorch -c fastai fastai
pip3 install torchvision
conda install -c pytorch -c fastai fastai
```

Check utilization of GPU
```
nvidia-smi -q -g 0 -d UTILIZATION -l
```
