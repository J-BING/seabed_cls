# Effective Geomorphology Classification model (EGC): A Deep Learning Model for Geomorphology Classification

This repo implements serveral CNN-based models to identify five unique geomorphic features. Specifically the proposed EGC model
adopts EfficientNet as the backbone.

## Environment Setup
This project is written by Python. To run the project, one must setup required python packages. Run following command to install packages:

```pip install -r requirements.txt```
  
## Models

Models can be found in models.py. Supported models include LetNet, VGG16, ResNet18, ResNet34, MobileNet, EfficientNet.

## Sandy Point Dredge Pit (SPDP) Dataset
To collect Sandy Point Dredge Pit (SPDP) dataset, we surveyed the Sandy Point dredge pit in September 2022 using a full suite of high-resolution geophysical instru-ments, including interferometric sonar for swath bathymetry, sidescan sonar, and CHIRP subbottom profiler.

After data cleaning, 385 sidescan sonar images are included in SPDP dataset, categorized into five classes, including: 
1. pit wall with rotational slump
2. pit wall without rotational slump
3. heterogenous pit bottom (sand-mud mixture)
4. homogenous pit bottom
5. homogenous seabed outside pit  

***SPDP dataset is not published. To get the dataset, please contact Wenqiang Zhang via wzhan46@lsu.edu.***

## Run Models

From the root directory, one needs to type the following command to train a model:

```python ./train.py --config <config name>```

\<config  name\> can be one of these `['vgg', 'vgg_imagenet', 'lenet', 'efficientnet', 'efficientnet_imagenet',
'mobilenet_small', 'mobilenet_small_imagenet', 'mobilenet_large', 'mobilenet_large_imagenet']`.

For example, to train EGC model, one can run `python ./train.py --config efficientnet`.

After the training finishes, model achieving the best validation loss is saved in `./results/` and the training logs are saved in `./ckpts/`. 

One can change the hyperparameters for any model in ```./configs/hyperpatrameters.py``` file.