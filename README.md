# Effective Geomorphology Classification model (EGC): A Deep Learning Model for Geomorphology Classification

This repo implements several CNN-based models to identify five unique geomorphic features. Specifically the proposed EGC model
adopts EfficientNet as the backbone.

## Environment Setup
This project is written in Python. To run the project, one must set up the required python packages. Run the following command to install packages:

```pip install -r requirements.txt```
  
## Models

Models can be found in models.py. Supported models include LetNet, VGG16, ResNet18, ResNet34, MobileNet, and EfficientNet.

## Sandy Point Dredge Pit (SPDP) Dataset
To collect the Sandy Point Dredge Pit (SPDP) dataset, we surveyed the Sandy Point dredge pit in September 2022 using a full suite of high-resolution geophysical instruments, including interferometric sonar for swath bathymetry, sidescan sonar, and CHIRP subbottom profiler.

After data cleaning, 385 sidescan sonar images are included in the SPDP dataset, categorized into five classes, including: 
1. pit wall with the rotational slump
2. pit wall without rotational slump
3. heterogenous pit bottom (sand-mud mixture)
4. homogenous pit bottom
5. homogenous seabed outside pit  

### Data Structure
The dataset is organized into five folders, each corresponding to one of the labeled classes. Each folder contains the SSS images in `jpg`.

### Data Download
You can download the entire dataset as a ZIP file from this [Google Drive](https://drive.google.com/file/d/10o5Gw7zQQ9FEStMFCVGx1lJKxseU8umv/view?usp=drive_link).

## Run Models

From the root directory, one needs to type the following command to train a model:

```python ./train.py --config <config name>```

\<config  name\> can be one of these `['vgg', 'vgg_imagenet', 'lenet', 'efficientnet', 'efficientnet_imagenet',
'mobilenet_small', 'mobilenet_small_imagenet', 'mobilenet_large', 'mobilenet_large_imagenet']`.

For example, to train EGC model, one can run `python ./train.py --config efficientnet`.

After the training finishes, model achieving the best validation loss is saved in `./results/` and the training logs are saved in `./ckpts/`. 

One can change the hyperparameters for any model in ```./configs/hyperpatrameters.py``` file.

## Citation
If you use this dataset in your research, please cite it as:

`Zhang W, Chen X, Zhou X, Chen J, Yuan J, Zhao T, Xu K. Deep Learning-Based Geomorphic Feature Identification in Dredge Pit Marine Environment. Journal of Marine Science and Engineering. 2024; 12(7):1091. https://doi.org/10.3390/jmse12071091`
