# Deep Learning-Based Geomorphic Feature Identification in Dredge Pit Marine Environments

This repository contains the code and dataset used in the research paper titled *"Deep Learning-Based Geomorphic Feature Identification in Dredge Pit Marine Environments"*, published in the *Journal of Marine Science and Engineering* (DOI: [10.3390/jmse12071091](https://doi.org/10.3390/jmse12071091)).

## Introduction
This project introduces a novel deep learning-based approach for identifying geomorphic features in dredge pits using sidescan sonar (SSS) data. The model presented, the Effective Geomorphology Classification (EGC) model, leverages convolutional neural networks (CNNs) to classify SSS images into different geomorphic categories, enhancing feature identification for underwater environments, particularly those affected by dredging activities.

## Sandy Point Dredge Pit (SPDP) Dataset
The dataset used in this study, the Sandy Point Dredge Pit (SPDP) dataset, consists of 385 sidescan sonar images collected from the west flank of the Mississippi bird-foot delta on the Louisiana inner shelf. The dataset is classified into five distinct geomorphic environments:
1. Pit wall with rotational slump
2. Pit wall without rotational slump
3. Homogenous pit bottom
4. Heterogenous seabed (sand-mud mixture) outside the pit
5. Homogenous seabed outside the pit

### Data Augmentation
To enhance model performance, we applied various data augmentation techniques, including rotation, scaling, and cropping, to simulate different environmental conditions and increase the diversity of the training set.

### Data Structure
The dataset is organized into five folders, each corresponding to one of the labeled classes. Each folder contains the SSS images in `jpg`.

### Data Download
You can download the entire dataset as a ZIP file from this [Google Drive](https://drive.google.com/file/d/10o5Gw7zQQ9FEStMFCVGx1lJKxseU8umv/view?usp=drive_link).

## Model Architecture
The EGC model is built on the EfficientNet-B0 backbone and is designed to balance performance and computational efficiency. The architecture comprises two main modules:
1. **Feature Extractor**: Extracts multi-scale geomorphological features from the SSS images.
2. **Classifier**: A two-layer fully connected network that classifies the input into one of the five geomorphic categories.

## Training and Evaluation
The model was trained on the SPDP dataset using the Adam optimizer with a cosine-decaying learning rate scheduler. The training process involved 400 epochs with a batch size of 8.

## Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/J-BING/seabed_cls.git
cd seabed_cls
pip install -r requirements.txt
```

## Usage

To train the model, use the following command:

```python ./train.py --config <config name>```

\<config  name\> can be one of these `['vgg', 'vgg_imagenet', 'lenet', 'efficientnet', 'efficientnet_imagenet',
'mobilenet_small', 'mobilenet_small_imagenet', 'mobilenet_large', 'mobilenet_large_imagenet']`.

For example, to train EGC model, one can run `python ./train.py --config efficientnet`.

After the training finishes, model achieving the best validation loss is saved in `./results/` and the training logs are saved in `./ckpts/`. 

One can change the hyperparameters for any model in ```./configs/hyperpatrameters.py``` file.

## Citation
If you use this code or dataset in your research, please cite our paper:

`Zhang W, Chen X, Zhou X, Chen J, Yuan J, Zhao T, Xu K. Deep Learning-Based Geomorphic Feature Identification in Dredge Pit Marine Environment. Journal of Marine Science and Engineering. 2024; 12(7):1091. https://doi.org/10.3390/jmse12071091`
