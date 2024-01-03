# Fracture Classification

Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. 
Automated chest radiograph interpretation at the level of practicing radiologists could provide substantial benefit in many medical settings, 
from improved workflow prioritization and clinical decision support to large-scale screening and global population health initiatives. In this project, I build a model to classify whether a X-Ray has fracture (labeled 
with `0`) or not (labeled with `1`).

# Dataset

Although the dataset I worked on is private, you can use the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), these two datasets are quite similar.

The following script will:
1. Download the CheXpert dataset from Kaggle and store them in *dataset* directory.
2. Split dataset into *train-validation-test* and export them to corresponding *.csv* files.

```bash
python read_data.py --json_path '/content/drive/MyDrive/Colab Notebooks/kaggle_API_credetial/kaggle.json'
```
*Note*: 
- The --json_path above is an example of storing it in Gdrive, you can modify it for your case.
- This project research on 2 case: only *Frontal X-Rays* and *Both Frontal+Lateral X-rays*. `read_data.py` will export 2 folders contains these 2 cases. 

# Training

```bash
python train.py --csv_dir ./Front --path ./frontal_checkpoints --mode frontal_CBAM --p_mixup 0 --p_masked 0 \
                --batchsize 16 --val_batchsize 256 --num_workers 8 --epoch 50
```

# Experiments
## Frontal
- Most of time we only have *Frontal images* to make predictions, we will assume that the training set doensn't have *Lateral*.
### 1. Default Version
- The backbone I've used is DenseNet161 with the pretrained weights from `torchvision.model.DenseNet161_Weights.IMAGENET1K_V1`, with the customized `Classifier` to downsize the number of classes to 1 (binary classification problem).
- Setting `--mode` to `frontal_nor` to use this version. 
### 2. Balanced Cross Entropy
- To deal with imbalanced dataset, one of possible solutions is using *Balanced Cross Entropy*, which use the distribution of classes in dataset to affect the loss function.
- However, it's not actually affect the gradient descent of loss function.
- You can use this loss by setting --weighted_score to True (Default: False). 
### 2. Assymetric Loss
- The positive-negative imbalance of *Fracture* and *No Fracture* dominates the optimization process, and can lead to under-emphasizing gradients from positive labels during training, resulting in poor accuracy.
- To tackle with this problem, I've used Assymetric Loss, as described in this [paper](https://arxiv.org/abs/2009.14119). In a nutshell, it decouple the focusing levels of the positive (*Fracture*) and negative (*No Fracture*), i.e using different gamma in [Focal Loss](https://arxiv.org/abs/1708.02002) for each class.
- By default, the model uses this loss in training.
### 2. Convolutional Block Attention Module - CBAM
- Recently, many different SOTA networks have leveraged this attention mechanism that have significantly improved and refined real-time results. Lightweight network and straightforward implementations have made it easier to incorporate directly into the feature extraction part of convolutional neural networks.
- These CBAM modules will be incorporated between `_DenseLayers` inside a `_DenseBlock` as well as between `_Transition` and `_DenseBlock`.
- Setting `--mode` to `frontal_CBAM` to use this version. 
## Multi-view approach
- We hypothesize that the model will get better performance when combining *Frontal view* with *Lateral view* as it will study features in both views.
- In order to do that, I've conducted a quite special `Dataloader` class, which is `CheXpertDualMoreBalancedDataSet()`. It allows us to get a pair of *Frontal* and *Lateral* X-Rays during training. Moreover, due to the unbalancing between *Fracture* and *No Fracture* classes, the probability of getting a pair of (Frontal, Lateral) with both *Fracture* is low, I've used these three paramters in hope to control the unbalanced problem.
  
| CheXpertDualMoreBalancedDataSet's important parameters |                                                   Usage                                                  |
|:------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|
| _frontal_balanced_ratio_                               | Control the portion of Fracture and No Fracture in Frontal images.                                       |
| _mask_flag_                                            | Probability of training with a pair of (frontal,torch.zeros_like(frontal)) instead of (frontal,lateral). |
| _balanced_flag_                                        | Probability of taking a Fracture Lateral image when the Frontal image is Fracture.                       |

- This "Dual Model" (sometimes I call "Tricked Model") will *stack* the features extracted from a *Shared Block*, then the *Classifier* will use these multi-view features to make prediction.
  
![image](https://github.com/LeTruongVu2k1/Fracture-Classification/assets/96435803/6d01bb42-1a45-4216-a8c7-50bdb8aca6e7)

- Setting `--mode` to `dual` to use this version.
  