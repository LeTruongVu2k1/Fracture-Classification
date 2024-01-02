from trainer import CheXpertTrainer, CheXpertTrainer_Asymmetric
from data_loader import CheXpertDataSet, CheXpertDataSet_Masking, CheXpertDataSet_Full
import torch
import torchvision.transforms as transforms
from models import DenseNet121, DenseNet161, DenseNet161_CBAM, DenseNet161_CBAM_longer_classifier, EfficientB4
from torch.utils.data import DataLoader
import albumentations.augmentations.transforms as albu
import albumentations.augmentations.crops 
import albumentations
import albumentations.pytorch 
import cv2
import argparse
import datetime
import os
import random
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # avoid warning 'nan' values when calculating f1-score 


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Note!!! 
# These albumentations.Compose return a dictionary    
def get_transforms(phase, image_size):
        if phase == 'train':
            return albumentations.Compose([
                # MinEdgeResize(size),
#                 albumentations.augmentations.transforms.ToFloat(),
                albumentations.augmentations.crops.RandomCrop(image_size, image_size, always_apply=True),
                albumentations.augmentations.geometric.transforms.HorizontalFlip(),
                albumentations.OneOf([
                    albu.RandomGamma(p=0.5),
                    albu.RandomBrightnessContrast(p=0.5),
                ], p=0.3),
#                 albumentations.augmentations.dropout.coarse_dropout.CoarseDropout(p=0.3),
                albumentations.augmentations.geometric.transforms.ShiftScaleRotate(shift_limit=0.01, rotate_limit=25, scale_limit=0.1, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                albu.Normalize(mean=(0.485), std=(0.229)),
                albumentations.pytorch.transforms.ToTensorV2()
            ])
        else:
            return albumentations.Compose([
#                 albumentations.augmentations.transforms.ToFloat(),
                # MinEdgeResize(size),
                albumentations.augmentations.crops.CenterCrop(image_size, image_size),
                albu.Normalize(mean=(0.485), std=(0.229)),
                albumentations.pytorch.transforms.ToTensorV2()
            ])
        
def train(args):
    epoch = args.epoch
    
    ################## Load FRONTAL dataset ##################  
    datasetTrain_front = CheXpertDataSet_Full(f'{args.csv_dir}/u_0_train.csv', get_transforms('train', 320), policy='zeroes')
#     datasetTest_front = CheXpertDataSet('u_0_test_front.csv', get_transforms('test', 320), policy='zeroes')
    datasetVal_front = CheXpertDataSet_Full(f'{args.csv_dir}/u_0_val.csv', get_transforms('val', 320), policy='zeroes')
#     datasetTrain_front = CheXpertDataSet_Masking('./heart_detection/u_0_train_front_masked.pkl', args.p_masked, get_transforms('train', 320), policy='zeroes')
#     datasetTest_front = CheXpertDataSet_Masking('./heart_detection/u_0_test_front_masked.pkl', args.p_masked, get_transforms('test', 320), policy='zeroes')
#     datasetVal_front = CheXpertDataSet_Masking('./heart_detection/u_0_val_front_masked.pkl', args.p_masked, get_transforms('val', 320), policy='zeroes')    
    
    ################## weighted loss ##################
    weighted_class = dict()
    if args.weighted_score:        
        weighted_class['pos_label'] = 0.5 * len(datasetTrain_front) / (datasetTrain_front.labels == 1).sum() 
        weighted_class['neg_label'] = 0.5 * len(datasetTrain_front) / (datasetTrain_front.labels == 0).sum()
        print(f"Pos-label's weight: {weighted_class['pos_label']}")
        print(f"Neg-label's weight: {weighted_class['neg_label']}")
        
    ################## Initializing DataLoader ##################
    trBatchSize = args.batchsize
    u_0_train_DL_front = DataLoader(dataset=datasetTrain_front, batch_size=trBatchSize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    u_0_val_DL_front = DataLoader(dataset=datasetVal_front, batch_size=512, shuffle=True, num_workers=args.num_workers, pin_memory=True)
#     u_0_test_DL_front = DataLoader(dataset=datasetTest_front, batch_size=trBatchSize*8, shuffle=True, num_workers=8, pin_memory=True)
        
    ################## Training ##################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nnClassCount = 1
    model = EfficientB4(nnClassCount).to(device) # Step 0: Initialize global model and load the model

    # # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    # for param in u_0_frontal_model.densenet121.features.parameters():
    #     param.requires_grad = False

    params, records = CheXpertTrainer_Asymmetric.train(args, model, u_0_train_DL_front, u_0_val_DL_front,
                                  nnClassCount, trMaxEpoch=epoch, checkpoint=args.checkpoint, device=device, mode=args.mode, 
                                                       model_path=args.path, weighted_class=weighted_class, p_mixup=args.p_mixup,
                                                       is_triple=args.is_triple)


        
if __name__ == '__main__':
    now = datetime.datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='./Front', help='Train-validation-test .csv files as exported by read_data.py')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epoch', required=True)
    parser.add_argument('--batchsize', '-bs', type=int, default=64, help='Train batch size', required=True)
    parser.add_argument('--val_batchsize', '-vbs', type=int, default=256, help='Val batch size', required=False)
    parser.add_argument('--num_workers', type=int, default=8, help='Self-explained', required=False)
    parser.add_argument('--path', '-p', type=str, default='./', help="Path for saving best model's checkpoint", required=True)
    parser.add_argument('--mode', '-m', type=str, default='checkpoint'+str(now), help="Mode of model", required=True)
    parser.add_argument('-device', type=str, default='cuda', help='Self-explained')
    parser.add_argument('--checkpoint', '-cp', type=str, default=None, help="Path of checkpoint for continuing training")
    parser.add_argument('--weighted_score', type=bool, default=False, help="Whether to use weighted-class technique in BCE loss function or not")
    parser.add_argument('--p_mixup', '-pmx', type=float, default=0.5, help='Probability of a batch being Mix Up', required=True)
    parser.add_argument('--is_triple', '-triple', type=bool, default=False, help="Whether to use triple loss or not")
    parser.add_argument('--p_masked', '-pm', type=float, default=0.5, help='Probability of masking at heart region of image', required=True)
    
    args = parser.parse_args()
    
    train(args)


    
    


