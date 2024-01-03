from trainer import CheXpertTrainer, CheXpertTrainer_Asymmetric
from data_loader import CheXpertDataSet, CheXpertDataSet_Masking, CheXpertDataSet_Full, CheXpertDualMoreBalancedDataSet
import torch
import torchvision.transforms as transforms
from models import DenseNet121, DenseNet161, DenseNet161_CBAM, DenseNet161_CBAM_longer_classifier, EfficientB4, DenseNet161_trick
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
import pandas as pd
from tqdm import tqdm

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
        
def inference(args):    
    ################## Load dataset ##################  
    if 'frontal' in args.mode:
        datasetTest = CheXpertDataSet(f'{args.in_path}', get_transforms('train', 320), policy='zeroes')
    else:
        datasetTest = CheXpertDualMoreBalancedDataSet(f'{args.in_path}', get_transforms('train', 320), policy='zeroes')

    ################## Initializing DataLoader ##################
    trBatchSize = args.batchsize
    u_0_test_DL = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ################## Initialize model ##################
    device = args.device
    nnClassCount = 1
    model_dict = {'frontal_CBAM': DenseNet161_CBAM(nnClassCount),
                 'frontal_nor': DenseNet161(nnClassCount),
                 'dual': DenseNet161_trick(nnClassCount)}
    model = model_dict[args.mode].to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])    
    model.eval()

    ################## Inference ##################

    outGT = torch.FloatTensor().cpu()
    outPRED = torch.FloatTensor().cpu()     

    with torch.no_grad():
        for batchID, (varInput, target) in enumerate(tqdm(u_0_test_DL)):            

            varInput = varInput.to(device)
            varTarget = target.to(device)

            varOutput = model(varInput, mode='val')

            outGT = torch.cat((outGT, varTarget.cpu()), 0)
            outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
            
    test_df = pd.read_csv(f'{args.in_path}')
    test_df['Logits'] = outPRED

    best_f1, best_threshold = CheXpertTrainer_Asymmetric.compute_f1_with_threshold(outGT, outPRED)
    best_threshold = torch.Tensor([best_threshold])
    outPRED = (outPRED > best_threshold).float() * 1
    test_df['Pred with best thresh'] = outPRED

    test_df.to_csv(f'{args.out_path}/out.csv', index=False)

    print(f'Best f1: {best_f1}')


        
if __name__ == '__main__':
    now = datetime.datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', '-i', type=str, default='./csv_files/Front/u_0_test.csv', help='Train-validation-test .csv files as exported by read_data.py')
    parser.add_argument('--batchsize', '-bs', type=int, default=64, help='Test batch size', required=True)
    parser.add_argument('--num_workers', type=int, default=8, help='Self-explained', required=False)
    parser.add_argument('--mode', '-m', type=str, default='checkpoint'+str(now), help="Mode of model", required=True)
    parser.add_argument('--device', type=str, default='cuda', help='Self-explained')
    parser.add_argument('--checkpoint', '-cp', type=str, default=None, help="Path of checkpoint for continuing training")
    parser.add_argument('--out_path', '-o', type=str, default='./', help='Out path of csv file')
    args = parser.parse_args()
    
    inference(args)


    
    


