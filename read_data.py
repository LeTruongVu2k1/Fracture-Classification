import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import argparse

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, 
                        default='/content/drive/MyDrive/Colab Notebooks/kaggle_API_credetial/kaggle.json',
                        help="Kaggle's Jason file needed to download Kaggle dataset")
    parser.add_argument('--csv_dir', type=str, default='./csv_files', help='Directory contains csv files')
    args = parser.parse_args()

    # os.makedirs('/root/.kaggle', exist_ok=True)
    # shutil.copy('/content/drive/MyDrive/Colab Notebooks/kaggle_API_credetial/kaggle.json', '/root/.kaggle')

    # os.system('kaggle datasets download ashery/chexpert --unzip -p dataset --quiet') # downloading kaggle dataset to "dataset" directory

    removed_columns = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Other', 'Pleural Effusion', 'AP/PA', 'No Finding',
        'Support Devices']

    val_data = pd.read_csv('dataset/valid.csv')
    fracture_val_data = val_data.drop(removed_columns,axis=1)
    fracture_val_data['Path'] = fracture_val_data['Path'].map(lambda x: x.replace('CheXpert-v1.0-small', 'dataset'))
    U_zeroes_val = fracture_val_data.replace(-1, 1)
    # U_zeroes_val.dropna(axis=0, subset=['Fracture'], inplace=True)
    U_zeroes_val.fillna(0, inplace=True)

    train_data = pd.read_csv('dataset/train.csv')
    fracture_train_data = train_data.drop(removed_columns,axis=1)
    fracture_train_data['Path'] = fracture_train_data['Path'].map(lambda x: x.replace('CheXpert-v1.0-small', 'dataset'))
    U_zeroes_train = fracture_train_data.replace(-1, 1)
    # U_zeroes_val.dropna(axis=0, subset=['Fracture'], inplace=True)
    U_zeroes_train.fillna(0, inplace=True)


    ############################# Splitting FRONTAL dataset #############################
    U_zeroes_train_front = U_zeroes_train[U_zeroes_train.Path.str.contains("frontal")]
    # Splitting train and val+test
    u_0_train_front, u_0_val_test_front = train_test_split(U_zeroes_train_front, test_size=0.2, random_state=19, stratify=U_zeroes_train_front['Fracture'])
    # Splitting val and test
    u_0_val_front, u_0_test_front = train_test_split(u_0_val_test_front, test_size=0.5, random_state=19, stratify=u_0_val_test_front['Fracture'])

    # Export to .csv 
    front_dir = f'{args.csv_dir}/Front'
    os.makedirs(front_dir, exist_ok=True)
    u_0_train_front.to_csv(f'{front_dir}/u_0_train.csv', index=False)
    u_0_val_front.to_csv(f'{front_dir}/u_0_val.csv', index=False)
    u_0_test_front.to_csv(f'{front_dir}/u_0_test.csv', index=False)

    ############################# Splitting BOTH FRONTAL & LATERAL dataset #############################
    # Splitting train and val+test
    u_0_train_trick, u_0_val_test_trick = train_test_split(U_zeroes_train, test_size=0.2, random_state=19, stratify=U_zeroes_train['Fracture'])
    # Splitting val and test
    u_0_val_trick, u_0_test_trick = train_test_split(u_0_val_test_trick, test_size=0.5, random_state=19, stratify=u_0_val_test_trick['Fracture'])

    # Export to .csv 
    trick_dir = f'{args.csv_dir}/Trick'
    os.makedirs(trick_dir, exist_ok=True)
    u_0_train_trick.to_csv(f'{trick_dir}/u_0_train.csv', index=False)
    u_0_val_trick.to_csv(f'{trick_dir}/u_0_val.csv', index=False)
    u_0_test_trick.to_csv(f'{trick_dir}/u_0_test.csv', index=False)