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
python train.py --csv_dir ./Front --path ./frontal_checkpoints --mode frontal --p_mixup 0 --p_masked 0 \
                --batchsize 16 --val_batchsize 256 --num_workers 8 --epoch 50
```
