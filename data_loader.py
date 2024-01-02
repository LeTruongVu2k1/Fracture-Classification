from torch.utils.data import Dataset
import csv
from PIL import Image
import torch    
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import random 
import os
from PIL import Image

def log_auto(u,L=255,y=0.5): #0.5->2.5
    c = (L-1)**(1-y)
    v = c*(u**y)
    return v.astype('uint8')

class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """
#         image_names = []
#         labels = []
#         images = []
        
#         with open(data_PATH, "r") as f:
#             csvReader = csv.reader(f)
#             print('******** Reading files ********')
#             next(csvReader, None) # skip the header
#             row_count = sum(1 for row in csvReader) 
#             f.seek(0) # move back to top of file
#             next(csvReader, None) # skip the header
#             for line in tqdm(csvReader, total=row_count):
#                 image_name = line[0]
# #                 image = Image.open(image_name).convert('RGB')
#                 label = [float(line[4])] # originally label is "str" type, torch.FloatTensor need 1-D array, not scalar
# #                 for i in range(2):
# #                     if label[i]:
# #                         a = float(label[i])
# #                         if a == 1:
# #                             label[i] = 1
# # #                         elif a == -1:
# # #                             if policy == "ones":
# # #                                 label[i] = 1
# # #                             elif policy == "zeroes":
# # #                                 label[i] = 0
# # #                             else:
# # #                                 label[i] = 0
# #                         else:
# #                             label[i] = 0
# #                     else:
# #                         label[i] = 0
                
#                 image_names.append(image_name)
# #                 images.append(image)
#                 labels.append(label)
                   
        df = pd.read_csv(data_PATH)
        self.image_names = df['path_new'].values 
#         self.images = images
        self.labels = df['Fracture'].values.reshape(-1,1).astype('float')
        self.transform = transform
        self.log_transform = log_auto
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
#         image_name = image_name.replace("CheXpert-v1.0-small", os.getcwd()+'/dataset')
        image = cv2.imread(image_name)
#         image = self.images[index]        
        label = self.labels[index]
        if self.transform is not None:
            image = self.log_transform(image)
            image = self.transform(image=image)['image']
            
        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.image_names)
    
    
class CheXpertDualBalancedDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """
        frontal_image_names = []
        lateral_image_names = deque()
        frontal_labels = []
        lateral_labels = deque()
        
        self.lateral_fracture_count = 0
        
        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[4]
                view = line[2] # frontal/lateral
                label = [float(line[3])] # originally label is "str" type, torch.FloatTensor need 1-D array, not scalar

#                 for i in range(2):
#                     if label[i]:
#                         a = float(label[i])
#                         if a == 1:
#                             label[i] = 1
# #                         elif a == -1:
# #                             if policy == "ones":
# #                                 label[i] = 1
# #                             elif policy == "zeroes":
# #                                 label[i] = 0
# #                             else:
# #                                 label[i] = 0
#                         else:
#                             label[i] = 0
#                     else:
#                         label[i] = 0
                if view == 'Frontal':
                    frontal_image_names.append(image_name)
                    frontal_labels.append(label)
                else:
                    if label == 1.0:
                        lateral_image_names.append(image_name)
                        lateral_labels.append(label)
                        lateral_fracture_count += 1
                        
                    else:
                        lateral_image_names.appendleft(image_name)
                        lateral_labels.appendleft(label)
                        
        self.frontal_image_names = frontal_image_names
        self.lateral_image_names = lateral_image_names
        
        self.frontal_labels = frontal_labels
        self.lateral_labels = lateral_labels
        
        self.lateral_no_fracture_count = len(self.lateral_image_names) - self.lateral_fracture_count
        self.transform = transform

        self.new_perm()

    def new_perm(self):
        x, y = len(self.lateral_image_names), len(self.frontal_image_names)
        self.randperm = torch.randperm(y)[:x]

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""        
         ######## frontal images ########
        frontal_image_name = self.frontal_image_names[self.randperm[index]]
        frontal_image = Image.open(frontal_image_name).convert('RGB')
        frontal_label = self.frontal_labels[index]

        if self.transform is not None:
            frontal_image = self.transform(image=frontal_image)
            
        ######## lateral images ########
        balanced_flag = random.random()

        if frontal_label == 1.0 and balanced_flag < 0.9: # getting 'fracture' samples in 'lateral view' if meeting the condition
            index = self.lateral_no_fracture_count + index % self.lateral_fracture_count
                
        lateral_image_name = self.lateral_image_names[index]
        lateral_image = Image.open(lateral_image_name).convert('RGB')
        lateral_label = self.lateral_labels[index]

        if self.transform is not None:
            lateral_image = self.transform(lateral_image)

        ######### Reset the randperm #########
        if index == len(self) - 1:
            self.new_perm()

        frontal_lateral_images = torch.cat((frontal_image, lateral_image), 0)
        frontal_lateral_label = torch.FloatTensor(lateral_label or frontal_label)

        return frontal_lateral_images, frontal_lateral_label

    def __len__(self):
        return min(len(self.frontal_image_names), len(self.lateral_image_names))
    
    
#############################################################################################    
    
    
class CheXpertDualMoreBalancedDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """
        data = pd.read_csv(data_PATH)
        
        ######## Path ########
        fracture_frontal_paths = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==1)].path_new.values
        no_fracture_frontal_paths = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==0)].path_new.values

        fracture_lateral_paths = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==1)].path_new.values
        no_fracture_lateral_paths = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==0)].path_new.values
        
        ######## Labels #######
        fracture_frontal_labels = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==1)].Fracture.values
        no_fracture_frontal_labels = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==0)].Fracture.values

        fracture_lateral_labels = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==1)].Fracture.values
        no_fracture_lateral_labels = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==0)].Fracture.values

        
        ######## self-assignment ########
        self.frontal_image_names = np.concatenate((no_fracture_frontal_paths, fracture_frontal_paths), axis=0)
        self.lateral_image_names = np.concatenate((no_fracture_lateral_paths, fracture_lateral_paths), axis=0)
        self.lateral_image_names = data[data['Frontal/Lateral']=='Lateral'].path_new.values
        
        self.frontal_labels = np.concatenate((no_fracture_frontal_labels, fracture_frontal_labels), axis=0)
        self.frontal_labels = self.frontal_labels.astype('float').reshape(len(self.frontal_labels), 1)
        
        self.lateral_labels = np.concatenate((no_fracture_lateral_labels, fracture_lateral_labels), axis=0)
        self.lateral_labels = data[data['Frontal/Lateral']=='Lateral'].Fracture.values
        self.lateral_labels = self.lateral_labels.astype('float').reshape(len(self.lateral_labels), 1)
        
        self.lateral_fracture_count = len(fracture_lateral_paths)
        self.lateral_no_fracture_count = len(self.lateral_image_names) - self.lateral_fracture_count
        self.frontal_fracture_count = len(fracture_frontal_paths) 
        self.frontal_no_fracture_count = len(self.frontal_image_names) - self.frontal_fracture_count
        
        self.transform = transform

        self.new_perm()
        


    def new_perm(self, frontal_balanced_ratio=0.5):
        '''
        - Sampling `frontal` samples based on `frontal_balanced_ratio`
        - The default `frontal_balanced_ratio` is 0.5, which means we will sample 50% `frontal` with `fracture` and 50% 'no fracture'
        '''
        self.frontal_balanced_ratio = frontal_balanced_ratio
        
        frontal_no_fracture_numb = int(self.frontal_balanced_ratio*len(self.lateral_image_names)) 
        frontal_fracture_numb = len(self.lateral_image_names) - frontal_no_fracture_numb
        
        frontal_no_fracture_indexes = random.choices(range(self.frontal_no_fracture_count), 
                                                     k=frontal_no_fracture_numb)
        
        frontal_fracture_indexes = random.choices(range(self.frontal_no_fracture_count, len(self.frontal_image_names)), 
                                                  k=frontal_fracture_numb)
        
        self.frontal_indexes = frontal_no_fracture_indexes + frontal_fracture_indexes # the length of `frontal_indexes` will equal to number of `lateral` samples
                                                                                      # which is len(self.lateral_image_names)
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""        
         ######## frontal images ########
        frontal_image_name = self.frontal_image_names[self.frontal_indexes[index]]
        frontal_image = cv2.imread(frontal_image_name)
        frontal_label = self.frontal_labels[self.frontal_indexes[index]]

        if self.transform is not None:
            frontal_image = self.transform(image=frontal_image)['image']
        ######## lateral images ########
        balanced_flag = random.random()
#         mask_flag = random.random()
        mask_flag = 0
    
        if mask_flag < 0.5:
            if frontal_label == [1.0] and balanced_flag < 0.9: # getting 'fracture' samples in 'lateral view' if meeting the condition
                index = self.lateral_no_fracture_count + index % self.lateral_fracture_count

            lateral_image_name = self.lateral_image_names[index]
            lateral_image = cv2.imread(lateral_image_name)
            lateral_label = self.lateral_labels[index]
            if self.transform is not None:
                lateral_image = self.transform(image=lateral_image)['image']
                    
        else:
            lateral_image = torch.zeros_like(frontal_image)
            lateral_label = frontal_label
        
        ######### Reset the randperm #########
        if index == len(self) - 1:
            self.new_perm()

#         frontal_lateral_images = torch.cat((frontal_image, lateral_image, frontal_image), 0)
        frontal_lateral_images = np.concatenate((frontal_image, lateral_image), 0) # np.concatenate faster than torch.cat!!!
        frontal_lateral_label = torch.from_numpy((lateral_label or frontal_label))
        return frontal_lateral_images, frontal_lateral_label

    def __len__(self):
        return min(len(self.frontal_image_names), len(self.lateral_image_names))
   

############################################################## Triple Loss ##############################################################

class CheXpertDualMoreBalancedDataSet_Triple(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """
        data = pd.read_csv(data_PATH)
        
        ######## Path ########
        fracture_frontal_paths = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==1)].path_new.values
        no_fracture_frontal_paths = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==0)].path_new.values

        fracture_lateral_paths = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==1)].path_new.values
        no_fracture_lateral_paths = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==0)].path_new.values
        
        ######## Labels #######
        fracture_frontal_labels = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==1)].Fracture.values
        no_fracture_frontal_labels = data[(data['Frontal/Lateral']=='Frontal') & (data['Fracture']==0)].Fracture.values

        fracture_lateral_labels = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==1)].Fracture.values
        no_fracture_lateral_labels = data[(data['Frontal/Lateral']=='Lateral') & (data['Fracture']==0)].Fracture.values

        
        ######## self-assignment ########
        self.frontal_image_names = np.concatenate((no_fracture_frontal_paths, fracture_frontal_paths), axis=0)
        self.lateral_image_names = np.concatenate((no_fracture_lateral_paths, fracture_lateral_paths), axis=0)
        self.lateral_image_names = data[data['Frontal/Lateral']=='Lateral'].path_new.values
        
        self.frontal_labels = np.concatenate((no_fracture_frontal_labels, fracture_frontal_labels), axis=0)
        self.frontal_labels = self.frontal_labels.astype('float').reshape(len(self.frontal_labels), 1)
        
        self.lateral_labels = np.concatenate((no_fracture_lateral_labels, fracture_lateral_labels), axis=0)
        self.lateral_labels = data[data['Frontal/Lateral']=='Lateral'].Fracture.values
        self.lateral_labels = self.lateral_labels.astype('float').reshape(len(self.lateral_labels), 1)
        
        self.lateral_fracture_count = len(fracture_lateral_paths)
        self.lateral_no_fracture_count = len(self.lateral_image_names) - self.lateral_fracture_count
        self.frontal_fracture_count = len(fracture_frontal_paths) 
        self.frontal_no_fracture_count = len(self.frontal_image_names) - self.frontal_fracture_count
        
        self.transform = transform

        self.new_perm()
        


    def new_perm(self, frontal_balanced_ratio=0.5):
        '''
        - Sampling `frontal` samples based on `frontal_balanced_ratio`
        - The default `frontal_balanced_ratio` is 0.5, which means we will sample 50% `frontal` with `fracture` and 50% 'no fracture'
        '''
        self.frontal_balanced_ratio = frontal_balanced_ratio
        
        frontal_no_fracture_numb = int(self.frontal_balanced_ratio*len(self.lateral_image_names)) 
        frontal_fracture_numb = len(self.lateral_image_names) - frontal_no_fracture_numb
        
        frontal_no_fracture_indexes = random.choices(range(self.frontal_no_fracture_count), 
                                                     k=frontal_no_fracture_numb)
        
        frontal_fracture_indexes = random.choices(range(self.frontal_no_fracture_count, len(self.frontal_image_names)), 
                                                  k=frontal_fracture_numb)
        
        self.frontal_indexes = frontal_no_fracture_indexes + frontal_fracture_indexes # the length of `frontal_indexes` will equal to number of `lateral` samples
                                                                                      # which is len(self.lateral_image_names)
        
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""        
         ######## frontal images ########
        frontal_image_name = self.frontal_image_names[self.frontal_indexes[index]]
        frontal_image = cv2.imread(frontal_image_name)
        frontal_label = self.frontal_labels[self.frontal_indexes[index]]

        if self.transform is not None:
            frontal_image = self.transform(image=frontal_image)['image']
        ######## lateral images ########
        balanced_flag = random.random()
#         mask_flag = random.random()
        mask_flag = 0
    
        if mask_flag < 0.5:
            if frontal_label == [1.0] and balanced_flag < 0.9: # getting 'fracture' samples in 'lateral view' if meeting the condition
                index = self.lateral_no_fracture_count + index % self.lateral_fracture_count

            lateral_image_name = self.lateral_image_names[index]
            lateral_image = cv2.imread(lateral_image_name)
            lateral_label = self.lateral_labels[index]
            if self.transform is not None:
                lateral_image = self.transform(image=lateral_image)['image']
                    
        else:
            lateral_image = torch.zeros_like(frontal_image)
            lateral_label = frontal_label
        
        ######### Reset the randperm #########
        if index == len(self) - 1:
            self.new_perm()

#         frontal_lateral_images = torch.cat((frontal_image, lateral_image, frontal_image), 0)
        frontal_lateral_images = np.concatenate((frontal_image, lateral_image), 0) # np.concatenate faster than torch.cat!!!
        frontal_lateral_labels = np.concatenate((frontal_label, lateral_label, frontal_label or lateral_label), 0)
        frontal_lateral_labels = torch.from_numpy(frontal_lateral_labels)
        
        return frontal_lateral_images, frontal_lateral_labels

    def __len__(self):
        return min(len(self.frontal_image_names), len(self.lateral_image_names))
    
############################################################## Masking ##############################################################
    
class CheXpertDataSet_Masking(Dataset):
    def __init__(self, data_PATH, mask_ratio, transform = None, policy = "ones", mode='train'):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """        
        self.mode = mode
        
        if self.mode == 'train':
            df = pd.read_pickle(data_PATH)
            self.cardiomegaly_top_boxes = df['cardiomegaly_top_box']        
        else:
            df = pd.read_csv(data_PATH)
            
        self.image_names = df['path_new'].values 
#         self.images = images
        self.labels = df['Fracture'].values.reshape(-1,1).astype('float')
        self.transform = transform

        self.log_transform = log_auto
        self.mask_ratio = mask_ratio
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = cv2.imread(image_name)
        label = self.labels[index]
        
        if self.mode == 'train':              
            ###################### Masking ######################
            heart_coord = self.cardiomegaly_top_boxes[index]
            mask_flag = random.random() < self.mask_ratio
            if mask_flag:
                width = heart_coord[3] - heart_coord[1]
                x1 = heart_coord[0]+int(width*30/165)  
                cv2.rectangle(image, (x1, heart_coord[1]), (x1+int(width*130/165), heart_coord[3]), 0, -1)
        
        if self.transform is not None:
            image = self.log_transform(image)
            image = self.transform(image=image)['image']
            
        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.image_names)    
    

############################################################## Full ##############################################################
class CheXpertDataSet_Full(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones", mode='train'):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels.
        """
        self.mode = mode
        
        read_functions = {'.csv': pd.read_csv,
                         '.xlsx': pd.read_excel,
                         '.txt': pd.read_csv,
                         '.parquet': pd.read_parquet,
                         '.json': pd.read_json,
                         '.pkl': pd.read_pickle,
                         '.feather': pd.read_feather}     
        extension = os.path.splitext(data_PATH)[1]
        df = read_functions[extension](data_PATH)
        
        self.image_names = df['path_new'].values
#         if self.mode != 'test' else df['new_path'].values
#         self.images = images
        self.labels = df['Fracture'].values.reshape(-1,1).astype('float')
        self.transform = transform
        self.log_transform = log_auto
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
#         image_name = image_name.replace("CheXpert-v1.0-small", os.getcwd()+'/dataset')
        if self.mode == 'test':
            image = cv2.imread(image_name)
            image = cv2.resize(image, (320, 320))
#             image = image[:,:,0] if len(image.shape) > 2 else image              
        else:
            image = np.load(image_name)
            image = np.dstack([image, image, image])
#         image = self.images[index]        
        label = self.labels[index]
        if self.transform is not None:
            image = self.log_transform(image)
            image = self.transform(image=image)['image']
            
        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.image_names)
    
    
