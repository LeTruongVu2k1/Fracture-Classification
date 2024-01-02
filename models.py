import torch.nn as nn
import torchvision
import torch
from cbam import CBAM

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # self.densenet121 = torchvision.models.densenet121(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet121_Weights.DEFAULT
        self.densenet121 = torchvision.models.densenet121(weights=self.weights)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
        )

    def forward(self, x, mode=None):
        if mode == 'train':
            x = self.densenet121(x)
        else:
            x = self.densenet121(x)
            x = nn.Sigmoid()(x)
        return x
    
class DenseNet161(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet161
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet161, self).__init__()
        # self.densenet161 = torchvision.models.densenet161(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet161_Weights.DEFAULT
        self.densenet161 = torchvision.models.densenet161(weights=self.weights)
        num_ftrs = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
        )

    def forward(self, x, mode='inference'):
        if mode == 'train':
            x = self.densenet161(x)
        else:
            x = self.densenet161(x)
            x = nn.Sigmoid()(x)
        return x
    

########################################## Trick Version ##########################################

class DenseNet121_trick(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121_trick, self).__init__()
        # self.densenet121 = torchvision.models.densenet121(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet121_Weights.DEFAULT
        self.densenet121 = torchvision.models.densenet121(weights=self.weights)
        self.features_extraction = self.densenet121.features

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.densenet121.features.norm5.num_features
        self.linear = nn.Linear(in_features*2, out_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, frontal_lateral_images):
        num_images = frontal_lateral_images.shape[1] // 3
        vectors = []

        for i in range(num_images):
            start_dim = i * 3
            end_dim = i * 3 + 3
            image = frontal_lateral_images[:,start_dim:end_dim,:,:]

            x = self.features_extraction(image)
            x = self.avgpool(x)
            x = x.squeeze()
            vectors.append(x)

        stack_vector = torch.cat(vectors, axis=1)
        logits = self.linear(stack_vector)

        pred = self.sigmoid(logits)

        return pred
   



class DenseNet161_trick(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet161
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet161_trick, self).__init__()
        # self.densenet161 = torchvision.models.densenet161(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet161_Weights.DEFAULT
        self.densenet161 = torchvision.models.densenet161(weights=self.weights)
        self.features_extraction = self.densenet161.features

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.densenet161.features.norm5.num_features
        self.linear = nn.Linear(in_features*2, out_size) # stack of 2 images during training
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, frontal_lateral_images, mode='inference'):
        if mode != 'train':
            frontal_lateral_images = torch.cat((frontal_lateral_images, torch.zeros_like(frontal_lateral_images)), 1)
            
        num_images = frontal_lateral_images.shape[1] // 3
        vectors = []

        for i in range(num_images):
            start_dim = i * 3
            end_dim = i * 3 + 3
            image = frontal_lateral_images[:,start_dim:end_dim,:,:]
            x = self.features_extraction(image)
            x = self.avgpool(x)
            x = x.squeeze()
            vectors.append(x)

        stack_vector = torch.cat(vectors, axis=1)
              
        if mode in ['train', 'val']:
            logits = self.linear(stack_vector)
            return logits
        
        logits = self.linear(stack_vector)
        pred = self.sigmoid(logits)

        return pred
    
    

class DenseNet161_trick_v2(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet161
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet161_trick_v2, self).__init__()
        # self.densenet161 = torchvision.models.densenet161(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet161_Weights.DEFAULT
        self.densenet161 = torchvision.models.densenet161(weights=self.weights)
        self.features_extraction = self.densenet161.features

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.densenet161.features.norm5.num_features
        self.linear = nn.Linear(in_features, out_size) # stack of 2 images during training
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, frontal_lateral_images, mode='inference'):
        if mode != 'train':
#             frontal_lateral_images = torch.cat((frontal_lateral_images, frontal_lateral_images), 1)
            frontal_lateral_images = torch.cat((frontal_lateral_images, torch.zeros_like(frontal_lateral_images)), 1)
    
        num_images = frontal_lateral_images.shape[1] // 3
        vectors = []

        for i in range(num_images):
            start_dim = i * 3
            end_dim = i * 3 + 3
            image = frontal_lateral_images[:,start_dim:end_dim,:,:]
            x = self.features_extraction(image)
            x = self.avgpool(x)
            x = x.squeeze()
            vectors.append(x)

        stack_vector = torch.stack(vectors, axis=1)
        pool_views, _ = torch.max(stack_vector, 1)
        
        if mode in ['train', 'val']:
            logits = self.linear(pool_views)
            return logits
        
        logits = self.linear(pool_views)
        pred = self.sigmoid(logits)

        return pred    
    
class DenseNet161_trick_v3(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet161
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet161_trick_v3, self).__init__()
        # self.densenet161 = torchvision.models.densenet161(pretrained = False) # `pretrained=False` is deprecated
        weights = torchvision.models.DenseNet161_Weights.DEFAULT
        densenet161 = torchvision.models.densenet161(weights=weights)
        
        self.features = nn.Module()
        self.features.add_module('shared_block', nn.Sequential())
        self.list_views = ['frontal', 'lateral']
        self.features.add_module('frontal_branch', nn.Sequential())
        self.features.add_module('lateral_branch', nn.Sequential())

        for i, (name, child) in enumerate(list(densenet161.features.named_children())):
            if i < 8:
                self.features.shared_block.add_module(name, child)
            else:
                self.features.frontal_branch.add_module(name, child)
                self.features.lateral_branch.add_module(name, child)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.features.frontal_branch.norm5.num_features
        self.linear = nn.Linear(in_features, out_size) # stack of 2 images during training
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, frontal_lateral_images, mode='inference'):
        if mode != 'train':
            frontal_lateral_images = torch.cat((frontal_lateral_images, torch.zeros_like(frontal_lateral_images)), 1)
            
        vectors = []
        for i, view in enumerate(self.list_views):
            start_dim = i * 3
            end_dim = i * 3 + 3
            image = frontal_lateral_images[:,start_dim:end_dim,:,:]
            x = self.features.shared_block(image)
            
            view_specific_branch = getattr(self.features, f'{view}_branch')
            x = view_specific_branch(x)
            
            x = self.avgpool(x)
            x = x.squeeze()
            vectors.append(x)

        stack_vector = torch.stack(vectors, axis=1)
        pool_views, _ = torch.max(stack_vector, 1)
        logits = self.linear(pool_views)      
        
        if mode in ['train', 'val']:          
            return logits        
        return self.sigmoid(logits)  

    
########################################## CBAM Version ##########################################
    

class DenseNet161_CBAM(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet161
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet161_CBAM, self).__init__()
        # self.densenet161 = torchvision.models.densenet161(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet161_Weights.DEFAULT
        self.densenet161 = torchvision.models.densenet161(weights=self.weights)
        for block in self.densenet161.features.children():
            if isinstance(block, torchvision.models.densenet._DenseBlock):
                for i, dense_layer in enumerate(block.children()):
                    dense_layer.add_module(f'dense_cbam', CBAM(gate_channels=dense_layer.conv2.out_channels))
                    block.add_module(f'denselayer{i+1}', dense_layer)

            elif isinstance(block, torchvision.models.densenet._Transition):
                block.add_module(f'trans_cbam', CBAM(gate_channels=block.conv.out_channels))        
        
        num_ftrs = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
        )

    def forward(self, x, mode='inference'):
        if mode == 'train':
            x = self.densenet161(x)
        else:
            x = self.densenet161(x)
            x = nn.Sigmoid()(x)
        return x
    
    
class DenseNet121_CBAM(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121_CBAM, self).__init__()
        # self.densenet121 = torchvision.models.densenet121(pretrained = False) # `pretrained=False` is deprecated
        self.weights = torchvision.models.DenseNet121_Weights.DEFAULT
        self.densenet121 = torchvision.models.densenet121(weights=self.weights)
        for block in self.densenet121.features.children():
            if isinstance(block, torchvision.models.densenet._DenseBlock):
                for i, dense_layer in enumerate(block.children()):
                    dense_layer.add_module(f'dense_cbam', CBAM(gate_channels=dense_layer.conv2.out_channels))
                    block.add_module(f'denselayer{i+1}', dense_layer)

            elif isinstance(block, torchvision.models.densenet._Transition):
                block.add_module(f'trans_cbam', CBAM(gate_channels=block.conv.out_channels))        
        
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
        )

    def forward(self, x, mode='inference'):
        if mode == 'train':
            x = self.densenet121(x)
        else:
            x = self.densenet121(x)
            x = nn.Sigmoid()(x)
        return x