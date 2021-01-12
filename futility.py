# Functions to load and preproces the data

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable

# Keras prediction model, "Height-Width-Depth" technique (https://keras.io/api/applications/)
# The training script uses two different architectures available from torchvision.models:vgg16 & densenet121
arch = {"vgg16": 25088, "densenet121": 1024}

# Load the data
def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
   
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
# Defining the transforms.
# Using the image datasets and the trainforms, define the dataloaders
# Loading the datasets with ImageFolder

    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32

    train_transforms = transforms.Compose([transforms.Resize(max_image_size+1),
                                    transforms.CenterCrop(max_image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(expected_means, expected_std)])

    valid_transforms = transforms.Compose([transforms.Resize(max_image_size+1),
                                    transforms.CenterCrop(max_image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(expected_means, expected_std)])

    test_transforms = transforms.Compose([transforms.Resize(max_image_size+1),
                                    transforms.CenterCrop(max_image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(expected_means, expected_std)])

    train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform= valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle= True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size= batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size= batch_size)
 
    return trainloader, validloader, testloader, train_data
