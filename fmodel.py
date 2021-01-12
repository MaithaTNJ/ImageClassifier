# Functions to set up the model network. 

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
import torchvision.models as models
from torch import nn, optim
import futility

# Keras prediction model, "Height-Width-Depth" technique (https://keras.io/api/applications/)
# The training script uses two different architectures available from torchvision.models:vgg16 & densenet121
arch = {"vgg16":25088,
        "densenet121":1024}
# Seting up the network
def setup_network(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='gpu'):
# Choosing the training the model on a GPU; CUDA. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Seting up two different architectures from torchvision.models as a training script
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict    

    model.classifier = nn.Sequential(
         nn.Linear(arch['vgg16'] , hidden_units),
         nn.ReLU(),
         nn.Dropout(dropout),
         nn.Linear(hidden_units, 102),
         nn.LogSoftmax(dim=1)
     )
    print(model)
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion

# Saving the checkpoint
def save_checkpoint(train_data, model= 0, path= 'checkpoint.pth',
                    structure = 'vgg16', hidden_units = 4096,
                    dropout= 0.3, lr= 0.001, epochs= 1):
    model.class_to_idx =  train_data.class_to_idx
# Saving directory   
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)

# Loading the checkpoint
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model,_ = setup_network(structure , dropout,hidden_units,lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):   
    model.to('cuda')
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probability = torch.exp(output).data
    
    return probability.topk(topk)


def process_image(image):

#PIL image, PyTorch model
    img_pil = Image.open(image)
# As a numpy array
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    image = img_transforms(img_pil)    
    return image
