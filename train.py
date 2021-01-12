
"""
AIPND project for Udacity.
"""
# train.py: Train a network then save the model as a checkpoint.

import argparse
import torch
import torchvision
import torchvision.transforms as transforms, torchvision.datasets as datasets, torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models

# Files has functions that created to run the classifier 
import futility
import fmodel


# Keras prediction model, "Height-Width-Depth" technique (https://keras.io/api/applications/)
# The training script uses two different architectures available from torchvision.models:vgg16 & densenet121
arch = {"vgg16":25088, "densenet121":1024} 

# Interfaces
parser = argparse.ArgumentParser(description = 'trainor Parser')

# Set the ehyperparameters for learning rate, number of hidden units, and training epochs.
parser.add_argument('--data_dir', action= "store", default= "./flowers/")
parser.add_argument('--save_dir', action= "store", default= "./checkpoint.pth")
parser.add_argument('--arch', action= "store", default= "vgg16")
parser.add_argument('--learning_rate', action= "store", type= float, default= 0.01)
parser.add_argument('--hidden_units', action= "store", dest= "hidden_units", type= int, default= 512)
parser.add_argument('--epochs', action= "store", default= 3, type= int)
parser.add_argument('--dropout', action= "store", type= float, default= 0.5)
parser.add_argument('--gpu', action= "store", default= "gpu")

# Assign the converted objects into attributes 
args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

# https://pytorch.org/docs/stable/notes/cuda.html
# Choosing the training the model on a GPU; CUDA. 
if power == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

    
def main():
    trainloader, validloader, testloader, train_data = futility.load_data(where)
    model, criterion = fmodel.setup_network(struct, dropout, hidden_units, lr, power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
# Training classifier processes
    steps = 0
    running_loss = 0
    print_every = 5
    print(f"Classifier training started: ")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1 #count the epoch steps
            if torch.cuda.is_available() and power =='gpu':
                inputs = inputs.to('cuda') #Move input to the default device
                labels = labels.to('cuda') #Label tensors to the default device
            optimizer.zero_grad()

# Passong Forward and Backward
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs = inputs.to('cuda')
                        labels = labels.to('cuda')
                        
# Measuring the validation loss
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

# Measuring the validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
# Printing the training loss, validation loss, and validation accuracy as a network trains
                print(f" Epoch {epoch+1} out of {epochs} "
                      f" Loss: {running_loss/print_every:.3f} "
                      f" Validation Loss: {valid_loss/len(validloader):.3f} "
                      f" Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
# Saving the classifier model to the checkpoint
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure': struct,
                'hidden_units': hidden_units,
                'dropout': dropout,
                'learning_rate': lr,
                'no_of_epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, path)
    
    print("Checkpoint saved.")
    
if __name__== "__main__":
    main()