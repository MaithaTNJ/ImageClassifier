"""
AIPND project for Udacity.
"""
# predict.py: Use the trained network to predict the class for the input image.

import argparse
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
import torch
from torch import nn, optim

# Files has functions that created to run the classifier 
import futility
import fmodel

# Interfaces
parser = argparse.ArgumentParser(description = 'predictor Parser')

#Set the ehyperparameters.
parser.add_argument('input', default= './flowers/test/1/image_06752.jpg', nargs= '?', action= "store", type = str)
parser.add_argument('--dir', action= "store",dest= "data_dir", default= "./flowers/")
parser.add_argument('checkpoint', default= './checkpoint.pth', nargs= '?', action= "store", type = str)
parser.add_argument('--top_k', default= 5, dest= "top_k", action= "store", type= int)
parser.add_argument('--category_names', dest= "category_names", action= "store", default= 'cat_to_name.json')
parser.add_argument('--gpu', default= "gpu", action="store", dest="gpu")

# Assign the converted objects into attributes 
args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu

path = args.checkpoint

def main():
    model=fmodel.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
# call predect function to predict the labels of the data values on the basis of the trained model 
    probabilities = fmodel.predict(path_image, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability_round = np.array(probabilities[0][0]*100)
    probability = np.around(probability_round,2)
    i= 0 
    while i < number_of_outputs:
        print("{} probability:    {}%".format(labels[i], np.around(probability[i])))
        i += 1
        
    print("\t . . .Predicting Process DONE. . . ")

    
if __name__== "__main__":
    main()
    