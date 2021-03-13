import torch

import argparse
import json

from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F 
from collections import OrderedDict 
from PIL import Image
import numpy as np
import pandas as pd 

from model_utils import author_model, train_the_model, sp_save_model

from data_utils import data_transforms


#generate parser
parser = argparse.ArgumentParser(description="Adds command line arguments to Train a new Neural Network using transfer learning in command line app")

#obtain file path to pictures 
parser.add_argument('data_dir', default='ImageClassifier/flowers', help="the file path of images to train the network")

# define the path where you will save the newly created neural network
# important note is that PyTorch will automatically generate the xyz.pth aspect of the file , given that
# you are pointing it to an approriate file path DIRECTORY . a directory is basically a folder

# a detailed_model_checkpoint

parser.add_argument('--save_detailed_model_to', default= 'ImageClassifier/saved_detailed_models' , help="Creates the path to save the detailed model checkpoint including the class_to_idx mapping")

# select architecture
parser.add_argument('--arch', default="vgg16", help="the arch used to train the model must be vgg16 or densenet161")

# allow changes to hyperparameters via the application
# this can be learn rate, hidden units, epochs, and batch size
parser.add_argument('--learn_rate', type=float, default=0.001, help ="enter learning rate")
parser.add_argument('--hidden_layer1', type=int, default=4096, help ="enter hidden layer 1")
parser.add_argument('--hidden_layer2', type=int, default=1024, help ="enter hidden layer 2")
parser.add_argument('--epochs', type=int, default=0, help ="enter epochs")
parser.add_argument('--batch_size', type=int, default=32, help ="enter batch size")

# define and ability to utilize the GPU. this one might be kinda tricky given needed parameters
parser.add_argument('--gpu', default=False, action='store_true', help="want to use the GPU? Default is set to False. User must type --gpu to set to True")

args = parser.parse_args()
data_dir = args.data_dir
save_detailed_location = args.save_detailed_model_to
arch = args.arch
learn_rate = args.learn_rate
hidden_layer1 = args.hidden_layer1
hidden_layer2 = args.hidden_layer2
epochs = args.epochs
batch_size = args.batch_size
use_gpu = args.gpu

 

# obtain image data from files and build your data loaders
image_datasets1, dataloaders1 = data_transforms(data_dir, batch_size)

training_dataloader1 = dataloaders1['train_loader']
val_dataloader1 = dataloaders1['val_loader']

training_datasets1 = image_datasets1['training']

print("The data loaders where successfully created from the flower images data...")

print("......")

## ask the program if the GPU is on, AND then assign the 'device' approrpriately as 'cuda' or 'cpu' 

print("Is the GPU element defined?")
print(args.gpu)
print("Will we try to use the GPU this time?")
print(use_gpu)
if use_gpu == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("the GPU is 'ON', IF the device shown below is 'cuda'")
    print("the GPU is 'OFF', IF the device shown below is 'cpu'")
else:
    device = torch.device('cpu')
    print("the GPU is off IF the device shown below is 'cpu'")

print("the device you are running is...")
print(device)

#define the number of output neurons
num_of_outputs = len(training_datasets1.class_to_idx)
print("The number of outputs is...")
print(num_of_outputs)

# author the base model as either vgg16 OR densenet161
# using author_model from sp_Artificial_Intelligence
model = author_model(arch, hidden_layer1, hidden_layer2, num_of_outputs)
print("your initial model was successfully created...")
#print(model)


# train the model and print out validation epochs
print(".......")
print("Training the model on the images from the dataloaders now....")
train_the_model(model, training_dataloader1, val_dataloader1, learn_rate, epochs, device)

print("now that the model has been trained, we will save the detailed model, as...")
print(" ..the path you entered after --save_detailed_model_to...")
print("...OR if there was an error, the default saved paths is....")
print(".......")
print("...ImageClassifier/saved_detailed_models....")

# because the variable num_of_inputs was defined inside of the xyz function
#we need to also simultaneously define it on this side of the program in trian.py
#so the equivilent variable scope can be applied below in sp_save_model
if arch.lower() == 'vgg16':    
    num_of_inputs = 25088
        
elif arch.lower() == 'densenet161':
    num_of_inputs = 2208  #(carry the same ratios between layers)
else:
    print("Unsupported architecture. Enter vgg16 or densenet161")
    
sp_save_model(model, training_datasets1, num_of_inputs, num_of_outputs, epochs, learn_rate, arch, hidden_layer1, hidden_layer2, save_detailed_location)
print(".......")
print("...the detailed model has been saved...")
# last edit num_of_outputs = len(training_datasets1.class_to_idx) 10:50pm Tue 3/9 
                    
                    