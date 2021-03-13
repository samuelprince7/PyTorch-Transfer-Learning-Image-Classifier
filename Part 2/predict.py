import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F 
from collections import OrderedDict 
from PIL import Image
import numpy as np
import pandas as pd 

from image_utils import process_image, imshow 

import json
from Load_Json_File import load_json_file

import argparse 

from data_utils import data_transforms 

from model_utils import simple_load_rebuild_model, predict1, sp_sanity_checker  

#generate a parser to take in specifications from the command line application
parser = argparse.ArgumentParser(description ="Load a pretrained network to test how it infers a single image")

#below are the 'arguments' for the user to imput via the command line
#description of the argument is listed in the help section
# this command line application was designed with simplicity in mind
# leveraging the model_utils, image_utils, and data_utils , programs
# to test newly trained neural networks

parser.add_argument('data_dir', help="path to the image, required")
parser.add_argument('detailed_checkpoint_pth', help="path to trained neural network DETAILED MODEL checkpoint as pth file, must contain class_to_idx data.")
parser.add_argument('--top_probs', default=5, type=int, help="number of most probable classes you wish to preview. this shows a number of flowers the pretrained model has identified as probable flower names")

parser.add_argument('--spec_labels_pth', default = 'ImageClassifier/cat_to_name.json',help="json file path to load flower names")
parser.add_argument('--gpu', default=False, action='store_true', help="user must type --gpu to use the GPU")

# enable input from the command line

args = parser.parse_args()
data_dir = args.data_dir
detailed_checkpoint = args.detailed_checkpoint_pth
top_probs = args.top_probs
names = args.spec_labels_pth
utilize_gpu = args.gpu

## ask the program if the GPU is on, AND then assign the 'device' approrpriately as 'cuda' or 'cpu' 
print("......")

print("Is the GPU element defined?")
print(args.gpu)
print("Will we try to use the GPU this time?")
print(utilize_gpu)
if utilize_gpu == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("the GPU is 'ON', IF the device shown below is 'cuda'")
    print("the GPU is 'OFF', IF the device shown below is 'cpu'")
else:
    device = torch.device('cpu')
    print("the GPU is off IF the device shown below is 'cpu'")

print("the device you are running is...")
print(device)
                       

# rebuild the model

rebuilt_detailed_model = simple_load_rebuild_model(detailed_checkpoint)

#get json data that holds the names of flowers to an idx mapping 
flower_names = load_json_file(names)
#run predict with the rebuilt model and this should
# output the top_k results of the rebuilt model in the terminal
# it is important to note that Pandas Dataframes will seemlessly display output in the command line prompt
# however Matplotlib will not
probs, classes = predict1(data_dir, device, rebuilt_detailed_model, topk=top_probs)
print("the probabilities are...")
print(probs)
print("the classes are...")
print(classes)

#flower_names should go in sanity checker along with the data_dir, device, rebuilt_detailed_model, and top_probs 


sp_sanity_checker(flower_names, data_dir, device, rebuilt_detailed_model, top_probs)
#789
#1:31am Feb 26, removed utilize_gpu from predict1, remains in predict
