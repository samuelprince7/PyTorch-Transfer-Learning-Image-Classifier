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

from data_utils import data_transforms 

    
def author_model(arch, hidden_layer1, hidden_layer2, num_of_outputs):
    '''Manufactures a new pretrained model using VGG16 or (Densenet161) & returns it
    
    Inputs: 
    arch - the pytorch architecture in all lower case
    hidden_layer1 = count of elements you want in the 1st hidden layer
    hidden_layer2 = count of elements you want in the 2nd hidden layer
    
    Output:
    model - newly authored pretrained model 
    
    '''
    
    #load in the pretrained network(vgg16 or densenet161)
    # define an untrained feed-forward network as a classifier
    
    print("Building model now....")
    
    #Load model
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_of_inputs = 25088
        
    elif arch.lower() == 'densenet161':
        model = models.densenet161(pretrained=True)
        num_of_inputs = 2208  #(carry the same ratios between layers)
        
    else:
        print("Unsupported architecture. Enter vgg16 or densenet161")
        
              
    # freeze parameters so there is no backpropigation happening          
    for param in model.parameters():
        param.requires_grad = False 
    
    classifier = nn.Sequential(nn.Linear(num_of_inputs, hidden_layer1),
                          nn.ReLU(),
                          nn.Dropout(p=0.5), #0.5 is good, returns half for standardization
                          nn.Linear(hidden_layer1, hidden_layer2),
                          nn.ReLU(),
                          nn.Dropout(p=0.5), #0.5 is good
                          nn.Linear(hidden_layer2, num_of_outputs), 
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier    
    print("This model has been created...\n")
    
    return model 
          
def train_the_model(model, training_dataloaders, val_dataloaders, learn_rate, epochs, device): 
    #TODO: still need to change the variables for the training_dataloaders & val_dataloaders    
    '''
    Trains the model. Prints output loss and accuracy
    
    Inputs:
    model - the newly authored model to now train
    training_dataloaders - transformed data for training
    val_dataloaders - transformed data for validation
    learn_rate - the learning rate
    epochs - how many epochs the training will run
    
    
    Outputs:
    Prints the training and validation losses and accuracies
    Trains the model
    
    '''
         
    #define the loss     
    criterion = nn.NLLLoss()
          
    #establish learning rate      
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate) #should almost always be 0.001  
    
    # allow the model to be trained on the device (either cpu or gpu) (realistically will only work on gpu)
    model.to(device)
          
    # defining the primary variables used in our training program
    epochs = epochs #(should be 20-25 for a good model) (use 0-1 to test the saving path)
    steps = 0
    running_loss = 0
    print_every = 40      #(assusmes 20-ish epochs)
          
    # build the 'training loop'
    # this loop says, "this is how we learn from the data "
    for epoch in range(epochs):
        
    # we use the images formated into tensors from dataloaders['train_loader']
        for images, labels in training_dataloaders: # dataloaders['train_loader']:
        #we have the model take a step in it's learning journey 
            steps += 1
            # move the tensor data over to the GPU so it can process things faster
            images, labels = images.to(device), labels.to(device)
            # zero out gradients
            optimizer.zero_grad()
            #obtain the log probabilities
            logps = model(images)
            #logps = model.forward(images)
            # now w/the log probabilities we can get the loss from the criterion in the labels
            loss = criterion(logps, labels)
            # do a backwards pass, or backpropigation so it can learn from it's own mistakes
            loss.backward()
            # now it learns one step forward, so we pass that as a forward step in the learning model
            # it's kind of like we are telling it how to think and learn from it's previous mistake(s)
            #found in the backprop 
            optimizer.step()
            # finally we have to increment the running loss, so we keep track of of our total 'training loss'
            # and thus the program learns from itself continually 
            # until the end of it's cycle 
            running_loss += loss.item()
        
            #####VALIDATION LOOP 
        
            if steps % print_every ==0:
                #turn our model into evaluation inference mode which turns off dropout
                # this allows using the network to make predictions
                model.eval() # turn off drop-out
                validation_loss = 0
                accuracy = 0
                # get the images and label tensors for the testing set in dataloaders['test_loader']
                for images, labels in val_dataloaders:  # dataloaders['val_loader']:
                    #transfer tensors over to the GPU
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                    # calculate the accuracy
                    ps = torch.exp(logps) # get probabilities
                    top_ps, top_class = ps.topk(1, dim=1) # returns the 1st largest values in the probabilities 
                    # check for equality by creating an equality tensor
                    equality = top_class == labels.view(*top_class.shape)
                    #calculate accuaracy from the equality 
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
        
                    #we want to keep track of our epochs so we'll use the f string format
                print(f"Epoch {epoch+1}/{epochs}..."
                    f"Train loss: {running_loss/print_every:.3f}..."
                    f"Validation loss: {validation_loss/len(val_dataloaders):.3f}..."  #len(dataloaders['val_loader'])
                    f"Validation accuracy: {accuracy/len(val_dataloaders):.3f}")  #len(dataloaders['val_loader'])
              
                running_loss = 0
                # Turn drop-out back on      
                model.train()     
          
def sp_save_model(model, training_datasets, input_size, output_size, epochs, learning_rate, arch, hidden_layer1, hidden_layer2, save_detailed_location):      
    '''extremely important output of the model captured including
     class_to_idx, state_dict '''
    #saves the model checkpoint
    # does not include the optimizer state
    # if you want to CONTINUALLY train the model then you need the optimizer state
    
    model.class_to_idx = training_datasets.class_to_idx  # image_datasets['training'].class_to_idx
    
    detailed_checkpoint = {'input_size': input_size,
             'output_size': output_size,
             'epochs': epochs,
             'learning_rate': learning_rate,
             'arch': arch,
             'class_to_idx': model.class_to_idx,
             'hidden_layer1': hidden_layer1,
             'hidden_layer2': hidden_layer2,          
             'state_dict': model.state_dict()}
    
    # we removed this from above
    # 'optimizer_state': optimizer.state_dict(),
    # the optimizer.state_dict() is LARGE and not needed in the program
    
    
    #saves the detailed checkpoint
    torch.save(detailed_checkpoint, save_detailed_location + '/detailed_checkpoint1.pth')
          

def simple_load_rebuild_model(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet161':
        model = models.densenet161(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1']),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(checkpoint['hidden_layer1'], checkpoint['hidden_layer2']),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(checkpoint['hidden_layer2'], checkpoint['output_size']),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier 
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model 
### 999   this was used to return the entire model, but that is not 
### 999   idustry standard practice with PyTorch
###999    instead, retail the class_to_idx, state_dict, (and optimizer for continued learning) 
###999    trained_model = torch.load(path)
###999    return trained_model      
          
          
def predict1(image_path, device, rebuilt_detailed_model, topk=5):    
    #previous def predict(image_path, rebuilt_model, gpu_is_on, topk=5):
    # prevouis def predict1(image_path, rebuilt_model, gpu_is_on, topk=top_probs):   
    ''' Predict the class (or classes) of an image using a trained deep learning model.
     You need to pass it rebuilt_model, NOT rebuilt_detailed_model...
     The detailed model is only used in the dictionaries of sanity_checker in order to 
     map out the flower names accordingly to the idx
    '''
    
    pic = process_image(image_path) #creates format of what was previously called 'test_image' from ee
    pic = pic.to(device)              
    pic = pic.unsqueeze(0)
    model = rebuilt_detailed_model
    model.to(device)
    model.eval()
    model = model.forward(pic) ###999 rebuilt_detailed_model.forward(pic)
    model = torch.exp(model)
    probs, classes = model.topk(topk, largest =True, sorted=True)
    probabilities = probs.data
    classes = classes.data
    probs_list = probabilities.tolist()
    classes_list = classes.tolist() 
    probs = probs_list[0]
    classes = classes_list[0]
    return probs, classes
          

#the sanity checker takes in BOTH 
# rebuilt_model   as it's model_checkpoint AND
# rebuilt_detailed_model  as it's detailed_checkpoint 
          
          
# for below enter BOTH rebuilt_model, rebuilt_detailed_model          
def sp_sanity_checker(cat_to_name, image_path, device, rebuilt_detailed_model, top_probs):
    
    #use the model_checkpoint
    probs, classes = predict1(image_path, device, rebuilt_detailed_model, topk=top_probs)
    
    # now use the detailed_model_checkpoint 
    #map key=classes to value=probabilities 
    p_c_dict = dict()
    for idx in range(0, len(probs)):
        p_c_dict[classes[idx]] = probs[idx]
        
    #build a dictionary including the class_to_idx from the model    
    # try recreating it from rebuilt_detailed_model
    # this succeeded by doing the following rebuilt_detailed_model.class_to_idx
    dict_origin_class2idx = rebuilt_detailed_model.class_to_idx
    dict_idx2class = dict((value, key) for key,value in dict_origin_class2idx.items())
    
### I HAVE INCLUDED NOTES BELOW ON DICTIONARY OPERATIONS AND 
### I HAVE ALSO COMMENTED ON ALTERNATIVE WAYS TO ACCOMPLISH THE SAME THING
### USING DICTONARIES IN PYTHON
### THIS WILL BE ESPECIALLY HELPFUL FOR SOMEONE (LIKE SELF) MOVING FROM BEGINNER TO INDERMEDIATE LEVEL 
### IN PYTHON DICTIONARIES
### EVERYONE CROSSES THIS BRIDGE AT SOME POINT, SOME SOONER THAN OTHERS

    #now map the flower IDnumber to the probability 
    new_order_prob_class_dict = dict()
    for k, v in p_c_dict.items():
        new_order_prob_class_dict[dict_idx2class[k]] = v


    #now map the flower name to the probability 
    flower_prob_list = []
    for key, value in new_order_prob_class_dict.items():
        flower_prob_dictionary = dict()

#999 You can get rid of this second for loop as dictionaries can be accessed using keys. You could do:
# 999   flower_prob_dictionary['name'] = cat_to_name[key]
#999    flower_prob_dictionary['prob'] = value
        
        for k, v in cat_to_name.items():
            if key == k:
                flower_prob_dictionary['name'] = v
                flower_prob_dictionary['prob'] = value
        flower_prob_list.append(flower_prob_dictionary)
        
    #now create a simple pandas dataframe from flower_prob_list
    #print(flower_prob_dictionary)
    #so it can be easily mapped and plotted to matplotlib OR displayed in the command line terminal
    dataframe = pd.DataFrame(flower_prob_list)
    #sort from lowest to highest probability 
    dataframe = dataframe.sort_values(by=['prob'])
    print("the most likely image class is...")
    print(dataframe.iloc[-1]['name'])
    print("and it's associated probability is seen below, together with")
    print("the top K classes, along with their respective associated probabilities")
    print(dataframe)
    #now plot the pic of the flower with it's name
    # and the approriated probabilities for the top 5 flowers
    # depending on the terminal format, matplotlib may not display
    # but you should still see the above pandas dataframe seemlessly display the approriate data inside of the terminal
    plt.figure(1)
    plt.subplot(121)
    plt.title(str(dataframe.iloc[-1]['name']))
    imshow((process_image(image_path)), plt, str(dataframe.iloc[-1]['name']))

    plt.figure(2)
    plt.subplot(222)

    plt.barh(range(len(dataframe['name'])),dataframe['prob'])
    plt.yticks(range(len(dataframe['name'])),dataframe['name'])
    plt.show()
          
# last update   dict_origin_class2idx = rebuilt_detailed_model.class_to_idx   at 8:06pm March 9    

     