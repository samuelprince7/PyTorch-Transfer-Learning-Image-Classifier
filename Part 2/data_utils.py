import torch
import torchvision
from torchvision import datasets, transforms, models

def data_transforms(data_dir, batch_sz):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training_transform_set': transforms.Compose([transforms.RandomRotation(30),
                                                      transforms.RandomResizedCrop(224),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])]),
    
        'validation_transform_set': transforms.Compose([transforms.Resize(255),
                                                       transforms.CenterCrop(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])]),
    
        'testing_transform_set': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])    }
    # Load in your datasets with ImageFolder
    #use datasets imported from torchvision 
    # dataset = datasets.ImageFolder('path/to/data', transform = transforms)
    image_datasets = {'training': datasets.ImageFolder(train_dir, transform=data_transforms['training_transform_set']),
                      'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transform_set']),
                      'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing_transform_set'])}
                 
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #once you have the dataset (with the file path and approriately transformed data)
    # we have to pass it into a dataloader 
    # we define a batch size, or the number of images you go through each 'batch' in this data loader
    #  dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # shuffle randomly shuffles the training data every time you start a new ephoch, 2nd time different order, 3rd time,etc
    # this dataloader loads up a GENERATOR . very important concept 
    dataloaders = {'train_loader':torch.utils.data.DataLoader(image_datasets['training'],batch_size=batch_sz,shuffle=True),
                  'val_loader':torch.utils.data.DataLoader(image_datasets['validation'],batch_size=batch_sz),
                   'test_loader':torch.utils.data.DataLoader(image_datasets['testing'],batch_size=batch_sz)}        
               
    return image_datasets, dataloaders
    
    #last mark 7:42pm