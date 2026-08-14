
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

class carClassifier(nn.Module):
    def __init__ (self, dataDirect, transform=None):
        self.data = ImageFolder(dataDirect, transform=transform)
    def __len__(self):
            return len(self.data)
    def __getitem__(self, index):
            return self.data[index]

    def classes(self):
            return self.data.classes

class toSquare(object):
    def __call__(self, image):
        w,h = image.size 
        maxwh = max(w,h)
        pad_left = (maxwh - w) // 2  #half the size of the biggest size - the smallest creating 2pads per side
        pad_top = (maxwh - h) // 2
        pad_right = maxwh - w - pad_left
        pad_bottom = maxwh - h - pad_top    
        padding = (pad_left, pad_top, pad_right, pad_bottom) #padding obj
        return F.pad(image, padding, fill=0, padding_mode='constant')

transform = transforms.Compose([
    #make the image a padded square
    toSquare(),
    transforms.Resize((128,128)),  #keeps image at 128,128
    transforms.ToTensor() #converts to a tensor matrix of data
      
])

dataDirect = r'C:\Users\harry\gitrepos\carClassifier.py\archive\Cars_Body_Type\train'
dataset = carClassifier(dataDirect, transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #this grabs first info, the batch of info, and if its randomized
#dataloading is neccesary to train a model in batches for more optimal ouput then full data runs 


class simpleCarClassifier(nn.Module): #this class is to initalize the model and its data the size it works with the features and foward ouput
    def __init__(self, numclasses=7): #num classes is the 7 things to identify coupe sedan etc 
            super().__init__()  #inherits the modules from nn.module which is required to create the model
            enet_out_size = 512
            self.model = timm.create_model(  #set the self or current mdodel to timm's
                "efficientnet_b0",
                pretrained=True, #pretrained means the data is trained to identify features like shadows and edges
                num_classes=numclasses #numclasses is for refrence of the model to know the 7 car classes
            )

    def forward(self, x): #map out the data via math / x passes the data as a parameter 
        return self.model(x)  #this returns scores for the classes / perdicitions


def predict_image(image, model, transform, device, class_names):
    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_index = outputs.argmax(dim=1).item()
    predicted_class = class_names[predicted_index]
    return predicted_class

if __name__ == "__main__":
    model = simpleCarClassifier(numclasses=7) #this creates the model and sets the number of classes to 7
    criterion = nn.CrossEntropyLoss() #criterion is important to calculate the loss of the pass, it defines the loss function, this uses cross entropy
    optimizer = torch.optim.AdamW( #optmizer can be altered based on model but decides the strength of learning rate 
        model.parameters(), #alter all trainable weights in the model
        lr=0.0001, #learning rate is set to 0.0001 which is part of the math to adjust the slope of data on the gradient 
        weight_decay=0.0001 #weight decay stops exponential growth of weights to jump past the right answer
    )

    train_folder = '../carClassifier.py/archive/Cars_Body_Type/train'
    valid_folder = '../carClassifier.py/archive/Cars_Body_Type/valid'
    test_folder = '../carClassifier.py/archive/Cars_Body_Type/test' #setup data per folder

    train_dataset = carClassifier(train_folder, transform=transform) #transform data as verticies
    val_dataset = carClassifier(valid_folder, transform=transform)  
    test_dataset = carClassifier(test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #load the data to be trained shuffled for difference
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #these are not shuffled 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_epochs = 7 #number of times to run through the data
    train_losses, val_losses = [], [] #empty lists for losses to be stored in epoches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #grab gpu else grab cpu 
    model.to(device) #move model to gpu or cpu

    for epoch in range(num_epochs): #for each epoch
        model.train() #set model to train each start of the epoch
        runningloss = 0.0 #resets the loss of each epoch

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) #move the data to gpu or cpu
            optimizer.zero_grad() #clears gradients for each batch
            ouputs = model(images)
            loss = criterion(ouputs, labels) #calculate the loss of the batch
            loss.backward() #calculate the gradients for the batch
            optimizer.step() #update the weights of the model based on the gradients
            runningloss += loss.item() #add the loss of the batch to the total loss of the epoch
        avg_train_loss = runningloss / len(train_loader.dataset)
        train_losses.append(avg_train_loss) #append the loss to the list of losses for each epoch

        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): #evulating the data without using gradients
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) #this calls foward to access the models math and make predicitons
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * images.size(0) #add this batch to the total
    #.size(0) is a python function for returning a useable integer or float
                predicted_classes = outputs.argmax(dim=1) #the highest score of the scores given via the evulation,  outputs = model(images) which project numbers
                correct += (predicted_classes == labels).sum().item() #compares the guesses to the valid answers for each set 
                #of data if label is on 4 and the guess is 4 we can say theres a pass
                total += labels.size(0) #add to the total number of guesses for the epoch
            #the loss of the validation compared to its length gives the average value in decimal of the losses per validation batch
            #accuracy can be determined after by the correct compared to the total  
        average_validation_loss = (validation_loss / len(val_loader.dataset))
        validation_accuracy = correct / total
        #this info is outside the for loop because its the average of the total added in the for loop


    
    print("Saving model...")
    print("Current folder:", os.getcwd())

    torch.save(
        model.state_dict(),
        "car_classifier.pth"
    )

    print("Saved:", os.path.exists("car_classifier.pth"))



