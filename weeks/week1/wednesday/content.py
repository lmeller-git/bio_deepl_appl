import os

import numpy as np

import pandas as pd



# for plotting

import matplotlib.pyplot as plt

import seaborn as sns 



# dealing with images

from PIL import Image



# Pytorch

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader, Dataset

from torchvision.utils import make_grid

import torchmetrics



# Pytorch Lightning

import lightning as L



# for visualization

import tensorboard
# check if we can use GPUs

torch.cuda.is_available()
####### TODO #########

# enter the path of the data directory

data_dir = "/path/of/datafolder" 



######################



# read the labels into a dataframe with pandas

labels_df = pd.read_csv(data_dir + "/metadata.csv", index_col=0)

print("labels_df = ", labels_df)
##### TODO ######

# get the percentage of normal and tumor classes in the dataset



normal_percentage = 

tumor_percentage = 



#################



print("Normal percentage: {}%".format(normal_percentage))

print("Tumor percentage: {}%".format(tumor_percentage))



sns.histplot(labels_df, x="class")
image_dir = data_dir + "/Brain Tumor Data Set/Brain Tumor Data Set/"

# seed everything 

torch.manual_seed(42) # set random seed to have reproducibility between the tutorials



# adding transforms to have same dimensions and some random rotations/flips to get more robust predictions

transform = transforms.Compose(

                [

                transforms.Resize((256,256)),

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomVerticalFlip(p=0.5),

                transforms.RandomRotation(30),

                transforms.ToTensor(),

                transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]) # from ImageNet

                ]  

            )



# load the complete dataset

full_dataset = torchvision.datasets.ImageFolder(image_dir, transform=transform) 



########## TODO ###############

# create the train val and test set, e.g. using a 70%, 15%, 15% split

# use a pytorch function to do this

train_set, val_set, test_set = 



# define the dataloaders for train, validation and test (use shuffle for train only)

batch_size_train = 256

batch_size = 128 # for eval and test



# usinf DataLoader from Pytorch

test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 16)

val_loader = 

train_loader = 



###############################
# plots n random brain MRI images from the passed dataset

def plot_images(dataset, n=16):

    CLA_label = {

    0 : 'Brain Tumor',

    1 : 'Healthy'

    } 

    cols, rows = 4, int(np.ceil(n/4))

    figure = plt.figure(figsize=(10, 10))

    for i in range(1, n + 1):

        sample_idx = torch.randint(len(dataset), size=(1,)).item()

        # read out image and label from item

        img, label = dataset[sample_idx]

        figure.add_subplot(rows, cols, i)

        plt.title(CLA_label[label])

        plt.axis("off")

        img_np = img.numpy().transpose((1, 2, 0))

        # Clip pixel values to [0, 1]

        img_valid_range = np.clip(img_np, 0, 1)

        plt.imshow(img_valid_range)

    plt.show()
plot_images(train_set)
# get first item from dataset, the item is just a tuple

first_item = train_set[0]



####### TODO #########

# get image at index 0 and label at index 1 from the item

image_tensor = 

label = 



# print the shape of the image tensor and the tensor 

print(image_tensor)

print("Shape: ", image_tensor.shape)

#####################
# showing tensor as image

img_valid_range = np.clip(image_tensor.numpy().transpose((1, 2, 0)), 0, 1)

plt.imshow(img_valid_range)
# get the output shape of our data after a convolution and pooling of a certain size



def get_conv2d_out_shape(tensor_shape, conv, pool=2):

    # return the new shape of the tensor after a convolution and pooling

    # tensor_shape: (channels, height, width)

    # convolution arguments

    kernel_size = conv.kernel_size

    stride=conv.stride # 2D array

    padding=conv.padding # 2D array

    dilation=conv.dilation # 2D array

    out_channels = conv.out_channels



    height_out = np.floor((tensor_shape[1]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)

    width_out = np.floor((tensor_shape[2]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    

    if pool:

        # adjust dimensions to pooling

        height_out/=pool

        width_out/=pool

        

    return int(out_channels),int(height_out),int(width_out)
# some simple tests

t1 = torch.randn(3, 256, 256)

t2 = torch.randn(5, 256, 256)

conv1 = nn.Conv2d(in_channels=3, out_channels = 256, kernel_size=10)

conv2 = nn.Conv2d(in_channels=3, out_channels = 64, kernel_size=4)

conv3 = nn.Conv2d(in_channels=5, out_channels = 13, kernel_size=7)

print(get_conv2d_out_shape(t1.shape, conv1, pool=2))

print(get_conv2d_out_shape(t1.shape, conv2, pool=1))

print(get_conv2d_out_shape(t2.shape ,conv3, pool=2))
class MRIModel(nn.Module):

    

    # Network Initialisation

    def __init__(self, params):

        

        super(MRIModel, self).__init__() #initialize parent pytorch module



        # read parameters

        shape_in = params["shape_in"]

        channels_out = params["initial_depth"] 

        fc1_size = params["fc1_size"]

        

        #### Convolution Layers



        # Max pooling layer

        self.pool = nn.MaxPool2d(2, 2)



        ##conv layer 1

        # convolution with kernel size 8, goes from three channels to 

        # number defined by initial_depth in params

        self.conv1 = nn.Conv2d(shape_in[0], channels_out, kernel_size=8)



        ############### TODO ################

        # get current shape after conv1, use helper function get_conv2d_out_shape, use pool=2

        current_data_shape = 



        ##conv layer 2

        # convolution with kernel size 4, double the amount of channels

        self.conv2 = nn.Conv2d(current_data_shape[0], current_data_shape[0]*2, kernel_size=4)

        # get current shape after conv2, use pool=2 again

        current_data_shape = 



        #### Fully connected layers

        # compute the flattened size as input for fully connected layer

        flat_size = current_data_shape[0] * current_data_shape[1] * current_data_shape[2]

        

        # linear layer reduces data from flat_size to fc1_size

        self.fc1 = 

        # last linear layer reduces data to output size 2

        self.fc2 = nn.Linear(fc1_size, 2)

        

        #####################################

        



    def forward(self,X):

        # our network's forward pass

        

        # Convolution & Pool Layers

        ############# TODO ###############

        # convolution (conv1), then relu, then max pool 

        X = F.relu(self.conv1(X))

        X = self.pool(X)

        # convolution (conv2), then relu, then max pool 

        X =



        X = torch.flatten(X, 1) # flatten all dimensions except batch



        # fully connected layer and ReLU

        X = 

        # second fully connected layer, no relu needed

        X = 



        #####################################

        # return log softmax to fit classification problem, no relu needed

        return F.log_softmax(X, dim=1)
# take first batch from the train loader

batch = next(iter(train_loader))[0]

# create the model

cnn_model = MRIModel(params={"shape_in":batch[0].shape,"initial_depth":4,"fc1_size":128})

# forward pass

out = cnn_model(batch)

# print shape of the input batch

print("Shape of the input batch: ", batch.shape)

# print the output shape

print("Shape of the output: ", out.shape)

# prediction output for first image, exp to get from log back to probabilities

print(torch.exp(out[0].detach()))
class LitMRIModel(L.LightningModule):

    def __init__(self, model_parameters, learning_rate=1e-2):

        super().__init__()

        ######## TODO ##########

        # Instantiate our model like above

        self.model = MRIModel(model_parameters)

        #pass the learning rate

        self.lr = 

        # define loss function

        self.loss_function = nn.NLLLoss(reduction="mean")

        # define accuracy metric (torchmetrics)

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)

        ########################



    def training_step(self, batch, batch_idx):

        # training_step defines the train loop.

        ######### TODO #############

        

        # read from batch

        x, y = 



        # run data through model

        predictions = 

        

        # compute loss

        loss = 

        # compute accuracy

        acc = 

        ##############################



        # logging the values (will appear in progress bar and on dashboard)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        self.log("train_acc", acc, on_epoch=True, prog_bar=True)



        return loss



    def configure_optimizers(self):

        ############## TODO ################

        # define the optimizer, let's use Adam

        optimizer = 

        ####################################

        return optimizer



    def test_step(self, batch, batch_idx):

        # this is the test loop



        ############### TODO #############

        # read from batch

        x, y = 



        # run data through model

        predictions = 

        

        # compute loss

        loss = 

        # compute accuracy

        acc = 

        ##############################



        # logging

        self.log("test_loss", loss, prog_bar=True)

        self.log("test_acc", acc, prog_bar=True)

        return loss, acc





    def validation_step(self, batch, batch_idx):

        # this is the validation loop

        ############### TODO #############

        # read from batch

        x, y = 



        # run data through model

        predictions = 

        

        # compute loss

        loss = 

        # compute accuracy

        acc = 

        ##############################



        # logging

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        return loss 

# create a tensorboard session

# new tab should open in your browser

# define parameters

model_parameters={

        "shape_in": (3,256,256), # size of our images

        "initial_depth": 4,    

        "fc1_size": 128}
# train model

########## TODO #############

# instantiate lightning model with the cnn_model and learning_rate=1e-3

model = 

############################



# instantiate the lightning trainer 

trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)

# train

trainer.fit(model, train_loader, val_loader)
# Test the model on the validation set

trainer.validate(model, val_loader)
from sklearn.metrics import classification_report, confusion_matrix



# get the predictions and plot a confusion matrix



# function to retrieve the predictions of the model and return them with the true labels

def get_predictions(val_loader, model):

    y_true = []

    y_pred = []



    for images, labels in val_loader:

        images = images#.to(device)

        labels = labels.numpy()

        outputs = model.model(images)

        _, pred = torch.max(outputs.data, 1)

        pred = pred.detach().cpu().numpy()

        

        y_true = np.append(y_true, labels)

        y_pred = np.append(y_pred, pred)

    

    return y_true, y_pred



########## TODO #############

# get predictions from the cnn_model on the val_loader

y_true, y_pred = 

############################



# print summary

print(classification_report(y_true, y_pred), '\n\n')

cm = confusion_matrix(y_true, y_pred)



sns.heatmap(cm, annot=True)
# Plot a ROC curve

from sklearn.metrics import roc_curve, auc



# get predictions (as probabilities)

def get_prediction_probs(val_loader, model):

    y_true = []

    y_pred = []



    for images, labels in val_loader:

        images = images#.to(device)

        labels = labels.numpy()

        outputs = model.model(images)

        # exp() because we use log softmax as last layer

        # get the probabilities for tomor class 

        prediction_probabilities = torch.exp(outputs)[:,1] 

        pred = prediction_probabilities.detach().cpu().numpy()

    

        y_true = np.append(y_true, labels)

        y_pred = np.append(y_pred, pred)

    

    return y_true, y_pred



y_true, y_pred_probabilities = get_prediction_probs(val_loader, model)



########## TODO #############

# compute ROC curve and ROC area for each class

# use sklearn roc_curve and auc functions

fpr, tpr, _ = 

roc_auc = au

##############################



# Plot ROC curve

plt.figure()

plt.plot(fpr, tpr, lw=2, label='AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color="grey")

plt.legend()
# train a model with a large learning rate (e.g. 1e-1)

# make sure to name your lightning model variable in a way to not overwrite the previously trained model



########## TODO #############

# instantiate lightning model

model_large_lr = 

# define trainer, 20 epochs

trainer =

# train



##############################

# train a model with a small learning rate (e.g. 1e-5)



########## TODO #############

# instantiate lightning model

model_small_lr = 

# define trainer, 20 epochs

trainer =

# train



##############################
# lightning model

model_small_batches = LitMRIModel(model_parameters, learning_rate=1e-3)



#### TODO ####

## create train dataloader with a small batch size

train_loader_small = 



# train model

trainer =

# train with smaller batch size

# lightning model

model_big_batches = LitMRIModel(model_parameters, learning_rate=1e-3)



#### TODO ####

# train with a large batch size, what's the largest batch size you can use?

train_loader_big = 



# train model

trainer = 

# train

# lightning model

model_long_training = LitMRIModel(model_parameters, learning_rate=1e-3)



# train the model for 100 epochs

trainer = 

# train on train_loader again

################# TODO ##################

# create a more complex model 

################# TODO ##################

# train 
################# TODO ##################

# validate
############### TODO #################

# pass the best performing model here

best_model =



# test your best performing model on the test set

trainer.test(best_model, test_loader)



# print the confusion matrix and classification report

y_true, y_pred = 



print(classification_report(y_true, y_pred), '\n\n')

# confusion matrix

cm = 



sns.heatmap(cm, annot=True)
