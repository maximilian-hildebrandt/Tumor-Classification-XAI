import shap
from data import get_img_dataset
from torchvision import transforms
import torch
from model import BaselineClf, Transfer_res50 
import torch.optim as optimize 
from utils import *
import argparse
import numpy as np
#argparse for getting inputs from the user
parser = argparse.ArgumentParser()
parser.add_argument('-p',"--model_to_load", type = str, \
        help = "The model_name.", default= "model_weights/task2_model_128_0.001_True.pth")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = BaselineClf()
    model.load_state_dict(torch.load(args.model_to_load)["model_state_dict"])
except:
    model = Transfer_res50()
    model.load_state_dict(torch.load(args.model_to_load)["model_state_dict"])    
#the dataset 
transform = [transforms.RandomRotation(90), transforms.RandomHorizontalFlip()]
train_dataset, val_dataset, test_dataset = get_img_dataset(transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle= True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle= False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle= False)
# Create sample images to train model background and compute shap values for a few examples from both classes 
num_batches = 6
# Merge several batches to get enough images for background
for i in range(num_batches):
    batch=next(iter(test_dataloader))
    images, labels = batch
    if i == 0:
        images_combined_background=images
        labels_combined_background=labels
    else:
        images_combined_background=torch.cat((images_combined_background, images), 0)
        labels_combined_background=torch.cat((labels_combined_background, labels), 0)
images_combined_background.shape

# Sample additional images which will be split by classes for specific examples by class
for i in range(num_batches):
    batch=next(iter(test_dataloader))
    images, labels = batch
    if i == 0:
        images_combined=images
        labels_combined=labels
    else:
        images_combined=torch.cat((images_combined, images), 0)
        labels_combined=torch.cat((labels_combined, labels), 0)
images_combined.shape

# Create image vectors for background, tumor and non-tumor series
background = images_combined_background[0:100]
test_images_tumor = images_combined[labels_combined==1]
test_images_notumor= images_combined[labels_combined==0]
test_images_tumor = test_images_tumor[0:5]
test_images_notumor = test_images_notumor[0:5]
test_images_tumor.shape

# Compute SHAP values via DeepExplainer for image classification
e = shap.DeepExplainer(model, background.to(model.device))
shap_values_tumor = e.shap_values(test_images_tumor.to(model.device))
shap_values_notumor= e.shap_values(test_images_notumor.to(model.device))

# Transform values to make them plottable
shap_numpy_tumor = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values_tumor]
test_numpy_tumor = np.swapaxes(np.swapaxes(test_images_tumor.numpy(), 1, -1), 1, 2)
shap_numpy_notumor = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values_notumor]
test_numpy_notumor = np.swapaxes(np.swapaxes(test_images_notumor.numpy(), 1, -1), 1, 2)

# Plot the feature attributions for tumor examples
shap.image_plot(shap_numpy_tumor, test_numpy_tumor)

# Plot the feature attributions for non-tumor examples
shap.image_plot(shap_numpy_notumor, test_numpy_notumor)