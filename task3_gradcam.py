#import libraries
from data import get_img_dataset
from torchvision import transforms
import torch
from model import BaselineClf, GradCam
import torch.optim as optimize 
from utils import *
import torch.nn as nn
import numpy as np
import cv2
import argparse
#parser to obtain inputs from user
parser = argparse.ArgumentParser()
parser.add_argument('-l',"--layer", type = int, \
        help = "The layer from which to visualize.", default= 12)
parser.add_argument('-p',"--prediction", type = int, \
    choices=[0, 1, -1], help="The prediction for which to check. -1 means highest prediction will be taken.", default=-1) 
args = parser.parse_args()

#Obtain the saved weights of a previous model
chkpnt = torch.load("model_weights/task2_model.pth")
model = BaselineClf()
model.load_state_dict(chkpnt["model_state_dict"])

#Obtain the dataset/ Image
transform = []#, transforms.ToTensor()]
train_dataset, val_dataset, test_dataset = get_img_dataset(transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle= False)
img, _ = next(iter(test_dataloader))

gradcam = GradCam(model, layer= args.layer) #Initialize Gradnorm
#The gradnorm algorithm
pred = gradcam(img)
img_required = img
pred_required = pred
if args.prediction==-1:
    value = torch.argmax(pred_required[0]).item()
else:
    value = args.prediction
pred_required[:, value].backward() #Gradients from the scalar
gradients = gradcam.get_activation_gradients()
pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])
activations = gradcam.get_activations(img_required).detach()
for i in range(gradients.shape[1]):
    activations[:, i, :,:]*= pooled_gradients[i]
heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = torch.relu(heatmap)
heatmap /=torch.max(heatmap) #heatmap generated

#Heatmap resized and superimposed on the image
img = img.squeeze().permute(1, 2, 0)
img = img.squeeze().cpu().numpy()*255
heatmap = heatmap.cpu().numpy()
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite(f'./map_{value}_{args.layer}.jpg', superimposed_img)