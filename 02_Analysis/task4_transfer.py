#import libraries
from data import get_img_dataset
from torchvision import transforms
import torch
from model import Transfer_res50
import torch.optim as optimize 
from utils import *
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
#argparse for getting inputs from the user
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs",type = int,  \
    help="The number of epochs it would run.", default=100)
parser.add_argument("-p","--patience",type = int,  \
    help="Patience before it terminates.", default=4)
parser.add_argument('-l',"--learningrate", type = float, \
        help = "The initial learning rate of the model.", default= 1e-3)
parser.add_argument('-g','--use_grid_search', action='store_true', \
        help="Whether to use grid search or not.")
parser.add_argument('-w','--use_weights', action='store_true',\
        help="Whether to consider the unequal distribution of classes.")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.use_weights:
    model_weights = torch.Tensor([3.0, 2.0])
else:
    model_weights = torch.Tensor([1.0, 1.0])

model_weights = model_weights.to(device)
def param_train(batch_size = 32, learning_rate = 1e-3, val_score = 1e10):
        #the dataset 
        transform = [transforms.RandomRotation(90), transforms.RandomHorizontalFlip()]
        train_dataset, val_dataset, test_dataset = get_img_dataset(transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle= True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle= False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle= False)

        #the model training pipeline
        model = Transfer_res50(model_weights).to(device)
        optimizer = optimize.Adam(model.parameters(), lr=args.learningrate)
        optimizer = optimize.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        train(model, train_dataloader, val_dataloader, optimizer, \
                scheduler, model_name = f"model_weights/task4_model_{batch_size}_{learning_rate}_{args.use_weights}.pth", epochs= args.epochs, patience=args.patience)
        return val_epoch(model, val_dataloader, epoch=-1)
grid_search_params = {'batch_size':[32, 64, 128], 'lr':[0.0001, 0.001, 0.01, 0.1, 1]}

if args.use_grid_search:
        best_batch_size = -1
        best_learning_rate = -1
        min_val_score = 1e10
        for batch_size in grid_search_params['batch_size']:
                for lr in grid_search_params['lr']:
                        val_score = param_train(batch_size, lr)
                        if min_val_score>val_score:
                                best_batch_size = batch_size
                                best_learning_rate = lr
        print(f"The best batch_size is {best_batch_size} and learning rate is {best_learning_rate}.")

else:
        param_train(128, args.learningrate)