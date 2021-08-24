# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:07:42 2021

@author: Miguel
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from tqdm import tnrange, notebook, tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from basededatos import LiTS
import time as t

from model_unet import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    generador_nombre,
    informe,
    fecha
)

# Hyperparameters etc.
LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 
IMAGE_WIDTH =  512  # 
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_DIR = "/content/entrenamiento/vol"
TRAIN_MASK_DIR = "/content/entrenamiento/seg"

#VAL_IMG_DIR = "/content/drive/MyDrive/SOFTWARE_TT/datos/LiTS/batch2/vol"
#VAL_MASK_DIR = "/content/drive/MyDrive/SOFTWARE_TT/datos/LiTS/batch2/seg"

VAL_IMG_DIR = "/content/validacion/vol"
VAL_MASK_DIR = "/content/validacion/seg"

#__________guardar información sobre el entrenamiento___________________________

mode = 'UNET'
data = 'LiTS'
shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
batch = BATCH_SIZE
ad = 'AD' # data aumentation
opti = 'ADAM'
nclass = 2

nombre = generador_nombre(mode, data, shape, batch, ad, opti, nclass)
DIR = '/content/drive/MyDrive/SOFTWARE_TT/datos/LiTS/'

CHECK_P_FILENAME = DIR + nombre + ".pth.tar"
INFO_FILENAME =  nombre

info = informe(DIR, INFO_FILENAME, LEARNING_RATE)

#info.agrega(loss, dice, acc,lr)

def train_fn(loader, model, optimizer, loss_fn, scaler,info):
    loop = notebook.tqdm(loader, desc = '=> Entrenando', leave = False)
   
    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            
            predictions = model(data)

            loss = loss_fn(predictions.float(), targets.float())
            
            info.agrega(loss.detach().cpu().numpy(), (0,0), (0,0))
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],  # modificar para que sean mas canales de entrada
                std=[1.0],    # modificar para que sean mas canales de entrada
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    model = UNET(in_channels=3, out_channels=2).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    #if LOAD_MODEL:
    #  load_checkpoint(torch.load(CHECK_P_FILENAME), model)
    info.load_checkpoint(model)
    #check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in notebook.tqdm(range(NUM_EPOCHS), desc = ' General'):

      info.it = epoch
      train_fn(train_loader, model, optimizer, loss_fn, scaler,info)
      

      # save model
      checkpoint = {
          "state_dict": model.state_dict(),
          "optimizer":optimizer.state_dict(),
          
      }

      save_checkpoint(checkpoint,CHECK_P_FILENAME)
      
      # check accuracy
    check_accuracy(val_loader, model, info, device=DEVICE)

      # print some examples to a folder
      #save_predictions_as_imgs(
      #    val_loader, model, folder="saved_images/", device=DEVICE    
      #)
if __name__ == "__main__":
    main()