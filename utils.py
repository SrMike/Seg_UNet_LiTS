# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:03:26 2021

@author: Miguel
"""
import torch
import torchvision
from basededatos import LiTS 
from torch.utils.data import DataLoader
import time as t
import pandas as pd
from tqdm import tqdm, notebook
import numpy as np

t.strftime("%H:%M:%S")
def fecha():
  return t.strftime("%d/%m/%y,%H:%M:%S")

def generador_nombre(model, data, shape, batch, ad, optim, nclass):
  nombre = '{0}_{1}_{2}x{3}_{4}_{5}_{6}_{7}C'.format(model, data, shape[0],shape[1], batch, ad, optim, nclass)
  return nombre
           
  
def save_checkpoint(state, filename="lits.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #dice.load_state_dict(checkpoint["dice"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = LiTS(     
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        
    )

    val_ds = LiTS(   
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        
    )

    return train_loader, val_loader
def dice_score(target, prediction):
  if type(target) == torch.Tensor:
    target = target.long()
  else:
    target = np.int32(target)
  if type(prediction) == torch.Tensor:
    prediction = prediction.long()
  else:
    prediction = np.int32(prediction)
  #target y prediction tienen las mismas dimenciones [lote, canales, filas, columnas]
  lote, canales, fil, col = target.shape

  dice = np.zeros([canales,1])
  for i in range(canales):
    preds = prediction[:,i,:,:]
    y =  target[:,i,:,:]
    dice[i] = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
  return dice

def accurrancy(target, prediction):
  if type(target) == torch.Tensor:
    target = target.long()
  else:
    target = np.int32(target)
  if type(prediction) == torch.Tensor:
    prediction = prediction.long()
  else:
    prediction = np.int32(prediction)

  #target y prediction tienen las mismas dimenciones [lote, canales, filas, columnas]
  bach,clases,fil,col = target.shape
  acc = np.zeros([clases,1])
  for i in range(clases):
    n_correct = np.sum(target[:,i,:,:])
    n_pred = np.sum(prediction[:,i,:,:])+ 1e-8
    acc[i] = (n_pred/n_correct)*100
  return acc
def check_accuracy(loader, model, info, device="cuda"):
    
    num_correct_1 = 0
    num_pixels_1 = 0
    num_correct_2 = 0
    num_pixels_2 = 0
    dice_higado = 0
    dice_tumor = 0
    model.eval()
    with torch.no_grad():
        c = 0
        loader = notebook.tqdm(loader,desc= '=> Checking accurrancy')
        
        for x, y in loader:
            inicio =t.time()
            c = c+1
            
            x = x.to(device)
            y = y.to(device)
            
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            numpy_y = y.detach().cpu().numpy()
            numpy_preds = preds.detach().cpu().numpy()
            dice = dice_score(numpy_y, numpy_preds)
            acc = accurrancy(numpy_y,numpy_preds)

            info.agrega(0, dice, acc)
            
            
          

import os

import pandas as pd
import numpy as np
class informe():
  def __init__(self, dir,nombre, lr):
    self.dir = dir
    self.nombre = nombre + '.csv'
    self.id = 0
    self.lr = lr
    if self.nombre in os.listdir(self.dir):
      print('Cargando datos de: '+ self.nombre + '...')
      self.frame = pd.read_csv(str(self.dir+self.nombre))
      print('\b Listo!')
      if self.frame.shape[0] == 0:
        self.id = 0
      else:
        self.id = self.frame.iloc[-1].ID
    else:
      dic = {'ID':[], 'FECHA':[], 'LOSS':[],'LR':[],'DICE_0':[], 'DICE_1':[], 'ACC_0':[], 'ACC_1':[]}
      frame = pd.DataFrame(dic)
      frame.to_csv(str(self.dir + self.nombre), header = True, index = False)
      print('Creando: '+ self.nombre)
      self.frame = pd.read_csv(str(self.dir+self.nombre))
      print('\b Listo!')
  def agrega(self, loss, dice, acc, lr = 'same'):
    if (lr == 'same'): lr = self.lr
    self.id = self.id + 1

    dic = {'ID':self.id, 'FECHA':fecha(), 'LOSS':loss,'LR':lr,'DICE_0':dice[0], 'DICE_1':dice[1], 'ACC_0':acc[0], 'ACC_1':acc[1]}
    self.frame = self.frame.append(dic, ignore_index = True)
    self.frame.to_csv(str(self.dir + self.nombre), header = True, index = False)
 



def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    print("=> Saving predictions")
    model.eval()
    loader = tqdm(loader)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        y = y.reshape_as(preds)
        
        
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}_{idx}.png")


        
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()

