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
t.strftime("%H:%M:%S")
def fecha():
  return t.strftime("%d/%m/%y,%H:%M:%S")
def generador_nombre(datos,
                     ancho, 
                     largo, 
                     batch,
                     aumento_datos = True,
                     optim= 'Adam', 
                     fech = False,
                     n_clases = 2):
  if aumento_datos:
    a_d = 'AD'
  else:
    a_d = 'NAD'
  if fech:
    f = fecha()+'-'
  else:
    f = ''
  nombre = datos  + '-' + f + str(ancho) + 'x'+str(largo) +'-'+str(batch) + '-'+str(a_d) + '-' + str(optim)+'-'+str(n_clases)+'_clases'
  return  nombre
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
        loader = notebook.tqdm(loader,desc= 'Checking accurrancy')
        
        for x, y in loader:
            inicio =t.time()
            c = c+1
            
            x = x.to(device)
            y = y.to(device)
            
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            preds = preds.reshape_as(y)
            numpy_y = y.detach().cpu().numpy()
            numpy_preds = preds.detach().cpu().numpy()
            dice = dice_score(numpy_y, numpy_preds)
            acc = accurrancy(numpy_y,numpy_preds)
            fin = t.time()
            info.agrega(c, 0, fin-inicio, fecha(), info.optim, info.ancho, info.largo, dice, acc)
            
            
          

import os

class informe():
  def __init__(self, dir,nombre):
    print(dir,nombre)
    self.dir = dir
    self.nombre = nombre
    if nombre in os.listdir(self.dir):
      print('Cargando datos de: '+ self.nombre + ' ...')
      self.frame = pd.read_csv(str(dir+nombre))#, index_col = 'Iteración')
    else:

      dic = {'Iteración':[], 
             'Loss':[],
             'Segundos':[], 
             'Fecha':[], 
             'Optim':[],
             'Ancho':[],
             'Largo':[],
             'Dice':[],
             'Acc':[]}
      frame = pd.DataFrame(dic)
      frame.to_csv(str(self.dir+self.nombre), header=True, index = False)
      print('Creando...'+nombre)
      self.frame = pd.read_csv(str(self.dir+self.nombre))#, index_col = 'Iteración')
      print(self.frame)
  def agrega(self,it, lo, se, fe, op, an, la, di, ac):
    
    dic = {'Iteración':it, 'Loss':lo, 'Segundos':se, 'Fecha':fe, 'Optim':op,'Ancho':an,'Largo':la,'Dice':di,'Acc':ac}
    self.frame = self.frame.append(dic, ignore_index = True)
    self.frame.to_csv(str(self.dir + self.nombre), header = True, index = False)
    self.optim = op
    self.ancho = an
    self.largo = la
    self.it = it



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

