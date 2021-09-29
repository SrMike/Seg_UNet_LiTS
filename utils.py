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
def dice_score(tar, prediction):
  if len(prediction.shape) > len(tar.shape):
    clases = prediction.shape[1]
    target = gen_dim_clases(tar,clases)
    
  else:
    target = tar

  if len(target.shape) == 3:
    target.unsqueeze_(0)
    prediction.unsqueeze_(0)
  
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

  #dice = np.zeros([canales,1])
  dice = []
  for i in range(canales):
    preds = prediction[:,i,:,:]
    y =  target[:,i,:,:]
    dice.append((2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8))
  return dice

def jaccard_index(tar, prediction):
  if len(prediction.shape) > len(tar.shape):
    clases = prediction.shape[1]
    target = gen_dim_clases(tar,clases)
  else:
    target = tar

  if len(prediction.shape) == 3:
    prediction.unsqueeze_(0)

  if len(target.shape) == 3:
    target.unsqueeze_(0)
  

  if type(target) == torch.Tensor:
    target = target.long()
  else:
    target = np.int32(target)

  if type(prediction) == torch.Tensor:
    prediction = prediction.long()
  else:
    prediction = np.int32(prediction)
    
  
  lote, canales, fil, col = target.shape
  #ji = np.zeros([canales,1])
  ji = []
  for i in range(canales):
    p = prediction[:,i,:,:]
    t = target[:,i,:,:]

    inter = (t*p).sum()
    union = t.sum()+p.sum()-inter + 1e-8
    ji.append(inter/union)
  return ji
def gen_dim_clases(target,clases):
  lote, filas, columnas = target.shape
  if type(target) == torch.Tensor:
    tar = torch.zeros((lote, clases, filas, columnas), dtype = torch.long, device = target.device.type)
    
  else:
    tar = np.zeros((lote, clases, filas, columnas), dtype = np.int32)
    
  for i in range(clases):
    tar[:,i,:,:] = target == i
  return tar
#def accurrancy(target, prediction):
#  if type(target) == torch.Tensor:
#    target = target.long()
#  else:
#    target = np.int32(target)
#  if type(prediction) == torch.Tensor:
#    prediction = prediction.long()
#  else:
#    prediction = np.int32(prediction)

  #target y prediction tienen las mismas dimenciones [lote, canales, filas, columnas]
#  bach,clases,fil,col = target.shape
#  acc = np.zeros([clases,1])
#  for i in range(clases):
#    n_correct = np.sum(target[:,i,:,:])
#    n_pred = np.sum(prediction[:,i,:,:])+ 1e-8
#    acc[i] = (n_pred/n_correct)*100
#  return acc
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
            a,b,c,d = preds.shape

            #for i in notebook.tqdm(range(a),desc = '=> Lote: '):
            #  dice = dice_score(y[i,:,:,:], preds[i,:,:,:])
            #  ji = jaccard_index(y[i,:,:,:],preds[i,:,:,:])
            #  info.agrega(-1, dice, ji)
            dice = dice_score(y, preds)
            ji = jaccard_index(y,preds)
            info.agrega(-1, (dice[0].detach().cpu().numpy(),dice[1].detach().cpu().numpy()), (ji[0].detach().cpu().numpy(),ji[1].detach().cpu().numpy()))
            
          

import os

import pandas as pd
import numpy as np
from utils import *
class informe():
  def __init__(self,nombre = 'name', lr = 0, dir = 'no_dir'):
    print(os.listdir())
    self.dir = dir
    self.nombre = nombre
    self.id = 0
    self.lr = lr
    self.id_val = 0
    self.val = 0
    # La información se guardará en 2 carpetas
    # trained_models guarda los modelos entrenados (checkpoint)
    # training_data guarda la información del entrenamiento.
    
    self.trained_model_folder = 'trained_models'
    self.training_data_folder = 'training_data'

    if dir != 'no_dir':  
      os.chdir(dir)
    self.trained_model_route = self.trained_model_folder + '/'
    self.training_data_route = self.training_data_folder + '/'
    
    self.trained_model_file = self.trained_model_route + nombre + '.pth.tar'
    self.training_data_file = self.training_data_route + nombre + '.csv'

    
    # Crea los folders si no existen.
    if not(self.trained_model_folder in os.listdir()):
      print('Creando: ', self.trained_model_folder)
      os.mkdir(self.trained_model_folder)

    if not(self.training_data_folder in os.listdir()):
      print('Creando: ', self.training_data_folder)
      os.mkdir(self.training_data_folder)
      

    
    if nombre + '.csv' in os.listdir(self.training_data_folder):

      print('Cargando datos de: '+ self.training_data_file + '...')
      self.frame = pd.read_csv(self.training_data_file)
      print('\b Listo!')
      if self.frame.shape[0] == 0:
        self.id = 0
      else:
        self.id = self.frame.iloc[-1].ID
        self.id_val = self.frame['VAL'].max()
        
    else:
      dic = {'ID':[], 'FECHA':[], 'LOSS':[],'LR':[],'DICE_0':[], 'DICE_1':[], 'JI_0':[], 'JI_1':[],'VAL':[]}
      frame = pd.DataFrame(dic)
      frame.to_csv(self.training_data_file, header = True, index = False)
      print('Creando: '+ self.nombre)
      self.frame = pd.read_csv(self.training_data_file)
      print('\b Listo!')
  def agrega(self, loss, dice, acc, lr = 'same'):
    if (lr == 'same'): lr = self.lr
    self.id = self.id + 1
    if (loss == -1)and(self.frame[self.frame['ID'] == self.id-1].LOSS.values[0] != -1):
      self.id_val = self.id_val + 1
      self.val = self.id_val
    elif (loss != -1):
      self.val = 0

    dic = {'ID':self.id, 'FECHA':fecha(), 'LOSS':loss,'LR':lr,'DICE_0':dice[0], 'DICE_1':dice[1], 'JI_0':acc[0], 'JI_1':acc[1],'VAL':self.val}
    self.frame = self.frame.append(dic, ignore_index = True)
    
    if (loss == -1): 
      self.frame.to_csv(self.training_data_file, header = True, index = False)

  def checkpoint(self, model, optimizer):
    checkpoint = {
          "state_dict": model.state_dict(),
          "optimizer":optimizer.state_dict(),
          
      }
    print("=> Saving checkpoint")
    torch.save(checkpoint, self.trained_model_file)
    self.frame.to_csv(self.training_data_file, header = True, index = False)

  def load_checkpoint(self, model):
    try: 
      checkpoint = torch.load(self.trained_model_file)
      model.load_state_dict(checkpoint["state_dict"])
    except FileNotFoundError:
      print('El modelo se entrenará desde 0 ')
    
  def guarda_graficas(self):
    return 0
  def genera_graficas(self):
    
    return 0

 



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

