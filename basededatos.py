# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:05:58 2021

@author: Miguel
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm, notebook
import torch
class LiTS(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir # directorio de los volumenes en formato string
        self.mask_dir = mask_dir   # directorio de las mascaras en formato string
        self.transform = transform
        
        self.images = self.ordena_lista(os.listdir(image_dir)) # lista con los nombres de cada uno
                                            # de los archivos #volumen-xx.nii


        self.mask = self.ordena_lista(os.listdir(mask_dir))    # lista con ['mascaras-00.nii',...]
        
        
        
        self.list_images = [] # son los volumenes
        self.list_mask = []   # son los segmentos
        self.tamaños = []     # Lista que contiene el número de frames por 
                              # archivo

        for i in range(len(self.images)-1): 
          #--- se cargan los archivos " .nii" utilizando nib.load()
          imag = nib.load(self.image_dir + '/' + self.images[i])
          mask = nib.load(self.mask_dir + '/' + self.mask[i])
          #_se guardan en las listas
          self.list_images.append(imag)
          self.list_mask.append(mask)
          #_se consulta el tamaño y se guarda en la lista tamaños
          self.tamaños.append(imag.shape[2])
        #self.total = sum(self.tamaños)
        self.index_array = np.zeros([sum(self.tamaños),2])
        self.cont = 0

        for i in notebook.tqdm(range(len(self.tamaños)), desc= '=> Cargando base de datos', leave = False):
          for j in notebook.tqdm(range(self.tamaños[i]), desc = self.images[i], leave = False):
            if self.list_mask[i].slicer[:,:,j:j+1].get_fdata().sum() != 0:
              self.index_array[self.cont,0] = i
              self.index_array[self.cont,1] = j
              self.cont = self.cont + 1

    def __len__(self):
        return self.cont

    def __getitem__(self, index):
      nlist,idx = np.int16(self.index_array[index])
      image = self.list_images[nlist].slicer[:,:,idx:idx+1].get_fdata()
      mask = self.list_mask[nlist].slicer[:,:,idx:idx+1].get_fdata()
      #image = image + np.abs(image.min())
      #image = (image/image.max())*255
      #image = np.uint8(image)
      #image = image[:,:,0]
      mat = np.zeros((mask.shape[0], mask.shape[1],2))
      mat[:,:,0] = mask[:,:,0] == 1
      mat[:,:,1] = mask[:,:,0] == 2
      mask = mat
        #img_path = os.path.join(self.image_dir, self.images[index])
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        #image = np.array(Image.open(img_path).convert("RGB"))
        #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask == 255.0] = 1.0

      if self.transform is not None:
          
          augmentations = self.transform(image=image, mask=mask)
          image = augmentations["image"]
          mask = augmentations["mask"]
          mat = torch.zeros((2,mask.shape[0], mask.shape[1]))
          mat[0,:,:] = mask[:,:,0]
          mat[1,:,:] = mask[:,:,1]
          mask = mat
         
      return image, mask
    def obtener_numero(self,nombre):
      a = nombre.find('-')
      if nombre[a+2]=='.':
        return int(nombre[a+1])
      elif nombre[a+3] == '.':
        return int(nombre[a+1]+nombre[a+2])
      else:
        return int(nombre[a+1]+nombre[a+2]+nombre[a+3])

    def ordena_lista(self,lista):
      # Esta función ordena de menor a mayor los nombres en formato string 
      # contenidos en la lista de entrada
      dic = {}
      for i in lista:
        n = self.obtener_numero(i)
        dic[n] = i
      lista = []
      for i in range(min(dic),max(dic)):
        lista.append(dic[i])
      return lista

