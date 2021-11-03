import sys
from xml.etree.ElementTree import ParseError
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox, QFileDialog
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtGui import QIcon
import sys 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import *

from PyQt5.QtGui import *

import sys

import ctypes

import os
from utils import *

try:
  import skimage
except ImportError:
  print("Trying to Install required module: requests\n")
  os.system('python -m conda install scikit-image')

from skimage import data


class Canvas(FigureCanvas):
    def __init__(self, parent,name, imagen = 'none' ):
        
        fig, self.ax = plt.subplots(figsize = (4,3), dpi = 100)
        super().__init__(fig)
        self.setParent(parent)
        if imagen == 'none':
            self.imagen = data.camera()
        else: 
            self.imagen = imagen
        
        #fig, ax = plt.subplots()
        self.ax.imshow(self.imagen)
        self.ax.set_title(name)
        #self.ax.grid(False)
        self.ax.axis('off')
        self.toolbar = NavigationToolbar(self, parent)
    def act_img(self, imagen, pixdim):
        self.pixdim = pixdim
        ima = np.int32(imagen*255)
        self.ima = np.zeros((ima.shape[0],ima.shape[1],3), dtype = np.int32)
        self.ima[:,:,0]  =   ima
        self.ima[:,:,1]  =   ima
        self.ima[:,:,2]  =   ima

        self.ax.imshow(self.ima, aspect=self.pixdim)
        
        self.draw()
    def act_rectangulo(self, punto, ancho, largo):
        self.ax.patches = []
        
        print(punto, ancho, largo)
        #self.ax.imshow(self.ima, aspect=self.pixdim)
        
        self.rectangle = patches.Rectangle(
            (punto[0],punto[1]),
            ancho,
            largo,
            edgecolor = 'red',
            facecolor = 'Blue',
            fill = False
        )

        self.ax.add_patch(
            self.rectangle
         )
        self.draw()
    def drop_rectangulo(self):
        self.ax.patches = []
class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi('gui_designer.ui', self)
        
        self.setWindowTitle('Prueba de integración número 4')
        #self.resize(800, 500)
        #==============================Actualiza Barra de estado==============================
        #self.barrastatus('Bienvenido')
        #================================Inicio Barra de menú=================================
        #objeto menu bar
        menu = self.menuBar()
        #menu padre
        menu_archivo = menu.addMenu('&Archivo')
        #menu padre
        menu_editar = menu.addMenu('&Editar')
        #============Opciónes Archivo============================================================
        #===============1. Abrir.nii=============================================================
        menu_archivo_abrir_nii = QAction(QIcon(), '&Abrir.nii', self)   # Se crea la acción con nombre
        menu_archivo_abrir_nii.setShortcut("Ctrl+o")                    # 
        menu_archivo_abrir_nii.setStatusTip('Abrir .nii')               #actualiza la barra de estatus

        menu_archivo_abrir_nii.triggered.connect(self.abrir_nibabel) #se vincula con el metodo menuArchivoAbrir
        menu_archivo.addAction(menu_archivo_abrir_nii)                  #Se agrega al menú padre

        #===============2. Abrir folder DICOM=====================================================
        menu_archivo_abrir_dicom = QAction(QIcon(), '&Folder DICOM', self)
        menu_archivo_abrir_dicom.setShortcut('Ctrl+d')
        menu_archivo_abrir_dicom.setStatusTip('Cargando archivos DICOM')

        menu_archivo_abrir_dicom.triggered.connect(self.abrir_dicom)
        menu_archivo.addAction(menu_archivo_abrir_dicom)
        #=======================Botones del bloque 'Cargar datos' ===============================
        self.pushNibabelabrir.clicked.connect(self.abrir_nibabel)
        self.pushDicomabrir.clicked.connect(self.abrir_dicom)
        #======================================Scroll Bars =====================================
        self.scrollbarAxial.valueChanged.connect(self.up_ima)
        self.scrollbarCoronal.valueChanged.connect(self.up_ima)
        self.scrollbarSagital.valueChanged.connect(self.up_ima)
        #=============================Botones del bloque Region de Interes'=============
        
        self.pushCrea.clicked.connect(self.crea_rdi)
        
        self.pushReset.clicked.connect(self.reset_rdi)
        self.pushAgrandar.clicked.connect(self.agranda_rdi)
        self.pushAchicar.clicked.connect(self.achica_rdi)
        self.pushSubir.clicked.connect(self.subir_rdi)
        self.pushBajar.clicked.connect(self.bajar_rdi)
        self.pushDerecha.clicked.connect(self.derecha_rdi)
        self.pushIzquierda.clicked.connect(self.izquierda_rdi)
        self.pushAceptar.clicked.connect(self.aceptar_rdi)

        
        #======================================Canvas ==========================================
        self.canvas_inicio()



        

        
        self.showMaximized()

#=======================Funciones Seleccionar RDI (Region de Interes)===========================
    def borra_rdi(self):
        self.axial.drop_rectangulo()
        self.coronal.drop_rectangulo()
        self.sagital.drop_rectangulo()
        self.groupRDI.setEnabled(False)

    def aceptar_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]
        self.volumen = self.volumen[s:s+largo,c:c+ancho,self.volumen.shape[2]-a:self.volumen.shape[2]-a+profundo]
        
        self.init_tools()
        self.borra_rdi()
        
    def achica_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]
        paso = np.int32(self.linePaso.text())

        if self.radioAxial.isChecked():
            ancho = ancho - paso
            largo = largo - paso
            #c = c - np.int32(ancho/4)
            #s = s - np.int32(ancho/4)
        elif self.radioCoronal.isChecked():
            profundo = profundo - paso
            #a = a - np.int32(ancho/4)
        elif self.radioSagital.isChecked():
            profundo = profundo - paso
            #a = a - np.int32(ancho/4)
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()

    def agranda_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]
        paso = np.int32(self.linePaso.text())

        if self.radioAxial.isChecked():
            ancho = ancho + paso
            largo = largo + paso
            #c = c - np.int32(ancho/4)
            #s = s - np.int32(ancho/4)
        elif self.radioCoronal.isChecked():
            profundo = profundo + paso
            #a = a - np.int32(ancho/4)
        elif self.radioSagital.isChecked():
            profundo = profundo + paso
            #a = a - np.int32(ancho/4)
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()

    def izquierda_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]
        paso = np.int32(self.linePaso.text())
        if self.radioAxial.isChecked():
            c = c-paso
        elif self.radioCoronal.isChecked():
            c = c-paso
        elif self.radioSagital.isChecked():
            s = s-paso
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()
        
    def derecha_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]

        paso = np.int32(self.linePaso.text())

        if self.radioAxial.isChecked():
            c = c+paso
        elif self.radioCoronal.isChecked():
            c = c+paso
        elif self.radioSagital.isChecked():
            s = s+paso
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()

    def bajar_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]

        paso = np.int32(self.linePaso.text())

        if self.radioAxial.isChecked():
            s = s+paso
        elif self.radioCoronal.isChecked():
            a = a+paso
        elif self.radioSagital.isChecked():
            a = a+paso
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()

    def subir_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]

        paso = np.int32(self.linePaso.text())

        if self.radioAxial.isChecked():
            s = s-paso
        elif self.radioCoronal.isChecked():
            a = a-paso
        elif self.radioSagital.isChecked():
            a = a-paso
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()

    def reset_rdi(self):
        a = np.int32(self.volumen.shape[2]/2)
        c = np.int32(self.volumen.shape[0]/2)
        s = np.int32(self.volumen.shape[1]/2)
        ancho = 20
        largo = ancho
        profundo = 10
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()
    def actualiza_rdi(self):
        a = self.RDIpunto[0]
        c = self.RDIpunto[1]
        s = self.RDIpunto[2]
        ancho = self.RDIpunto[3]
        largo = self.RDIpunto[4]
        profundo = self.RDIpunto[5]

        self.axial.act_rectangulo([c,s], ancho, largo)
        self.coronal.act_rectangulo([c, a-profundo], ancho, profundo)
        self.sagital.act_rectangulo([s, a-profundo], largo, profundo)
        self.labelRdipunto.setText(str(self.RDIpunto))
        # programar las funciones de los botones de flechas




    def init_bloqueRDI(self):
        self.groupRDI.setEnabled(True)

    def crea_rdi(self):
        #==============Se habilitan y se inicializa el punto del RDI=====================
        self.groupPlandes.setEnabled(True)
        self.pushReset.setEnabled(True)
        self.groupNave.setEnabled(True)
        a = np.int32(self.volumen.shape[2]/2)
        c = np.int32(self.volumen.shape[0]/2)
        s = np.int32(self.volumen.shape[1]/2)
        ancho = 20
        largo = ancho
        profundo = 10
        self.RDIpunto = [a, c, s, ancho, largo, profundo]
        self.actualiza_rdi()
    


    #=Funciones_movimiento
# ==========================Actualiza imagen ============================================
    def up_ima(self):
        #self.gridAxial.axial.act_img(self.volumen[:,:,0])

        #self.canvas_inicio()
        
        #Las dimenciones son: [coronal, sagital, axial]
        #
        valueAxial = self.scrollbarAxial.value()
        
        valueSagital = self.scrollbarSagital.value()
        valueCoronal = self.scrollbarCoronal.value()
        
        imagenAxial = self.volumen[:,:,valueAxial]
        imagenCoronal = np.rot90(self.volumen[::-1,:,:][valueCoronal,:,:], k = 1)
        imagenSagital = np.rot90(self.volumen[:,::-1,:][:,valueSagital,:], k = 1)
        
        #imagenCoronal = np.rot90(self.volumen[self.volumen.shape[0]-valueCoronal-1,:,:], k = 1)
        #imagenSagital = np.rot90(self.volumen[:,self.volumen.shape[1]-valueSagital-1,:], k = 1)
        
        self.axial.act_img(imagenAxial,pixdim = self.axial_aspect_ratio)
        self.coronal.act_img(imagenCoronal,pixdim = self.coronal_aspect_ratio)
        self.sagital.act_img(imagenSagital,pixdim = self.sagital_aspect_ratio)

        self.labelAxial.setText(str(valueAxial))
        self.labelCoronal.setText(str(self.volumen.shape[0]-valueCoronal-1))
        self.labelSagital.setText(str(self.volumen.shape[1]-valueSagital-1))
        
        #self.axial.ax.imshow(ima)
        
# ========================Funciones para abrir imagen en formato dicom o en para nibabel============= 
    def init_tools(self):
        #===========Inicializa valores maximos de scrollBars========================================
        self.scrollbarAxial.setMaximum(self.volumen.shape[2]-1)
        self.scrollbarCoronal.setMaximum(self.volumen.shape[0]-1)
        self.scrollbarSagital.setMaximum(self.volumen.shape[1]-1)
        #===========Calcula la relación de aspecto de las imagenes=================================
        self.axial_aspect_ratio = self.pixdim[1]/self.pixdim[0]
        self.sagital_aspect_ratio = self.pixdim[2]/self.pixdim[1]
        self.coronal_aspect_ratio = self.pixdim[2]/self.pixdim[0]
        #===========Despliega/actualiza información en el bloque 'Información'
        self.labelShape.setText(str(self.volumen.shape))
        self.labelPixdim.setText(str(self.pixdim))
        
        self.labelAxialratio.setText(str(self.axial_aspect_ratio))
        self.labelCoronalratio.setText(str(self.coronal_aspect_ratio))
        self.labelSagitalratio.setText(str(self.sagital_aspect_ratio))
        #self.scrollbarAxial.setmaximum(self.volumen.shape[])
        #=================Desbloquea el bloque de botones y lineas 'Seleccionar RDI'==============
        self.init_bloqueRDI()
        #==================Actualiza los Canvas===================================================
        self.up_ima()

    def abrir_dicom(self):
        filename = QFileDialog.getExistingDirectory()
        if filename == '':

            print('No_filename')
        else: 
            self.filename = filename
        
            self.labelRuta.setText(self.filename)
            #self.volumen, self.pixdim, self.mod = 
            #==========================Obtiene información básica del caso==============
            a = info_dicom(self.filename)
            self.volumen = a[0]
            self.original_volumen = a[0].copy
            self.pixdim = a[1]
            self.mod = a[2]
            #==========================Actualiza las herramientas =====================
            self.init_tools()
            

    def abrir_nibabel(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Buscar Volumen')
        if filename == '':
            print('No_filename')
        else: 

            self.filename = filename
            self.labelRuta.setText(self.filename)
            #=========================Obtiene información basica volumen y relaciónes==========
            self.volumen, self.pixdim = info_nibabel(self.filename)
            
            #print(self.volumen.shape)
            #self.volumen = self.volumen.T
            #zeros = np.zeros((self.volumen.shape[1],self.volumen.shape[2], self.volumen.shape[0]))
            #zeros[:,:,0:self.volumen.shape[0]] = self.volumen[0:self.volumen.shape[0], :,:]
            #=========================Corrije la posicion del volumen ==========================
            #print(np.rot90(self.volumen, k = 1, axes = (0,1)).shape)
            
            self.volumen = np.rot90(self.volumen, k = 1, axes = (0,1))
            self.volumen = self.volumen[:,::-1,:]
            #==========================Actualiza las herramientas =====================
            self.init_tools()
            
            
            
#==================================== Inicialización del canvas ==================================
    def canvas_inicio(self):     
        
        self.canvas3D = Canvas(self,'3D')
        self.gridCanvas3D.addWidget(self.canvas3D.toolbar)
        self.gridCanvas3D.addWidget(self.canvas3D)

        self.axial = Canvas(self,'axial')
        self.gridAxial.addWidget(self.axial.toolbar)
        self.gridAxial.addWidget(self.axial)

        self.coronal = Canvas(self,'coronal')
        self.gridCoronal.addWidget(self.coronal.toolbar)
        self.gridCoronal.addWidget(self.coronal)

        self.sagital = Canvas(self, 'sagital')
        self.gridSagital.addWidget(self.sagital.toolbar)
        self.gridSagital.addWidget(self.sagital)
        




        
        
        #self.setMaximumSize(200,200)
        #char.move(50,50)
        #print(QWheelEvent.angleDelta())
        

    #def wheelEvent(self,event):
        #self.x =self.x + QEvent.angleDelta()/120
        #print(self.x)
        #self.labelWheel.setText("Total Steps: "+ QString.number(self.x))


app = QApplication(sys.argv)
window = Window()
window.show()
app.exec_()