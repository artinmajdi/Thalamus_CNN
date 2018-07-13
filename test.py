from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6_1 import TestData3
from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef



# def TestData4(net , MultByThalamusFlag, Directory_Nuclei_Test0 , Dir_NucleiModelOut , Thalamus_OriginalSeg , Thalamus_PredSeg , Nuclei_Image , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum):

# Dir_NucleiModelOut_cptk = Dir_NucleiModelOut + 'model.cpkt'

dir = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/temp/'

# ----------------- temo -<<<<<<<<<<<<<<<<<<<<
MultByThalamusFlag = 1
net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"
trainer = unet.Trainer(net) # , optimizer = "adam",learning_rate=0.03
TrainData = image_util.ImageDataProvider( dir + 'Test/*.tif')
path = trainer.train(TrainData, dir + 'Results', training_iters=2, epochs=2, display_step=500) #   restore=True


Nuclei_Image = nib.load(dir + 'vimp2_964_08092013_TG/WMnMPRAGE_bias_corr_Deformed.nii.gz')
Nuclei_Label = nib.load(dir + 'vimp2_964_08092013_TG/Manual_Delineation_Sanitized/1-THALAMUS_deformed.nii.gz')
Nuclei_Image = Nuclei_Image.get_data()
Nuclei_Label = Nuclei_Label.get_data()
# ------------------>>>>>>>>>>>>>>>>>>

sz_Orig = Nuclei_Image.shape
SliceNumbers = range(107,140)
CropDim = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]+1] ])

# temp !!!!!!!!!
Thalamus_PredSeg = Nuclei_Label
gpuNum = 'nan'
Dir_NucleiModelOut_cptk = path
padSizeFull = 90
padSize = int(padSizeFull/2)

Init = {'Slice_Numbers':SliceNumbers , 'CropDim':CropDim , 'gpuNum':gpuNum , 'padSize':padSize}

SliceNumbers = Init['Slice_Numbers']
CropDim      = Init['CropDim']
gpuNum       = Init['gpuNum']
padSize      = Init['padSize']

# >>>>>>>>>>>>>>>>>> Prediction >>>>>>>>>>>>>>>>>>>>>>>>>.........
# Dir_NucleiModelOut_cptk = dir + 'model/model.cpkt'

Nuclei_Image  = Nuclei_Image[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]

Nuclei_Image = np.pad(Nuclei_Image,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
Nuclei_Image = np.transpose(Nuclei_Image,[2,0,1])
if gpuNum != 'nan':
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_Image[...,np.newaxis], GPU_Num=gpuNum)
else:
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_Image[...,np.newaxis])

prediction = np.transpose(prediction,[1,2,0,3])
prediction = prediction[...,1]

prediction_Logical = np.zeros(prediction.shape)

for i in range(len(SliceNumbers)):
    try:
        Thresh_Mult = max(filters.threshold_otsu(prediction[...,i]),0.2)
    except:
        Thresh_Mult = 0.2

    prediction_Logical[...,i]  = prediction[...,i] > Thresh_Mult

prediction_3D = np.zeros(sz_Orig)
prediction_3D_Logical = np.zeros(sz_Orig)
prediction_3D[ CropDim[0,0]:CropDim[0,1] , CropDim[1,0]:CropDim[1,1] , SliceNumbers ] = prediction
prediction_3D_Logical[ CropDim[0,0]:CropDim[0,1] , CropDim[1,0]:CropDim[1,1] , SliceNumbers ] = prediction_Logical


# >>>>>>>>>>>>>>>>>> if mult by thalamus >>>>>>>>>>>>>>>>>>>>>>>>>.........

if MultByThalamusFlag != 0:
    prediction_3D_Mult_Logical = Thalamus_PredSeg * prediction_3D_Logical


# >>>>>>>>>>>>>>>>>> Dice >>>>>>>>>>>>>>>>>>>>>>>>>.........

DiceCoefficient = DiceCoefficientCalculator(prediction_3D_Logical,Nuclei_Label)  # 20 is for zero padding done for input

if MultByThalamusFlag != 0:
    DiceCoefficient = DiceCoefficientCalculator(prediction_3D_Mult_Logical,Nuclei_Label)  # 20 is for zero padding done for input
    DiceCoefficient = np.append(DiceCoefficient,DiceCoefficient)




# >>>>>>>>>>>>>>>>>> saving >>>>>>>>>>>>>>>>>>>>>>>>>.........

np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient.txt',DiceCoefficient)


Prediction3D_nifti = nib.Nifti1Image(prediction_3D,Affine)
Prediction3D_nifti.get_header = Header
nib.save(Prediction3D_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '.nii.gz')

Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Logical,Affine)
Prediction3D_logical_nifti.get_header = Header
nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '_Logical.nii.gz')

if MultByThalamusFlag != 0:
    Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Mult_Logical,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '_Logical.nii.gz')
