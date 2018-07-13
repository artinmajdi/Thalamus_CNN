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

def TestData4(net , MultByThalamusFlag, Directory_Nuclei_Test0 , Dir_NucleiModelOut , Thalamus_OriginalSeg , Thalamus_PredSeg , Nuclei_Image , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum):

# MultByThalamusFlag = 0
Dir_NucleiModelOut_cptk = Dir_NucleiModelOut + 'model.cpkt'

# dir = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/temp/'

# net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"
# trainer = unet.Trainer(net) # , optimizer = "adam",learning_rate=0.03
# TrainData = image_util.ImageDataProvider( dir + 'Test/*.tif')
# path = trainer.train(TrainData, dir + 'Results', training_iters=2, epochs=2, display_step=500) #   restore=True
#
#
# Nuclei_Image = nib.load(dir + 'vimp2_964_08092013_TG/WMnMPRAGE_bias_corr_Deformed.nii.gz')
# Nuclei_Label = nib.load(dir + 'vimp2_964_08092013_TG/Manual_Delineation_Sanitized/1-THALAMUS_deformed.nii.gz')
#
# Nuclei_Image = Nuclei_Image.get_data()
# Nuclei_Label = Nuclei_Label.get_data()

sz_Orig = Nuclei_Image.shape
SliceNumbers = range(107,140)
CropDim = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

Nuclei_Image  = Nuclei_Image[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]
Nuclei_Label = Nuclei_Label[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]
if MultByThalamusFlag != 0:
    Thalamus_PredSeg = [CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]


sz_Crop = Nuclei_Image.shape

# *******   padding and transposing
padSizeFull = 90
padSize = int(padSizeFull/2)

Nuclei_Image = np.pad(Nuclei_Image,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
Nuclei_Label = np.pad(Nuclei_Label,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )

Nuclei_Image = np.transpose(Nuclei_Image,[2,0,1])
Nuclei_Label = np.transpose(Nuclei_Label,[2,0,1])



# *******   predict
if gpuNum != 'nan':
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_Image[...,np.newaxis], GPU_Num=gpuNum)
else:
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_Image[...,np.newaxis])


prediction = np.transpose(prediction,[1,2,0,3])
prediction = prediction[...,1]


prediction_Logical = np.zeros(sz_Crop)
if MultByThalamusFlag != 0:
    prediction_Mult = np.zeros(sz_Crop)

DiceCoefficient = np.zeros(sz_Crop[2]+1)
DiceCoefficient_Mult = np.zeros(sz_Crop[2]+1)
LogLoss = np.zeros(sz_Crop[2]+1)
LogLoss_Mult = np.zeros(sz_Crop[2]+1)
for i in range(sz_Crop[2]):
    try:
        Thresh_Mult = max(filters.threshold_otsu(prediction[...,i]),0.2)
    except:
        Thresh_Mult = 0.2

    prediction_Logical[...,i]  = prediction[...,i] > Thresh_Mult
    DiceCoefficient[i] = DiceCoefficientCalculator(prediction_Logical[...,i],Nuclei_Label[...,i])  # 20 is for zero padding done for input
    Loss = unet.error_rate(prediction_Logical[...,i],Nuclei_Label[...,i])
    LogLoss[i] = np.log10(Loss)


    if MultByThalamusFlag != 0:
        prediction_Logical_Mult[...,i] = np.multiply(prediction_Logical[...,i],Thalamus_PredSeg[...,i])
        DiceCoefficient_Mult[i] = DiceCoefficientCalculator(prediction_Logical_Mult[...,i],Nuclei_Label[...,i])  # 20 is for zero padding done for input
        Loss = unet.error_rate(prediction_Logical_Mult[...,i],Nuclei_Label[...,i])
        LogLoss_Mult[i] = np.log10(Loss)

DiceCoefficient[sz_Crop[2]] = DiceCoefficientCalculator(prediction_Logical,Nuclei_Label)
if MultByThalamusFlag != 0:
    DiceCoefficient[sz_Crop[2]] = DiceCoefficientCalculator(prediction_Logical_Mult,Nuclei_Label)

prediction_3D = np.zeros(sz_Orig)
prediction_Logical_3D = np.zeros(sz_Orig)
if MultByThalamusFlag != 0:
    prediction_Logical_3D_Mult = np.zeros(sz_Orig)

prediction_3D[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0]:CropDim[2,1] ] = prediction
prediction_Logical_3D[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0]:CropDim[2,1] ] = prediction_Logical

prediction_Logical_3D_Mult[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0]:CropDim[2,1] ] = prediction_Logical_Mult


np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient.txt',DiceCoefficient)
np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient_Mult.txt',DiceCoefficient_Mult)
np.savetxt(Directory_Test_Results_Thalamus + 'LogLoss.txt',LogLoss)
np.savetxt(Directory_Test_Results_Thalamus + 'LogLoss.txt_Mult',LogLoss_Mult)


Prediction3D_nifti = nib.Nifti1Image(prediction_3D,Affine)
Prediction3D_nifti.get_header = Header
nib.save(Prediction3D_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '.nii.gz')

Prediction3D_logical_nifti = nib.Nifti1Image(prediction_Logical_3D,Affine)
Prediction3D_logical_nifti.get_header = Header
nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '_Logical.nii.gz')

if MultByThalamusFlag != 0:

    Prediction3D_logical_nifti = nib.Nifti1Image(prediction_Logical_3D_Mult,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '_Logical.nii.gz')
