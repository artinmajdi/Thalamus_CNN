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
import matplotlib.pyplot as plt
def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def testNme(A,ii):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'
        return TestName

def initialDirectories(ind = 1, mode = 'oldDataset'):

    # 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
    # 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
    # 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz

    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 2:
        NucleusName = '2-AV'
    elif ind == 4567:
        NucleusName = '4567-VL'
    elif ind == 4:
        NucleusName = '4-VA'
    elif ind == 5:
        NucleusName = '5-VLa'
    elif ind == 6:
        NucleusName = '6-VLP'
    elif ind == 7:
        NucleusName = '7-VPL'
    elif ind == 8:
        NucleusName = '8-Pul'
    elif ind == 9:
        NucleusName = '9-LGN'
    elif ind == 10:
        NucleusName = '10-MGN'
    elif ind == 11:
        NucleusName = '11-CM'
    elif ind == 12:
        NucleusName = '12-MD-Pf'
    elif ind == 13:
        NucleusName = '13-Hb'


    if mode == 'oldDatasetV2':
        NeucleusFolder = 'oldDatasetV2/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'oldDatasetV2/CNN1_THALAMUS_2D_SanitizedNN'
    elif mode == 'oldDataset':
        NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    elif mode == 'newDataset':
        NeucleusFolder = 'newDataset/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'newDataset/CNN1_THALAMUS_2D_SanitizedNN'


    if mode == 'localMachine':
        Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
        Dir_Prior = ''
    elif (mode == 'oldDataset') | (mode == 'oldDatasetV2'):
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'
    elif mode == 'newDataset':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS/'

    return NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior


Init = {'init':1}
ind = 1
mode = 'oldDatasetV2'
NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior = initialDirectories(ind , mode)

# dir = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/Manual_Delineation_Sanitized_Full/vimp2_ctrl_920_07122013_SW/Manual_Delineation_Sanitized/'
# msk = nib.load(dir + NucleusName + '_deformed.nii.gz')
#
# msk = msk.get_data()
# msk.shape
# a = msk.sum(axis=0)
# a = a.sum(axis=0)
# a.shape
# np.where(a>0)

A = [[0,0],[6,1],[1,2],[1,3],[4,1]] # [4,3],

Init['SliceNumbers'] = range(107,140)
Init['Dice_Flag'] = 1
Init['MultThlms_Flag'] = 0
Init['optimizer'] = "momentum" # "adam"
Init['CropDim'] = np.array([ [50,198] , [130,278] , [ Init['SliceNumbers'][0] , Init['SliceNumbers'][ len(Init['SliceNumbers'])-1 ] ] ])
# Init['gpuNum'] = '2'
Init['padSize'] = int(90/2)
Init['NucleusName'] = NucleusName

dir = '/media/groot/Seagate Backup Plus Drive/code/mine/test/'
Nuclei_Image = nib.load(dir + 'WMnMPRAGE_bias_corr_Deformed.nii.gz')
Init['Dir_NucleiModelOut'] = dir + 'model_Backup/'
Init['Nuclei_Label'] = nib.load(dir + '1-THALAMUS_deformed.nii.gz')
Init['Dir_NucleiTestSamples'] = dir
Init['gpuNum'] = 'nan'
net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

# -------------------------------------------------------------------------------------------------------
# def TestData4(net , Init , Nuclei_Image):

ResultFldName = 'Results_momentum'


CropDim      = Init['CropDim']
padSize      = Init['padSize']
SliceNumbers = Init['SliceNumbers']

Dir_NucleiModelOut_cptk = Init['Dir_NucleiModelOut'] + 'model.cpkt'

dir_ResultOut = mkDir(Init['Dir_NucleiTestSamples'] + 'Results_' + Init['optimizer'] + '/')

if Init['MultThlms_Flag'] != 0:
    dir_ResultOut_Mlt = mkDir(Init['Dir_NucleiTestSamples'] + 'Results_' + Init['optimizer'] + '_MultByThlms/')

Nuclei_ImageD = Nuclei_Image.get_data()
Header  = Nuclei_Image.header
Affine  = Nuclei_Image.affine
sz_Orig = Nuclei_ImageD.shape




# ........>>>>>>>>>>>>>>>>>> Prediction >>>>>>>>>>>>>>>>>>>>>>>>>.........

Nuclei_ImageD  = Nuclei_ImageD[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]

Nuclei_ImageD = np.pad(Nuclei_ImageD,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
Nuclei_ImageD = np.transpose(Nuclei_ImageD,[2,0,1])

if Init['gpuNum'] != 'nan':
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_ImageD[...,np.newaxis], GPU_Num=Init['gpuNum'])
else:
    prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_ImageD[...,np.newaxis])
b = Nuclei_ImageD[...,np.newaxis]
a = Nuclei_ImageD[10,:,:,np.newaxis]
a = a[np.newaxis,...]
a.shape
prediction = net.predict( Dir_NucleiModelOut_cptk, b)
prediction.shape


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

if Init['MultThlms_Flag'] != 0:
    Thalamus_PredSegD = Init['Thalamus_PredSeg'].get_data()
    prediction_3D_Mult_Logical = Thalamus_PredSegD * prediction_3D_Logical





# >>>>>>>>>>>>>>>>>> Dice >>>>>>>>>>>>>>>>>>>>>>>>>.........

Dice = [0,0]
if Init['Dice_Flag'] == 1:
    Dice[0] = DiceCoefficientCalculator(prediction_3D_Logical, Init['Nuclei_Label'].get_data())

    if Init['MultThlms_Flag'] != 0:
        Dice[1] = DiceCoefficientCalculator(prediction_3D_Mult_Logical,Nuclei_LabelD)
        # DiceCoefficient = np.append(DiceCoefficient,DiceM)

    print(Dice)
    np.savetxt(dir_ResultOut + 'DiceCoefficient.txt',Dice )

    dir_ResultOut




    # >>>>>>>>>>>>>>>>>> saving >>>>>>>>>>>>>>>>>>>>>>>>>.........

    Prediction3D_nifti = nib.Nifti1Image(prediction_3D,Affine)
    Prediction3D_nifti.get_header = Header
    nib.save(Prediction3D_nifti , dir_ResultOut + Init['subFolders'] + '_' + Init['NucleusName'] + '.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Logical,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti , dir_ResultOut + Init['subFolders'] + '_' + Init['NucleusName'] + '_Logical.nii.gz')

    if Init['MultThlms_Flag'] != 0:
        Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Mult_Logical,Affine)
        Prediction3D_logical_nifti.get_header = Header
        nib.save(Prediction3D_logical_nifti , dir_ResultOut_Mlt + Init['subFolders'] + '_' + Init['NucleusName'] + '_Logical.nii.gz')
