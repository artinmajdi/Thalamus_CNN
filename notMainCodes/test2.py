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


# def TestData3(net , MultThlms_Flag, Directory_Nuclei_Test0 , Dir_NucleiModelOut , Thalamus_OriginalSeg , Thalamus_PredSeg , Nuclei_ImageD , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum):
def TestData4(net , Init , Nuclei_Image, Nuclei_Label ):

    ResultFldName = 'Results_momentum'


    CropDim    = Init['CropDim']
    gpuNum     = Init['gpuNum']
    padSize    = Init['padSize']
    subFolders = Init['subFolders']
    NucleusName        = Init['NucleusName']
    SliceNumbers       = Init['Slice_Numbers']
    ResultFldName      = Init['ResultFldName']
    MultThlms_Flag     = Init['MultThlms_Flag']
    Thalamus_PredSeg   = Init['Thalamus_PredSeg']
    Dir_NucleiModelOut = Init['Dir_NucleiModelOut']


    Dir_NucleiModelOut_cptk = Dir_NucleiModelOut + 'model.cpkt'
    Directory_Test_Results_Nuclei = mkDir(Directory_Nuclei_Test0 + ResultFldName + '/')

    if MultThlms_Flag != 0:
        Directory_Test_Results_Thalamus = mkDir(Directory_Nuclei_Test0 + ResultFldName + '_MultByThlms/')

    Nuclei_ImageD = Nuclei_Image.get_data()
    Header  = Nuclei_Image.header
    Affine  = Nuclei_Image.affine
    sz_Orig = Nuclei_ImageD.shape


    # ........>>>>>>>>>>>>>>>>>> Prediction >>>>>>>>>>>>>>>>>>>>>>>>>.........

    Nuclei_ImageD  = Nuclei_ImageD[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],SliceNumbers]

    Nuclei_ImageD = np.pad(Nuclei_ImageD,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
    Nuclei_ImageD = np.transpose(Nuclei_ImageD,[2,0,1])

    if gpuNum != 'nan':
        prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_ImageD[...,np.newaxis], GPU_Num=gpuNum)
    else:
        prediction = net.predict( Dir_NucleiModelOut_cptk, Nuclei_ImageD[...,np.newaxis])

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

    if MultThlms_Flag != 0:
        prediction_3D_Mult_Logical = Thalamus_PredSeg * prediction_3D_Logical


    # >>>>>>>>>>>>>>>>>> Dice >>>>>>>>>>>>>>>>>>>>>>>>>.........

    DiceCoefficient = DiceCoefficientCalculator(prediction_3D_Logical,Nuclei_Label.get_data())

    if MultThlms_Flag != 0:
        DiceM = DiceCoefficientCalculator(prediction_3D_Mult_Logical,Nuclei_LabelD)
        DiceCoefficient = np.append(DiceCoefficient,DiceM)




    # >>>>>>>>>>>>>>>>>> saving >>>>>>>>>>>>>>>>>>>>>>>>>.........

    np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient.txt',DiceCoefficient)


    Prediction3D_nifti = nib.Nifti1Image(prediction_3D,Affine)
    Prediction3D_nifti.get_header = Header
    nib.save(Prediction3D_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Logical,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '_Logical.nii.gz')

    if MultThlms_Flag != 0:
        Prediction3D_logical_nifti = nib.Nifti1Image(prediction_3D_Mult_Logical,Affine)
        Prediction3D_logical_nifti.get_header = Header
        nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '_Logical.nii.gz')
