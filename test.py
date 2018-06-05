from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6 import TestData3
from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf

gpuNum = 'nan'
NeucleusFolder = 'CNN8_Pul_2D_SanitizedNN'  #  'CNN1_THALAMUS_2D_SanitizedNN' #'  CNN4567_VL_2D_SanitizedNN
NucleusName = '8-Pul'  # '1-THALAMUS' #'6-VLP' #
ManualDir = '/Manual_Delineation_Sanitized/' #ManualDelineation

SliceNumbers = range(107,140)

Directory_main = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
Directory_Nuclei_Full = Directory_main + NeucleusFolder

TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'

subFolders = os.listdir(Directory_Nuclei)
Directory_Nuclei_Test  = Directory_Nuclei + subFolders[0] + '/Test/'

priorDir = Directory_main + 'Manual_Delineation_Sanitized_Full/'

TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)

L = len(SliceNumbers)
DiceCoefficient  = np.zeros(L)
DiceCoefficient_Mult  = np.zeros(L)
LogLoss  = np.zeros(L)
LogLoss_Mult  = np.zeros(L)

SliceIdx = np.zeros(L)

for sliceNum in range(L):
    Stng = TestData.data_files[sliceNum]
    d = Stng.find('_Slice')
    SliceIdx[sliceNum] = int(Stng[d+6:].split('.')[0])

print(TestData.data_files)
print(SliceIdx)

SliceIdxArg = np.argsort(SliceIdx)
Data , Label = TestData(L)
