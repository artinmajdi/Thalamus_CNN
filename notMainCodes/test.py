from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6_2 import TestData4
from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf


# tf.train.import_meta_graph(
#     meta_graph_or_file,
#     clear_devices=True,
#     import_scope=None,
# )

gpuNum = '4' # nan'
mode = 'oldDatasetV2'

def testNme(A,ii):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    return TestName

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    return subFolders

def initialDirectories(ind = 1, mode = 'oldDatasetV2'):

    if ind == 1:
        NucleusName = '1-THALAMUS'
        SliceNumbers = range(103,147)
        # SliceNumbers = range(107,140) # original one
    elif ind == 2:
        NucleusName = '2-AV'
        SliceNumbers = range(126,143)
    elif ind == 4567:
        NucleusName = '4567-VL'
        SliceNumbers = range(114,143)
    elif ind == 4:
        NucleusName = '4-VA'
        SliceNumbers = range(116,140)
    elif ind == 5:
        NucleusName = '5-VLa'
        SliceNumbers = range(115,133)
    elif ind == 6:
        NucleusName = '6-VLP'
        SliceNumbers = range(115,145)
    elif ind == 7:
        NucleusName = '7-VPL'
        SliceNumbers = range(114,141)
    elif ind == 8:
        NucleusName = '8-Pul'
        SliceNumbers = range(112,141)
    elif ind == 9:
        NucleusName = '9-LGN'
        SliceNumbers = range(105,119)
    elif ind == 10:
        NucleusName = '10-MGN'
        SliceNumbers = range(107,121)
    elif ind == 11:
        NucleusName = '11-CM'
        SliceNumbers = range(115,131)
    elif ind == 12:
        NucleusName = '12-MD-Pf'
        SliceNumbers = range(115,140)
    elif ind == 13:
        NucleusName = '13-Hb'
        SliceNumbers = range(116,129)

    NeucleusFolder  = mode + '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    ThalamusFolder  = mode + '/CNN1_THALAMUS_2D_SanitizedNN'

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]]
    # SliceNumbers = range(107,140)

    if mode == 'localMachine':
        Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
        Dir_Prior = Dir_AllTests + 'Manual_Delineation_Sanitized_Full/'

    elif mode == 'oldDatasetV2':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'


    return NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, SliceNumbers, A

def input_GPU_Ix():

    gpuNum = '5'  # 'nan'
    IxNuclei = 1
    testMode = 'EnhancedSeperately' # 'AllTrainings'

    for input in sys.argv:
        if input.split('=')[0] == 'nuclei':
            IxNuclei = int(input.split('=')[1])
        elif input.split('=')[0] == 'gpu':
            gpuNum = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            testMode = input.split('=')[1] # 'AllTrainings'

    return gpuNum, IxNuclei, testMode


Init = {'init':1}
ind = 1
mode = 'oldDataset'
NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, SliceNumbers, A = initialDirectories(ind , mode)


A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

Init['SliceNumbers'] = SliceNumbers
Init['Dice_Flag'] = 1
Init['MultThlms_Flag'] = 0
Init['optimizer'] = "adam" # "momentum" #
Init['CropDim'] = np.array([ [50,198] , [130,278] , [ Init['SliceNumbers'][0] , Init['SliceNumbers'][ len(Init['SliceNumbers'])-1 ] ] ])
Init['gpuNum'] = 'nan'
Init['padSize'] = int(90/2)
Init['NucleusName'] = NucleusName


Dir = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2/vimp2_964_08092013_TG/WMnMPRAGE_bias_corr_Deformed.nii.gz'
Nuclei_Image = nib.load(Dir)
Init['subFolders'] = 'vimp2_964_08092013_TG'


TestData4(net , Init , Nuclei_Image)
