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


gpuNum = '4' # nan'
mode = 'oldDataset'

def testNme(A,ii):
    return (
        'Test_WMnMPRAGE_bias_corr_Deformed'
        if ii == 0
        else f'Test_WMnMPRAGE_bias_corr_Sharpness_{str(A[ii][0])}_Contrast_{str(A[ii][1])}_Deformed'
    )

def subFoldersFunc(Dir_Prior):
    subFlds = os.listdir(Dir_Prior)
    return [subFlds[i] for i in range(len(subFlds)) if subFlds[i][:5] == 'vimp2']

def initialDirectories(ind = 1, mode = 'oldDataset'):

    if ind == 1:
        NucleusName = '1-THALAMUS'
        # SliceNumbers = range(106,143)
        SliceNumbers = range(103,147)
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

    elif ind == 2:
        NucleusName = '2-AV'
        SliceNumbers = range(126,143)
    elif ind == 4:
        NucleusName = '4-VA'
        SliceNumbers = range(116,140)
    elif ind == 4567:
        NucleusName = '4567-VL'
        SliceNumbers = range(114,143)
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
    # if mode == 'oldDatasetV2':
    NeucleusFolder = (
        f'{mode}/CNN' + NucleusName.replace('-', '_') + '_2D_SanitizedNN'
    )
    ThalamusFolder = f'{mode}/CNN1_THALAMUS_2D_SanitizedNN'
    # else if mode == 'oldDataset':
    #     NeucleusFolder  = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    #     ThalamusFolder  = 'CNN1_THALAMUS_2D_SanitizedNN'

    A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
    # SliceNumbers = range(107,140)

    if mode == 'localMachine':
        Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
        Dir_Prior = f'{Dir_AllTests}Manual_Delineation_Sanitized_Full/'

    elif mode == 'oldDataset':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'


    return NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, SliceNumbers, A


ManualDir = '/Manual_Delineation_Sanitized/'

padSize = 90
MultByThalamusFlag = 0
for ind in [1]: # [1,4,6,8,10,12]


    NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, SliceNumbers, A = initialDirectories(ind , 'oldDataset')

    for ii in range(1): # len(A)):

        TestName = testNme(A,ii)

        Dir_AllTests_Nuclei_EnhancedFld = Dir_AllTests + NeucleusFolder + '/' + TestName + '/'
        Dir_AllTests_Thalamus_EnhancedFld = Dir_AllTests + ThalamusFolder + '/' + TestName + '/'

        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)

        for sFi in range(1): # len(subFolders)):
            # try:
            Dir_Prior_NucleiSample = Dir_Prior +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
            Dir_Prior_ThalamusSample = Dir_Prior +  subFolders[sFi] + ManualDir +'1-THALAMUS' + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

            Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Test/'
            Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'
            Dir_NucleiModelOut = f'{Dir_NucleiTrainSamples}model/'
            Dir_ResultsOut = f'{Dir_NucleiTestSamples}Results/'

            Dir_ThalamusTestSamples  = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Test/'
            Dir_ThalamusModelOut = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Train/model/'


            TrainData = image_util.ImageDataProvider(f"{Dir_NucleiTrainSamples}*.tif")

            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

            NucleiOrigSeg = nib.load(Dir_Prior_NucleiSample)
            ThalamusOrigSeg = nib.load(Dir_Prior_ThalamusSample)

            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3(net , MultByThalamusFlag, Dir_NucleiTestSamples , Dir_NucleiModelOut     , ThalamusOrigSeg , NucleiOrigSeg , subFolders[sFi], CropDimensions , padSize , Dir_ThalamusTestSamples , Dir_ThalamusModelOut , NucleusName , SliceNumbers , gpuNum)
