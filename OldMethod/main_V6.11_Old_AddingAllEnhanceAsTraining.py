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
import sys



def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    return subFolders

def testNme(A,ii):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    return TestName

def initialDirectories(ind = 1, mode = 'oldDatasetV2'):

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]] # [4,3],

    if ind == 1:
        NucleusName = '1-THALAMUS'
        # SliceNumbers = range(106,143)
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


    # mode:= oldDatasetV2 newDataset
    NeucleusFolder = mode + '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    ThalamusFolder = mode + '/CNN1_THALAMUS_2D_SanitizedNN'

    # if mode == 'oldDatasetV2':
    #     NeucleusFolder = 'oldDatasetV2/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    #     ThalamusFolder = 'oldDatasetV2/CNN1_THALAMUS_2D_SanitizedNN'
    # # elif mode == 'oldDataset':
    # #     NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    # #     ThalamusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    # elif mode == 'newDataset':
    #     NeucleusFolder = 'newDataset/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    #     ThalamusFolder = 'newDataset/CNN1_THALAMUS_2D_SanitizedNN'

    if mode == 'localMachine':
        Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
        Dir_Prior = ''
    elif (mode == 'oldDataset') | (mode == 'oldDatasetV2'):
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'
    elif mode == 'newDataset':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS/'

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


gpuNum, IxNuclei, testMode = input_GPU_Ix()
# gpuNum = '5' # nan'

for ind in [IxNuclei]:

    NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, SliceNumbers, A = initialDirectories(ind , 'oldDatasetV2')

    L = 1 if testMode == 'AllTrainings' else len(A)  # [1,4]: #
    for ii in range(L):

        TestName = 'Test_AllTrainings' if testMode == 'AllTrainings' else testNme(A,ii)
        # if testMode == 'AllTrainings':
        #     TestName = 'Test_AllTrainings'
        # else:
        #     TestName = testNme(A,ii)

        Dir_AllTests_Nuclei_EnhancedFld = Dir_AllTests + NeucleusFolder + '/' + TestName + '/'
        Dir_AllTests_Thalamus_EnhancedFld = Dir_AllTests + ThalamusFolder + '/' + TestName + '/'

        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)


        for sFi in range(len(subFolders)):

            K = 'Test_' if testMode == 'AllTrainings' else 'Test_WMnMPRAGE_bias_corr_'
            print(NucleusName,TestName.split(K)[1],subFolders[sFi])

            Dir_Prior_NucleiSample = Dir_Prior +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' + NucleusName + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
            Dir_Prior_ThalamusSample = Dir_Prior +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' +'1-THALAMUS' + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

            K = '/Test0/' if testMode == 'AllTrainings' else '/Test/'
            Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + K
            Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'
            Dir_NucleiModelOut = Dir_NucleiTrainSamples + 'model/'
            Dir_ResultsOut   = Dir_NucleiTestSamples  + 'Results/'

            Dir_ThalamusTestSamples  = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Test/'
            Dir_ThalamusModelOut = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Train/model/'


            Dir_NucleiModelOut = mkDir(Dir_NucleiModelOut)
            Dir_ResultsOut = mkDir(Dir_ResultsOut)

            TrainData = image_util.ImageDataProvider(Dir_NucleiTrainSamples + "*.tif")

            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # unet.config = config
            cost_kwargs = {'class_weights':[0,1]}
            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs) # , cost="dice_coefficient"

            trainer = unet.Trainer(net, optimizer = "adam") # ,learning_rate=0.03
            if gpuNum != 'nan':
                path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=200, epochs=100, display_step=500, GPU_Num=gpuNum ,prediction_path=Dir_ResultsOut) #  restore=True
            else:
                path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=200, epochs=150, display_step=500 ,prediction_path=Dir_ResultsOut) #   restore=True

            NucleiOrigSeg = nib.load(Dir_Prior_NucleiSample)
            ThalamusOrigSeg = nib.load(Dir_Prior_ThalamusSample)

            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90
            MultByThalamusFlag = 0
            [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3(net , MultByThalamusFlag, Dir_NucleiTestSamples , Dir_NucleiModelOut , ThalamusOrigSeg , NucleiOrigSeg , subFolders[sFi], CropDimensions , padSize , Dir_ThalamusTestSamples , Dir_ThalamusModelOut , NucleusName , SliceNumbers , gpuNum)
