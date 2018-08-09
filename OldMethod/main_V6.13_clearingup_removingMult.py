from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6_1 import TestData3_cleanedup
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

eps = np.finfo(float).eps

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

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

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'old'):

    Params = {}

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

    if 'local' in mode:

        Params['modelFormat'] = 'ckpt'
        if 'old' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/7T_MS'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'

    elif 'server' in mode:

        Params['modelFormat'] = 'ckpt'
        if 'old' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/newPriors/7T_MS'

        Dir_AllTests  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method' # 'oldDataset' #


    Params['A'] = [[0,0],[6,1],[1,2],[1,3],[4,1]]
    Params['Flag_cross_entropy'] = 0
    Params['NeucleusFolder'] = '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Params['ThalamusFolder'] = '/CNN1_THALAMUS_2D_SanitizedNN'
    Params['Dir_Prior']    = Dir_Prior
    Params['Dir_AllTests'] = Dir_AllTests
    Params['SliceNumbers'] = SliceNumbers
    Params['NucleusName']  = NucleusName
    Params['optimizer'] = 'adam'
    Params['CropDim'] = np.array([ [50,198] , [130,278] , [Params['SliceNumbers'][0] , Params['SliceNumbers'][len(Params['SliceNumbers'])-1]] ])

    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] = '6'  # '5'  #
    UserEntries['method'] = 'old'
    UserEntries['dataset'] = 'old'
    UserEntries['IxNuclei'] = 1
    UserEntries['testMode'] = 'EnhancedSeperately' # 'AllTrainings'

    for input in sys.argv:
        if input.split('=')[0] == 'nuclei':
            UserEntries['IxNuclei'] = int(input.split('=')[1])
        elif input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1] # 'AllTrainings'

    return UserEntries


UserEntries = input_GPU_Ix()
# gpuNum = '5' # nan'

for ind in [UserEntries['IxNuclei']]:

    Params = initialDirectories(ind = ind, mode = 'server' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']


    L = 1 if UserEntries['testMode'] == 'AllTrainings' else len(Params['A'])  # [1,4]: #
    for ii in range(L):

        TestName = 'Test_AllTrainings' if UserEntries['testMode'] == 'AllTrainings' else testNme(Params['A'],ii)

        Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + TestName + '/'

        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)

        subFolders = ['vimp2_ctrl_921_07122013_MP']
        for sFi in range(len(subFolders)):

            K = 'Test_' if UserEntries['testMode'] == 'AllTrainings' else 'Test_WMnMPRAGE_bias_corr_'
            print(Params['NucleusName'],TestName.split(K)[1],subFolders[sFi])

            K = '/Test0/' if UserEntries['testMode'] == 'AllTrainings' else '/Test/'
            Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + K
            Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'

            Dir_ThalamusTestSamples = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/' + subFolders[sFi] + '/Test/'
            # Dir_ThalamusModelOut    = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/' + subFolders[sFi] + '/Train/model/'


            Dir_NucleiModelOut = mkDir(Dir_NucleiTrainSamples + 'model_CE/')
            Dir_ResultsOut = mkDir(Dir_NucleiTestSamples  + 'Results_CE/')

            TrainData = image_util.ImageDataProvider(Dir_NucleiTrainSamples + "*.tif")
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


            if Params['Flag_cross_entropy'] == 1:
                cost_kwargs = {'class_weights':[0.7,0.3]}
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
            else:
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"


            trainer = unet.Trainer(Params['net'], optimizer = "adam")
            if Params['gpuNum'] != 'nan':
                path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=200, epochs=150, display_step=500, GPU_Num=Params['gpuNum'] ,prediction_path=Dir_ResultsOut) #  restore=True
            else:
                path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=50, epochs=4, display_step=500 ,prediction_path=Dir_ResultsOut) #   restore=True


            NucleiOrigSeg   = nib.load( Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz' )
            ThalamusOrigSeg = nib.load( Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' +        '1-THALAMUS'   + '_deformed.nii.gz' )

            info = {}
            info['Dir_NucleiTestSamples'] = Dir_NucleiTestSamples
            info['Dir_ResultsOut']     = Dir_ResultsOut
            info['MultByThalamusFlag'] = 0
            info['Dir_NucleiModelOut'] = Dir_NucleiModelOut + 'model.' + Params['modelFormat']
            info['padSize']      = 90
            info['CropDim']      = Params['CropDim']
            info['subFolders']   = subFolders[sFi]
            info['NucleusName']  = Params['NucleusName']
            info['SliceNumbers'] = Params['SliceNumbers']
            info['gpuNum']       = UserEntries['gpuNum']
            info['net']          = Params['net']

            [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3_cleanedup(info , NucleiOrigSeg)
