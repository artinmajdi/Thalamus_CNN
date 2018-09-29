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
from skimage import filters

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]


def copyPreviousModel(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

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
    #elif ind == 14:
        #NucleusName = '14-MTT'
        #SliceNumbers = range(116,129)

    Params['modelFormat'] = 'ckpt'
    if 'localLT' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/Unlabeled'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'
        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    elif 'localPC' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/Unlabeled'

        Dir_AllTests  = '/media/data1/artin/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'

    elif 'server' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/Unlabeled'

        Dir_AllTests  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    Params['A'] = A
    Params['Flag_cross_entropy'] = 0
    Params['NeucleusFolder'] = '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Params['ThalamusFolder'] = '/CNN1_THALAMUS_2D_SanitizedNN'
    Params['Dir_Prior']    = Dir_Prior
    Params['Dir_AllTests'] = Dir_AllTests
    Params['SliceNumbers'] = SliceNumbers
    Params['NucleusName']  = NucleusName
    Params['optimizer'] = 'adam'
    Params['CropDim'] = np.array([ [50,198] , [130,278] , [Params['SliceNumbers'][0] , Params['SliceNumbers'][len(Params['SliceNumbers'])-1]] ])


    if Params['Flag_cross_entropy'] == 1:
        Params['modelName'] = 'model_CE/'
        Params['resultName'] = 'Results_CE/'
    else:
        Params['modelName'] = 'model/'
        Params['resultName'] = 'Results/'

    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '7'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'old'
    UserEntries['testmode'] = 'EnhancedSeperately' # 'combo'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['mode'] = 'server'
    UserEntries['init'] = 0

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testmode':
            UserEntries['testmode'] = input.split('=')[1] # 'combo'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]
        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]
        elif 'init' in input:
            UserEntries['init'] = 1

        elif input.split('=')[0] == 'nuclei':
            if 'all' in input.split('=')[1]:
                a = range(4,14)
                UserEntries['IxNuclei'] = np.append([1,2,4567],a)

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['IxNuclei'] = [int(k) for k in B]

            else:
                UserEntries['IxNuclei'] = [int(input.split('=')[1])]

        elif input.split('=')[0] == 'enhance':
            if 'all' in input.split('=')[1]:
                UserEntries['enhanced_Index'] = range(len(A))

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['enhanced_Index'] = [int(k) for k in B]

            else:
                UserEntries['enhanced_Index'] = [int(input.split('=')[1])]

        # elif input.split('=')[0] == 'training_iters':
        #     UserEntries['training_iters'] = input.split('=')[1] # 'combo'
        # elif input.split('=')[0] == 'epochs':
        #     UserEntries['epochs'] = input.split('=')[1] # 'combo'
        # elif input.split('=')[0] == 'temp_Slice':
        #     UserEntries['temp_Slice'] = input.split('=')[1] # 'combo'

    return UserEntries


UserEntries = input_GPU_Ix()
# gpuNum = '5' # nan'

for ind in UserEntries['IxNuclei']:
    print('ind',ind)
    Params = initialDirectories(ind = ind, mode = 'server' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']


    L = [0] if UserEntries['testmode'] == 'combo' else UserEntries['enhanced_Index'] # len(Params['A'])  # [1,4]: #
    for ii in L:

        TestName = 'Test_AllTrainings' if UserEntries['testmode'] == 'combo' else testNme(Params['A'],ii)

        Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + TestName + '/'

        if 'Unlabeled' in UserEntries['dataset']:
            Params['restorePath'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + TestName + '/' + 'vimp2_901_07052013_AS' + '/Train/' + 'model/' # Params['modelName']
        else:
            Params['restorePath'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + TestName + '/' + 'vimp2_1519_04212015' + '/Train/' + 'model/' # Params['modelName']

        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)

        # subFolders = ['vimp2_ctrl_921_07122013_MP'] # vimp2_ctrl_920_07122013_SW'] #
        # aaa = range(14,len(subFolders))
        # aaa = np.append([0,1],aaa)
        L = [0] if UserEntries['testmode'] == 'combo' else range(len(subFolders))
        for sFi in L:

            K = 'Test_' if UserEntries['testmode'] == 'combo' else 'Test_WMnMPRAGE_bias_corr_'

            if UserEntries['testmode'] != 'combo':
                print(Params['NucleusName'],TestName.split(K)[1],subFolders[sFi])

            K = 'Test/' if UserEntries['testmode'] == 'combo' else '/Test/'

            if UserEntries['testmode'] == 'combo':
                Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + K
                Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + 'Train/'
            else:
                Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + K
                Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'

                # Dir_ThalamusTestSamples = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/' + subFolders[sFi] + '/Test/'
                # Dir_ThalamusModelOut    = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/' + subFolders[sFi] + '/Train/model/'


            Dir_NucleiModelOut = mkDir(Dir_NucleiTrainSamples + Params['modelName'])
            Dir_ResultsOut = mkDir(Dir_NucleiTestSamples  + Params['resultName'])

            TrainData = image_util.ImageDataProvider(Dir_NucleiTrainSamples + "*.tif")
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


            if Params['Flag_cross_entropy'] == 1:
                cost_kwargs = {'class_weights':[0.7,0.3]}
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
            else:
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"



            trainer = unet.Trainer(Params['net'], optimizer = "adam")

            Params['init'] = 1
            if Params['init'] == 1:

                copyPreviousModel( Params['restorePath'], Dir_NucleiModelOut )
                if Params['gpuNum'] != 'nan':
                    # path2 = ''
                    path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=400, epochs=150, display_step=500, GPU_Num=Params['gpuNum'] ,prediction_path=Dir_ResultsOut , restore='True')
                else:
                    path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=50, epochs=4, display_step=500 ,prediction_path=Dir_ResultsOut , restore='True')

            else:
                if Params['gpuNum'] != 'nan':
                    # path2 = ''
                    path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=400, epochs=150, display_step=500, GPU_Num=Params['gpuNum'] ,prediction_path=Dir_ResultsOut) #  restore=True
                else:
                    path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=50, epochs=4, display_step=500 ,prediction_path=Dir_ResultsOut) #   restore=True


            if UserEntries['testmode'] != 'combo':

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
