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
        Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + 'old' + 'Dataset_' + 'old' +'Method'


    elif 'flash' in mode:

        machine = 'artin' # groot
        Params['modelFormat'] = 'ckpt'
        Dir_Prior = '/media/' + machine + '/aaa/Manual_Delineation_Sanitized_Full'
        Dir_AllTests  = '/media/' + machine + '/aaa/AllTests/' + dataset + 'Dataset_' + method +'Method'
        Params['Dir_AllTests_restore']  = '/media/' + machine + '/aaa/AllTests/' + 'old' + 'Dataset_' + 'old' +'Method'

    elif 'server' in mode:

        Params['modelFormat'] = 'ckpt' # cpkt
        hardDrive = 'ssd'
        if 'old' in dataset:
            Dir_Prior = '/array/' + hardDrive + '/msmajdi/data/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/array/' + hardDrive + '/msmajdi/data/newPriors/7T_MS'

        Dir_AllTests  = '/array/' + hardDrive + '/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method' # 'oldDataset' #
        Params['Dir_AllTests_restore']  ='/array/' + hardDrive + '/msmajdi/Tests/Thalamus_CNN/' + 'old' + 'Dataset_' + 'old' +'Method'

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
    padSizeFull = 90
    Params['padSize'] = int(padSizeFull/2)

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
    UserEntries['Flag_cross_entropy'] = 0

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


    if UserEntries['Flag_cross_entropy'] == 1:
        UserEntries['cost_kwargs'] = {'class_weights':[0.7,0.3]}
        UserEntries['modelName'] = 'model_CE/'
        UserEntries['resultName'] = 'Results_CE/'
    else:
        UserEntries['modelName'] = 'model/'
        UserEntries['resultName'] = 'Results/'

    return UserEntries

def ReadingTestImage(Params,subFolders):
    TestImage = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/' + Params['TestName'].split('Test_')[1] + '.nii.gz').get_data()
    TestImage = TestImage[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ]
    TestImage = np.pad(TestImage,((Params['padSize'],Params['padSize']),(Params['padSize'],Params['padSize']),(0,0)),'constant' )

    TestImage = np.transpose(TestImage,[2,0,1])
    TestImage = TestImage[...,np.newaxis]

    label = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz')

    return TestImage, label

def testFunc(Params):

    if Params['Flag_cross_entropy'] == 1:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=UserEntries['cost_kwargs']) # , cost="dice_coefficient"
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

    net = Params['net']

    # Data = Params['TestImage'][np.newaxis,:,:,:]
    Data = Params['TestImage']
    mn = np.min(Data)
    mx = np.max(Data)
    Data = (Data - mn) / (mx - mn)

    if Params['gpuNum'] != 'nan':
        prediction2 = net.predict( Params['Dir_NucleiTrainSamples'] + '/' + Params['modelName'] + 'model.' + Params['modelFormat'], np.asarray(Data,dtype=np.float32), GPU_Num=Params['gpuNum'])
    else:
        prediction2 = net.predict( Params['Dir_NucleiTrainSamples'] + '/' + Params['modelName'] + 'model.' + Params['modelFormat'], np.asarray(Data,dtype=np.float32))


    try:
        Thresh = max( filters.threshold_otsu(prediction2[...,1]) ,0.2)
    except:
        print('---------------------------error Thresholding------------------')
        Thresh = 0.2

    PredictedSeg = prediction2[...,1] > Thresh
    PredictedSeg = np.transpose(PredictedSeg,[1,2,0])
    prediction2 = np.transpose(prediction2[...,1], [1,2,0])

    return prediction2 , PredictedSeg

def trainFunc(Params):

    TrainData = image_util.ImageDataProvider(Params['Dir_NucleiTrainSamples'] + "*.tif")

    trainer = unet.Trainer(Params['net'], optimizer = "adam")
    if Params['gpuNum'] != 'nan':
        # path2 = ''
        path = trainer.train(TrainData, Params['Dir_NucleiModelOut'], training_iters=200, epochs=150, display_step=500, GPU_Num=Params['gpuNum'] ,prediction_path=Params['Dir_ResultsOut']) #  restore=True
    else:
        path = trainer.train(TrainData, Params['Dir_NucleiModelOut'], training_iters=50, epochs=4, display_step=500 ,prediction_path=Params['Dir_ResultsOut']) #   restore=True

    return path

def copyPreviousModel(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

UserEntries = input_GPU_Ix()


for ind in UserEntries['IxNuclei']:
    print('ind',ind)
    Params = initialDirectories(ind = ind, mode = 'server' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']


    # L = [0] if UserEntries['testmode'] == 'combo' else UserEntries['enhanced_Index'] # len(Params['A'])  # [1,4]: #
    L = UserEntries['enhanced_Index']
    for ii in L:

        Params['TestName'] = testNme(Params['A'],ii)

        # if UserEntries['testmode'] == 'combo':
        #     Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + 'Test_AllTrainings' + '/'
        # else:
        Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/'

        # subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)
        subFolders = subFoldersFunc(Params['Dir_Prior'])

        # subFolders = ['vimp2_ctrl_921_07122013_MP'] # vimp2_ctrl_920_07122013_SW'] #
        # L = [0] if UserEntries['testmode'] == 'combo' else range(len(subFolders))
        L = range(len(subFolders))
        for sFi in L:

            # K = 'Test_' if UserEntries['testmode'] == 'combo' else 'Test_WMnMPRAGE_bias_corr_'

            # if UserEntries['testmode'] != 'combo':
                # print(Params['NucleusName'],Params['TestName'].split(K)[1],subFolders[sFi])

            # K = 'Test/' if UserEntries['testmode'] == 'combo' else '/Test/'

            # if UserEntries['testmode'] == 'combo':
            #     Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + 'Test/'
            #     Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + 'Train/'
            # else:
            Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Test/'
            Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'

                # Dir_ThalamusTestSamples = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + Params['TestName'] + '/' + subFolders[sFi] + '/Test/'
                # Dir_ThalamusModelOut    = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + Params['TestName'] + '/' + subFolders[sFi] + '/Train/model/'

            TestImage, label = ReadingTestImage(Params,subFolders[sFi])

            output = np.zeros(label.shape)
            output_Lgc = np.zeros(label.shape)

            Params['TestImage'] = TestImage


            Params['Dir_NucleiModelOut'] = mkDir(Params['Dir_NucleiTrainSamples'] + Params['modelName'])
            Params['Dir_ResultsOut'] = mkDir(Params['Dir_NucleiTestSamples']  + Params['resultName'])

            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

            if Params['Flag_cross_entropy'] == 1:
                cost_kwargs = {'class_weights':[0.7,0.3]}
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
            else:
                Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"


            # path = trainFunc(Params)
            if UserEntries['testmode'] == 'combo':
                Params['restorePath_full'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + 'Test_AllTrainings' + '/Train'
            else:
                subFolder_trainModel = 'vimp2_824_05212013_JS'
                Params['restorePath_full'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/' + subFolder_trainModel + '/Train'

            Params['restorePath'] = Params['restorePath_full'] + '/' + Params['modelName']
            copyPreviousModel( Params['restorePath'], Params['Dir_NucleiModelOut'] )


            # if 0:
            pred , pred_Lgc = testFunc(Params)
            output[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ] = pred
            output_Lgc[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ] = pred_Lgc

            # ---------------------------  showing -----------------------------------
            Lbl = label.get_data()[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ]
            dice = [0]
            dice[0] = DiceCoefficientCalculator(pred_Lgc , Lbl )
            np.savetxt(Params['Dir_ResultsOut'] + 'DiceCoefficient.txt',dice)


            output2 = nib.Nifti1Image(output,label.affine)
            output2.get_header = label.header
            nib.save(output2 , Params['Dir_ResultsOut'] + subFolders[sFi] + '_' + Params['NucleusName'] + '.nii.gz')

            output_Lgc2 = nib.Nifti1Image(output_Lgc,label.affine)
            output_Lgc2.get_header = label.header
            nib.save(output_Lgc2 , Params['Dir_ResultsOut'] + subFolders[sFi] + '_' + Params['NucleusName'] + '_Logical.nii.gz')

            diceF = [0]
            diceF[0] = DiceCoefficientCalculator(output_Lgc,label.get_data())
            np.savetxt(Params['Dir_ResultsOut'] + 'DiceCoefficientF.txt',diceF)




            # if UserEntries['testmode'] != 'combo':
            #
            #     NucleiOrigSeg   = nib.load( Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz' )
            #     ThalamusOrigSeg = nib.load( Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' +        '1-THALAMUS'   + '_deformed.nii.gz' )
            #
            #     info = {}
            #     info['Dir_NucleiTestSamples'] = Params['Dir_NucleiTestSamples']
            #     info['Dir_ResultsOut']     = Params['Dir_ResultsOut']
            #     info['MultByThalamusFlag'] = 0
            #     info['Dir_NucleiModelOut'] = Params['Dir_NucleiModelOut'] + 'model.' + Params['modelFormat']
            #     info['padSize']      = 90
            #     info['CropDim']      = Params['CropDim']
            #     info['subFolders']   = subFolders[sFi]
            #     info['NucleusName']  = Params['NucleusName']
            #     info['SliceNumbers'] = Params['SliceNumbers']
            #     info['gpuNum']       = UserEntries['gpuNum']
            #     info['net']          = Params['net']
            #
            #     [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3_cleanedup(info , NucleiOrigSeg)
