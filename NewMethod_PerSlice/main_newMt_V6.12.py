from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
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

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'new'):

    Params = {}

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


    if 'local' in mode:

        if 'oldDGX' not in dataset:
            Params['modelFormat'] = 'ckpt'
            if 'old' in dataset:
                Dir_Prior = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2'
            elif 'new' in dataset:
                Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/7T_MS'

            Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'
        else:
            Params['modelFormat'] = 'cpkt'
            Dir_Prior = '/media/groot/aaa/Manual_Delineation_Sanitized_Full'
            Dir_AllTests  = '/media/groot/aaa/AllTests/' + dataset + 'Dataset_' + method +'Method'

    elif 'server' in mode:

        Params['modelFormat'] = 'cpkt'
        if 'old' in dataset:
            Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'

        Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method' # 'oldDataset' #



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


    if Params['Flag_cross_entropy'] == 1:
        cost_kwargs = {'class_weights':[0.7,0.3]}
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"



    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] = 'nan'  # '5'  #
    UserEntries['IxNuclei'] = 1
    UserEntries['dataset'] = 'oldDGX' # 'old'
    UserEntries['method'] = 'new'
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

def DiceCoefficientCalculator(msk1,msk2):

    DiceCoef = np.zeros(1)
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef[0] = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def trainFunc(Params , slcIx):

    # if Params['IxNuclei'] == 9:
    #     if (slcIx < 2) | (slcIx > len(Params['SliceNumbers'])-2  ):
    #         Params['epochNum'] = 30
    #     else:
    #         Params['epochNum'] = 10
    # else:
    #     Params['epochNum'] = 100

    sliceNum = Params['SliceNumbers'][slcIx]

    Dir_NucleiModelOut = mkDir( Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum) + '/model/' )
    Dir_ResultsOut = mkDir( Params['Dir_NucleiTestSamples']  + '/Slice_' + str(sliceNum) + '/Results/' )

    TrainData = image_util.ImageDataProvider(Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum) + '/*.tif')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    trainer = unet.Trainer(Params['net'], optimizer = Params['optimizer']) # ,learning_rate=0.03
    if Params['gpuNum'] != 'nan':
        path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochNum'], display_step=100 , prediction_path=Dir_ResultsOut , GPU_Num=Params['gpuNum']) #  restore=True
    else:
        path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochNum'], display_step=100 , prediction_path=Dir_ResultsOut) #   restore=True

    return path

def testFunc(Params , slcIx):

    net = Params['net']
    sliceNumSubFld = Params['SliceNumbers'][slcIx]
    TestData = image_util.ImageDataProvider(  Params['Dir_NucleiTestSamples']  + '/Slice_' + str(sliceNumSubFld) + '/*.tif',shuffle_data=False)

    L = len(TestData.data_files)
    data,label = TestData(L)

    sliceNum = []
    for l in range(L):
        sliceNum.append(int(TestData.data_files[l].split('_Slice_')[1].split('.tif')[0]))

    if (len(sliceNum) > 1) | (sliceNum[0] != sliceNumSubFld):
        print('error , test files incompatible with new method')

    else:

        Data , Label = TestData(L)
        if Params['gpuNum'] != 'nan':
            prediction2 = net.predict( Params['Dir_NucleiTrainSamples']  + '/Slice_' + str(sliceNumSubFld) + '/model/model.' + Params['modelFormat'], np.asarray(Data,dtype=np.float32), GPU_Num=Params['gpuNum'])
        else:
            prediction2 = net.predict( Params['Dir_NucleiTrainSamples']  + '/Slice_' + str(sliceNumSubFld) + '/model/model.' + Params['modelFormat'], np.asarray(Data,dtype=np.float32))


        try:
            Thresh = max(filters.threshold_otsu(prediction2[0,...,1]),0.2)
        except:
            Thresh = 0.2

        PredictedSeg = prediction2[0,...,1] > Thresh

    return prediction2 , PredictedSeg






UserEntries = input_GPU_Ix()


for ind in [1]: # [UserEntries['IxNuclei']]:

    Params = initialDirectories(ind = ind, mode = 'local' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']
    Params['IxNuclei'] = UserEntries['IxNuclei']


    L = 1 if UserEntries['testMode'] == 'AllTrainings' else len(Params['A'])  # [1,4]: #
    for ii in range(1): # L):

        TestName = 'Test_AllTrainings' if UserEntries['testMode'] == 'AllTrainings' else testNme(Params['A'],ii)

        Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + TestName + '/'
        Dir_AllTests_Thalamus_EnhancedFld = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/'
        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)

        subFolders = ['vimp2_ANON724_03272013'] #
        for sFi in range(1): # len(subFolders)):
            K = 'Test_' if UserEntries['testMode'] == 'AllTrainings' else 'Test_WMnMPRAGE_bias_corr_'
            print(Params['NucleusName'],TestName.split(K)[1],subFolders[sFi])

            Dir_Prior_NucleiSample = Params['Dir_Prior'] +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'
            Dir_Prior_ThalamusSample = Params['Dir_Prior'] +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' +'1-THALAMUS' + '_deformed.nii.gz'

            K = '/Test0' if UserEntries['testMode'] == 'AllTrainings' else '/Test'
            Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + K
            Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train'


            # ---------------------------  main part-----------------------------------

            label  = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz')
            output = np.zeros(label.shape)
            Params['epochNum'] = 40
            Params['training_iters'] = 100
            for slcIx in range(4,5): # len(Params['SliceNumbers'])): # 1): #

                # ---------------------------  training -----------------------------------
                path = trainFunc(Params , slcIx)

                # ---------------------------  testing -----------------------------------
                _ , pred = testFunc(Params , slcIx)
                output[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'][slcIx] ] = pred


            a = label.get_data()[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'][slcIx] ]
            print('dice' , DiceCoefficientCalculator(pred , a) )  #  epoch:40 iter 300 dice:0.55 ;;; epoch:40 iter 100 dice:0.54
            ax,fig = plt.subplots(1,2)
            fig[0].imshow(pred,cmap='gray')
            fig[1].imshow(a,cmap='gray')
            plt.show()
            # ---------------------------  writing -----------------------------------
            # output2 = nib.Nifti1Image(output,label.affine)
            # output2.get_header = label.header
            # nib.save(output2 , Params['Dir_NucleiTestSamples'] + '/' + subFolders[sFi] + '_' + Params['NucleusName'] + '.nii.gz')
            #
            # Dice = DiceCoefficientCalculator(output,label.get_data())
            # np.savetxt(Params['Dir_NucleiTestSamples'] + '/DiceCoefficient.txt',Dice)
