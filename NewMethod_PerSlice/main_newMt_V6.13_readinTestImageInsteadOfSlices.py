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

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

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
    print(ind)
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

        Params['modelFormat'] = 'ckpt'
        if 'old' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/7T_MS'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'


    elif 'flash' in mode:

        machine = 'artin' # groot
        Params['modelFormat'] = 'ckpt'
        Dir_Prior = '/media/' + machine + '/aaa/Manual_Delineation_Sanitized_Full'
        Dir_AllTests  = '/media/' + machine + '/aaa/AllTests/' + dataset + 'Dataset_' + method +'Method'

    elif 'server' in mode:

        Params['modelFormat'] = 'ckpt' # cpkt
        hardDrive = 'ssd'
        if 'old' in dataset:
            Dir_Prior = '/array/' + hardDrive + '/msmajdi/data/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/array/' + hardDrive + '/msmajdi/data/newPriors/7T_MS'

        Dir_AllTests  = '/array/' + hardDrive + '/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method' # 'oldDataset' #
        Params['Dir_AllTests_restore']  = '/array/' + hardDrive + '/msmajdi/Tests/Thalamus_CNN/' + 'old' + 'Dataset_' + 'old' +'Method'

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
        cost_kwargs = {'class_weights':[0.7,0.3]}
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"

        Params['modelName'] = 'model_CE/'
        Params['resultName'] = 'Results_CE/'
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"
        Params['modelName'] = 'model/'
        Params['resultName'] = 'Results/'

    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = 1
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'new'
    UserEntries['testMode'] = 'EnhancedSeperately' # 'AllTrainings'
    UserEntries['enhanced_Index'] = range(len(A))

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]

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
        #     UserEntries['training_iters'] = input.split('=')[1] # 'AllTrainings'
        # elif input.split('=')[0] == 'epochs':
        #     UserEntries['epochs'] = input.split('=')[1] # 'AllTrainings'
        # elif input.split('=')[0] == 'temp_Slice':
        #     UserEntries['temp_Slice'] = input.split('=')[1] # 'AllTrainings'

    return UserEntries

def DiceCoefficientCalculator(msk1,msk2):

    DiceCoef = np.zeros(1)
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef[0] = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def trainFunc(Params , slcIx):



    sliceNum = Params['SliceNumbers'][slcIx]


    Dir_NucleiModelOut = mkDir( Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum) + '/' + Params['modelName'] )
    Dir_ResultsOut = mkDir( Params['Dir_NucleiTestSamples']  + '/Slice_' + str(sliceNum) + '/' + Params['resultName'] )
    print(Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum))
    TrainData = image_util.ImageDataProvider(Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum) + '/*.tif')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    if Params['Flag_cross_entropy'] == 1:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"


    trainer = unet.Trainer(Params['net'], optimizer = Params['optimizer']) # ,learning_rate=0.03
    if Params['gpuNum'] != 'nan':

        print('----------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------')

        if slcIx > -10:
            # copyPreviousModel( Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum-1) + '/' + Params['modelName'] , Dir_NucleiModelOut )
            copyPreviousModel( Params['restorePath'], Dir_NucleiModelOut )
            path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochs'], display_step=500 , prediction_path=Dir_ResultsOut , GPU_Num=Params['gpuNum'] , restore='True') # , write_graph=True
        else:
            path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochs'], display_step=500 , prediction_path=Dir_ResultsOut , GPU_Num=Params['gpuNum'])
    else:
        if slcIx > -10:
            # copyPreviousModel( Params['Dir_NucleiTrainSamples'] + '/Slice_' + str(sliceNum-1) + '/' + Params['modelName'] , Dir_NucleiModelOut )
            copyPreviousModel( Params['restorePath'], Dir_NucleiModelOut )
            path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochs'], display_step=500 , prediction_path=Dir_ResultsOut , restore=True) #  , write_graph=True

        else:
            path = trainer.train(TrainData , Dir_NucleiModelOut , training_iters=Params['training_iters'] , epochs=Params['epochs'], display_step=500 , prediction_path=Dir_ResultsOut) #   restore=True

    return path

def testFunc(Params , slcIx):

    if Params['Flag_cross_entropy'] == 1:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=cost_kwargs) # , cost="dice_coefficient"
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

    net = Params['net']
    sliceNumSubFld = Params['SliceNumbers'][slcIx]


    readingFromSlices = 0
    if readingFromSlices == 0:
        Data = Params['TestSliceImage']
        Label = Params['TestSliceLabel']
        sliceNum = [sliceNumSubFld]
    else:
        TestData = image_util.ImageDataProvider(  Params['Dir_NucleiTestSamples']  + '/Slice_' + str(sliceNumSubFld) + '/*.tif',shuffle_data=False)

        L = len(TestData.data_files)
        Data,Label = TestData(L)

        sliceNum = []
        for l in range(L):
            sliceNum.append(int(TestData.data_files[l].split('_Slice_')[1].split('.tif')[0]))



    if (len(sliceNum) > 1) | (sliceNum[0] != sliceNumSubFld):
        print('error , test files incompatible with new method')

    else:

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

def paramIterEpoch(Params , slcIx):

    Params['training_iters'] = 200

    if Params['IxNuclei'] == [9]:
        if (slcIx < 2) | (slcIx > len(Params['SliceNumbers'])-2  ):
            Params['epochs'] = 30
        else:
            Params['epochs'] = 10

    elif Params['IxNuclei'] == [1]:
        if (slcIx < 5) | (slcIx > len(Params['SliceNumbers'])-5  ):
            Params['epochs'] = 3 # 40
        else:
            Params['epochs'] = 3 # 60
    else:
        Params['epochs'] = 50

    return Params

def copyPreviousModel(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def ReadingTestImage(Params,subFolders,TestName):
    TestImage = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/' + TestName.split('Test_')[1] + '.nii.gz').get_data()
    TestImage = TestImage[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ]
    TestImage = np.pad(TestImage,((Params['padSize'],Params['padSize']),(Params['padSize'],Params['padSize']),(0,0)),'constant' )
    TestImage = TestImage[:,:,np.newaxis,:]

    label = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz')
    TestLabel = label.get_data()[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ]
    TestLabel = np.pad(TestLabel,((Params['padSize'],Params['padSize']),(Params['padSize'],Params['padSize']),(0,0)),'constant' )

    B = 1 - TestLabel
    a = np.append(B[...,np.newaxis],TestLabel[...,np.newaxis],axis=3)
    TestLabel = np.transpose(a,[0,1,3,2])

    return TestImage, TestLabel, label

UserEntries = input_GPU_Ix()


for ind in UserEntries['IxNuclei']:

    Params = initialDirectories(ind = ind, mode = 'server' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']
    Params['IxNuclei'] = UserEntries['IxNuclei']




    L = [0] if UserEntries['testMode'] == 'AllTrainings' else UserEntries['enhanced_Index'] # len(Params['A'])  # [1,4]: #
    for ii in L: # range(2): #@ L):

        TestName = 'Test_AllTrainings' if UserEntries['testMode'] == 'AllTrainings' else testNme(Params['A'],ii)

        Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + TestName + '/'
        Dir_AllTests_Thalamus_EnhancedFld = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + TestName + '/'
        subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)

        subFolders = ['vimp2_ANON724_03272013'] #
        for sFi in range(len(subFolders)):
            K = 'Test_' if UserEntries['testMode'] == 'AllTrainings' else 'Test_WMnMPRAGE_bias_corr_'
            print(Params['NucleusName'],TestName.split(K)[1],subFolders[sFi])

            Dir_Prior_NucleiSample = Params['Dir_Prior'] +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'
            Dir_Prior_ThalamusSample = Params['Dir_Prior'] +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' +'1-THALAMUS' + '_deformed.nii.gz'

            K = '/Test0' if UserEntries['testMode'] == 'AllTrainings' else '/Test'
            Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + K
            Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train'
            if Params['gpuNum'] != 'nan':
                Params['restorePath'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + TestName + '/' + subFolders[sFi] + '/Train/' + Params['modelName']


            # ---------------------------  main part-----------------------------------

            TestImage, TestLabel, label = ReadingTestImage(Params,subFolders[sFi],TestName)
            output = np.zeros(label.shape)

            # Params['epochs'] = int(UserEntries['epochs']) # 40
            # Params['training_iters'] = int(UserEntries['training_iters']) # 100

            dice = np.zeros(len(Params['SliceNumbers'])+1)
            for slcIx in range(len(Params['SliceNumbers'])):

                Params = paramIterEpoch(Params , slcIx)
                print('epochs',Params['epochs'],'IxNuclei',Params['IxNuclei'],'iter',Params['training_iters'])

                # ---------------------------  training -----------------------------------
                # path = trainFunc(Params , slcIx)

                # ---------------------------  testing -----------------------------------
                Params['TestSliceImage'] = TestImage[np.newaxis,...,slcIx]
                Params['TestSliceLabel'] = TestLabel[np.newaxis,...,slcIx]

                _ , pred = testFunc(Params , slcIx)
                output[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'][slcIx] ] = pred

                # ---------------------------  showing -----------------------------------
                # print('-------------------------------------------------------------------')
                Lbl = label.get_data()[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'][slcIx] ]
                dice[slcIx] = DiceCoefficientCalculator(pred , Lbl )
                np.savetxt(Params['Dir_NucleiTestSamples'] + '/DiceCoefficient.txt',dice)
                # ax,fig = plt.subplots(1,2)
                # fig[0].imshow(pred,cmap='gray')
                # fig[1].imshow(a,cmap='gray')
                # plt.show()



            # ---------------------------  writing -----------------------------------
            output2 = nib.Nifti1Image(output,label.affine)
            output2.get_header = label.header
            nib.save(output2 , Params['Dir_NucleiTestSamples'] + '/' + subFolders[sFi] + '_' + Params['NucleusName'] + '.nii.gz')

            dice[len(Params['SliceNumbers'])] = DiceCoefficientCalculator(output,label.get_data())
            np.savetxt(Params['Dir_NucleiTestSamples'] + '/DiceCoefficient.txt',dice)
