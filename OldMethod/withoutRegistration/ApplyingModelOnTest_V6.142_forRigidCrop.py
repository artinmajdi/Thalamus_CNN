from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
# from TestData_V6_1 import TestData3_cleanedup
# from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf
import sys
from skimage import filters

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

do_I_want_Upsampling = 0

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

def NucleiSelection(ind):

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
    elif ind == 14:
        NucleusName = '14-MTT'
        SliceNumbers = range(104,135)

    return NucleusName , SliceNumbers

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'old'):

    Params = {}
    NucleusName , SliceNumbers = NucleiSelection(ind)


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
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + 'old' + 'Dataset_' + 'old' +'Method'
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
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + 'old' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'

    elif 'server' in mode:

        if 'old' in dataset:
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
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + 'old' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    Params['A'] = A
    Params['Flag_cross_entropy'] = 0
    Params['NeucleusFolder'] = '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Params['ThalamusFolder'] = '/CNN1_THALAMUS_2D_SanitizedNN'
    Params['Dir_Prior']    = Dir_Prior
    Params['Dir_AllTests'] = Dir_AllTests

    Params['NucleusName']  = NucleusName
    Params['optimizer'] = 'adam'
    Params['registrationFlag'] = 0

    if Params['registrationFlag'] == 1:
        Params['SliceNumbers'] = SliceNumbers
        Params['CropDim'] = np.array([ [50,198] , [130,278] , [Params['SliceNumbers'][0] , Params['SliceNumbers'][len(Params['SliceNumbers'])-1]] ])
    # else:

        # d1 = [105,192]
        # d2 = [67,184]
        # SN = [129,251]
        # Params['SliceNumbers'] = range(SN[0],SN[1])
        # Params['CropDim'] = np.array([ d1 , d2 , [Params['SliceNumbers'][0] , Params['SliceNumbers'][len(Params['SliceNumbers'])-1]] ])

    padSizeFull = 90
    Params['padSize'] = int(padSizeFull/2)

    if Params['Flag_cross_entropy'] == 1:
        Params['cost_kwargs'] = {'class_weights':[0.7,0.3]}
        Params['modelName'] = 'model_CE/'
        Params['resultName'] = 'Results_CE/'
    else:
        Params['modelName'] = 'model_LR1m4/'
        Params['resultName'] = 'Results_LR1m4/'

    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = 'nnnnnnnnnn' #'oldDGX' #
    UserEntries['method'] = 'old'
    UserEntries['testmode'] = 'normal' # 'combo' 'onetrain'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['mode'] = 'server'
    UserEntries['Flag_cross_entropy'] = 0
    UserEntries['onetrain_testIndexes'] = [1,5,10,14,20]


    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testmode':
            UserEntries['testmode'] = input.split('=')[1] # 'combo' 'onetrain'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]
        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]
        elif 'init' in input:
            UserEntries['init'] = 1

        elif 'onetrain_testIndexes' in input:
            UserEntries['testmode'] = 'onetrain'
            if input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['onetrain_testIndexes'] = [int(k) for k in B]
            else:
                UserEntries['onetrain_testIndexes'] = [int(input.split('=')[1])]


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


    return UserEntries

def funcNormalize(im):
    # return (im-im.mean())/im.std()
    im = np.float32(im)
    return ( im-im.min() )/( im.max() - im.min() )

def funcCropping_FromThalamus(im ,  CropMaskTh , CropMask, Params):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 1)[0]
    c2 = np.where(np.sum(ss,axis=0) > 1)[0]

    ss = np.sum(CropMaskTh,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 1)[0]

    gap = 0
    gap2 = 1
    d1 = [  c1[0]-gap  , c1[ c1.shape[0]-1 ]+gap   ]
    d2 = [  c2[0]-gap  , c2[ c2.shape[0]-1 ]+gap   ]
    SN = [  c3[0]-gap2 , c3[ c3.shape[0]-1 ]+gap2  ]
    Params['SliceNumbers'] = range(SN[0],SN[1])

    Params['CropDim'] = np.array([d1 , d2 , [SN[0],SN[1]] ])
    im = im[ d1[0]:d1[1],d2[0]:d2[1],Params['SliceNumbers'] ] # Params['SliceNumbers']]

    return im , Params

def funcCropping(im , CropMask, Params):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]
    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    d1 = [  c1[0] , c1[ c1.shape[0]-1 ]  ]
    d2 = [  c2[0] , c2[ c2.shape[0]-1 ]  ]
    SN = [  c3[0] , c3[ c3.shape[0]-1 ]  ]
    Params['SliceNumbers'] = range(SN[0],SN[1])

    # Params['RigidCrop'] = [d1 , d2 , SliceNumbers]
    Params['CropDim'] = np.array([d1 , d2 , [SN[0],SN[1]] ])
    im = im[ d1[0]:d1[1],d2[0]:d2[1], Params['SliceNumbers'] ] # Params['SliceNumbers']]

    return im , Params

def funcPadding(im,Params):
    sz = im.shape
    df = 238 - sz[0]
    p1 = [int(df/2) , df - int(df/2)]

    df = 238 - sz[1]
    p2 = [int(df/2) , df - int(df/2)]

    Params['padding'] = np.array([p1 , p2])
    im = np.pad(im,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )

    return im,Params

def funcFlipLR_Upsampling(Params, im):
    if 'Unlabeled' in Params['dataset']:

        for i in range(im.shape[2]):
            im[...,i] = np.fliplr(im[...,i])

        # if do_I_want_Upsampling == 1:
        #     mask = ndimage.zoom(mask,(1,1,2),order=0)
        #     im = ndimage.zoom(im,(1,1,2),order=3)
    else:
        im   = np.transpose(im,[0,2,1])

        # if im.shape[2] == 200:
        #     im = ndimage.zoom(im,(1,1,2),order=3)
        #     mask = ndimage.zoom(mask,(1,1,2),order=0)

    return im

def ReadingTestImage(Params,subFolders):

    TestImage = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/' + Params['TestName'].split('Test_')[1] + '.nii.gz').get_data()
    TestImage = funcNormalize( TestImage )
    CropMask = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/' + 'MyCrop2_Gap20.nii.gz').get_data()

    if '1-THALAMUS' in Params['NucleusName']:
        TestImage , Params = funcCropping(TestImage , CropMask , Params)
    else:
        # mskTh = nib.load(Params['Dir_AllTests'] + '/CNN1_THALAMUS_2D_SanitizedNN/' + Params['TestName'] + '/OneTrain_MultipleTest' + '/TestCases/' + subFolders + '/Test/Results/' + subFolders[sFi] + '_1-THALAMUS' + '_Logical.nii.gz').get_data()
        # TestImage , Params = funcCropping_FromThalamus(TestImage , mskTh , Params)

        try:
            mskTh = nib.load(Params['Dir_Prior'] + '/'  + subFolders + '/Test/Results/' + subFolders +'_1-THALAMUS_Logical.nii.gz').get_data()
            TestImage , Params = funcCropping_FromThalamus(TestImage , mskTh , CropMask , Params)
        except:
            print('*************** unable to read full thalamus ***************')

            TestImage , Params = funcCropping(TestImage , CropMask, Params)




    TestImage = funcFlipLR_Upsampling(Params, TestImage)

    TestImage,Params = funcPadding(TestImage,Params)

    TestImage = np.transpose(TestImage,[2,0,1])
    TestImage = TestImage[...,np.newaxis]

    return TestImage , Params

def testFunc(Params):

    if Params['Flag_cross_entropy'] == 1:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True , cost_kwargs=Params['cost_kwargs']) # , cost="dice_coefficient"
    else:
        Params['net'] = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

    net = Params['net']

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

def saveImageDice(label , Params , pred , pred_Lgc , subFolders):
    output = np.zeros(label.shape)
    output_Lgc = np.zeros(label.shape)

    # print('----Params[CropDim]-----',Params['CropDim'])
    # print('predshape',pred.shape)

    labelF = label.get_data()
    pred = funcFlipLR_Upsampling(Params, pred)
    pred_Lgc = funcFlipLR_Upsampling(Params, pred_Lgc)

    output[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ] = pred
    output_Lgc[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ] = pred_Lgc


    # ---------------------------  showing -----------------------------------
    Lbl = labelF[ Params['CropDim'][0,0]:Params['CropDim'][0,1] , Params['CropDim'][1,0]:Params['CropDim'][1,1] , Params['SliceNumbers'] ]
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
    diceF[0] = DiceCoefficientCalculator(output_Lgc,labelF)
    np.savetxt(Params['Dir_ResultsOut'] + 'DiceCoefficientF.txt',diceF)

savingThalamusPredOnTrainData = 0
UserEntries = input_GPU_Ix()

for ind in UserEntries['IxNuclei']:
    print('ind',ind)
    Params = initialDirectories(ind = ind, mode = UserEntries['mode'] , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Params['gpuNum'] = UserEntries['gpuNum']
    Params['dataset'] = UserEntries['dataset']

    L = [0] if UserEntries['testmode'] == 'combo' else UserEntries['enhanced_Index'] # len(Params['A'])  # [1,4]: #
    # L = UserEntries['enhanced_Index']
    for ii in L:

        if Params['registrationFlag'] == 1:
            Params['TestName'] = testNme(Params['A'],ii)
        else:
            Params['TestName'] = testNme(Params['A'],ii).split('_Deformed')[0]

        if UserEntries['testmode'] == 'combo':
            Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + 'Test_AllTrainings' + '/'
        else:
            Dir_AllTests_Nuclei_EnhancedFld = Params['Dir_AllTests'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/'

        # subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld)
        if UserEntries['testmode'] == 'onetrain':

            if (savingThalamusPredOnTrainData == 1) & ('1-THALAMUS' in Params['NucleusName']):
                subFolders = subFoldersFunc( Params['Dir_Prior'] )
                # subFolders = subFolders[:5]
            else:
                subFolders = subFoldersFunc(Dir_AllTests_Nuclei_EnhancedFld + 'OneTrain_MultipleTest' + '/TestCases/')

        else:
            subFolders = subFoldersFunc(Params['Dir_Prior'])


        if UserEntries['testmode'] == 'MW':
            subFolders_TrainedModels = subFoldersFunc(Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/')


        # subFolders = ['vimp2_ctrl_921_07122013_MP'] # vimp2_ctrl_920_07122013_SW'] #
        L = [0] if UserEntries['testmode'] == 'combo' else range(len(subFolders))
        # L = range(len(subFolders))
        for sFi in L:

            # K = 'Test_' if UserEntries['testmode'] == 'combo' else 'Test_WMnMPRAGE_bias_corr_'

            # if UserEntries['testmode'] != 'combo':
                # print(Params['NucleusName'],Params['TestName'].split(K)[1],subFolders[sFi])

            # K = 'Test/' if UserEntries['testmode'] == 'combo' else '/Test/'

            if UserEntries['testmode'] == 'combo':
                Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + 'Test/'
                Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + 'Train/'

            elif UserEntries['testmode'] == 'onetrain':
                if (savingThalamusPredOnTrainData == 1) & ('1-THALAMUS' in Params['NucleusName']):
                    subFolders = subFoldersFunc( Params['Dir_Prior'] )
                    Params['Dir_NucleiTrainSamples']  = mkDir( Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Train/')
                    Params['Dir_NucleiTestSamples']   = Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Test/'
                else:
                    Params['Dir_NucleiTrainSamples']  = mkDir(Dir_AllTests_Nuclei_EnhancedFld + 'OneTrain_MultipleTest' + '/TestCases/' + subFolders[sFi] + '/Train/')
                    Params['Dir_NucleiTestSamples']   = Dir_AllTests_Nuclei_EnhancedFld + 'OneTrain_MultipleTest' + '/TestCases/' + subFolders[sFi] + '/Test/'

            elif UserEntries['testmode'] == 'normal':
                Params['Dir_NucleiTestSamples']  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Test/'
                Params['Dir_NucleiTrainSamples'] = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'

                # Dir_ThalamusTestSamples = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + Params['TestName'] + '/' + subFolders[sFi] + '/Test/'
                # Dir_ThalamusModelOut    = Params['Dir_AllTests'] + Params['ThalamusFolder'] + '/' + Params['TestName'] + '/' + subFolders[sFi] + '/Train/model/'

            TestImage, Params = ReadingTestImage(Params,subFolders[sFi])
            label = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz')

            Params['TestImage'] = TestImage


            Params['Dir_NucleiModelOut'] = mkDir(Params['Dir_NucleiTrainSamples'] + Params['modelName'])
            # print('-------------------------------------------------------')
            # print('-------------------------------------------------------')
            # print('Dir_NucleiTrainSamples',Params['Dir_NucleiTrainSamples'])
            # print('Dir_NucleiModelOut',Params['Dir_NucleiModelOut'])
            # print('modelName',Params['modelName'])


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
                Params['restorePath'] = Params['restorePath_full'] + '/' + Params['modelName']
                copyPreviousModel( Params['restorePath'], Params['Dir_NucleiModelOut'] )

                pred , pred_Lgc = testFunc(Params)

            elif UserEntries['testmode'] == 'MW':
                cp = Params['CropDim']
                predFull_Lgc = np.zeros((cp[0,1]-cp[0,0] , cp[1,1]-cp[1,0] , len(Params['SliceNumbers']),len(subFolders_TrainedModels)))
                predFull = np.zeros((cp[0,1]-cp[0,0] , cp[1,1]-cp[1,0] , len(Params['SliceNumbers']),len(subFolders_TrainedModels)))

                for sFi_tr in range(len(subFolders_TrainedModels)):

                    subFolder_trainModel = subFolders_TrainedModels[sFi_tr]
                    Params['restorePath_full'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/' + subFolder_trainModel + '/Train'
                    Params['restorePath'] = Params['restorePath_full'] + '/' + Params['modelName']
                    copyPreviousModel( Params['restorePath'], Params['Dir_NucleiModelOut'] )
                    predTp , pred_LgcTp = testFunc(Params)
                    predFull_Lgc[...,sFi_tr] = pred_LgcTp
                    predFull[...,sFi_tr] = predTp

                pred = np.mean(predFull,axis=3)
                pred_Lgc = np.sum(predFull_Lgc,axis=3) > int(len(subFolders_TrainedModels)/2)

            elif UserEntries['testmode'] == 'normal':
                subFolder_trainModel = 'vimp2_1519_04212015' # 'vimp2_ANON724_03272013'

                Params['restorePath_full'] = Params['Dir_AllTests_restore'] + Params['NeucleusFolder'] + '/' + Params['TestName'] + '/' + subFolder_trainModel + '/Train'
                Params['restorePath'] = Params['restorePath_full'] + '/' + Params['modelName']
                copyPreviousModel( Params['restorePath'], Params['Dir_NucleiModelOut'] )
                pred , pred_Lgc = testFunc(Params)

            elif UserEntries['testmode'] == 'onetrain':
                Params['restorePath'] = Dir_AllTests_Nuclei_EnhancedFld + 'OneTrain_MultipleTest' + '/Train/' + Params['modelName']
                copyPreviousModel( Params['restorePath'], Params['Dir_NucleiModelOut'] )
                pred , pred_Lgc = testFunc(Params)

                p1 = Params['padding'][0]
                p2 = Params['padding'][1]

                pred     =     pred[  p1[0]-45:148-(p1[1]-45) , p2[0]-45:148-(p2[1]-45) , :  ]
                pred_Lgc = pred_Lgc[  p1[0]-45:148-(p1[1]-45) , p2[0]-45:148-(p2[1]-45) , :  ]




            saveImageDice(label , Params , pred , pred_Lgc , subFolders)





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
