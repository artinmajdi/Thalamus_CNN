from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
import tensorflow as tf
# from TestData import TestData


def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    return intersection.sum()*2/(msk1.sum()+msk2.sum())

def ReadMasks(DirectoryMask,SliceNumbers):

    mask = nib.load(DirectoryMask)
    maskD = mask.get_data()

    Header = mask.header
    Affine = mask.affine

    msk = maskD

    msk[maskD<0.5]  = 0
    msk[msk>=0.5] = 1

    return msk , Header , Affine

def SumMasks(DirectorySubFolders):

    i = 1
    Directory_Nuclei_Label = '/6_VLP_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD = np.zeros(msk.shape)
    maskD[msk == 1] = 10*i

    i += 1
    Directory_Nuclei_Label = '/7_VPL_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    i += 1
    Directory_Nuclei_Label = '/8_Pul_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    i += 1
    Directory_Nuclei_Label = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    return maskD , Header , Affine

NeucleusFolder = 'CNN7_Pul_2D' #'CNN_Thalamus' #
NucleusName = '1-THALAMUS' #'6-VLP' #

A = [[0,0]] # ,[4,3],[6,1],[1,2],[1,3],[4,1]] #
SliceNumbers = range(107,140)

Directory_main = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' # '/localhome/msmajdi_local/Tests/Thalamus_CNN/'
Directory_Nuclei_Full = Directory_main + NeucleusFolder

for ii in range(len(A)):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = f'Test_WMnMPRAGE_bias_corr_Sharpness_{str(A[ii][0])}_Contrast_{str(A[ii][1])}_Deformed'

    Directory_Nuclei = f'{Directory_Nuclei_Full}/{TestName}/'

    # with open(Directory_Nuclei_Full + '/OriginalDeformedPriors/subFolderList.txt' ,"rb") as fp:
    #     subFolders = pickle.load(fp)
    subFolders = os.listdir(Directory_Nuclei)
    # print len(subFolders)


    for sFi in range(len(subFolders)):
        print(f'Test: {str(A[ii])}Subject: {str(subFolders[sFi])}')
    # sFi = 1

        Directory_Nuclei_Label = f'/array/hdd/msmajdi/data/priors_forCNN/{subFolders[sFi]}/ManualDelineation/{NucleusName}_Deformed.nii.gz'

        Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/'
        Directory_Nuclei_Train = Directory_Nuclei + subFolders[sFi] + '/Train/'
        Directory_Nuclei_Train_Model = f'{Directory_Nuclei_Train}model/'
        TestResults_Path = f'{Directory_Nuclei_Test}results/'

        try:
            os.stat(Directory_Nuclei_Train_Model)
        except:
            os.makedirs(Directory_Nuclei_Train_Model)

        try:
            os.stat(TestResults_Path)
        except:
            os.makedirs(TestResults_Path)

        if os.path.isfile(f'{Directory_Nuclei_Train_Model}checkpoint'):
            print(f'*---  Already Done:   {Directory_Nuclei_Train_Model}  ---*')
            continue
        else:
            print(f'*---  Not Done:   {Directory_Nuclei_Train_Model}' + '  ---*')

            TrainData = image_util.ImageDataProvider(Directory_Nuclei_Train + "*.tif")
            # print Directory_Nuclei_Test
            # TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)
            # print len(TrainData.data_files)
            # data , label = TrainData(len(TrainData.data_files))
            # plt.imshow(label[1,:,:,1],cmap='gray')
            # plt.show()
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # session = tf.Session(config=config, ...)
            unet.config = config
            # with tf.device('/device:GPU:2'):
            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

            trainer = unet.Trainer(net)
            path = trainer.train(TrainData, Directory_Nuclei_Train_Model, training_iters=100, epochs=100, display_step=500) #, training_iters=100, epochs=100, display_step=10
            # tensorboard --logdir=~/artin/data/Thalamus/ForUnet_Test2_IncreasingNumSlices/TestSubject0/train/model
            # OriginalSeg , Header , Affine = SumMasks(Directory_OriginalData+subFolders[sFi])
            OriginalSegFull = nib.load(Directory_Nuclei_Label)
            Header = OriginalSegFull.header
            Affine = OriginalSegFull.affine
            OriginalSeg = OriginalSegFull.get_data()


            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90

            # [data,label,prediction] = TestData(net , Directory_Nuclei_Test , Directory_Nuclei_Train_Model , OriginalSeg , Header , Affine , subFolders[sFi] , CropDimensions , padSize)
    # #   TestData2_MultipliedByWholeThalamus(net , Directory_Nuclei_Test , Directory_Nuclei_Train_Model , OriginalSeg , subFolders[sFi], CropDimensions , padSize , Test_Path_Thalamus , Trained_Model_Path_Thalamus):

