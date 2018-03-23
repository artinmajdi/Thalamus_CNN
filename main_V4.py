from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData import TestData


def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

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
    SegmentName = '/6_VLP_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD = np.zeros(msk.shape)
    maskD[msk == 1] = 10*i

    i = i + 1
    SegmentName = '/7_VPL_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    i = i + 1
    SegmentName = '/8_Pul_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    i = i + 1
    SegmentName = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    return maskD , Header , Affine

NeucleusFolder = 'CNN_VLP4_2DSlicesSeperately' #'CNN_Thalamus' #
NucleusName = '6-VLP' #'1-THALAMUS' #

A = [[0,0],[4,3]] # ,[6,1],[1,2],[1,3],[4,1]
SliceNumbers = range(107,140)

Directory_Nuclei = '/media/data1/artin/data/Thalamus/' + NeucleusFolder
Directory_Thalamus = '/media/data1/artin/data/Thalamus/' + 'CNN_Thalamus'


for ii in range(len(A)):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Test_Directory_Nuclei = Directory_Nuclei + '/' + TestName + '/'

    # with open(Directory_Nuclei + '/OriginalDeformedPriors/subFolderList.txt' ,"rb") as fp:
    #     subFolders = pickle.load(fp)
    subFolders = os.listdir(Test_Directory_Nuclei)
    print len(subFolders)


    for sFi in range(len(subFolders)):
        print('Test: ' + str(A[ii]) + 'Subject: ' + str(subFolders[sFi]))
    # sFi = 1

        Segments_Directory = Directory_Nuclei + '/OriginalDeformedPriors/' +  subFolders[sFi] + '/ManualDelineation'
        SegmentName = Segments_Directory +'/' + NucleusName + '_Deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped


        for sliceInd in range(32):

            Test_Path_Nuclei  = Test_Directory_Nuclei + subFolders[sFi] + '/Test/Slice' + str(sliceInd) + '/'
            Train_Path_Nuclei = Test_Directory_Nuclei + subFolders[sFi] + '/Train/Slice' + str(sliceInd) + '/'
            Trained_Model_Path_Nuclei = Train_Path_Nuclei + 'model/'
            TestResults_Path_Nuclei   = Test_Path_Nuclei  + 'results/'

            try:
                os.stat(Trained_Model_Path_Nuclei)
            except:
                os.makedirs(Trained_Model_Path_Nuclei)

            try:
                os.stat(TestResults_Path_Nuclei)
            except:
                os.makedirs(TestResults_Path_Nuclei)


            TrainData = image_util.ImageDataProvider(Train_Path_Nuclei + "*.tif")
            print TrainData.n_class
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

            trainer = unet.Trainer(net)
            path = trainer.train(TrainData, Trained_Model_Path_Nuclei, training_iters=50, epochs=25, display_step=500) #, training_iters=100, epochs=100, display_step=10

            # tensorboard --logdir=~/artin/data/Thalamus/ForUnet_Test2_IncreasingNumSlices/TestSubject0/train/model

            # OriginalSeg , Header , Affine = SumMasks(Directory_OriginalData+subFolders[sFi])
            OriginalSegFull = nib.load(SegmentName)
            Header = OriginalSegFull.header
            Affine = OriginalSegFull.affine
            OriginalSeg = OriginalSegFull.get_data()


            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90

            # [data,label,prediction] = TestData(net , Test_Path_Nuclei , Trained_Model_Path_Nuclei , OriginalSeg , Header , Affine , subFolders[sFi] , CropDimensions , padSize)
        # #   TestData2_MultipliedByWholeThalamus(net , Test_Path_Nuclei , Trained_Model_Path_Nuclei , OriginalSeg , subFolders[sFi], CropDimensions , padSize , Test_Path_Thalamus , Trained_Model_Path_Thalamus):
            # [data,label,prediction,OriginalSeg] = TestData2_MultipliedByWholeThalamus(net , Test_Path_Nuclei , Trained_Model_Path_VLP , OriginalSeg , subFolders[sFi] , CropDimensions , padSize , Test_Path_Thalamus , Trained_Model_Path_Thalamus , NucleusName)
