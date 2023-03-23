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
    SegmentName = '/6_VLP_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD = np.zeros(msk.shape)
    maskD[msk == 1] = 10*i

    i += 1
    SegmentName = '/7_VPL_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    i += 1
    SegmentName = '/8_Pul_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    i += 1
    SegmentName = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+SegmentName)
    maskD[msk == 1] = 10*i

    return maskD , Header , Affine


SegmentName = '/PulNeucleusSegDeformed.nii.gz'  # ThalamusSegDeformed   ThalamusSegDeformed_Croped  PulNeucleusSegDeformed PulNeucleusSegDeformed_Croped
TestName = 'ForUnet_Test16_Enhanced_Pul'
Directory = '/media/data1/artin/data/Thalamus/'
Directory_OriginalData = f'{Directory}OriginalData/'
with open(f"{Directory_OriginalData}subFolderList.txt", "rb") as fp:
    subFolders = pickle.load(fp)

for sFi in range(len(subFolders)):
# sFi = 1
    Test_Path  = Directory + TestName + '/TestSubject'+str(sFi)+'/test/'
    Train_Path = Directory + TestName + '/TestSubject'+str(sFi)+'/train/'
    Trained_Model_Path = f'{Train_Path}model/'
    TestResults_Path = f'{Test_Path}results/'

    try:
        os.stat(Trained_Model_Path)
    except:
        os.makedirs(Trained_Model_Path)

    try:
        os.stat(TestResults_Path)
    except:
        os.makedirs(TestResults_Path)

    TrainData = image_util.ImageDataProvider(f"{Train_Path}*.tif")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

    trainer = unet.Trainer(net)
    path = trainer.train(TrainData, Trained_Model_Path, training_iters=100, epochs=100, display_step=25) #, training_iters=100, epochs=100, display_step=10

    # tensorboard --logdir=~/artin/data/Thalamus/ForUnet_Test2_IncreasingNumSlices/TestSubject0/train/model

    # OriginalSeg , Header , Affine = SumMasks(Directory_OriginalData+subFolders[sFi])
    OriginalSegFull = nib.load(Directory_OriginalData+subFolders[sFi]+SegmentName)
    Header = OriginalSegFull.header
    Affine = OriginalSegFull.affine
    OriginalSeg = OriginalSegFull.get_data()


    CropDimensions = np.array([ [50,198] , [130,278]])
    padSize = 90

    [data,label,prediction] = TestData(net , Test_Path , Trained_Model_Path , OriginalSeg , Header , Affine , subFolders[sFi] , CropDimensions , padSize)
