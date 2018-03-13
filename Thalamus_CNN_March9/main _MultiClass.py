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


SegmentName = '/ThalamusSegDeformed.nii.gz'  # ThalamusSegDeformed   ThalamusSegDeformed_Croped  PulNeucleusSegDeformed PulNeucleusSegDeformed_Croped
TestName = 'ForUnet_Test14_MultiClass_Enhanced'  # 'ForUnet_Test2_IncreasingNumSlices'
Directory = '/media/data1/artin/data/Thalamus/'
Directory_OriginalData = Directory + 'OriginalData/'
with open(Directory_OriginalData + "subFolderList.txt" ,"rb") as fp:
    subFolders = pickle.load(fp)

# for sFi in range(len(subFolders)):
sFi = 1
Test_Path  = Directory + TestName + '/TestSubject'+str(sFi)+'/test/'
Train_Path = Directory + TestName + '/TestSubject'+str(sFi)+'/train/'
Trained_Model_Path = Train_Path + 'model/'
TestResults_Path   = Test_Path  + 'results/'

try:
    os.stat(Trained_Model_Path)
except:
    os.makedirs(Trained_Model_Path)

try:
    os.stat(TestResults_Path)
except:
    os.makedirs(TestResults_Path)

TrainData = image_util.ImageDataProvider(Train_Path + "*.tif",n_class=5)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

data,label =
print TrainData(1)
net = unet.Unet(layers=4, features_root=16, channels=1, n_class=5 , summaries=True) #  , cost="dice_coefficient"

trainer = unet.Trainer(net)
path = trainer.train(TrainData, Trained_Model_Path, training_iters=2, epochs=2, display_step=25) #, training_iters=100, epochs=100, display_step=10

# tensorboard --logdir=~/artin/data/Thalamus/ForUnet_Test2_IncreasingNumSlices/TestSubject0/train/model

OriginalSeg , Header , Affine = SumMasks(Directory_OriginalData+subFolders[sFi])
# OriginalSeg = nib.load(Directory_OriginalData+subFolders[sFi]+SegmentName)


CropDimensions = np.array([ [50,198] , [130,278]])
padSize = 90

[data,label,prediction,OriginalSeg] = TestData(net , Test_Path , Trained_Model_Path , OriginalSeg , Header , Affine , subFolders[sFi] , CropDimensions , padSize)
