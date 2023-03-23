from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6 import TestData3
from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf


def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
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

# define worker function
def calculate(process_name, tasks, results):
    print(f'[{process_name}] evaluation routine starts')

    while True:
        new_value = tasks.get()
        if new_value < 0:
            print(f'[{process_name}] evaluation routine quits')

            # Indicate finished
            results.put(-1)
            break
        else:
            # Compute result and mimic a long-running task
            compute = new_value * new_value
            sleep(0.02*new_value)

            # Output which process received the value
            # and the calculation result
            print('[%s] received value: %i' % (process_name, new_value))
            print('[%s] calculated value: %i' % (process_name, compute))

            # Add result to the queue
            results.put(compute)

    return

#def Main(sFi):

print("start ")
gpuNum = "6"
NeucleusFolder = 'CNN5_Thalamus_2D_VTK' #'CNN_Thalamus' #
NucleusName = '1-THALAMUS' #'6-VLP' #

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
SliceNumbers = range(107,140)

Directory_main = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' # '/localhome/msmajdi_local/Tests/Thalamus_CNN/'
Directory_Nuclei_Full = Directory_main + NeucleusFolder
Directory_Thalamus_Full = f'{Directory_main}CNN5_Thalamus_2D_VTK'

# ii = 1
for ii in range(4,len(A)): # len(A)):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = f'Test_WMnMPRAGE_bias_corr_Sharpness_{str(A[ii][0])}_Contrast_{str(A[ii][1])}_Deformed'

    Directory_Nuclei = f'{Directory_Nuclei_Full}/{TestName}/'
    Directory_Thalamus = f'{Directory_Thalamus_Full}/{TestName}/'
    # print Directory_Nuclei
    # with open(Directory_Nuclei_Full + '/OriginalDeformedPriors/subFolderList.txt' ,"rb") as fp:
    #     subFolders = pickle.load(fp)
    subFolders = os.listdir(Directory_Nuclei)
    # print len(subFolders)


    for sFi in range(len(subFolders)):

        Directory_Nuclei_Label = f'/array/hdd/msmajdi/data/priors_forCNN/{subFolders[sFi]}/ManualDelineation/{NucleusName}_Deformed.nii.gz'
        Directory_Thalamus_Label = f'/array/hdd/msmajdi/data/priors_forCNN/{subFolders[sFi]}/ManualDelineation/1-THALAMUS_Deformed.nii.gz'

        # sliceInd = 25
        Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/'
        Directory_Nuclei_Train = Directory_Nuclei + subFolders[sFi] + '/Train/'
        Directory_Nuclei_Train_Model = f'{Directory_Nuclei_Train}model/'
        TestResults_Path   = Directory_Nuclei_Test  + 'Results/'

        Directory_Thalamus_Test  = Directory_Thalamus + subFolders[sFi] + '/Test/'
        Directory_Thalamus_Train = Directory_Thalamus + subFolders[sFi] + '/Train/'
        Directory_Thalamus_Train_Model = Directory_Thalamus_Train + 'model/'


        try:
            os.stat(Directory_Nuclei_Train_Model)
        except:
            os.makedirs(Directory_Nuclei_Train_Model)
        try:
            os.stat(TestResults_Path)
        except:
            os.makedirs(TestResults_Path)

        if os.path.isfile(TestResults_Path + 'DiceCoefficient.txt'):
            print('*---  Already Done:   ' + Directory_Nuclei_Train_Model + '  ---*')
            continue
        else:
            print('*---  Not Done:   ' + Directory_Nuclei_Train_Model + '  ---*')
            TrainData = image_util.ImageDataProvider(Directory_Nuclei_Train + "*.tif")

            TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)
            # data , label = TrainData(len(TrainData.data_files))
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # unet.config = config

            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

            trainer = unet.Trainer(net)
            path = trainer.train(TrainData, Directory_Nuclei_Train_Model, training_iters=100, epochs=100, display_step=500, GPU_Num=gpuNum) #  , cost="dice_coefficient" restore=True

            OriginalSegFull = nib.load(Directory_Nuclei_Label)

            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90
            MultByThalamusFlag = 0
            [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3(net , MultByThalamusFlag, Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSegFull , subFolders[sFi], CropDimensions , padSize , Directory_Thalamus_Test , Directory_Thalamus_Train_Model , NucleusName , SliceNumbers , gpuNum)

