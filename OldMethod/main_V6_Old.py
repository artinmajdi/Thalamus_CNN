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

gpuNum = '4' # nan'

# 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
# 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
# 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz

ind = 8
if ind == 1:
    NeucleusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    NucleusName = '1-THALAMUS'
elif ind == 4:
    NeucleusFolder = 'CNN4567_VL_2D_SanitizedNN' # 'CNN12_MD_Pf_2D_SanitizedNN' #  'CNN1_THALAMUS_2D_SanitizedNN' 'CNN6_VLP_2D_SanitizedNN'  #
    NucleusName = '4567-VL'
elif ind == 6:
    NeucleusFolder = 'CNN6_VLP_2D_SanitizedNN'
    NucleusName = '6-VLP'
elif ind == 8:
    NeucleusFolder = 'CNN8_Pul_2D_SanitizedNN'
    NucleusName = '8-Pul'
elif ind == 12:
    NeucleusFolder = 'CNN12_MD_Pf_2D_SanitizedNN'
    NucleusName = '12-MD-Pf'


ManualDir = '/Manual_Delineation_Sanitized/' #ManualDelineation

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
SliceNumbers = range(107,140)

Directory_main = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' #
#Directory_main = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'

Directory_Nuclei_Full = Directory_main + NeucleusFolder
Directory_Thalamus_Full = Directory_main + 'CNN1_THALAMUS_2D_SanitizedNN'

#priorDir = Directory_main + 'Manual_Delineation_Sanitized_Full/'
priorDir =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'

# subFolders = list(['vimp2_915_07112013_LC', 'vimp2_943_07242013_PA' ,'vimp2_964_08092013_TG'])

for ii in range(len(A)):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
    Directory_Thalamus = Directory_Thalamus_Full + '/' + TestName + '/'
    # print Directory_Nuclei
    # with open(Directory_Nuclei_Full + '/OriginalDeformedPriors/subFolderList.txt' ,"rb") as fp:
    #     subFolders = pickle.load(fp)
    subFolders = []
    subFlds = os.listdir(Directory_Nuclei)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    for sFi in range(len(subFolders)):

        Directory_Nuclei_Label = priorDir +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
        Directory_Thalamus_Label = priorDir +  subFolders[sFi] + ManualDir +'1-THALAMUS' + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

        print(Directory_Nuclei_Label) # sliceInd = 25
        Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/'
        Directory_Nuclei_Train = Directory_Nuclei + subFolders[sFi] + '/Train/'
        Directory_Nuclei_Train_Model = Directory_Nuclei_Train + 'model/'
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

        if os.path.isfile(TestResults_Path + 'DiceCoefficient__.txt'):
            print('*---  Already Done:   ' + Directory_Nuclei_Train_Model + '  ---*')
            continue
        else:
            print('*---  Not Done:   ' + Directory_Nuclei_Train_Model + '  ---*')
            TrainData = image_util.ImageDataProvider(Directory_Nuclei_Train + "*.tif")

            # TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)
            # data , label = TrainData(len(TrainData.data_files))
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # unet.config = config

            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) # , cost="dice_coefficient"

            trainer = unet.Trainer(net, optimizer = "adam")
            if gpuNum != 'nan':
                path = trainer.train(TrainData, Directory_Nuclei_Train_Model, training_iters=200, epochs=150, display_step=500, GPU_Num=gpuNum ) #  restore=True
            else:
                path = trainer.train(TrainData, Directory_Nuclei_Train_Model, training_iters=200, epochs=150, display_step=500) #   restore=True

            NucleiOrigSeg = nib.load(Directory_Nuclei_Label)
            ThalamusOrigSeg = nib.load(Directory_Thalamus_Label)

            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90
            MultByThalamusFlag = 1
            [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3(net , MultByThalamusFlag, Directory_Nuclei_Test , Directory_Nuclei_Train , ThalamusOrigSeg , NucleiOrigSeg , subFolders[sFi], CropDimensions , padSize , Directory_Thalamus_Test , Directory_Thalamus_Train_Model , NucleusName , SliceNumbers , gpuNum)
