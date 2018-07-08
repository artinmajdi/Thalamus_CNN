from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V6_1_newDataset  import TestData3
from tf_unet import unet, util, image_util
import multiprocessing
import tensorflow as tf

gpuNum = '4' # nan'

# 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
# 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
# 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz

ind = 1
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
elif ind == 10:
    NeucleusFolder = 'CNN10_MGN_2D_SanitizedNN'
    NucleusName = '10-MGN'
elif ind == 12:
    NeucleusFolder = 'CNN12_MD_Pf_2D_SanitizedNN'
    NucleusName = '12-MD-Pf'


ManualDir = '/Manual_Delineation_Sanitized/' #ManualDelineation

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
SliceNumbers = range(107,140)

Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' #
#Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'

Name_allTests_Nuclei = Dir_AllTests + NeucleusFolder
Name_allTests_Thalamus = Dir_AllTests + 'CNN1_THALAMUS_2D_SanitizedNN'

# Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'

Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'
# Dir_Prior = '/array/hdd/msmajdi/data/test'

for ii in range(len(A)):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Dir_AllTests_Nuclei_EnhancedFld = Name_allTests_Nuclei + '/' + TestName + '/'
    Dir_AllTests_Thalamus_EnhancedFld = Name_allTests_Thalamus + '/' + TestName + '/'

    subFolders = []
    subFlds = os.listdir(Dir_AllTests_Nuclei_EnhancedFld)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    for sFi in range(len(subFolders)):
        try:
            Dir_Prior_NucleiSamples = Dir_Prior +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
            Dir_Prior_ThalamusSamples = Dir_Prior +  subFolders[sFi] + ManualDir +'1-THALAMUS' + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

            Dir_NucleiTestSamples  = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Test/'
            Dir_NucleiTrainSamples = Dir_AllTests_Nuclei_EnhancedFld + subFolders[sFi] + '/Train/'
            Dir_NucleiModelOut = Dir_NucleiTrainSamples + 'model/'
            Dir_ResultsOut   = Dir_NucleiTestSamples  + 'Results/'

            Dir_ThalamusTestSamples  = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Test/'
            Dir_ThalamusTrainSamples = Dir_AllTests_Thalamus_EnhancedFld + subFolders[sFi] + '/Train/'
            Dir_ThalamusModelOut = Dir_ThalamusTrainSamples + 'model/'


            if os.path.isfile(Dir_ResultsOut + 'DiceCoefficient__.txt'):
                print('*---  Already Done:   ' + Dir_NucleiModelOut + '  ---*')
                continue
            else:
                print('*---  Not Done:   ' + Dir_NucleiModelOut + '  ---*')
                TrainData = image_util.ImageDataProvider(Dir_NucleiTrainSamples + "*.tif")

                logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
                # config = tf.ConfigProto()
                # config.gpu_options.allow_growth = True
                # config.gpu_options.per_process_gpu_memory_fraction = 0.4
                # unet.config = config

                net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

                # trainer = unet.Trainer(net)
                # if gpuNum != 'nan':
                #     path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=200, epochs=150, display_step=500, GPU_Num=gpuNum) #  , cost="dice_coefficient" restore=True
                # else:
                #     path = trainer.train(TrainData, Dir_NucleiModelOut, training_iters=200, epochs=150, display_step=500) #  , cost="dice_coefficient" restore=True

                NucleiOrigSeg = nib.load(Dir_Prior_NucleiSamples)
                ThalamusOrigSeg = nib.load(Dir_Prior_ThalamusSamples)

                CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

                padSize = 90
                MultByThalamusFlag = 0
                [Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical] = TestData3(net , MultByThalamusFlag, Dir_NucleiTestSamples , Dir_NucleiTrainSamples , ThalamusOrigSeg , NucleiOrigSeg , subFolders[sFi], CropDimensions , padSize , Dir_ThalamusTestSamples , Dir_ThalamusModelOut , NucleusName , SliceNumbers , gpuNum)
        except:
            print('-------------------------------------------------------')
            print('subFolders: ',subFolders[sFi])
