import os
import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import pickle

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

def initialDirectories(ind = 1, mode = 'oldDataset'):

    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 4:
        NucleusName = '4567-VL'
    elif ind == 6:
        NucleusName = '6-VLP'
    elif ind == 8:
        NucleusName = '8-Pul'
    elif ind == 10:
        NucleusName = '10-MGN'
    elif ind == 12:
        NucleusName = '12-MD-Pf'


    if mode == 'oldDatasetV2':
        NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    elif mode == 'oldDataset':
        NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    elif mode == 'newDataset':
        NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
        ThalamusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'

    if mode == 'localMachine':
        Dir_AllTests = '/media/artin-laptop/D0E2340CE233F5761/Thalamus_Segmentation/Data/'
        Dir_Prior = ''

    elif mode == 'oldDataset':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'

    elif mode == 'oldDatasetV2':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/oldDatasetV2/'
        Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'

    elif mode == 'newDataset':
        Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/newDataset/'
        Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS/'

    return NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior

def subFolderList(Dir_AllTests_nucleiFld):
    TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    Dir_AllTests_nucleiFld_Ehd = Dir_AllTests_nucleiFld + '/' + TestName + '/'
    subFolders = os.listdir(Dir_AllTests_nucleiFld_Ehd)

    listt = []
    for i in range(len(subFolders)):
        if subFolders[i][:4] == 'vimp':
            listt.append(subFolders[i])

    return listt

gpuNum = '4' # nan'

# 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
# 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
# 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz
for ind in [1,6,8,10,12]:

    mode = 'newDataset'
    NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior = initialDirectories(ind , mode)


    ManualDir = '/Manual_Delineation_Sanitized/' #ManualDelineation

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]] # [4,3],
    SliceNumbers = range(107,140)


    # Dir_AllTests = '/array/hdd/msmajdi/Tests/Thalamus_CNN/newDataset/' #
    # Dir_AllTests = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/'
    Dir_AllTests_nucleiFld = Dir_AllTests + NeucleusFolder
    Dir_AllTests_ThalamusFld = Dir_AllTests + 'CNN1_THALAMUS_2D_SanitizedNN'
    Dir_SaveMWFld  = Dir_AllTests + 'Folder_MajorityWoting/'
    try:
        os.stat(Dir_SaveMWFld)
    except:
        os.makedirs(Dir_SaveMWFld)

    #Dir_Prior = Dir_AllTests + 'Manual_Delineation_Sanitized_Full/'
    # Dir_Prior =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'
    # Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS/'
    # subFolders = list(['vimp2_915_07112013_LC', 'vimp2_943_07242013_PA' ,'vimp2_964_08092013_TG'])

    subFolders = subFolderList(Dir_AllTests_nucleiFld)

    for reslt in ['Results_momentum']: # 'Results' ,  , 'Results_MultByManualThalamus'

        print(reslt + '--------------->>>>>---------------')
        Dice = np.zeros((len(subFolders), len(A)+1))

        Directory_Nuclei_Label = Dir_Prior +  subFolders[0] + ManualDir + NucleusName + '_deformed.nii.gz'
        Label = nib.load(Directory_Nuclei_Label)
        Label = Label.get_data()
        sz = Label.shape

        # subFolders = ['vimp2_ctrl_921_07122013_MP']
        for sFi in range(len(subFolders)):
            print(str(sFi) + ': ' + str(subFolders[sFi]))
            # sFi = 1
            Directory_Nuclei_Label = Dir_Prior +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'
            Label = nib.load(Directory_Nuclei_Label)
            Label = Label.get_data()

            Dir_save = Dir_SaveMWFld + NeucleusFolder + '/' + subFolders[sFi] + '/'
            try:
                os.stat(Dir_save)
            except:
                os.makedirs(Dir_save)
                
            Prediction_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
            Er = 0
            for ii in range(len(A)):
                # ii = 1
                if ii == 0:
                    TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
                else:
                    TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

                Dir_AllTests_nucleiFld_Ehd = Dir_AllTests_nucleiFld + '/' + TestName + '/'
                Dir_AllTests_ThalamusFld_Ehd = Dir_AllTests_ThalamusFld + '/' + TestName + '/'


                Directory_Nuclei_Test  = Dir_AllTests_nucleiFld_Ehd + subFolders[sFi] + '/Test/' + reslt + '/'
                Dir_NucleiPredSample = Directory_Nuclei_Test + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz'

                try:
                    PredictionF = nib.load(Dir_NucleiPredSample)
                    Prediction = PredictionF.get_data()

                    # print('k')
                    # Thresh = max(filters.threshold_otsu(Prediction),0.2)
                    Dice[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)

                    Prediction_full[:,:,:,ii] = Prediction > 0.5
                    np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')
                except:
                    Er = Er + 1

            # print(len(A))
            Prediction2 = np.sum(Prediction_full,axis=3)
            predictionMV = np.zeros(Prediction2.shape)
            predictionMV[:,:,:] = Prediction2 > 2-Er

            Dice[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV)
            np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')


            Header = PredictionF.header
            Affine = PredictionF.affine

            # Dir_AllTests_nucleiFld2 = Dir_AllTests_nucleiFld + '/Folder_MajorityVoting_Results_Atom'




            predictionMV_nifti = nib.Nifti1Image(predictionMV,Affine)
            predictionMV_nifti.get_header = Header
            AA =  Dir_save + subFolders[sFi] + '_' + NucleusName + '_MW.nii.gz'
            nib.save(predictionMV_nifti ,AA)

        Dice2 = np.zeros((len(subFolders)+1, len(A)+1))
        Dice2[:len(subFolders),:] = Dice
        Dice2[len(subFolders),:] = np.mean(Dice,axis=0)
        np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/DiceMW_' + reslt + '.txt',100*Dice2, fmt='%2.1f')
        # np.savetxt(Dir_AllTests_nucleiFld + '/subFolders_Python.txt',subFolders)

        with open(Dir_SaveMWFld + NeucleusFolder + "/subFoldersList_MW_"+reslt+".txt" ,"wb") as fp:
            pickle.dump(subFolders,fp)

    # a = np.random.random((3,4)) > 0.5
    # b = np.random.random((3,4)) > 0.5
    #
    # sz = a.shape
    # c = np.zeros((sz[0],sz[1],2))
    # c[:,:,0] = a
    # c[:,:,1] = b
    # print(a)
    # print(b)
    # print(np.sum(c,axis=2)>1)
