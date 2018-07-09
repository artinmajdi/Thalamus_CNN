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


gpuNum = '4' # nan'

# 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
# 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
# 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz

for ind in [1,4,6,8,12]:
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

    Directory_main = '/array/hdd/msmajdi/Tests/Thalamus_CNN/newDataset/' #
    # Directory_main = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/'
    Directory_Nuclei_Full = Directory_main + NeucleusFolder
    Directory_Thalamus_Full = Directory_main + 'CNN1_THALAMUS_2D_SanitizedNN'

    #priorDir = Directory_main + 'Manual_Delineation_Sanitized_Full/'
    # priorDir =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'
    Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS/'

    # subFolders = list(['vimp2_915_07112013_LC', 'vimp2_943_07242013_PA' ,'vimp2_964_08092013_TG'])

    TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
    subFolders = os.listdir(Directory_Nuclei)

    listt = []
    for i in range(len(subFolders)):
        if subFolders[i][:4] == 'vimp':
            listt.append(subFolders[i])

    subFolders = listt

    for reslt in ['Results']: #  , 'Results_MultByManualThalamus'

        print(reslt + '--------------->>>>>---------------')
        Dice = np.zeros((len(subFolders), len(A)+1))

        Directory_Nuclei_Label = priorDir +  subFolders[0] + ManualDir + NucleusName + '_deformed.nii.gz'
        Label = nib.load(Directory_Nuclei_Label)
        Label = Label.get_data()
        sz = Label.shape

        # subFolders = ['vimp2_ctrl_921_07122013_MP']
        for sFi in range(len(subFolders)):
            print(str(sFi) + ': ' + str(subFolders[sFi]))
            # sFi = 1
            Directory_Nuclei_Label = priorDir +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'
            Label = nib.load(Directory_Nuclei_Label)
            Label = Label.get_data()

            Prediction_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
            Er = 0
            for ii in range(len(A)):
                # ii = 1
                if ii == 0:
                    TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
                else:
                    TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

                Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
                Directory_Thalamus = Directory_Thalamus_Full + '/' + TestName + '/'


                Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/' + reslt + '/'
                Dirr = Directory_Nuclei_Test + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz'

                try:
                    PredictionF = nib.load(Dirr)
                    Prediction = PredictionF.get_data()

                    # print('k')
                    # Thresh = max(filters.threshold_otsu(Prediction),0.2)
                    Dice[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)

                    Prediction_full[:,:,:,ii] = Prediction > 0.5
                    np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python'+reslt+'.txt',100*Dice, fmt='%2.1f')
                except:
                    Er = Er + 1

            # print(len(A))
            Prediction2 = np.sum(Prediction_full,axis=3)
            predictionMV = np.zeros(Prediction2.shape)
            predictionMV[:,:,:] = Prediction2 > 3-Er

            Dice[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV)
            np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python'+reslt+'.txt',100*Dice, fmt='%2.1f')


            Header = PredictionF.header
            Affine = PredictionF.affine

            Directory_Nuclei_Full2 = Directory_Nuclei_Full + '/MajorityVoting_Results'
            try:
                os.stat(Directory_Nuclei_Full2)
            except:
                os.makedirs(Directory_Nuclei_Full2)

            Directory_Nuclei_Full3 = Directory_Nuclei_Full2 + '/' + subFolders[sFi]
            try:
                os.stat(Directory_Nuclei_Full3)
            except:
                os.makedirs(Directory_Nuclei_Full3)

            #print(type(predictionMV))
            #print(Affine)
            predictionMV_nifti = nib.Nifti1Image(predictionMV,Affine)
            predictionMV_nifti.get_header = Header
            AA =  Directory_Nuclei_Full3 + '/' + subFolders[sFi] + '_' + NucleusName + '.nii.gz'
            #print(AA)
            nib.save(predictionMV_nifti ,AA)

        # print(Dice)
        Dice2 = np.zeros((len(subFolders)+1, len(A)+1))
        Dice2[:len(subFolders),:] = Dice
        Dice2[len(subFolders),:] = np.mean(Dice,axis=0)
        np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python'+reslt+'.txt',100*Dice2, fmt='%2.1f')
        # np.savetxt(Directory_Nuclei_Full + '/subFolders_Python.txt',subFolders)

        with open(Directory_Nuclei_Full + "/subFoldersList_Python"+reslt+".txt" ,"wb") as fp:
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
