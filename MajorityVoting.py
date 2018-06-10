import os
import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters


def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef


NeucleusFolder = 'CNN8_Pul_2D_SanitizedNN'  #  'CNN1_THALAMUS_2D_SanitizedNN' #'  CNN4567_VL_2D_SanitizedNN
NucleusName = '8-Pul'  # '1-THALAMUS' #'6-VLP' #
ManualDir = '/Manual_Delineation_Sanitized/' #ManualDelineation

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
SliceNumbers = range(107,140)

Directory_main = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' #
# Directory_main = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/'
Directory_Nuclei_Full = Directory_main + NeucleusFolder
Directory_Thalamus_Full = Directory_main + 'CNN1_THALAMUS_2D_SanitizedNN'

#priorDir = Directory_main + 'Manual_Delineation_Sanitized_Full/'
priorDir =  '/array/hdd/msmajdi/data/priors_forCNN_Ver2/'

# subFolders = list(['vimp2_915_07112013_LC', 'vimp2_943_07242013_PA' ,'vimp2_964_08092013_TG'])

TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
subFolders = os.listdir(Directory_Nuclei)

Dice = np.zeros((len(subFolders), len(A)+1))

Directory_Nuclei_Label = priorDir +  subFolders[0] + ManualDir + NucleusName + '_deformed.nii.gz'
Label = nib.load(Directory_Nuclei_Label)
Label = Label.get_data()
sz = Label.shape



for sFi in range(len(subFolders)):
    print(str(sFi) + ': ' + str(subFolders[sFi]))
    # sFi = 1
    Directory_Nuclei_Label = priorDir +  subFolders[sFi] + ManualDir + NucleusName + '_deformed.nii.gz'
    Label = nib.load(Directory_Nuclei_Label)
    Label = Label.get_data()

    Label_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
    for ii in range(len(A)):
        # ii = 1
        if ii == 0:
            TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
        else:
            TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

        Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
        Directory_Thalamus = Directory_Thalamus_Full + '/' + TestName + '/'


        Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/Results/'
        Dirr = Directory_Nuclei_Test + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz'
        Prediction = nib.load(Dirr)
        Prediction = Prediction.get_data()


        # Thresh = max(filters.threshold_otsu(Prediction),0.2)
        Dice[sFi,ii] = DiceCoefficientCalculator(Label,Prediction > 0.5)

        Label_full[:,:,:,ii] = Label
        np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient.txt',Dice)

    Label2 = np.sum(Label_full,axis=3) > 3
    Dice[sFi,ii+1] = DiceCoefficientCalculator(Label2,Prediction > 0.5)
    np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient.txt',Dice)


















print('llllllllllll')
