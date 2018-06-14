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

Flag_Result = 1
Flag_Result_Mult = 1

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

listt = []
for i in range(len(subFolders)):
    if subFolders[i][:4] == 'vimp':
        listt.append(subFolders[i])

subFolders = listt

Dice = np.zeros((len(subFolders), len(A)+1))
Dice_Mult = np.zeros((len(subFolders), len(A)+1))


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

    if Flag_Result == 1:
        Prediction_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
        Er  = 0

    if Flag_Result_Mult == 1:
        Prediction_full_Mult = np.zeros((sz[0],sz[1],sz[2],len(A)))
        Er2 = 0

    
    for ii in range(len(A)):
        # ii = 1
        if ii == 0:
            TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
        else:
            TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

        Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
        Directory_Thalamus = Directory_Thalamus_Full + '/' + TestName + '/'


        Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/Results/'
        Directory_Nuclei_Test_Mult  = Directory_Nuclei + subFolders[sFi] + '/Test/Results_MultByManualThalamus/'
        Dirr = Directory_Nuclei_Test + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz'
        Dirr_Mult = Directory_Nuclei_Test_Mult + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz'

        print('sFi: ',str(sFi),' Aii: ', str(ii))
        if Flag_Result == 1:
            try:
                PredictionF = nib.load(Dirr)
                Prediction = PredictionF.get_data()
                Dice[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)
                Prediction_full[:,:,:,ii] = Prediction > 0.5
                np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python.txt',100*Dice, fmt='%2.1f')
            except:
                print('Exceptioon   sFi: ',str(sFi),' Aii: ', str(ii))
                Er = Er + 1

        if Flag_Result_Mult == 1:
            try:
                PredictionF = nib.load(Dirr_Mult)
                Prediction = PredictionF.get_data()
                Dice_Mult[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)
                Prediction_full_Mult[:,:,:,ii] = Prediction > 0.5
                np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python_Mult.txt',100*Dice_Mult, fmt='%2.1f')
            except:
                print('Exceptioon   sFi: ',str(sFi),' Aii: ', str(ii))
                Er2 = Er2 + 1


    if Flag_Result == 1:
        Prediction2 = np.sum(Prediction_full,axis=3)
        predictionMV = np.zeros(Prediction2.shape)
        predictionMV[:,:,:] = Prediction2 > 3-Er

    if Flag_Result_Mult == 1:
        Prediction2 = np.sum(Prediction_full_Mult,axis=3)
        predictionMV_Mult = np.zeros(Prediction2.shape)
        predictionMV_Mult[:,:,:] = Prediction2 > 3-Er2

    if Flag_Result == 1:
        Dice[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV)
        np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python.txt',100*Dice, fmt='%2.1f')

    if Flag_Result_Mult == 1:
        Dice_Mult[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV_Mult)
        np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python_Mult.txt',100*Dice_Mult, fmt='%2.1f')


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

    if Flag_Result == 1:
        predictionMV_nifti = nib.Nifti1Image(predictionMV,Affine)
        predictionMV_nifti.get_header = Header
        AA =  Directory_Nuclei_Full3 + '/' + subFolders[sFi] + '_' + NucleusName + '.nii.gz'
        nib.save(predictionMV_nifti ,AA)

    if Flag_Result_Mult == 1:
        predictionMV_nifti = nib.Nifti1Image(predictionMV_Mult,Affine)
        predictionMV_nifti.get_header = Header
        AA =  Directory_Nuclei_Full3 + '/' + subFolders[sFi] + '_' + NucleusName + '_Mult.nii.gz'
        nib.save(predictionMV_nifti ,AA)


if Flag_Result == 1:
    Dice2 = np.zeros((len(subFolders)+1, len(A)+1))
    Dice2[:len(subFolders),:] = Dice
    Dice2[len(subFolders),:] = np.mean(Dice,axis=0)
    np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python.txt',100*Dice2, fmt='%2.1f')

if Flag_Result_Mult == 1:
    Dice2_Mult = np.zeros((len(subFolders)+1, len(A)+1))
    Dice2_Mult[:len(subFolders),:] = Dice_Mult
    Dice2_Mult[len(subFolders),:] = np.mean(Dice_Mult,axis=0)
    np.savetxt(Directory_Nuclei_Full + '/DiceCoefficient_Python_Mult.txt',100*Dice2_Mult, fmt='%2.1f')


# np.savetxt(Directory_Nuclei_Full + '/subFolders_Python.txt',subFolders)

with open(Directory_Nuclei_Full + "/subFoldersList_Python.txt" ,"wb") as fp:
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














# print('llllllllllll')
