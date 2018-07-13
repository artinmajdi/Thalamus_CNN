import numpy as np
import matplotlib.pyplot as plt
# import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter
import xlwt



def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef


SliceNumbers = range(107,140)

nulcieFull = [
'1-THALAMUS_deformed.nii.gz' ,
'2-AV_deformed.nii.gz' ,
'4-VA_deformed.nii.gz' ,
'5-VLa_deformed.nii.gz' ,
'6-VLP_deformed.nii.gz' ,
'7-VPL_deformed.nii.gz',
'8-Pul_deformed.nii.gz',
'9-LGN_deformed.nii.gz' ,
'10-MGN_deformed.nii.gz',
'11-CM_deformed.nii.gz'	,
'12-MD-Pf_deformed.nii.gz' ,
'13-Hb_deformed.nii.gz'] #  , '4567-VL_deformed.nii.gz'


Dir_oldPriors = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/Manual_Delineation_Sanitized_Full'
Dir_newPriors = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/NewPriors/test/7T_MS'
Dir_save = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/NewPriors/test'

results = xlwt.Workbook(encoding="utf-8")
sheet = results.add_sheet('Majority Voting')

lst_Old = os.listdir(Dir_oldPriors)
lst_newTemp = os.listdir(Dir_newPriors)

lst_new = []
for i in range(len(lst_newTemp)):
    if lst_newTemp[i][:5] == 'vimp2':
        lst_new.append(lst_newTemp[i])

sheet.write(0, len(lst_new)+1, 'Average')

for n in range(len(nulcieFull)):

    nuclei = nulcieFull[n]
    sheet.write(n+1, 0, nuclei.split('_deformed.nii.gz')[0])

    print('nuclei: ' + nuclei)
    for l in range(len(lst_Old)):

        A = nib.load(Dir_oldPriors + '/' +  lst_Old[l] + '/Manual_Delineation_Sanitized'  + '/' + nuclei)
        A = A.get_data()
        A = A[50:198,130:278,SliceNumbers]

        if l == 0:
            Full_Priors_old = A[...,np.newaxis]
        else:
            Full_Priors_old = np.append(Full_Priors_old,A[...,np.newaxis],axis=3)

    MW_Seg = np.sum(Full_Priors_old,axis=3)
    MW_Seg = MW_Seg > 10

    Dice = np.zeros(len(lst_new))
    for l in range(len(lst_new)):
        A = nib.load(Dir_newPriors + '/' +  lst_new[l] + '/Manual_Delineation_Sanitized'  + '/' + nuclei)
        A = A.get_data()
        A = A[50:198,130:278,SliceNumbers]

        Dice[l] = DiceCoefficientCalculator(A,MW_Seg)
        if n == 0:
            sheet.write(0, l+1, lst_new[l].split('vimp2_')[1])
        sheet.write(n+1, l+1,  Dice[l])


    Dice = np.append(Dice , np.mean(Dice))

    sheet.write(n+1, len(lst_new)+1, Dice[len(lst_new)])
    np.savetxt(Dir_save + '/DiceCoefficient_' + nuclei.split('_deformed.nii.gz')[0] + '.txt', 100*Dice, fmt='%2.1f')

    results.save(Dir_save + '/DiceCoefficient.xls')
