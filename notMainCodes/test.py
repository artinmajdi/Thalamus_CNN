from skimage import filters
import nibabel as nib
import numpy as np

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

name = 'vimp2_824_05212013_JS'

dir1 = '/media/artin/D0E2340CE233F576/Results/' + name + '_10-MGN.nii.gz'
dir2 = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/Manual_Delineation_Sanitized_Full/' + name + '/Manual_Delineation_Sanitized/9-LGN_deformed.nii.gz'

pred = nib.load(dir1).get_data()
mask = nib.load(dir2).get_data()


Thresh = max(filters.threshold_otsu(pred),0.2)
print(Thresh)
DiceCoefficientCalculator(pred > 0.31 , mask)
