from skimage import filters
import nibabel as nib
import numpy as np

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

name = 'vimp2_ctrl_920_07122013_SW'
# name = 'vimp2_ctrl_921_07122013_MP'

dir1 = '/media/artin/D0E2340CE233F576/Results_Epch150_Th02_' + name + '/' + name + '_9-LGN.nii.gz'
dir2 = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/Manual_Delineation_Sanitized_Full/' + name + '/Manual_Delineation_Sanitized/9-LGN_deformed.nii.gz'

pred = nib.load(dir1).get_data()
mask = nib.load(dir2).get_data()


Thresh = max(filters.threshold_otsu(pred),0.2)
print(Thresh)
DiceCoefficientCalculator(pred > 0.31 , mask)
