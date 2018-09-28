from tf_unet import unet, util, image_util
import nibabel as nib
import numpy as np


dir2 = '/media/data1/artin/thomas/priors/20priors/819_05172013_DS/'
name = 'WMnMPRAGE_bias_corr.nii.gz'



TrainData = image_util.ImageDataProvider(dir2 + "*.tif")
data , label = TrainData(10)
print(data.shape)
