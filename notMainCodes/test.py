import os
import nibabel as nib
import numpy as np


dir2 = '/media/data1/artin/thomas/priors/20priors'

sub = os.listdir(dir2)
sub
i = 1
im = nib.load(dir2 + '/' + sub[i] + '/WMnMPRAGEdeformed.nii.gz').get_data()




for i in range(len(sub)):
    im = nib.load(dir2 + '/' + sub[i] + '/WMnMPRAGEdeformed.nii.gz'


TrainData = image_util.ImageDataProvider(dir2 + "*.tif")
data , label = TrainData(10)
print(data.shape)
