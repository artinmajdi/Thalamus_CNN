import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

dir2 = '/media/data1/artin/thomas/priors/ET/7T'

sub = os.listdir(dir2)
sub

# i = 1
# im = nib.load(dir2 + '/' + sub[i] + '/Manual_Delineation_Sanitized/14-MTT_deformed.nii.gz')

# imm = im.get_data()[...,:100]
# b = np.sum(imm,axis=1)
# b = np.sum(b,axis=0)
# b.shape

# plt.imshow(imm[...,122],cmap='gray')
# plt.show()

BB = np.zeros((256,len(sub)))
for i in range(len(sub)):
    print('i',i)
    im = nib.load(dir2 + '/' + sub[i] + '/Manual_Delineation_Sanitized/5-VLa_deformed.nii.gz').get_data()
    b = np.sum(im,axis=1)
    b = np.sum(b,axis=0)
    BB[:,i] = b

np.where(b>0)
