import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import tifffile

im = nib.load('/media/data1/artin/thomas/origtemplate.nii.gz')
# mask = nib.load('/media/data1/artin/code/Thalamus_CNN/notMainCodes/RigidRegistration/1-THALAMUS.nii.gz').get_data()


im.shape
# mask.shape


mask = np.zeros(im.shape)
mask.shape

cp = [[72,146],[154,236],[85,162]]

mask[cp[0][0]:cp[0][1] , cp[1][0]:cp[1][1] , cp[2][0]:cp[2][1] ] = 1

output = nib.Nifti1Image(mask,im.affine)
output.get_header = im.header
nib.save(output , '/media/data1/artin/mask.nii.gz')


fig , ax = plt.subplots(1,2)
ax[0].imshow(im[:,200,:],cmap='gray')
ax[1].imshow(mask[:,200,:],cmap='gray')
plt.show()


# 3rd: 105 , 142
# 2nd:  164 , 226
# 1st 92 , 126
