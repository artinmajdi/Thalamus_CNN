from tf_unet import unet, util, image_util
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


SliceNumbers = range(115,145)
dir = '/media/artin/dataLocal1/dataThalamus/AllTests/oldDataset_newMethod/CNN6_VLP_2D_SanitizedNN/Test_WMnMPRAGE_bias_corr_Deformed/vimp2_ANON724_03272013/Train/Slice_135/'
image = nib.load('/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2/vimp2_ANON724_03272013/Manual_Delineation_Sanitized/6-VLP_deformed.nii.gz')
image = image.get_data()[...,135]
s = 1 - image
np.unique(image + s)
plt.imshow(image,cmap='gray')

TestData = image_util.ImageDataProvider( dir + '/*.tif',shuffle_data=False)

data,label = TestData(10)
label.shape
label[:,:,np.newaxis,:,:].shape
np.unique(label)

v = np.transpose(label,[1,2,3,0])
v.shape

im = label[0,...]
A = im[...,1]
B = 1 - A

a = np.append(B[...,np.newaxis],A[...,np.newaxis],axis=2)
