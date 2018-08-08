from tf_unet import unet, util, image_util
import numpy as np
import matplotlib.pyplot as plt

dir = '/media/artin/dataLocal1/dataThalamus/AllTests/oldDataset_newMethod/CNN6_VLP_2D_SanitizedNN/Test_WMnMPRAGE_bias_corr_Deformed/vimp2_ANON724_03272013/Train/Slice_135/'
TestData = image_util.ImageDataProvider( dir + '/*.tif',shuffle_data=False)

data,label = TestData(10)
label.shape
label[:,:,np.newaxis,:,:].shape


v = np.transpose(label,[1,2,3,0])
v.shape

im = label[0,...]
A = im[...,1]
B = 1 - A

a = np.append(B[...,np.newaxis],A[...,np.newaxis],axis=2)
