from tf_unet import unet, util, image_util
import nibabel as nib
import numpy as np

dir2 = '/media/artin/dataLocal1/dataThalamus/AllTests/oldDataset_newMethod/CNN6_VLP_2D_SanitizedNN/Test_WMnMPRAGE_bias_corr_Deformed/vimp2_ANON724_03272013/Train/Slice_117/'

TrainData = image_util.ImageDataProvider(dir2 + "*.tif")
data , label = TrainData(10)
print(data.shape)


TestImage = nib.load('/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2/vimp2_819_05172013_DS/WMnMPRAGE_bias_corr_Deformed.nii.gz').get_data()

a = TestImage[...,:10]

print(a.shape)

b = np.transpose(a,[2,0,1])
b.shape
