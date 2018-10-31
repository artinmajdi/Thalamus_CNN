import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import tifffile



dir = '/media/data1/artin/thomas/origtemplate.nii.gz'
im = nib.load(dir)

mask = np.zeros(im.shape)

gap = 20
gap2 = 2
mask[92-gap:126+gap,164-gap2:226+gap2, 105-gap:142+gap] = 1

maskF2 = nib.Nifti1Image(mask,im.affine)
maskF2.get_header = im.header
nib.save(maskF2,'/media/data1/artin/code/Thalamus_CNN/notMainCodes/RigidRegistration/MyCrop_Template2_Gap20.nii.gz' )


# def funcNormalize(im):
#     # return (im-im.mean())/im.std()
#     im = np.float32(im)
#     return ( im-im.min() )/( im.max() - im.min() )
#
# def cropDimensions(im , mask , CropMask):
#     ss = np.sum(CropMask,axis=2)
#     c1 = np.where(np.sum(ss,axis=1) > 10)[0]
#     c2 = np.where(np.sum(ss,axis=0) > 10)[0]
#     ss = np.sum(CropMask,axis=1)
#     c3 = np.where(np.sum(ss,axis=0) > 10)[0]
#
#     d1 = [  c1[0] , c1[ c1.shape[0]-1 ]  ]
#     d2 = [  c2[0] , c2[ c2.shape[0]-1 ]  ]
#     SN = [  c3[0] , c3[ c3.shape[0]-1 ]  ]
#     SliceNumbers = range(SN[0],SN[1])
#
#     im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers] # Params['SliceNumbers']]
#     mask = mask[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers] # Params['SliceNumbers']]
#
#     return im , mask
#
# dir = '/media/data1/artin/vimp2_0699_04302014'
#
# mask    = nib.load(dir + '/' + 'Manual_Delineation_Sanitized/' + '1-THALAMUS' + '.nii.gz').get_data()
# imF       = nib.load(dir + '/' + 'WMnMPRAGE_bias_corr.nii.gz' )
# CropMask = nib.load(dir + '/' + 'MyCrop.nii.gz').get_data()
#
# Thalamus = nib.load('/media/data1/artin/Results/vimp2_1530_04232015_1-THALAMUS_Logical.nii.gz')
#
# ss = np.sum(Thalamus.get_data(),axis=2)
# c1 = np.where(np.sum(ss,axis=1) > 1)[0]
# c2 = np.where(np.sum(ss,axis=0) > 1)[0]
# ss = np.sum(Thalamus.get_data(),axis=1)
# c3 = np.where(np.sum(ss,axis=0) > 1)[0]
#
# d1 = [  c1[0] , c1[ c1.shape[0]-1 ]  ]
# d2 = [  c2[0] , c2[ c2.shape[0]-1 ]  ]
# SN = [  c3[0] , c3[ c3.shape[0]-1 ]  ]
# SliceNumbers = range(SN[0],SN[1])
#
# mask = np.zeros(CropMask.shape)
# gap = 5
# mask[ d1[0]-gap:d1[1]+gap,d2[0]-gap:d2[1]+gap,SliceNumbers] = 1
#
# maskF2 = nib.Nifti1Image(mask,Thalamus.affine)
# maskF2.get_header = Thalamus.header
# nib.save(maskF2,'/media/data1/artin/Results/mask_from_Thalamus.nii.gz' )
# im = funcNormalize( imF.get_data() )
#
# im , mask = cropDimensions(im , mask , CropMask)
#
#
# for i in range(mask.shape[2]):
#     mask[...,i] = np.fliplr(mask[...,i])
#     im[...,i] = np.fliplr(im[...,i])
#
#
# sz = mask.shape
# df = 238 - sz[0]
# p1 = [int(df/2) , df - int(df/2)]
#
# df = 238 - sz[1]
# p2 = [int(df/2) , df - int(df/2)]
#
# imD_padded = np.pad(im,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )
# maskD_padded = np.pad(mask,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )
#
# maskD_padded.shape
#
# slc = 80
# fig , ax = plt.subplots(1,3)
# ax[0].imshow(imD[...,slc],cmap='gray')
# ax[1].imshow(maskD[...,slc],cmap='gray')
# ax[2].imshow(CropMask[...,slc],cmap='gray')
# plt.show()
#
#
#
#
# a = np.zeros((238,238,46))
# b = np.zeros((238,238,49))
#
# np.concatenate((a,b),axis=2).shape
#
#
#
#
# a = [5,4,1,-2]
