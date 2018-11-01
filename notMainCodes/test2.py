import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import tifffile
from scipy.misc import imrotate
from scipy import ndimage

range(1,10,2)

def MyTranspose(dir,order):
    im = nib.load(dir)
    im2   = np.transpose(im.get_data(),[0,2,1])
    # im2 = ndimage.zoom(im2,(1,1,0.5),order=order)
    # im2 = im2[...,range(0,im.shape[2],2)]

    im2 = nib.Nifti1Image(im2,im.affine)
    im2.get_header = im.header
    dir2 = dir.split('.nii.gz')[0] + '_2.nii.gz'
    nib.save(im2,dir2)

    return im2


im = MyTranspose('/media/data1/artin/vimp2_Test_ANON724_03272013/WMnMPRAGE_bias_corr.nii.gz',3)
crop = MyTranspose('/media/data1/artin/vimp2_Test_ANON724_03272013/MyCrop2_Gap20.nii.gz',0)
mask = MyTranspose('/media/data1/artin/vimp2_Test_ANON724_03272013/Manual_Delineation_Sanitized/1-THALAMUS.nii.gz',0)


im = nib.load('/media/data1/artin/vimp2_Test_964_08092013_TG/WMnMPRAGE_bias_corr.nii.gz')
im2 = nib.load('/media/data1/artin/thomas/priors/Unlabeled/vimp2_804_06042014
               /WMnMPRAGE_bias_corr.nii.gz')
im2.shape
fig,axs = plt.subplots(1,2)
axs[0].imshow(im.get_data()[...,200],cmap='gray')
axs[1].imshow(np.fliplr(im2.get_data()[...,100]),cmap='gray')
plt.show()



do_I_want_Upsampling = 0

def funcNormalize(im):
    # return (im-im.mean())/im.std()
    im = np.float32(im)
    return ( im-im.min() )/( im.max() - im.min() )

def funcCropping(im , mask , CropMask2):
    ss = np.sum(CropMask2,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]
    ss = np.sum(CropMask2,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    d1 = [  c1[0] , c1[ c1.shape[0]-1 ]  ]
    d2 = [  c2[0] , c2[ c2.shape[0]-1 ]  ]
    SN = [  c3[0] , c3[ c3.shape[0]-1 ]  ]
    SliceNumbers = range(SN[0],SN[1])

    im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers] # Params['SliceNumbers']]
    mask = mask[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers] # Params['SliceNumbers']]

    return im , mask

def funcFlipLR_Upsampling(Params, im , mask):
    if 'Unlabeled' in Params['dataset']:

        for i in range(mask.shape[2]):
            im[...,i] = np.fliplr(im[...,i])
            mask[...,i] = np.fliplr(mask[...,i])

        if do_I_want_Upsampling == 1:
            mask = ndimage.zoom(mask,(1,1,2),order=0)
            im = ndimage.zoom(im,(1,1,2),order=3)
    else:
        im   = np.transpose(im,[0,2,1])
        mask = np.transpose(mask,[0,2,1])

        if im.shape[2] == 200:
            im = ndimage.zoom(im,(1,1,2),order=3)
            mask = ndimage.zoom(mask,(1,1,2),order=0)

    return im , mask

def funcPadding(im, mask):
    sz = mask.shape
    df = 238 - sz[0]
    p1 = [int(df/2) , df - int(df/2)]

    df = 238 - sz[1]
    p2 = [int(df/2) , df - int(df/2)]

    im = np.pad(im,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )
    mask = np.pad(mask,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )

    return im , mask

def funcRotating(im, mask, CropMask):

    angle = np.random.random_integers(45)

    for i in range(im.shape[2]):
        im[...,i] = imrotate(im[...,i],angle)
        mask[...,i] = imrotate(mask[...,i],angle)
        CropMask[...,i] = imrotate(CropMask[...,i],angle)

    return im, mask, CropMask

def funcShifting(im, mask):

    shftX = np.random.random_integers(15)
    shftY = np.random.random_integers(15)

    im = np.roll(im,shftX,axis=0)
    im = np.roll(im,shftY,axis=1)

    mask = np.roll(mask,shftX,axis=0)
    mask = np.roll(mask,shftY,axis=1)

    return im, mask


dir = '/media/data1/artin/vimp2_1621_10162015'
Params = {'dataset':'Unlabeled'}

im2   = nib.load(dir + '/' + 'WMnMPRAGE_bias_corr.nii.gz' ).get_data()
mask2 = nib.load(dir + '/' + 'Manual_Delineation_Sanitized/' + '1-THALAMUS' + '.nii.gz').get_data()
CropMask2 = nib.load(dir + '/' + 'MyCropMask2_Gap20.nii.gz' ).get_data()

im2 = funcNormalize(im2)

im = im2.copy()
mask = mask2.copy()
CropMask = CropMask2.copy()

im2, mask2 = funcCropping(im2, mask2, CropMask2)
im2, mask2 = funcFlipLR_Upsampling(Params, im2 , mask2)
imP2, maskP2 = funcPadding(im2, mask2)

im, mask, CropMask = funcRotating(im, mask, CropMask)
im, mask = funcCropping(im, mask, CropMask)
im, mask = funcFlipLR_Upsampling(Params, im , mask)
imP, maskP = funcPadding(im, mask)
imP, maskP = funcShifting(imP, maskP)

slc = 20
fig, axs = plt.subplots(2,2)
axs[0,0].imshow(imP2[...,slc],cmap='gray')
axs[0,1].imshow(imP[...,slc],cmap='gray')
axs[1,0].imshow(maskP2[...,slc],cmap='gray')
axs[1,1].imshow(maskP[...,slc],cmap='gray')
plt.show()











# maskF2 = nib.Nifti1Image(mask,Thalamus.affine)
# maskF2.get_header = Thalamus.header
# nib.save(maskF2,'/media/data1/artin/Results/mask_from_Thalamus.nii.gz' )
