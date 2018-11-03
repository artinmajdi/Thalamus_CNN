import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import tifffile
from scipy.misc import imrotate
from scipy import ndimage
import os

def MyTranspose(dir,order):
    im = nib.load(dir)
    im2   = np.transpose(im.get_data(),[0,2,1])
    im2 = ndimage.zoom(im2,(1,1,0.5),order=order)
    # im2 = im2[...,range(0,im.shape[2],2)]

    im2 = nib.Nifti1Image(im2,im.affine)
    im2.get_header = im.header
    dir2 = dir.split('_Orig.nii.gz')[0] + '.nii.gz'
    nib.save(im2,dir2)

    return im2

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    return subFolders


dir = '/array/ssd/msmajdi/data/ET/3T'
subFolders = subFoldersFunc(dir)


for sFi in range(len(subFolders)):

    print('sFi',sFi,subFolders[sFi])
    im = MyTranspose(dir + '/' + subFolders[sFi] + '/WMnMPRAGE_bias_corr_Orig.nii.gz',3)
    crop = MyTranspose(dir + '/' + subFolders[sFi] + '/MyCrop2_Gap20_Orig.nii.gz',0)
    mask = MyTranspose(dir + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/1-THALAMUS_Orig.nii.gz',0)
