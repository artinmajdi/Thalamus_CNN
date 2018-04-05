import nifti
import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

Directory = '/media/data1/artin/data/Thalamus/OriginalData/'


subFolders = os.listdir(Directory)

subFolders2 = []
i = 0
for o in range(len(subFolders)-1):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;
print subFolders2
subFolders = subFolders2
with open(Directory+"subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

SliceNumbers = range(119,139)
sFi = 0

im = nib.load(Directory+subFolders2[sFi]+'/WMnMPRAGEdeformed.nii.gz')
Header = im.header
Affine = im.affine
imD = im.get_data()
maskD = mask.get_data()

imD = imD/7
imDcrpd = imD[50:198,130:278,:]
im3 = imD.copy()
for sliceNum in range(imDcrpd.shape[2]):

    im2 = Image.fromarray(imDcrpd[:,:,sliceNum])
    im2 = im2.convert('L')
    # im3 = im2.filter(ImageFilter.SHARPEN)

    # enh = ImageEnhance.Contrast(im2)
    enh = ImageEnhance.Sharpness(im2)
    im3[50:198,130:278,sliceNum] = enh.enhance(1.5)


Prediction3D_nifti = nib.Nifti1Image(im3,Affine)
Prediction3D_nifti.get_header = Header

nib.save(Prediction3D_nifti,Directory+subFolders2[sFi]+'/WMnMPRAGEdeformed_Sharped.nii.gz')
