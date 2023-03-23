
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import regionprops , label
import cv2
from PIL import Image
import os
import numpy as np

def CroppingImage(im,mask):
    imData = im.get_data()
    maskData = mask.get_data()

    labelImg = label(maskData)
    obj = regionprops(labelImg)
    inds = obj[0].bbox

    return imData[inds[0]:inds[3],inds[1]:inds[4],inds[2]:inds[5]]

Directory = '/media/artin/data/documents/UofA/courses/Research/data/Thalamus/'

subFolders = os.listdir(Directory)
subFolders2 = []
i = 0
for o in range(len(subFolders)-1):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;

i = 0
mask = nib.load(Directory+'mask_templ_93x187x68.nii.gz')
# for i in range(len(subFolders2))
im = nib.load(Directory+subFolders2[i]+'/WMnMPRAGEdeformed.nii.gz')
imLabel = nib.load(Directory+subFolders2[i]+'/WholeThalamusSegment_TemplateDomain.nii.gz')


imCroped = CroppingImage(im,mask)
imLabelCroped = CroppingImage(imLabel,mask)
print im.shape
# Image.SAVE()
aa = np.sum(imLabelCroped==1)
print aa

# nib.save(imCroped, os.path.join('build',Directory+subFolders2[i]+'/imCroped.nii.gz'))

# aa = nib.Nifti1Image(dataobj=imDataCroped,header=im.get_header)
# nib.save(imDataCroped,Directory+subFolders2[i]+'/WMnMPRAGEdeformed_croped.nii.gz')
# print imDataCroped.shape
sliceNum = 34
fig , ax = plt.subplots(1,2,sharey=True)
ax[0].imshow(imCroped[:,:,sliceNum],cmap = plt.get_cmap("gray") )
ax[1].imshow(imLabelCroped[:,:,sliceNum],cmap = plt.get_cmap("gray") )
plt.show()

# nib.save(imDataCroped)
# obj = ski.regionprops(maskData[:,:,0])
# plt.imshow(imData[:,:,100])
# plt.show()
