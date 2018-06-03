import nifti
import numpy as np
import matplotlib.pyplot as plt
import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter


def EnhanceSlices(imD):
    imD = Image.fromarray(imD)
    imD = imD.convert('L')
    enh = ImageEnhance.Contrast(imD)
    # enh = ImageEnhance.Sharpness(im2)
    imD = enh.enhance(1.5)

    return imD

Enhance_Flag = 1

Directory = '/media/data1/artin/thomas/priors/'
TestName  = 'CNN_Enhanced_VPL'
SegmentName = '/ThalamusSegDeformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
subFolders = os.listdir(Directory)

subFolders2 = []
i = 0
for o in range(len(subFolders)-1):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;

subFolders = subFolders2
with open(Directory+"subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

# with open(Directory+"subFolderList" ,"rb") as fp:
#     ssff2 = pickle.load(fp)

# SliceNumbers = range(119,139)
SliceNumbers = range(107,140)
# SliceNumbers = range(21,47)


for sFi in range(len(subFolders)):
    # sFi = 0
    mask = nib.load(Directory+subFolders2[sFi]+SegmentName)
    im = nib.load(Directory+subFolders2[sFi]+'/WMnMPRAGEdeformed.nii.gz')

    imD = im.get_data()
    maskD = mask.get_data()
    Header = im.header
    Affine = im.affine


    if Enhance_Flag == 1:
        for sliceInd in range(107,143):
            imD[80:175,160:270,sliceInd] = EnhanceSlices(imD[80:175,160:270,sliceInd]/7)
            imD[80:175,160:270,sliceInd] = 7*imD[80:175,160:270,sliceInd]


    imD = imD[50:198,130:278,SliceNumbers]
    maskD = maskD[50:198,130:278,SliceNumbers]

    padSizeFull = 90
    padSize = padSizeFull/2
    imD_padded = np.pad(imD,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) #
    maskD_padded = np.pad(maskD,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) # , constant_values=(5)

    # print imD.shape
    # imD = np.reshape(imD,[572,572,10])

    for p in range(len(subFolders)):

        #     os.makedirs(SaveDirectorySegment)
        if sFi == p:
            SaveDirectoryImage = Directory+'../'+TestName+'/TestSubject'+str(p)+'/test/'
        else:
            SaveDirectoryImage = Directory+'../'+TestName+'/TestSubject'+str(p)+'/train/'
        # SaveDirectorySegment = Directory+'ForUnet/Segments/'+subFolders2[sFi]+'/'

        try:
            os.stat(SaveDirectoryImage)
        except:
            os.makedirs(SaveDirectoryImage)

        for sliceInd in range(imD_padded.shape[2]):

            tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'.tif',7*imD_padded[:,:,sliceInd])
            tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'_mask.tif',maskD_padded[:,:,sliceInd])
