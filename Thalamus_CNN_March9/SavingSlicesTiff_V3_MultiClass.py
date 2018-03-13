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

def ReadMasks(DirectoryMask,SliceNumbers):

    mask = nib.load(DirectoryMask)
    maskD = mask.get_data()

    msk = maskD[50:198,130:278,SliceNumbers]

    # msk[msk<0.5]  = 0
    # msk[msk>=0.5] = 1

    return msk

def SumMasks(DirectorySubFolders,SliceNumbers):

    i = 0
    SegmentName = '/6_VLP_NeucleusSegDeformed.nii.gz'
    msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    maskD = np.zeros((msk.shape[0],msk.shape[1],5,msk.shape[2]))
    maskD[:,:,i,:] = msk

    i = i + 1
    SegmentName = '/7_VPL_NeucleusSegDeformed.nii.gz'
    msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    maskD[:,:,i,:] = msk

    i = i + 1
    SegmentName = '/8_Pul_NeucleusSegDeformed.nii.gz'
    msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    maskD[:,:,i,:] = msk

    i = i + 1
    SegmentName = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    maskD[:,:,i,:] = msk

    # i = 1
    # SegmentName = '/6_VLP_NeucleusSegDeformed.nii.gz'
    # msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    # maskD = np.zeros(msk.shape)
    # maskD[msk == 1] = 10*i
    #
    # i = i + 1
    # SegmentName = '/7_VPL_NeucleusSegDeformed.nii.gz'
    # msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    # maskD[msk == 1] = 10*i
    #
    # i = i + 1
    # SegmentName = '/8_Pul_NeucleusSegDeformed.nii.gz'
    # msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    # maskD[msk == 1] = 10*i
    #
    # i = i + 1
    # SegmentName = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    # msk = ReadMasks(DirectorySubFolders+SegmentName,SliceNumbers)
    # maskD[msk == 1] = 10*i

    return maskD

Enhance_Flag = 1

Directory = '/media/data1/artin/data/Thalamus/OriginalData/'
TestName  = 'ForUnet_Test14_MultiClass_Enhanced'
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

SliceNumbers = range(119,139)


for sFi in range(len(subFolders)):
# sFi = 0

    maskD = SumMasks(Directory+subFolders2[sFi],SliceNumbers)

    im = nib.load(Directory+subFolders2[sFi]+'/WMnMPRAGEdeformed.nii.gz')
    imD = im.get_data()
    Header = im.header
    Affine = im.affine

    if Enhance_Flag == 1:
        for sliceInd in range(107,143):
            imD[80:175,160:270,sliceInd] = EnhanceSlices(imD[80:175,160:270,sliceInd]/7)
            imD[80:175,160:270,sliceInd] = 7*imD[80:175,160:270,sliceInd]


    imD = imD[50:198,130:278,SliceNumbers]

    padSizeFull = 90
    padSize = padSizeFull/2
    imD_padded = np.pad(imD,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) #
    maskD_padded = np.pad(maskD,((padSize,padSize),(padSize,padSize),(0,0),(0,0)),'constant' ) # , constant_values=(5)

    # plt.imshow(maskD_padded[:,:,10],cmap='gray')
    # plt.show()
    # print maskD_padded.max()
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

            im = Image.fromarray(np.uint8(7*imD_padded[:,:,sliceInd]/256))
            msk = Image.fromarray(np.uint8(maskD_padded[:,:,:,sliceInd]))

            # tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'.tif',7*imD_padded[:,:,sliceInd])
            # tifffile.imsave(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'_mask.tif',maskD_padded[:,:,sliceInd])

            im.save(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'.tif')
            msk.save(SaveDirectoryImage+subFolders2[sFi]+'_slice'+str(SliceNumbers[sliceInd])+'_mask.tif')
