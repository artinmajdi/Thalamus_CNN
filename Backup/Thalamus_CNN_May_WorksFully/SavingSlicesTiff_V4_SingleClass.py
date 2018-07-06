import nifti
import numpy as np
import matplotlib.pyplot as plt
import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

NeucleusFolder = 'CNN_VLP2'
NeuclusName = '6-VLP'

Directory_Priors = '/media/data1/artin/data/Thalamus/'+ NeucleusFolder + '/OriginalDeformedPriors'
SegmentName = 'ManualDelineation/' + NeuclusName + '_Deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

subFolders = os.listdir(Directory_Priors)

subFolders2 = []
i = 0
for o in range(len(subFolders)):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;

subFolders = subFolders2
with open(Directory_Priors+"subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

print(len(subFolders))
print(subFolders[19])
# print subFolders
# IndxX = range(92,127) # range(22,36)
# IndxY = range(151,211) #range(81,106)
# IndxZ = range(109,151) # range(28,49)
A = [[6,1],[1,2],[1,3],[4,1]] # [0,0],[4,3],
print len(A)
SliceNumbers = range(107,140)



for ii in range(len(A)):
    if ii == 0:
        TestName = 'WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Directory_Test = '/media/data1/artin/data/Thalamus/'+ NeucleusFolder + '/Test_' + TestName

    inputName = TestName + '.nii.gz'


    for sFi in range(len(subFolders)):
        # sFi = 0

        mask   = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + SegmentName)
        im     = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + inputName)
        print im.shape
        print subFolders2[sFi]
        imD    = im.get_data()
        maskD  = mask.get_data()
        Header = im.header
        Affine = im.affine

        # imD2 = imD
        # maskD2 = maskD

        imD2 = imD[50:198,130:278,SliceNumbers]
        maskD2 = maskD[50:198,130:278,SliceNumbers]

        padSizeFull = 90
        padSize = padSizeFull/2
        imD_padded = np.pad(imD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) #
        maskD_padded = np.pad(maskD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) # , constant_values=(5)

        # imD2 = imD[IndxX,:,:]
        # imD2 = imD2[:,IndxY,:]
        # imD2 = imD2[:,:,IndxZ]
        #
        # maskD2 = maskD[IndxX,:,:]
        # maskD2 = maskD2[:,IndxY,:]
        # maskD2 = maskD2[:,:,IndxZ]

        for p in range(len(subFolders)):


            if sFi == p:
                SaveDirectoryImage = Directory_Test + '/' + subFolders2[sFi] + '/' + '/Test'
            else:
                SaveDirectoryImage = Directory_Test + '/' + subFolders2[sFi] + '/' + '/Train'

            try:
                os.stat(SaveDirectoryImage)
            except:
                os.makedirs(SaveDirectoryImage)

            for sliceInd in range(imD2.shape[2]):

                # tifffile.imsave(SaveDirectoryImage + '/' +'Slice'+str(IndxZ[sliceInd])+'.tif',imD2[:,:,sliceInd])
                # tifffile.imsave(SaveDirectoryImage + '/' +'Slice'+str(IndxZ[sliceInd])+'_mask.tif',maskD2[:,:,sliceInd])

                tifffile.imsave(SaveDirectoryImage + '/' + subFolders2[p] + '_Slice'+str(sliceInd)+'.tif',imD_padded[:,:,sliceInd])
                tifffile.imsave(SaveDirectoryImage + '/' + subFolders2[p] + '_Slice'+str(sliceInd)+'_mask.tif',maskD_padded[:,:,sliceInd])
