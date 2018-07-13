# import nifti
import numpy as np
import matplotlib.pyplot as plt
# import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

# 10-MGN_Deformed.nii.gz	  1-THALAMUS_Deformed.nii.gz  5-VLa_Deformed.nii.gz  9-LGN_Deformed.nii.gz
# 11-CM_Deformed.nii.gz	  2-AV_Deformed.nii.gz	      6-VLP_Deformed.nii.gz
# 12-MD-Pf_Deformed.nii.gz  4567-VL_Deformed.nii.gz     7-VPL_Deformed.nii.gz
# 13-Hb_Deformed.nii.gz	  4-VA_Deformed.nii.gz	      8-Pul_Deformed.nii.gz


NeuclusName = '6-VLP' #'8-Pul' # '4567-VL' #
NeuclusNameFull = ['12-MD-Pf'] # '6-VLP' , '1-THALAMUS' , '8-Pul' , '4567-VL']
# Directory_Priors = '/media/data1/artin/data/Thalamus/'+ NeucleusFolder + '/OriginalDeformedPriors'
Directory_Priors = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
Directory_Tests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN'



subFolders = os.listdir(Directory_Priors)

subFolders2 = []
i = 0
for o in range(len(subFolders)):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])

        i = i+1;

subFolders = subFolders2
with open(Directory_Priors + "subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

print(len(subFolders))
# print(subFolders[19])
A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]] #
# print len(A)
SliceNumbers = range(107,140)

methodMode = 'Old_Method'

for NeuclusName in NeuclusNameFull:


    NeucleusFolder = 'CNN' + NeuclusName.replace('-','_') + '_2D_SanitizedNN'
    SegmentName = 'Manual_Delineation_Sanitized/' + NeuclusName + '_deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped
    for ii in range(len(A)):
        if ii == 0:
            TestName = 'WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
        else:
            TestName = 'WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

        Directory_Test = Directory_Tests + '/' + NeucleusFolder + '/Test_' + TestName

        inputName = TestName + '.nii.gz'
        print(inputName)

        for sFi in range(len(subFolders)):
            # sFi = 0
            if (ii == 1) & (sFi == 18):
                print('error')
            else:
                print('ii '+str(ii) + ' sfi ' + str(sFi))
                mask   = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + SegmentName)
                im     = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + inputName)
                # print im.shape
                print str(sFi) + subFolders2[sFi]
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

                for p in range(len(subFolders)):

                    if sFi == p:
                        SaveDirectoryImage = Directory_Test + '/' + subFolders2[p] + '/Test'
                    else:
                        SaveDirectoryImage = Directory_Test + '/' + subFolders2[p] + '/Train'

                    try:
                        os.stat(SaveDirectoryImage)
                    except:
                        os.makedirs(SaveDirectoryImage)

                    for sliceInd in range(imD2.shape[2]):

                        # if methodMode == 'New_Method':
                        #     SaveDirectoryImageSlice = SaveDirectoryImage + '/Slice' + str(sliceInd)
                        # else:
                        SaveDirectoryImageSlice = SaveDirectoryImage

                        # try:
                        #     os.stat(SaveDirectoryImageSlice)
                        # except:
                        #     os.makedirs(SaveDirectoryImageSlice)

                        # if sFi == p:
                        tifffile.imsave(SaveDirectoryImageSlice + '/' + subFolders2[sFi] + '_Slice'+str(sliceInd)+'.tif',imD_padded[:,:,sliceInd])
                        tifffile.imsave(SaveDirectoryImageSlice + '/' + subFolders2[sFi] + '_Slice'+str(sliceInd)+'_mask.tif',maskD_padded[:,:,sliceInd])
                        # else:
                        #     for i in range(max(0,sliceInd-1),min(sliceInd+2,imD2.shape[2])):
                        #         tifffile.imsave(SaveDirectoryImageSlice + '/' + subFolders2[p] + '_Slice'+str(i)+'.tif',imD_padded[:,:,i])
                        #         tifffile.imsave(SaveDirectoryImageSlice + '/' + subFolders2[p] + '_Slice'+str(i)+'_mask.tif',maskD_padded[:,:,i])
