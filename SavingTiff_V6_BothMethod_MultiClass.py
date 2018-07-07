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

ind = 1
if ind == 1:
    NeucleusFolder = 'CNN1_THALAMUS_2D_SanitizedNN'
    NucleusName = '1-THALAMUS'
elif ind == 4:
    NeucleusFolder = 'CNN4567_VL_2D_SanitizedNN' # 'CNN12_MD_Pf_2D_SanitizedNN' #  'CNN1_THALAMUS_2D_SanitizedNN' 'CNN6_VLP_2D_SanitizedNN'  #
    NucleusName = '4567-VL'
elif ind == 6:
    NeucleusFolder = 'CNN6_VLP_2D_SanitizedNN'
    NucleusName = '6-VLP'
elif ind == 8:
    NeucleusFolder = 'CNN8_Pul_2D_SanitizedNN'
    NucleusName = '8-Pul'
elif ind == 10:
    NeucleusFolder = 'CNN10_MGN_2D_SanitizedNN'
    NucleusName = '10-MGN'
elif ind == 12:
    NeucleusFolder = 'CNN12_MD_Pf_2D_SanitizedNN'
    NucleusName = '12-MD-Pf'

NeuclusName = '6-VLP' #'8-Pul' # '4567-VL' #
NeuclusNameFull = ['8-Pul' , '6-VLP' , '4567-VL'] # '12-MD-Pf'] # '6-VLP' , '1-THALAMUS' , '8-Pul' , '4567-VL']


# Directory_Priors = '/media/data1/artin/data/Thalamus/'+ NeucleusFolder + '/OriginalDeformedPriors'
Directory_Priors = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
Directory_Tests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN'



subFolders = os.listdir(Directory_Priors)

subFolders2 = []

for o in range(len(subFolders)):
    if "vimp2_" in subFolders[o]:
        subFolders2.append(subFolders[o])

subFolders = subFolders2
with open(Directory_Priors + "subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

#print(len(subFolders))
# print(subFolders[19])
A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]] #
# print len(A)
SliceNumbers = range(107,140)

methodMode = 'Old_Method'
padSizeFull = 90
padSize = padSizeFull/2


SegmentName = []
L = len(NeuclusNameFull)
for N in range(L):
    NeuclusName = NeuclusNameFull[N]
    # NeucleusFolder = 'CNN' + NeuclusName.replace('-','_') + '_2D_SanitizedNN'
    SegmentName.append('Manual_Delineation_Sanitized/' + NeuclusName + '_deformed.nii.gz')   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped


for ii in range(len(A)):
    if ii == 0:
        TestName = 'WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    print(TestName)
    Directory_Test = Directory_Tests + '/MultiClass/Test_' + TestName

    inputName = TestName + '.nii.gz'

    for sFi in range(len(subFolders)):
        # sFi = 0
        try:
            # print('ii '+str(ii) + ' sfi ' + str(sFi))

            maskDFull = []
            for N in range(L):
                mask = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + SegmentName[N])
                maskD = mask.get_data()
                maskD_B = maskD[50:198,130:278,SliceNumbers]
                maskD_padded = np.pad(maskD_B,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
                maskDFull.append(maskD_padded)

            print(str(sFi) + ':  ' + subFolders2[sFi])
            im     = nib.load(Directory_Priors + '/'  + subFolders2[sFi] + '/' + inputName)
            imD    = im.get_data()
            Header = im.header
            Affine = im.affine

            imD2 = imD[50:198,130:278,SliceNumbers]
            imD_padded = np.pad(imD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' ) #
            sz = imD_padded.shape


            for p in range(len(subFolders)):

                if sFi == p:
                    SaveDirectoryImage = Directory_Test + '/' + subFolders2[sFi] + '/Test'
                else:
                    SaveDirectoryImage = Directory_Test + '/' + subFolders2[sFi] + '/Train'

                try:
                    os.stat(SaveDirectoryImage)
                except:
                    os.makedirs(SaveDirectoryImage)


                for sliceInd in range(imD2.shape[2]):

                    msk_MultyClass = np.zeros((sz[0],sz[1],L))
                    for N in range(L):
                        msk_MultyClass[...,N] = maskDFull[N][:,:,sliceInd]

                    tifffile.imsave(SaveDirectoryImage + '/' + subFolders2[p] + '_Slice'+str(sliceInd)+'.tif',imD_padded[:,:,sliceInd])
                    tifffile.imsave(SaveDirectoryImage + '/' + subFolders2[p] + '_Slice'+str(sliceInd)+'_mask.tif',msk_MultyClass)
        except:
            print('Error ' + str(sFi) + ':  ' + subFolders2[sFi])
