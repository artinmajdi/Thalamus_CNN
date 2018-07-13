# import nifti
import numpy as np
import matplotlib.pyplot as plt
# import Image
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])
    return subFolders


for ind in [2,4,5,7,9,11,13]: # 1,6,8,10,12
    # ind = 1
    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 2:
        NucleusName = '2-AV'
    elif ind == 4567:
        NucleusName = '4567-VL'
    elif ind == 4:
        NucleusName = '4-VA'
    elif ind == 5:
        NucleusName = '5-VLa'
    elif ind == 6:
        NucleusName = '6-VLP'
    elif ind == 7:
        NucleusName = '7-VPL'
    elif ind == 8:
        NucleusName = '8-Pul'
    elif ind == 9:
        NucleusName = '9-LGN'
    elif ind == 10:
        NucleusName = '10-MGN'
    elif ind == 11:
        NucleusName = '11-CM'
    elif ind == 12:
        NucleusName = '12-MD-Pf'
    elif ind == 13:
        NucleusName = '13-Hb'


    # Dir_Prior = '/media/data1/artin/data/Thalamus/'+ Name_allTests_Nuclei + '/OriginalDeformedPriors'
    # Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'

    Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'
    # Dir_Prior = '/array/hdd/msmajdi/data/test'

    Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN'

    subFolders = subFoldersFunc(Dir_Prior)

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]] # [4,3],
    SliceNumbers = range(107,140)

    ManualDir = 'Manual_Delineation_Sanitized/'

    Name_allTests_Nuclei  = 'newDataset/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Name_priors_San_Label = ManualDir + NucleusName + '_deformed.nii.gz'


    for ii in range(len(A)):

        if ii == 0:
            TestName = 'WMnMPRAGE_bias_corr_Deformed'
        else:
            TestName = 'WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

        Dir_AllTests_Nuclei_EnhancedFld = Dir_AllTests + '/' + Name_allTests_Nuclei + '/Test_' + TestName

        inputName = TestName + '.nii.gz'

        print('---------------------------------------')
        # subFolders = ['vimp2_765_04162013_AW']
        for sFi in range(len(subFolders)):

            print(NucleusName,inputName.split('WMnMPRAGE_bias_corr_')[1].split('nii.gz')[0] , str(sFi) + ' ' + subFolders[sFi])

            mask   = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + Name_priors_San_Label)
            im     = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + inputName)
            imD    = im.get_data()
            maskD  = mask.get_data()
            Header = im.header
            Affine = im.affine

            imD2 = imD[50:198,130:278,SliceNumbers]
            maskD2 = maskD[50:198,130:278,SliceNumbers]

            padSizeFull = 90
            padSize = padSizeFull/2
            imD_padded = np.pad(imD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
            maskD_padded = np.pad(maskD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )


            Dir_TestSamples = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders[sFi] + '/Test'


            try:
                os.stat(Dir_TestSamples)
            except:
                os.makedirs(Dir_TestSamples)

            for sliceInd in range(imD2.shape[2]):

                Name_PredictedImage = subFolders[sFi] + '_Slice_'+str(SliceNumbers[sliceInd])
                tifffile.imsave( Dir_TestSamples + '/' + Name_PredictedImage + '.tif' , imD_padded[:,:,sliceInd] )
                tifffile.imsave( Dir_TestSamples + '/' + Name_PredictedImage + '_mask.tif' , maskD_padded[:,:,sliceInd] )
