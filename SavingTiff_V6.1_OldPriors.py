import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

# 10-MGN_Deformed.nii.gz	  1-THALAMUS_Deformed.nii.gz  5-VLa_Deformed.nii.gz  9-LGN_Deformed.nii.gz
# 11-CM_Deformed.nii.gz	  2-AV_Deformed.nii.gz	      6-VLP_Deformed.nii.gz
# 12-MD-Pf_Deformed.nii.gz  4567-VL_Deformed.nii.gz     7-VPL_Deformed.nii.gz
# 13-Hb_Deformed.nii.gz	  4-VA_Deformed.nii.gz	      8-Pul_Deformed.nii.gz

for ind in [1,6,8,10,12]:
    # ind = 1
    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 4:
        NucleusName = '4567-VL'
    elif ind == 6:
        NucleusName = '6-VLP'
    elif ind == 8:
        NucleusName = '8-Pul'
    elif ind == 10:
        NucleusName = '10-MGN'
    elif ind == 12:
        NucleusName = '12-MD-Pf'


    # Dir_Prior = '/media/data1/artin/data/Thalamus/'+ Name_allTests_Nuclei + '/OriginalDeformedPriors'
    # Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'

    Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'
    # Dir_Prior = '/array/hdd/msmajdi/data/test'


    Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN/oldDatasetV2'


    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])


    A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
    SliceNumbers = range(107,140)

    ManualDir = 'Manual_Delineation_Sanitized/'

    Name_allTests_Nuclei  = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Name_priors_San_Label = ManualDir + NucleusName + '_deformed.nii.gz'


    for ii in range(len(A)):

        if ii == 0:
            TestName = 'WMnMPRAGE_bias_corr_Deformed'
        else:
            TestName = 'WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

        Dir_AllTests_Nuclei_EnhancedFld = Dir_AllTests + '/' + Name_allTests_Nuclei + '/Test_' + TestName

        inputName = TestName + '.nii.gz'
        print(inputName)

        # subFolders = ['vimp2_765_04162013_AW']
        for sFi in range(len(subFolders)):

            print('ii '+str(ii) + ' sfi ' + str(sFi))
            mask   = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + Name_priors_San_Label)
            im     = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + inputName)
            print(str(sFi) + ' ' + subFolders[sFi])
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

            if sFi == 0:
                imFull = imD_padded[...,np.newaxis]
                mskFull = maskD_padded[...,np.newaxis]
            else:
                imFull = np.append(imFull,imD_padded[...,np.newaxis],axis=3)
                mskFull = np.append(mskFull,maskD_padded[...,np.newaxis],axis=3)

            Dir = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders[sFi] + '/Test'
            try:
                os.stat(Dir)
            except:
                os.makedirs(Dir)

            Dir = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders[sFi] + '/Train'
            try:
                os.stat(Dir)
            except:
                os.makedirs(Dir)



        for sFi_parent in range(len(subFolders)):
            for sFi_child in range(len(subFolders)):

                if sFi_parent == sFi_child:
                    Dir = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders[sFi_child] + '/Test'
                else:
                    Dir = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders[sFi_child] + '/Train'


                for slcIx in range(imFull.shape[2]):

                    Name_PredictedImage = subFolders[sFi_parent] + '_Slice_' + str(SliceNumbers[slcIx])
                    tifffile.imsave( Dir + '/' + Name_PredictedImage + '.tif' , imFull[:,:,slcIx,sFi_parent] )
                    tifffile.imsave( Dir + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )
