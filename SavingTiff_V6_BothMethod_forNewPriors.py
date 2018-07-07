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
    Name_allTests_Nuclei = 'CNN1_THALAMUS_2D_SanitizedNN'
    NucleusName = '1-THALAMUS'
elif ind == 4:
    Name_allTests_Nuclei = 'CNN4567_VL_2D_SanitizedNN' # 'CNN12_MD_Pf_2D_SanitizedNN' #  'CNN1_THALAMUS_2D_SanitizedNN' 'CNN6_VLP_2D_SanitizedNN'  #
    NucleusName = '4567-VL'
elif ind == 6:
    Name_allTests_Nuclei = 'CNN6_VLP_2D_SanitizedNN'
    NucleusName = '6-VLP'
elif ind == 8:
    Name_allTests_Nuclei = 'CNN8_Pul_2D_SanitizedNN'
    NucleusName = '8-Pul'
elif ind == 10:
    Name_allTests_Nuclei = 'CNN10_MGN_2D_SanitizedNN'
    NucleusName = '10-MGN'
elif ind == 12:
    Name_allTests_Nuclei = 'CNN12_MD_Pf_2D_SanitizedNN'
    NucleusName = '12-MD-Pf'


# Dir_Prior = '/media/data1/artin/data/Thalamus/'+ Name_allTests_Nuclei + '/OriginalDeformedPriors'

# Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'
Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN'



subFolders = os.listdir(Dir_Prior)

subFolders2 = []
i = 0
for o in range(len(subFolders)):
    if "." not in subFolders[o]:
        subFolders2.append(subFolders[o])
        i = i+1;

subFolders = subFolders2
with open(Dir_Prior + "subFolderList.txt" ,"wb") as fp:
    pickle.dump(subFolders,fp)

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]] #
SliceNumbers = range(107,140)


Name_allTests_Nuclei = 'newDataset/CNN' + NeuclusName.replace('-','_') + '_2D_SanitizedNN'

Name_priors_San_Label = 'Manual_Delineation_Sanitized/' + NeuclusName + '_deformed.nii.gz'


for ii in range(len(A)):
    if ii == 0:
        TestName = 'WMnMPRAGE_Deformed' # _Deformed_Cropped
    else:
        TestName = 'WMnMPRAGE_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Dir_AllTests_Nuclei_EnhancedFld = Dir_AllTests + '/' + Name_allTests_Nuclei + '/Test_' + TestName

    inputName = TestName + '.nii.gz'
    print(inputName)

    for sFi in range(len(subFolders)):

        print('ii '+str(ii) + ' sfi ' + str(sFi))
        mask   = nib.load(Dir_Prior + '/'  + subFolders2[sFi] + '/' + Name_priors_San_Label)
        im     = nib.load(Dir_Prior + '/'  + subFolders2[sFi] + '/' + inputName)
        print(str(sFi) + subFolders2[sFi])
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


        Dir_TestSamples = Dir_AllTests_Nuclei_EnhancedFld + '/' + subFolders2[sFi] + '/Test'


        try:
            os.stat(Dir_TestSamples)
        except:
            os.makedirs(Dir_TestSamples)

        for sliceInd in range(imD2.shape[2]):

            Name_PredictedImage = subFolders2[sFi] + '_Slice_'+str(SliceNumbers[sliceInd])
            tifffile.imsave( Dir_TestSamples + '/' + Name_PredictedImage + '.tif' , imD_padded[:,:,sliceInd] )
            tifffile.imsave( Dir_TestSamples + '/' + Name_PredictedImage + '_mask.tif' , maskD_padded[:,:,sliceInd] )
