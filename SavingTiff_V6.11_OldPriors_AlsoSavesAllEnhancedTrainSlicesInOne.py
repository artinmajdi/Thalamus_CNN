import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter

def testNme(A,ii):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    return TestName

def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    return subFolders

def initialDirectories(ind = 1, mode = 'oldDatasetV2'):

    if ind == 1:
        NucleusName = '1-THALAMUS'
        # SliceNumbers = range(106,143)
        SliceNumbers = range(103,147)
        # SliceNumbers = range(107,140) # original one
    elif ind == 2:
        NucleusName = '2-AV'
        SliceNumbers = range(126,143)
    elif ind == 4567:
        NucleusName = '4567-VL'
        SliceNumbers = range(114,143)
    elif ind == 4:
        NucleusName = '4-VA'
        SliceNumbers = range(116,140)
    elif ind == 5:
        NucleusName = '5-VLa'
        SliceNumbers = range(115,133)
    elif ind == 6:
        NucleusName = '6-VLP'
        SliceNumbers = range(115,145)
    elif ind == 7:
        NucleusName = '7-VPL'
        SliceNumbers = range(114,141)
    elif ind == 8:
        NucleusName = '8-Pul'
        SliceNumbers = range(112,141)
    elif ind == 9:
        NucleusName = '9-LGN'
        SliceNumbers = range(105,119)
    elif ind == 10:
        NucleusName = '10-MGN'
        SliceNumbers = range(107,121)
    elif ind == 11:
        NucleusName = '11-CM'
        SliceNumbers = range(115,131)
    elif ind == 12:
        NucleusName = '12-MD-Pf'
        SliceNumbers = range(115,140)
    elif ind == 13:
        NucleusName = '13-Hb'
        SliceNumbers = range(116,129)


    # Dir_Prior = '/media/data1/artin/data/Thalamus/'+ Name_allTests_Nuclei + '/OriginalDeformedPriors'
    Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
    # Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'
    # Dir_Prior = '/array/hdd/msmajdi/data/test'

    Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN/oldDatasetV2'

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]] # [4,3],

    return NucleusName, Dir_AllTests, Dir_Prior, SliceNumbers, A


for ind in [1,2,9,10,13]: # ,4,5,7,9,11,13]: # 1,6,8,10,12

    NucleusName, Dir_AllTests, Dir_Prior, SliceNumbers, A = initialDirectories(ind , 'oldDatasetV2')
    subFolders = subFoldersFunc(Dir_Prior)

    Name_priors_San_Label = 'Manual_Delineation_Sanitized/' + NucleusName + '_deformed.nii.gz'


    for ii in range(len(A)):

        TestName = testNme(A,ii)

        Dir_EachTraining = Dir_AllTests + '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN/' + TestName
        Dir_AllTrainings = Dir_AllTests + '/CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN/' + 'Test_AllTrainings'

        inputName = TestName.split('Test_')[1] + '.nii.gz'

        print('---------------------------------------')
        for sFi in range(len(subFolders)): # subFolders = ['vimp2_765_04162013_AW']

            print('Reading Images:  ',NucleusName,inputName.split('WMnMPRAGE_bias_corr_')[1].split('nii.gz')[0] , str(sFi) + ' ' + subFolders[sFi])
            mask   = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + Name_priors_San_Label)
            im     = nib.load(Dir_Prior + '/'  + subFolders[sFi] + '/' + inputName)

            imD    = im.get_data()
            maskD  = mask.get_data()
            Header = im.header
            Affine = im.affine

            imD2 = imD[50:198,130:278,SliceNumbers]
            maskD2 = maskD[50:198,130:278,SliceNumbers]

            padSizeFull = 90
            padSize = int(padSizeFull/2)
            imD_padded = np.pad(imD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )
            maskD_padded = np.pad(maskD2,((padSize,padSize),(padSize,padSize),(0,0)),'constant' )

            if sFi == 0:
                imFull = imD_padded[...,np.newaxis]
                mskFull = maskD_padded[...,np.newaxis]
            else:
                imFull = np.append(imFull,imD_padded[...,np.newaxis],axis=3)
                mskFull = np.append(mskFull,maskD_padded[...,np.newaxis],axis=3)

            mkDir(Dir_EachTraining + '/' + subFolders[sFi] + '/Test')
            mkDir(Dir_EachTraining + '/' + subFolders[sFi] + '/Train')

            mkDir(Dir_AllTrainings + '/' + subFolders[sFi] + '/Test' + str(ii))
            mkDir(Dir_AllTrainings + '/' + subFolders[sFi] + '/Train')

        print('---------------------------------------')


        for sFi_parent in range(len(subFolders)):
            print('Writing Images:  ',NucleusName,str(sFi_parent) + ' ' + subFolders[sFi_parent])
            for sFi_child in range(len(subFolders)):

                if sFi_parent == sFi_child:
                    Dir_Each = Dir_EachTraining + '/' + subFolders[sFi_child] + '/Test'
                    Dir_All  = Dir_AllTrainings + '/' + subFolders[sFi_child] + '/Test' + str(ii)

                else:
                    Dir_Each = Dir_EachTraining + '/' + subFolders[sFi_child] + '/Train'
                    Dir_All  = Dir_AllTrainings + '/' + subFolders[sFi_child] + '/Train'


                for slcIx in range(imFull.shape[2]):

                    Name_PredictedImage = subFolders[sFi_parent] + '_Sh' + str(A[ii][0]) + '_Ct' + str(A[ii][1]) + '_Slice_' + str(SliceNumbers[slcIx])
                    tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '.tif' , imFull[:,:,slcIx,sFi_parent] )
                    tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )

                    # if (ii == 0) | (sFi_parent != sFi_child) :  # the first argument will save both test and train files in the non enhanced version . the second argument will only save the train files for the enhanced version
                    tifffile.imsave( Dir_All + '/' + Name_PredictedImage + '.tif' , imFull[:,:,slcIx,sFi_parent] )
                    tifffile.imsave( Dir_All + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )
