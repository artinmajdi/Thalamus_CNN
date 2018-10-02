import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter
import sys

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

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

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'old'):

    Params = {}

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
    elif ind == 14:
        NucleusName = '14-MTT'
        SliceNumbers = range(104,135)


    Params['modelFormat'] = 'ckpt'
    if 'localLT' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/Unlabeled'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'
        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    elif 'localPC' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/Unlabeled'

        Dir_AllTests  = '/media/data1/artin/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'

    elif 'server' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/Unlabeled'

        Dir_AllTests  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'



    Params = {}
    Params['A'] = [[0,0],[6,1],[1,2],[1,3],[4,1]]
    Params['Dir_Prior']    = Dir_Prior
    Params['Dir_AllTests'] = Dir_AllTests
    Params['SliceNumbers'] = SliceNumbers
    Params['NucleusName']  = NucleusName

    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'old'
    UserEntries['testMode'] = 'EnhancedSeperately' # 'combo'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['mode'] = 'server'

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'combo'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]
        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]

        elif input.split('=')[0] == 'nuclei':
            if 'all' in input.split('=')[1]:
                a = range(4,14)
                UserEntries['IxNuclei'] = np.append([1,2,4567],a)

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['IxNuclei'] = [int(k) for k in B]

            else:
                UserEntries['IxNuclei'] = [int(input.split('=')[1])]

        elif input.split('=')[0] == 'enhance':
            if 'all' in input.split('=')[1]:
                UserEntries['enhanced_Index'] = range(len(A))

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['enhanced_Index'] = [int(k) for k in B]

            else:
                UserEntries['enhanced_Index'] = [int(input.split('=')[1])]

    return UserEntries


UserEntries = input_GPU_Ix()
# gpuNum = 'nan'
for ind in UserEntries['IxNuclei']: # 1,2,8,9,10,13]: #

    Params = initialDirectories(ind = ind, mode = UserEntries['mode'] , dataset = UserEntries['dataset'] , method = UserEntries['method'] )
    print(Params['Dir_Prior'])
    subFolders = subFoldersFunc(Params['Dir_Prior'])
    print(subFolders[0])
    Name_priors_San_Label = 'Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'


    for ii in UserEntries['enhanced_Index']: # len( Params['A'] ):

        TestName = testNme(Params['A'],ii)

        Dir_EachTraining = Params['Dir_AllTests'] + '/CNN' + Params['NucleusName'].replace('-','_') + '_2D_SanitizedNN/' + TestName
        Dir_AllTrainings = Params['Dir_AllTests'] + '/CNN' + Params['NucleusName'].replace('-','_') + '_2D_SanitizedNN/' + 'Test_AllTrainings'
        Dir_All  = mkDir(Dir_AllTrainings + '/Train')

        inputName = TestName.split('Test_')[1] + '.nii.gz'


        for sFi in range(len(subFolders)):

            print('Reading Images:  ',Params['NucleusName'],inputName.split('WMnMPRAGE_bias_corr_')[1].split('nii.gz')[0] , str(sFi) + ' ' + subFolders[sFi])
            mask   = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + Name_priors_San_Label)
            im     = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + inputName)

            imD    = im.get_data()
            maskD  = mask.get_data()
            Header = im.header
            Affine = im.affine

            imD2 = imD[50:198,130:278,Params['SliceNumbers']]
            maskD2 = maskD[50:198,130:278,Params['SliceNumbers']]

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

            #mkDir(Dir_EachTraining + '/' + subFolders[sFi] + '/Test')
            #mkDir(Dir_EachTraining + '/' + subFolders[sFi] + '/Train')



        print('---------------------------------------')


        for sFi_parent in range(len(subFolders)):


            print('Writing Images:  ',Params['NucleusName'],str(sFi_parent) + ' ' + subFolders[sFi_parent])
            for sFi_child in range(len(subFolders)):
                mkDir(Dir_EachTraining + '/' + subFolders[sFi_child] + '/Test')
                mkDir(Dir_EachTraining + '/' + subFolders[sFi_child] + '/Train')
                if sFi_parent == sFi_child: # in [1,5,10,14,20]: #
                    Dir_Each = Dir_EachTraining + '/' + subFolders[sFi_child] + '/Test'

                else:
                    Dir_Each = Dir_EachTraining + '/' + subFolders[sFi_child] + '/Train'

                for slcIx in range(imFull.shape[2]):

                    Name_PredictedImage = subFolders[sFi_parent] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
                    tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi_parent] )
                    tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )

                    # tifffile.imsave( Dir_All + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi_parent] )
                    # tifffile.imsave( Dir_All + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )
