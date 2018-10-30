import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import tifffile
import pickle
from PIL import ImageEnhance , Image , ImageFilter
import sys
from scipy import ndimage

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

def testNme(Params,ii):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(Params['A'][ii][0]) + '_Contrast_' + str(Params['A'][ii][1])

    if Params['registrationFlag'] == 1:
        TestName = TestName + '_Deformed'

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

def NucleiSelection(ind):

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

    return NucleusName , SliceNumbers

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'old'):

    Params = {}
    NucleusName , SliceNumbers = NucleiSelection(ind)

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
    Params['registrationFlag'] = 0


    if Params['registrationFlag'] == 1:
        Params['SliceNumbers'] = SliceNumbers
    else:
        Params['SliceNumbers'] = range(129,251)

    Params['NucleusName']  = NucleusName


    return Params

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'old'
    UserEntries['testmode'] = 'EnhancedSeperately' # 'combo' 'onetrain'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['mode'] = 'server'
    UserEntries['onetrain_testIndexes'] = [1,5,10,14,20]
    UserEntries['Flag_cross_entropy'] = 0

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testmode':
            UserEntries['testmode'] = input.split('=')[1] # 'combo' 'onetrain'
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

        elif 'onetrain_testIndexes' in input:
            UserEntries['testmode'] = 'onetrain'
            if input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['onetrain_testIndexes'] = [int(k) for k in B]
            else:
                UserEntries['onetrain_testIndexes'] = [int(input.split('=')[1])]

        elif input.split('=')[0] == 'enhance':
            if 'all' in input.split('=')[1]:
                UserEntries['enhanced_Index'] = range(len(A))

            elif input.split('=')[1][0] == '[':
                B = input.split('=')[1].split('[')[1].split(']')[0].split(",")
                UserEntries['enhanced_Index'] = [int(k) for k in B]

            else:
                UserEntries['enhanced_Index'] = [int(input.split('=')[1])]

        if 'Unlabeled' in UserEntries['dataset']:
            UserEntries['testmode'] = 'onetrain'
            UserEntries['onetrain_testIndexes'] = [1,5,10,14,20]

    return UserEntries

def normal_Cross_Validation(Params , subFolders , imFull , mskFull, ii):
    for sFi_parent in range(len(subFolders)):

        print('Writing Images:  ',Params['NucleusName'],str(sFi_parent) + ' ' + subFolders[sFi_parent])
        for sFi_child in range(len(subFolders)):
            mkDir(Params['Dir_EachTraining'] + '/' + subFolders[sFi_child] + '/Test')
            mkDir(Params['Dir_EachTraining'] + '/' + subFolders[sFi_child] + '/Train')
            if sFi_parent == sFi_child: # in [1,5,10,14,20]: # sFi_parent
                Dir_Each = Params['Dir_EachTraining'] + '/' + subFolders[sFi_child] + '/Test'

            else:
                Dir_Each = Params['Dir_EachTraining'] + '/' + subFolders[sFi_child] + '/Train'

            for slcIx in range(imFull.shape[2]):
                # print( '------' , 'slcIx' , slcIx , 'ii' , ii , 'len(subFolders)' , len(subFolders) , 'sFi_parent' , sFi_parent , 'imSh' , imFull.shape , Params['A'][ii],len(Params['SliceNumbers']) )
                # print(Params['SliceNumbers'])

                Name_PredictedImage = subFolders[sFi_parent] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
                tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi_parent] )
                tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )

                tifffile.imsave( Params['Dir_All'] + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi_parent] )
                tifffile.imsave( Params['Dir_All'] + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi_parent] )

# def old_OneTrain_MultipleTest(UserEntries , Params , subFolders,imFull,mskFull, ii):
#     print( '------------------' , 'Test' , '------------------' )
#
#     for sFi in UserEntries['onetrain_testIndexes']:
#
#         Dir_Each = mkDir(Params['Dir_EachTraining'] + '/OneTrain_MultipleTest' + '/TestCases/' + subFolders[sFi] + '/Test/')
#
#         for slcIx in range(imFull.shape[2]):
#             Name_PredictedImage = subFolders[sFi] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
#             tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi] )
#             tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi] )
#
#     print( '------------------' , 'Train' , '------------------' )
#
#     Dir_Each = mkDir(Params['Dir_EachTraining'] + '/OneTrain_MultipleTest' + '/Train')
#     for sFi in range(len(subFolders)):
#         if sFi not in UserEntries['onetrain_testIndexes']:
#
#             for slcIx in range(imFull.shape[2]):
#                 Name_PredictedImage = subFolders[sFi] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
#                 tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imFull[:,: ,slcIx,sFi] )
#                 tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , mskFull[:,:,slcIx,sFi] )

def OneTrain_MultipleTest(UserEntries , Params , subFolders,imD_padded,maskD_padded, ii, sFi):

    if sFi in UserEntries['onetrain_testIndexes']:
        Dir_Each = mkDir(Params['Dir_EachTraining'] + '/OneTrain_MultipleTest' + '/TestCases/' + subFolders[sFi] + '/Test/')
        for slcIx in range(imD_padded.shape[2]):
            Name_PredictedImage = subFolders[sFi] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
            tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imD_padded[:,: ,slcIx] )
            tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , maskD_padded[:,:,slcIx] )

    else:
        Dir_Each = mkDir(Params['Dir_EachTraining'] + '/OneTrain_MultipleTest' + '/Train')
        for slcIx in range(imD_padded.shape[2]):
            Name_PredictedImage = subFolders[sFi] + '_Sh' + str(Params['A'][ii][0]) + '_Ct' + str(Params['A'][ii][1]) + '_Slice_' + str(Params['SliceNumbers'][slcIx])
            tifffile.imsave( Dir_Each + '/' + Name_PredictedImage +      '.tif' , imD_padded[:,: ,slcIx] )
            tifffile.imsave( Dir_Each + '/' + Name_PredictedImage + '_mask.tif' , maskD_padded[:,:,slcIx] )

do_I_want_Upsampling = 0

def funcNormalize(im):
    # return (im-im.mean())/im.std()
    im = np.float32(im)
    return ( im-im.min() )/( im.max() - im.min() )

def funcCropping_FromThalamus(im , mask , CropMask):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 1)[0]
    c2 = np.where(np.sum(ss,axis=0) > 1)[0]
    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 1)[0]

    gap = 20
    gap2 = 3
    d1 = [  c1[0]-gap  , c1[ c1.shape[0]-1 ]+gap   ]
    d2 = [  c2[0]-gap  , c2[ c2.shape[0]-1 ]+gap   ]
    SN = [  c3[0]-gap2 , c3[ c3.shape[0]-1 ]+gap2  ]
    SliceNumbers = range(SN[0],SN[1])


    im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ] # Params['SliceNumbers']]
    mask = mask[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ] # Params['SliceNumbers']]

    return im , mask , SliceNumbers

def funcCropping(im , mask , CropMask):
    ss = np.sum(CropMask,axis=2)
    c1 = np.where(np.sum(ss,axis=1) > 10)[0]
    c2 = np.where(np.sum(ss,axis=0) > 10)[0]
    ss = np.sum(CropMask,axis=1)
    c3 = np.where(np.sum(ss,axis=0) > 10)[0]

    d1 = [  c1[0] , c1[ c1.shape[0]-1 ]  ]
    d2 = [  c2[0] , c2[ c2.shape[0]-1 ]  ]
    SN = [  c3[0] , c3[ c3.shape[0]-1 ]  ]
    SliceNumbers = range(SN[0],SN[1])


    im = im[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ] # Params['SliceNumbers']]
    mask = mask[ d1[0]:d1[1],d2[0]:d2[1],SliceNumbers ] # Params['SliceNumbers']]

    return im , mask , SliceNumbers

def funcPadding(im, mask):
    sz = mask.shape
    df = 238 - sz[0]
    p1 = [int(df/2) , df - int(df/2)]

    df = 238 - sz[1]
    p2 = [int(df/2) , df - int(df/2)]

    im = np.pad(im,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )
    mask = np.pad(mask,( (p1[0],p1[1]),(p2[0],p2[1]),(0,0) ),'constant' )

    return im , mask

def funcFlipLR_Upsampling(Params, im , mask):
    if 'Unlabeled' in Params['dataset']:

        for i in range(mask.shape[2]):
            im[...,i] = np.fliplr(im[...,i])
            mask[...,i] = np.fliplr(mask[...,i])

        if do_I_want_Upsampling == 1:
            mask = ndimage.zoom(mask,(1,1,2),order=0)
            im = ndimage.zoom(im,(1,1,2),order=3)
    else:
        im   = np.transpose(im,[0,2,1])
        mask = np.transpose(mask,[0,2,1])

        if im.shape[2] == 200:
            im = ndimage.zoom(im,(1,1,2),order=3)
            mask = ndimage.zoom(mask,(1,1,2),order=0)

    return im , mask

def readingImages(Params , subFolders,sFi):

    # for sFi in range(len(subFolders)):

    inputName = Params['TestName'].split('Test_')[1] + '.nii.gz'

    if Params['registrationFlag'] == 1:
        inputName = inputName + '_Deformed'

    print('Reading Images:  ',Params['NucleusName'],inputName.split('nii.gz')[0] , str(sFi) + ' ' + subFolders[sFi])

    maskF = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + Params['Name_priors_San_Label'])
    mask  = maskF.get_data()
    imF   = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + inputName )
    im    = imF.get_data()

    im = funcNormalize( im )

    if '1-THALAMUS' in Params['NucleusName']:
        print('--------------MyCrop2_Gap20')
        CropMask = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + 'MyCrop2_Gap20.nii.gz').get_data()
        im , mask , SliceNumbers = funcCropping(im , mask , CropMask)
    else:
        try:
            mskTh = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/Test/Results/' + subFolders[sFi] +'_1-THALAMUS_Logical.nii.gz').get_data()
            im , mask , SliceNumbers = funcCropping_FromThalamus(im , mask , mskTh)
        except:
            print('*************** unable to read full thalamus ***************')
            CropMask = nib.load(Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + 'MyCrop.nii.gz').get_data()
            im , mask , SliceNumbers = funcCropping(im , mask , CropMask)




    im , mask = funcFlipLR_Upsampling(Params, im , mask)
    # if 'Unlabeled' in Params['dataset']:
    #
    #     for i in range(mask.shape[2]):
    #         im[...,i] = np.fliplr(im[...,i])
    #         mask[...,i] = np.fliplr(mask[...,i])
    #
    #
    #     if do_I_want_Upsampling == 1:
    #         mask = ndimage.zoom(mask,(1,1,2),order=0)
    #         im = ndimage.zoom(im,(1,1,2),order=3)
    # else:
    #     im   = np.transpose(im,[0,2,1])
    #     mask = np.transpose(mask,[0,2,1])
    #
    #     if im.shape[2] == 200:
    #         im = ndimage.zoom(im,(1,1,2),order=3)
    #         mask = ndimage.zoom(mask,(1,1,2),order=0)

    if do_I_want_Upsampling == 1:
        maskF2 = nib.Nifti1Image(mask,maskF.affine)
        maskF2.get_header = maskF.header
        nib.save(maskF2,Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + Params['Name_priors_San_Label'].split('.nii.gz')[0] + '_US.nii.gz' )

        imF2 = nib.Nifti1Image(im,imF.affine)
        imF2.get_header = imF.header
        nib.save(imF2,Params['Dir_Prior'] + '/'  + subFolders[sFi] + '/' + inputName.split('.nii.gz')[0] + '_US.nii.gz' )

    imD_padded, maskD_padded = funcPadding(im, mask)


    return imD_padded, maskD_padded, SliceNumbers

UserEntries = input_GPU_Ix()

for ind in UserEntries['IxNuclei']: # 1,2,8,9,10,13]: #

    Params = initialDirectories(ind = ind, mode = UserEntries['mode'] , dataset = UserEntries['dataset'] , method = UserEntries['method'] )
    subFolders = subFoldersFunc(Params['Dir_Prior'])
    # subFolders = subFolders[:2]

    if Params['registrationFlag'] == 1:
        Params['Name_priors_San_Label'] = 'Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'
    else:
        Params['Name_priors_San_Label'] = 'Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz'

    for ii in UserEntries['enhanced_Index']: # len( Params['A'] ):

        Params['TestName'] = testNme(Params,ii)

        Params['Dir_EachTraining'] = mkDir(Params['Dir_AllTests'] + '/CNN' + Params['NucleusName'].replace('-','_') + '_2D_SanitizedNN/' + Params['TestName'])
        Params['Dir_All']  = mkDir(Params['Dir_AllTests'] + '/CNN' + Params['NucleusName'].replace('-','_') + '_2D_SanitizedNN/' + 'Test_AllTrainings' + '/Train')

        Params['dataset'] = UserEntries['dataset']

        for sFi in range(len(subFolders)):
            imD_padded, maskD_padded, SliceNumbers = readingImages(Params , subFolders,sFi)
            Params['SliceNumbers'] = SliceNumbers

            if UserEntries['testmode'] == 'onetrain':
                # print('----',UserEntries['testmode'])
                OneTrain_MultipleTest(UserEntries,Params,subFolders,imD_padded, maskD_padded, ii, sFi)

            # else:
            #     print('----',UserEntries['testmode'])
            #     normal_Cross_Validation(Params , subFolders , imFull , mskFull, ii)
