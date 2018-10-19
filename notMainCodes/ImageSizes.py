import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
from skimage.measure import regionprops
from skimage.morphology import label

dir2 = '/media/data1/artin/thomas/priors/ET/7T'
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

def WhichDataset(d):
    if d == 1:
        return '20priors'
    elif d == 2:
        return 'MS'
    else:
        return 'ET_7T'

def WhichDimension(d):
    if d == 0:
        return [0,2,1]
    elif d == 1:
        return [1,0,1]
    else:
        return [2,1,0]

def subFoldersFunc(Dir_Prior):
    subFolders = []
    subFlds = os.listdir(Dir_Prior)
    for i in range(len(subFlds)):
        if subFlds[i][:5] == 'vimp2':
            subFolders.append(subFlds[i])

    return subFolders

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
    Params['SliceNumbers'] = SliceNumbers
    Params['NucleusName']  = NucleusName

    return Params

def upsampleImage(im):
    im = scipy.ndimage.zoom(im,(1,1,2),order=3) > 0.1
    return im


# nuclei = 1
# Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = 'MS' , method = 'old' )
# sub = subFoldersFunc(Params['Dir_Prior'])
# im_MS = nib.load(Params['Dir_Prior'] + '/' + sub[1] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
# im_MS = np.transpose(im_MS,[0,2,1])
#
# Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = '20priors' , method = 'old' )
# sub = subFoldersFunc(Params['Dir_Prior'])
# im_20priors = nib.load(Params['Dir_Prior'] + '/' + sub[1] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
# im_20priors = np.transpose(im_20priors,[0,2,1])
#
# Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = 'ET_7T' , method = 'old' )
# sub = subFoldersFunc(Params['Dir_Prior'])
# im_ET_7T = nib.load(Params['Dir_Prior'] + '/' + sub[1] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
# im_ET_7T = np.transpose(im_ET_7T,[0,2,1])
#
# Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = 'Unlabeled' , method = 'old' )
# sub = subFoldersFunc(Params['Dir_Prior'])
# im_Unlabeled = nib.load(Params['Dir_Prior'] + '/' + sub[1] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
# for i in range(im_Unlabeled.shape[2]):
#     im_Unlabeled[...,i] = np.fliplr(im_Unlabeled[...,i])
#
# im_Unlabeled = scipy.ndimage.zoom(im_Unlabeled,(1,1,2),order=3)
#
# fig , axs = plt.subplots(1,4)
# axs[0].imshow(im_20priors[63:147,67:174,170],cmap='gray')
# axs[1].imshow(im_MS[63:147,67:174,185],cmap='gray')
# axs[2].imshow(im_ET_7T[63:147,67:174,190],cmap='gray')
# axs[3].imshow(im_Unlabeled[63:147,67:174,200],cmap='gray')
# plt.show()

# fig , axs = plt.subplots(1,4)
# axs[0].imshow(im_20priors[...,170],cmap='gray')
# axs[1].imshow(im_MS[...,195],cmap='gray')
# axs[2].imshow(im_ET_7T[...,190],cmap='gray')
# axs[3].imshow(im_Unlabeled[...,200],cmap='gray')
# plt.show()


dataset = 'Unlabeled''
Params = initialDirectories(ind = 1, mode = 'localPC' , dataset =  dataset , method = 'old' )
sub = subFoldersFunc(Params['Dir_Prior'])
Params['dataset'] = dataset
# im = nib.load(Params['Dir_Prior'] + '/' + sub[1] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz').get_data()

Results = np.zeros((len(sub),3,3))
for i in range(len(sub)):

    name = Params['NucleusName'] + '.nii.gz'

    mask = nib.load(Params['Dir_Prior'] + '/' + sub[i] + '/Manual_Delineation_Sanitized/' + name).get_data()
    # im = nib.load(Params['Dir_Prior'] + '/' + sub[i] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()

    if 'Unlabeled' in Params['dataset']:
        for k in range(mask.shape[2]):
            # im[...,i]   = np.fliplr(im[...,i])
            mask[...,k] = np.fliplr(mask[...,k])
    else:
        # im   = np.transpose(im,[0,2,1])
        mask = np.transpose(mask,[0,2,1])

        if mask.shape[2] == 200:
            # im = scipy.ndimage.zoom(im,(1,1,2),order=3)
            mask = scipy.ndimage.zoom(mask,(1,1,2),order=3) > 0.1

    a = np.where(mask != 0)
    for dd in range(3):
        Results[i,dd,:] = [a[dd].min() , a[dd].max() , mask.shape[dd]]
    print('sub ',i , mask.shape , sub[i])



for dd in range(3):
    print(Results[:,dd,0].min(axis=0),Results[:,dd,1].max(axis=0),Results[:,dd,2].min(axis=0))

# 20priors
#  87, 130, 256  -> 43
# 100, 156, 256  -> 56
# 145, 234, 392  -> 89

# Unlabeled
# 125 , 172 , 256  -> 47
#  87 , 164 , 256   -> 77
#  67 , 123 , 180   -> 56

# ET_7T
#  83, 130, 256    -> 38
#  94, 160, 256    -> 62
# 156, 240, 439    -> 55


# MS
#  87, 127, 256   -> 35
#  99, 154, 256   -> 44
# 150, 233, 392   -> 49

A = np.zeros((4,3,3))      # 20priors
A[0,0,:] = [87,130,256]
A[0,1,:] = [99,154,256]
A[0,2,:] = [150,233,392]

A[1,0,:] = [125,172,256]   # Unlabeled
A[1,1,:] = [87,164,256]
A[1,2,:] = [67*2,123*2,180*2]

A[2,0,:] = [83,130,256]   # ET_7T
A[2,1,:] = [94,160,256]
A[2,2,:] = [156,240,439]

A[3,0,:] = [87,127,256]   # MS
A[3,1,:] = [99,154,256]
A[3,2,:] = [150,233,392]

A[...,0]
np.min(A[:,:,0],axis=0)
A[...,1]
np.max(A[:,:,1],axis=0)

A[...,2]
np.max(A[...,2],axis=0)
