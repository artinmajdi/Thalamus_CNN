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
    im = scipy.ndimage.zoom(im,(1,2,1),order=3) > 0.1

    return im


# for DT in range(1,2):
# for nuclei in [1]:
nuclei = 1
# dataset = WhichDataset(DT)
Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = 'Unlabeled' , method = 'old' )
sub = subFoldersFunc(Params['Dir_Prior'])

Results = np.zeros((len(sub),3,4))
for i in range(len(sub)):

    im = nib.load(Params['Dir_Prior'] + '/' + sub[i] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz').get_data()
    # im = nib.load(Params['Dir_Prior'] + '/' + sub[i] + '/WMnMPRAGE_bias_corr.nii.gz').get_data()
    print(im.shape)
    # print('sub ',i , im.shape , sub[i])

    # flag = 0
    # if im.shape[1] == 200:
    #     im = upsampleImage(im)
    #     flag = 1
    #
    # a = np.where(im != 0)
    # for dd in range(3):
    #     Results[i,dd,:] = [flag , a[dd].min() , a[dd].max() , im.shape[dd]]
    # print('sub ',i , im.shape , sub[i])


# im.shape
# Results[:,1,:]
#
# BoundingBox = np.zeros((3,2))
# Shape = np.zeros((3,2))
# for i in range(3):
#
#     Min = np.min(Results[:,i,1],axis=0)
#     # Mean1 = np.mean(Results[:,i,1],axis=0)
#     # Std1 = np.std(Results[:,i,1],axis=0)
#
#     Max = np.max(Results[:,i,2],axis=0)
#     # Mean2 = np.mean(Results[:,i,2],axis=0)
#     # Std2 = np.std(Results[:,i,2],axis=0)
#
#
#     BoundingBox[i,:] = [Min , Max]
#     Shape[i,:] = [np.min(Results[:,i,3],axis=0) , np.max(Results[:,i,3],axis=0)]
#
# BoundingBox
# Shape
# # Results[:,1,:]

































print('ll')
# plt.imshow(imm[...,120],cmap='gray')
# plt.show()
