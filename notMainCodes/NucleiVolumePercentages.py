import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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

Results = np.zeros(15)

for DT in range(1,2): # ,4):
    for nuclei in [1,2,4,5,6,7,8,9,10,11,12,13,14]:
        # for dimension in range(3):

        dataset = WhichDataset(DT)

        Params = initialDirectories(ind = nuclei, mode = 'localPC' , dataset = dataset , method = 'old' )
        sub = subFoldersFunc(Params['Dir_Prior'])
        # im = nib.load(Params['Dir_Prior'] + '/' + sub[0] + '/Manual_Delineation_Sanitized/' + '5-VLa.nii.gz').get_data()
        # sz = im.shape
        # A = WhichDimension(dimension)
        #
        #
        # BB = np.zeros((sz[A[0]],len(sub)))

        Volume = []
        for i in range(len(sub)):
            print('nuclei: ',nuclei , 'sub ',i)
            im = nib.load(Params['Dir_Prior'] + '/' + sub[i] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz').get_data()
            sz = im.shape
            # print('nuclei: ',nuclei , 'dataset: ',dataset , 'dimension: ',dimension,'priors' + str(i),im.shape)
            # Results[dataset,sub[i]] = im.shape  # ,'dim: ' + str(dimension)

            if sz[1] > 250:
                Volume.append(np.sum(np.sum(np.sum(im))))

        print(Volume)
        Results[nuclei] = np.mean(Volume)

        # #     b = np.sum(im,axis=A[1])
        # #     b = np.sum(b,axis=A[2])
        # #     BB[:,i] = b
        # #
        # # mn = np.where(b>0)[0].min()
        # # mx = np.where(b>0)[0].max()
        # #
        # # Results[Params['NucleusName'],dataset,'dim: ' + str(dimension)] = [mn,mx]

Th = Results[1]
for nuclei in [1,2,4,5,6,7,8,9,10,11,12,13,14]:
    Results[nuclei] = Results[nuclei] / Th

Results*100
