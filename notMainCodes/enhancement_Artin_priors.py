import nibabel as nib
# import Image
import matplotlib.pylab as plt
from PIL import ImageEnhance , Image , ImageFilter
import numpy as np
import nifti
import os
from glob import glob

def initialDirectories(mode = 'local', dataset = 'new12'):

    if 'localLT' in mode:


        if '20Priors' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/20priors'
        elif '7T_MS' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/Unlabeled'

    elif 'localPC' in mode:

        if '20Priors' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors'
        elif '7T_MS' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/Unlabeled'

    elif 'flash' in mode:

        Dir_Prior = '/media/artin/aaa/Manual_Delineation_Sanitized_Full'

    elif 'server' in mode:

        if '20Priors' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/20priors'
        elif '7T_MS' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/Unlabeled'


    return Dir_Prior

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = 1
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'new'
    UserEntries['testMode'] = 'EnhancedSeperately' # 'AllTrainings'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['epochs'] = 'nan'
    UserEntries['Flag_cross_entropy'] = 0


    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]

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

        elif input.split('=')[0] == 'epochs':
            UserEntries['epochs'] = int(input.split('=')[1])

        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]

        elif input.split('=')[0] == '--cross_entropy':
            UserEntries['Flag_cross_entropy'] = 1


    if UserEntries['Flag_cross_entropy'] == 1:
        UserEntries['cost_kwargs'] = {'class_weights':[0.7,0.3]}
        UserEntries['modelName'] = 'model_CE/'
        UserEntries['resultName'] = 'Results_CE/'
    else:
        UserEntries['modelName'] = 'model/'
        UserEntries['resultName'] = 'Results/'

    return UserEntries

UserEntries = input_GPU_Ix()
Directory = initialDirectories(mode = UserEntries['mode'], dataset = UserEntries['dataset'])

# EnhMethod = 'Sharpness' #'Contrast' # Sharpness   +'_int16DivideMultipliedBy7'  # Contrast
def enhancing(im , scaleEnhance):
    # im = im.astype(dtype=np.int16)
    im = Image.fromarray(im)
    im = im.convert('L')

    if scaleEnhance[0] != 1:
        im2 = ImageEnhance.Sharpness(im)
        im = im2.enhance(scaleEnhance[0])

    if scaleEnhance[1] != 1:
        im2 = ImageEnhance.Contrast(im)
        im = im2.enhance(scaleEnhance[1])

    return im
# Directory = '/media/artin/D0E2340CE233F576/Thalamus_Segmentation/Data/NewPriors/test/ET'


subDirsFull = glob(Directory+'/*/')

# subDirs = subDirsFull[0]
scaleEnhance = [[1,2],[1,3],[4,1],[6,1],[4,2],[4,3]]


for subDirs in subDirsFull:
    print(subDirs)
    for Nm in ['WMnMPRAGE_bias_corr_Deformed']: #  'WMnMPRAGE_Deformed' 'WMnMPRAGE_bias_corr' , 'WMnMPRAGEdeformed']:

        # a = glob(subDirs + Nm + '.nii.gz')
        im = nib.load(subDirs + Nm + '.nii.gz')
        # name = a[0].split(subDirs)[1].split('.nii.gz')[0]

        imD = im.get_data()
        MaxValue = imD.max()
        # imD = imD.astype(float)*256/imD.max()
        imD = imD*256/imD.max()

        sz = imD.shape

        for s in scaleEnhance:
            print(s)
            imEnhanced = np.zeros(sz)
            for i in range(sz[2]):
                imEnhanced[:,:,i] = enhancing(imD[:,:,i] , s)

            imEnhanced = imEnhanced/256*MaxValue
            imEnhanced_nifti = nib.Nifti1Image(imEnhanced , im.affine , im.header)

            string = subDirs + 'WMnMPRAGE_bias_corr' + '_' + 'Sharpness_' + str(s[0]) + '_Contrast_' + str(s[1]) + '_Deformed.nii.gz'
            print(string)

            nib.save(imEnhanced_nifti,string)

            # sliceNum = 40
            # fig , axes = plt.subplots(1,2 , figsize=(10,5))
            # axes[0].imshow(imEnhanced_nifti.get_data()[:,:,sliceNum],cmap='gray',aspect='auto')
            # axes[1].imshow(imD[:,:,sliceNum],cmap='gray',aspect='auto')
            # plt.show()

EnhanceThomas = 0
if EnhanceThomas == 1:
    Directory = '/media/data1/artin/thomas/'
    scaleEnhance = [[1,2],[1,3],[4,1],[6,1],[4,2],[4,3]]
    for name in ['templ_93x187x68']:

        im = nib.load(Directory + name + '.nii.gz')
        imD = im.get_data()
        MaxValue = imD.max()
        # imD = imD.astype(float)*256/imD.max()
        imD = imD*256/imD.max()

        sz = imD.shape

        for s in scaleEnhance:
            print(s)
            imEnhanced = np.zeros(sz)
            for i in range(sz[2]):
                imEnhanced[:,:,i] = enhancing(imD[:,:,i] , s)

            imEnhanced = imEnhanced/256*MaxValue
            imEnhanced_nifti = nib.Nifti1Image(imEnhanced , im.affine , im.header)

            string = Directory + name + '_' + 'Sharpness_' + str(s[0]) + '_Contrast_' + str(s[1]) + '.nii.gz'
            print(string)

            nib.save(imEnhanced_nifti,string)
