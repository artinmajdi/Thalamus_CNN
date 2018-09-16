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
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2'
        elif '7T_MS' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/7T_MS'
        elif 'ET' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/ET'

    elif 'localPC' in mode:

        if '20Priors' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors'
        elif '7T_MS' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/NewPriors/7T_MS'
        elif 'ET' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/NewPriors/ET'


    elif 'flash' in mode:

        Dir_Prior = '/media/artin/aaa/Manual_Delineation_Sanitized_Full'

    elif 'server' in mode:

        if '20Priors' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/priors_forCNN_Ver2'
        elif '7T_MS' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/newPriors/7T_MS'
        elif 'ET' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/newPriors/ET'


    return Dir_Prior

Directory = initialDirectories(mode = 'localPC', dataset = 'ET')

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
