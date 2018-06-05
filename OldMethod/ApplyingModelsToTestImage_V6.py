from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import pickle
import nibabel as nib
from TestData_V6 import TestData3
import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util
from skimage import filters

eps = np.finfo(float).eps

def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)
    return DiceCoef

def ThalamusExtraction(net , Directory_Nuclei_Test , Directory_Nuclei_Train , subFolders, CropDim , padSize , gpuNum):


    Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train + 'model.cpkt'


    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    SliceIdx = np.zeros(L)

    for sliceNum in range(L):
        Stng = TestData.data_files[sliceNum]
        d = Stng.find('Slice')
        SliceIdx[sliceNum] = int(Stng[d+5:].split('.')[0])

    SliceIdxArg = np.argsort(SliceIdx)
    Data , Label = TestData(len(SliceIdx))


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0
    PredictionFull_Thalamus = np.zeros((szD[0],148,148,2))
    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction = net.predict( Directory_Nuclei_Train_Model_cpkt, data, GPU_Num=gpuNum)
        PredictionFull_Thalamus[sliceNum,:,:,:] = prediction

    return PredictionFull_Thalamus

# 10-MGN_deformed.nii.gz	  13-Hb_deformed.nii.gz       4567-VL_deformed.nii.gz  6-VLP_deformed.nii.gz  9-LGN_deformed.nii.gz
# 11-CM_deformed.nii.gz	  1-THALAMUS_deformed.nii.gz  4-VA_deformed.nii.gz     7-VPL_deformed.nii.gz
# 12-MD-Pf_deformed.nii.gz  2-AV_deformed.nii.gz	      5-VLa_deformed.nii.gz    8-Pul_deformed.nii.gz


print("start ")
gpuNum = "5"
# NeucleusFolder = 'CNN12_MD_Pf_2D_SanitizedNN' # 'CNN5_Thalamus_2D_VTK' #'CNN_Thalamus' #
NucleusName = '8-Pul' # '6-VLP' #'4567-VL'  # '12-MD-Pf' # '1-THALAMUS' #
NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'


A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
SliceNumbers = range(107,140)

Directory_main = '/array/hdd/msmajdi/data/priors_forCNN/'
Directory_main_Test = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'

Directory_Nuclei_Full = Directory_main_Test + NeucleusFolder
Directory_Thalamus_Full = Directory_main_Test + 'CNN1_THALAMUS_2D_SanitizedNN'

for ii in range(len(A)):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
    Directory_Thalamus = Directory_Thalamus_Full + '/' + TestName + '/'

    # print Directory_Nuclei
    subFolders = os.listdir(Directory_Nuclei)

    for sFi in range(len(subFolders)):
        # sFi = 0
        print('-----------------------------------------------------------')
        print('Test: ' + str(A[ii]) + 'Subject: ' + str(subFolders[sFi]))
        print('-----------------------------------------------------------')

        Directory_Nuclei_Label   = Directory_main +  subFolders[sFi] + '/ManualDelineation/' + NucleusName + '_Deformed.nii.gz'
        Directory_Thalamus_Label = Directory_main +  subFolders[sFi] + '/ManualDelineation/' + '1-THALAMUS'+ '_Deformed.nii.gz'

        # print Directory_Nuclei_Test
        Directory_Nuclei_Test = Directory_Nuclei + subFolders[sFi] + '/Test/'
        Directory_Thalamus_Test = Directory_Thalamus + subFolders[sFi] +  '/Test/'

        Directory_Nuclei_Train = Directory_Nuclei + subFolders[sFi] + '/Train/'
        Directory_Thalamus_TrainedModel = Directory_Thalamus + subFolders[sFi] +  '/Train/model/'

	
        OriginalSeg = nib.load(Directory_Thalamus_Label) # ThalamusSegDeformed_Croped PulNeucleusSegDeformed_Croped
        OriginalSegNuclei = nib.load(Directory_Nuclei_Label)
        net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2)


        CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

        padSize = 90
        MultByThalamusFlag = 1
        [Prediction3D_Mult, Prediction3D_Mult_logical] = TestData3(net , MultByThalamusFlag , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSeg , subFolders[sFi] , CropDimensions , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum)


# OrigLabel = OriginalSegNuclei.get_data()[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1] , SliceNumbers]
# outputSeg = Prediction3D_Mult[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1] , SliceNumbers]
# outputSegLogic = Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1] , SliceNumbers]
#
# sn = 8
# # snB = CropDim[2,0] + int(SliceNumbers[sn])
# f, ax = plt.subplots(2,2)
# Thresh = max(filters.threshold_otsu(outputSeg[:,:,sn]),0.1)
# print Thresh
# ax[0,0].imshow(outputSeg[:,:,sn], cmap = 'gray')
# ax[0,1].imshow(OrigLabel[:,:,sn], cmap = 'gray')
# ax[1,0].imshow(outputSeg[:,:,sn]>Thresh, cmap = 'gray')
# ax[1,1].imshow(outputSegLogic[:,:,sn], cmap = 'gray')
# plt.show()
