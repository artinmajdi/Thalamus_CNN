from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import pickle
import nibabel as nib
from TestData_V5 import TestData3_PerSlice
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


gpuNum = "6"
NeucleusFolder = 'CNN5_Thalamus_2D_PerSlice' #'CNN_Thalamus' #
NucleusName = '1-THALAMUS' #'6-VLP' #

A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]] #
SliceNumbers = range(107,140)
Directory_main = '/array/hdd/msmajdi/data/priors_forCNN/'
Directory_main_Test = '/array/hdd/msmajdi/Tests/Thalamus_CNN/'

Directory_Nuclei_Full = Directory_main_Test + NeucleusFolder
Directory_Thalamus_Full = Directory_main_Test + 'CNN5_Thalamus_2D_PerSlice'

for ii in range(0,len(A)):
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
        MultByThalamusFlag = 0
        [Prediction3D_Mult, Prediction3D_Mult_logical] = TestData3_PerSlice(net , MultByThalamusFlag , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSeg , subFolders[sFi] , CropDimensions , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum)
    ## ---------------------------------------------

    # CropDim = CropDimensions
    #
    # OriginalSeg_Data = OriginalSeg.get_data()
    # Header = OriginalSeg.header
    # Affine = OriginalSeg.affine
    #
    # Prediction3D_Mult_logical = np.zeros(OriginalSeg_Data.shape)
    # Prediction3D_Mult = np.zeros(OriginalSeg_Data.shape)
    #
    # # Prediction3D_PureNuclei = np.zeros(OriginalSeg_Data.shape)
    #
    # # output_PureNuclei = np.zeros((33,148,148,2))
    # # output_Mult = np.zeros((33,148,148,2))
    # # InputData = np.zeros((33,148+90,148+90,1))
    # # InputLabel = np.zeros((33,148+90,148+90,2))
    #
    # trainer = unet.Trainer(net)
    #
    # PredictionFull_Thalamus = ThalamusExtraction(net , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , subFolders[sFi], CropDim , padSize , gpuNum)
    #
    # AA = OriginalSeg_Data[50:198 , 130:278 , SliceNumbers]
    #
    # L = len(SliceNumbers)
    #
    # for sliceInd in range(L):
    #     PredictionFull_Thalamus[sliceInd,:,:,1] = AA[:,:,sliceInd]
    #     PredictionFull_Thalamus[sliceInd,:,:,0] = 1 - AA[:,:,sliceInd]
    #
    #
    #
    # DiceCoefficient  = np.zeros(L)
    # LogLoss  = np.zeros(L)
    #
    # for sliceInd in range(L):
    #
    #     if sliceInd == 32:
    #         Directory_Nuclei_Train2 = Directory_Nuclei_Train + 'Slice' + str(31) + '/'
    #     else:
    #         Directory_Nuclei_Train2 = Directory_Nuclei_Train + 'Slice' + str(sliceInd) + '/'
    #
    #
    #     Directory_Nuclei_Test2  = Directory_Nuclei_Test  + 'Slice' + str(sliceInd) + '/'
    #
    #     Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train2 + 'model/model.cpkt'
    #     Directory_Nuclei_Test_Results   = Directory_Nuclei_Test2  + 'results_MultThalamus/'
    #
    #     try:
    #         os.stat(Directory_Nuclei_Test_Results)
    #     except:
    #         os.makedirs(Directory_Nuclei_Test_Results)
    #
    #     TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test2 + '*.tif' , shuffle_data=False)
    #     SliceIdx = np.zeros(L)
    #
    #     data , label = TestData(1)
    #
    #     # InputData[sliceInd,...] = data
    #     # InputLabel[sliceInd,...] = label
    #
    #     shiftFlag = 0
    #
    #     if shiftFlag == 1:
    #         shiftX = 0
    #         shiftY = 0
    #         data = np.roll(data,[0,shiftX,shiftY,0])
    #         label = np.roll(label,[0,shiftX,shiftY,0])
    #
    #     prediction2 = net.predict( Directory_Nuclei_Train_Model_cpkt, data)
    #
    #     prediction = np.zeros(prediction2.shape)
    #     # prediction[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[ SliceNumbers[sliceInd] ,:,:,:])
    #     prediction[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[sliceInd,:,:,:])
    #
    #     Prediction3D_Mult[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = prediction[0,...,1]
    #     # output_Mult[ sliceInd , ... ] = prediction[0,...]
    #     # output_PureNuclei[ sliceInd , ... ] = prediction2[0,...]
    #
    #     Thresh = max(filters.threshold_otsu(prediction[0,...,1]),0.2)
    #     # print 'Otsu Thresh: ' + str(Thresh)
    #     PredictedSeg = prediction[0,...,1] > Thresh
    #     Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = PredictedSeg
    #
    #     # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
    #
    #     sz = label.shape
    #     AA  = prediction*0
    #     print prediction.shape
    #     AA[0,:,:,:] = PredictionFull_Thalamus[sliceInd,:,:,:]
    #
    #     A = (padSize/2)
    #     imgCombined = util.combine_img_prediction(data, label, prediction)
    #     imgCombined2 = util.combine_img_prediction(data, label, AA)
    #     DiceCoefficient[sliceInd] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
    #     util.save_image(imgCombined, Directory_Nuclei_Test_Results+"prediction_slice"+ str(SliceNumbers[sliceInd]) + ".jpg")
    #     util.save_image(imgCombined2, Directory_Nuclei_Test_Results+"prediction_slice"+ str(SliceNumbers[sliceInd]) + "2.jpg")
    #
    #     Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
    #     LogLoss[sliceInd] = np.log10(Loss)
    # print PredictionFull_Thalamus.shape
    # print PredictionFull_Thalamus.shape
    # Directory_Nuclei_TestOutput = Directory_Nuclei_Test + 'ResultsMultThalamus/'
    # # print Directory_Nuclei_TestOutput
    # try:
    #     os.stat(Directory_Nuclei_TestOutput)
    # except:
    #     os.makedirs(Directory_Nuclei_TestOutput)
    #
    # np.savetxt(Directory_Nuclei_TestOutput + 'DiceCoefficient.txt',DiceCoefficient)
    # np.savetxt(Directory_Nuclei_TestOutput + 'LogLoss.txt',LogLoss)
    #
    # Prediction3D_nifti = nib.Nifti1Image(Prediction3D_Mult,Affine)
    # Prediction3D_nifti.get_header = Header
    #
    # nib.save(Prediction3D_nifti,Directory_Nuclei_TestOutput + subFolders[sFi] + '_' + NucleusName + '.nii.gz')
    #
    # Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_Mult_logical,Affine)
    # Prediction3D_logical_nifti.get_header = Header
    #
    # nib.save(Prediction3D_logical_nifti,Directory_Nuclei_TestOutput + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz')



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
