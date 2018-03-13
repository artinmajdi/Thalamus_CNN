import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util

eps = np.finfo(float).eps

def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef


def TestData(net , Test_Path , Train_Path , OriginalSeg_Data , Header , Affine , subFolders, CropDim , padSize):


    Trained_Model_Path = Train_Path + 'model.cpkt'
    TestResults_Path   = Test_Path  + 'results/'

    try:
        os.stat(TestResults_Path)
    except:
        os.makedirs(TestResults_Path)


    # OriginalSeg_Data = OriginalSeg.get_data()
    # Header = OriginalSeg.header
    # Affine = OriginalSeg.affine


    Prediction3D_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D = np.zeros(OriginalSeg_Data.shape)

    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(  Test_Path + '*.tif',shuffle_data=False)

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
    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction = net.predict( Trained_Model_Path, data)
        print('prediction size:' + str(prediction.shape) )
        
        Prediction3D[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceIdx[sliceNum])] = prediction[0,...,1]
        Prediction3D_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceIdx[sliceNum])] = prediction[0,...,1] > 0.2

        # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
        PredictedSeg = prediction[0,...,1] > 0.2
        sz = label.shape

        A = (padSize/2)
        imgCombined = util.combine_img_prediction(data, label, prediction)
        DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(imgCombined, TestResults_Path+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")


        Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[sliceNum] = np.log10(Loss+eps)

    np.savetxt(TestResults_Path+'DiceCoefficient.txt',DiceCoefficient)
    np.savetxt(TestResults_Path+'LogLoss.txt',LogLoss)

    Prediction3D_nifti = nib.Nifti1Image(Prediction3D,Affine)
    Prediction3D_nifti.get_header = Header

    nib.save(Prediction3D_nifti,TestResults_Path + subFolders + '_PredictedSegment.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_logical,Affine)
    Prediction3D_logical_nifti.get_header = Header

    nib.save(Prediction3D_logical_nifti,TestResults_Path + subFolders + '_PredictedSegment_logical.nii.gz')


    return data,label,prediction
    # ax[0].imshow(data[0,20:sz[1]-20,20:sz[2]-20,0], aspect="auto",cmap = 'gray')
    # ax[1].imshow(label[0,20:sz[1]-20,20:sz[2]-20,1], aspect="auto",cmap = 'gray')
    # ax[2].imshow(prediction[0,...,1], aspect="auto",cmap = 'gray')
    # ax[3].imshow(OriginalSeg, aspect="auto",cmap = 'gray')
    # ax[0].set_title("Input")
    # ax[1].set_title("Ground truth")
    # ax[2].set_title("Prediction")
    # ax[3].set_title("Prediction OriginalSeg")
    # plt.show()

def ThalamusExtraction(net , Test_Path , Train_Path , subFolders, CropDim , padSize):


    Trained_Model_Path = Train_Path + 'model.cpkt'


    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(  Test_Path + '*.tif',shuffle_data=False)

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    SliceIdx = np.zeros(L)

    for sliceNum in range(L):
        Stng = TestData.data_files[sliceNum]
        d = Stng.find('slice')
        SliceIdx[sliceNum] = int(Stng[d+5:].split('.')[0])

    SliceIdxArg = np.argsort(SliceIdx)
    Data , Label = TestData(len(SliceIdx))


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0
    PredictionFull = np.zeros((szD[0],148,148,2))
    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction = net.predict( Trained_Model_Path, data)
        PredictionFull[sliceNum,:,:,:] = prediction

    return PredictionFull

def TestData2_MultipliedByWholeThalamus(net , Test_Path , Train_Path , OriginalSeg , subFolders, CropDim , padSize , Test_Path_Thalamus , Trained_Model_Path_Thalamus):


    Trained_Model_Path = Train_Path + 'model.cpkt'
    TestResults_Path   = Test_Path  + 'results/'

    try:
        os.stat(TestResults_Path)
    except:
        os.makedirs(TestResults_Path)


    OriginalSeg_Data = OriginalSeg.get_data()
    Header = OriginalSeg.header
    Affine = OriginalSeg.affine


    Prediction3D_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D = np.zeros(OriginalSeg_Data.shape)

    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(  Test_Path + '*.tif',shuffle_data=False)

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    SliceIdx = np.zeros(L)

    for sliceNum in range(L):
        Stng = TestData.data_files[sliceNum]
        d = Stng.find('slice')
        SliceIdx[sliceNum] = int(Stng[d+5:].split('.')[0])

    SliceIdxArg = np.argsort(SliceIdx)
    Data , Label = TestData(len(SliceIdx))


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0

    PredictionFull = ThalamusExtraction(net , Test_Path_Thalamus , Trained_Model_Path_Thalamus , subFolders, CropDim , padSize)
    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction2 = net.predict( Trained_Model_Path, data)

        prediction = np.zeros(prediction2.shape)
        prediction[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull[sliceNum,:,:,:])

        Prediction3D[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceIdx[sliceNum])] = prediction[0,...,1]
        Prediction3D_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceIdx[sliceNum])] = prediction[0,...,1] > 0.5

        # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
        PredictedSeg = prediction[0,...,1] > 0.2
        sz = label.shape

        A = (padSize/2)
        imgCombined = util.combine_img_prediction(data, label, prediction)
        DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(imgCombined, TestResults_Path+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")


        Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[sliceNum] = np.log10(Loss)

    np.savetxt(TestResults_Path+'DiceCoefficient.txt',DiceCoefficient)
    np.savetxt(TestResults_Path+'LogLoss.txt',LogLoss)

    Prediction3D_nifti = nib.Nifti1Image(Prediction3D,Affine)
    Prediction3D_nifti.get_header = Header

    nib.save(Prediction3D_nifti,TestResults_Path + subFolders + '_ThalamusSegDeformed_Croped_Predicted.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_logical,Affine)
    Prediction3D_logical_nifti.get_header = Header

    nib.save(Prediction3D_logical_nifti,TestResults_Path + subFolders + '_ThalamusSegDeformed_Croped_Predicted_logical.nii.gz')


    return data,label,prediction,OriginalSeg
    # ax[0].imshow(data[0,20:sz[1]-20,20:sz[2]-20,0], aspect="auto",cmap = 'gray')
    # ax[1].imshow(label[0,20:sz[1]-20,20:sz[2]-20,1], aspect="auto",cmap = 'gray')
    # ax[2].imshow(prediction[0,...,1], aspect="auto",cmap = 'gray')
    # ax[3].imshow(OriginalSeg, aspect="auto",cmap = 'gray')
    # ax[0].set_title("Input")
    # ax[1].set_title("Ground truth")
    # ax[2].set_title("Prediction")
    # ax[3].set_title("Prediction OriginalSeg")
    # plt.show()
