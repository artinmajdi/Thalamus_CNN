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

def TestData(net , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSeg , subFolders, CropDim , padSize , gpuNum):


    Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train + 'model/model.cpkt'
    Directory_Nuclei_Test_Results     = Directory_Nuclei_Test  + 'results/'

    try:
        os.stat(Directory_Nuclei_Test_Results)
    except:
        os.makedirs(Directory_Nuclei_Test_Results)


    OriginalSeg_Data = OriginalSeg.get_data()
    Header = OriginalSeg.header
    Affine = OriginalSeg.affine


    Prediction3D_Mult_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_Mult = np.zeros(OriginalSeg_Data.shape)

    trainer = unet.Trainer(net)

    TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)

    L = len(TestData.data_files)
    DiceCoefficient  = np.zeros(L)
    LogLoss  = np.zeros(L)
    SliceIdx = np.zeros(L)

    # for sliceNum in range(L):
    #     Stng = TestData.data_files[sliceNum]
    #     d = Stng.find('Slice')
    #     SliceIdx[sliceNum] = int(Stng[d+5:].split('.')[0])
    #
    # SliceIdxArg = np.argsort(SliceIdx)
    Data , Label = TestData(1)


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0
    # for sliceNum in SliceIdxArg:
    sliceNum = 0

    data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
    label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

    if shiftFlag == 1:
        shiftX = 0
        shiftY = 0
        data = np.roll(data,[0,shiftX,shiftY,0])
        label = np.roll(label,[0,shiftX,shiftY,0])

    prediction = net.predict( Directory_Nuclei_Train_Model_cpkt, data, GPU_Num=gpuNum)

    # PredictedSeg = prediction[0,...,1] > 0.2
    Thresh = max(filters.threshold_otsu(prediction[0,...,1]),0.2)
    PredictedSeg = prediction[0,...,1] > Thresh


    Prediction3D_Mult[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = prediction[0,...,1]
    Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = PredictedSeg

    # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
    sz = label.shape

    A = (padSize/2)
    imgCombined = util.combine_img_prediction(data, label, prediction)
    DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
    util.save_image(imgCombined, Directory_Nuclei_Test_Results+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")


    Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
    LogLoss[sliceNum] = np.log10(Loss+eps)

    np.savetxt(Directory_Nuclei_Test_Results+'DiceCoefficient.txt',DiceCoefficient)
    np.savetxt(Directory_Nuclei_Test_Results+'LogLoss.txt',LogLoss)

    # Prediction3D_nifti = nib.Nifti1Image(Prediction3D_Mult,Affine)
    # Prediction3D_nifti.get_header = Header

    # nib.save(Prediction3D_nifti,Directory_Nuclei_Test_Results + subFolders + '_PredictedSegment.nii.gz')

    # Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_Mult_logical,Affine)
    # Prediction3D_logical_nifti.get_header = Header
    #
    # nib.save(Prediction3D_logical_nifti,Directory_Nuclei_Test_Results + subFolders + '_PredictedSegment_logical.nii.gz')


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

def TestData2_MultThalamus(net , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSeg , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , gpuNum):


    Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train + 'model/model.cpkt'
    Directory_Nuclei_Test_Results   = Directory_Nuclei_Test  + 'results/'

    try:
        os.stat(Directory_Nuclei_Test_Results)
    except:
        os.makedirs(Directory_Nuclei_Test_Results)


    OriginalSeg_Data = OriginalSeg.get_data()
    Header = OriginalSeg.header
    Affine = OriginalSeg.affine


    Prediction3D_Mult_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_Mult = np.zeros(OriginalSeg_Data.shape)

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

    PredictionFull_Thalamus = ThalamusExtraction(net , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , subFolders, CropDim , padSize)
    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction2 = net.predict( Directory_Nuclei_Train_Model_cpkt, data, GPU_Num=gpuNum)

        prediction = np.zeros(prediction2.shape)
        prediction[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[sliceNum,:,:,:])

        # PredictedSeg = prediction[0,...,1] > 0.2
        Thresh = max(filters.threshold_otsu(prediction[0,...,1]),0.2)
        PredictedSeg = prediction[0,...,1] > Thresh

        Prediction3D_Mult[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = prediction[0,...,1]
        Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = PredictedSeg

        # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))



        sz = label.shape

        A = (padSize/2)
        imgCombined = util.combine_img_prediction(data, label, prediction)
        DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(imgCombined, Directory_Nuclei_Test_Results+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")


        Loss = unet.error_rate(prediction,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[sliceNum] = np.log10(Loss)

    np.savetxt(Directory_Nuclei_Test_Results + 'DiceCoefficient.txt',DiceCoefficient)
    np.savetxt(Directory_Nuclei_Test_Results + 'LogLoss.txt',LogLoss)

    Prediction3D_nifti = nib.Nifti1Image(Prediction3D_Mult,Affine)
    Prediction3D_nifti.get_header = Header

    nib.save(Prediction3D_nifti,Directory_Nuclei_Test_Results + subFolders + '_' + NucleusName + '.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_Mult_logical,Affine)
    Prediction3D_logical_nifti.get_header = Header

    nib.save(Prediction3D_logical_nifti,Directory_Nuclei_Test_Results + subFolders + '_' + NucleusName + '_Logical.nii.gz')


    return data,label,prediction,OriginalSeg

    # f, ax = plt.subplots(1, 3, sharey=True)
    # ax[0].imshow(data[0,20:sz[1]-20,20:sz[2]-20,0], aspect="auto",cmap = 'gray')
    # ax[1].imshow(label[0,20:sz[1]-20,20:sz[2]-20,1], aspect="auto",cmap = 'gray')
    # ax[2].imshow(prediction[0,...,1], aspect="auto",cmap = 'gray')
    # ax[3].imshow(OriginalSeg, aspect="auto",cmap = 'gray')
    # ax[0].set_title("Input")
    # ax[1].set_title("Ground truth")
    # ax[2].set_title("Prediction")
    # ax[3].set_title("Prediction OriginalSeg")
    # plt.show()

def TestData3_PerSlice(net , MultByThalamusFlag, Directory_Nuclei_Test0 , Directory_Nuclei_Train0 , OriginalSeg , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum):

    # MultByThalamusFlag :
    # 0: Not Multiplied by Thalamus
    # 1: Multiplied by predicted Thalamus
    # 2: Multiplied by manual Thalamus

    OriginalSeg_Data = OriginalSeg.get_data()
    Header = OriginalSeg.header
    Affine = OriginalSeg.affine

    Prediction3D_Mult_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_Mult = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_PureNuclei = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_PureNuclei_logical = np.zeros(OriginalSeg_Data.shape)
    sz = OriginalSeg_Data.shape
    PredictionFull_Thalamus = np.zeros((len(SliceNumbers),148,148,2))

    trainer = unet.Trainer(net)

    if MultByThalamusFlag == 1:
        try:
            PredictionFull_Thalamus = ThalamusExtraction(net , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , subFolders, CropDim , padSize)
            outputFolder = 'Results_MultByPredictedThalamus/'
        except:
            MultByThalamusFlag = 0
            print('Error: Could not load Thalamus')

    elif MultByThalamusFlag == 2:
        AA = OriginalSeg_Data[50:198 , 130:278 , SliceNumbers]
        L = len(SliceNumbers)
        for sliceInd in range(L):
            PredictionFull_Thalamus[sliceInd,:,:,1] = AA[:,:,sliceInd]
            PredictionFull_Thalamus[sliceInd,:,:,0] = 1 - AA[:,:,sliceInd]


    L = len(SliceNumbers)
    DiceCoefficient  = np.zeros(L)
    DiceCoefficient_Mult  = np.zeros(L)
    LogLoss  = np.zeros(L)
    LogLoss_Mult  = np.zeros(L)

    for sliceInd in range(L):

        # if sliceInd == 32:
        #     Directory_Nuclei_Train = Directory_Nuclei_Train0 + 'Slice' + str(31) + '/'
        # else:
        #     Directory_Nuclei_Train = Directory_Nuclei_Train0 + 'Slice' + str(sliceInd) + '/'

        Directory_Nuclei_Train = Directory_Nuclei_Train0 + 'Slice' + str(sliceInd) + '/'
        Directory_Nuclei_Test  = Directory_Nuclei_Test0 + 'Slice' + str(sliceInd) + '/'

        Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train + 'model/model.cpkt'

        if MultByThalamusFlag != 0:
            Directory_Test_Results_Thalamus = Directory_Nuclei_Test0 + 'Results_MultByManualThalamus/'
            try:
                os.stat(Directory_Test_Results_Thalamus)
            except:
                os.makedirs(Directory_Test_Results_Thalamus)

        Directory_Test_Results_Nuclei = Directory_Nuclei_Test + 'Results/'

        try:
            os.stat(Directory_Test_Results_Nuclei)
        except:
            os.makedirs(Directory_Test_Results_Nuclei)

        TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)
        SliceIdx = np.zeros(L)

        data , label = TestData(1)

        shiftFlag = 0

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction2 = net.predict( Directory_Nuclei_Train_Model_cpkt, data, GPU_Num=gpuNum)

        if MultByThalamusFlag != 0:

            prediction_Mult = np.zeros(prediction2.shape)
            # prediction_Mult[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[ SliceNumbers[sliceInd] ,:,:,:])
            prediction_Mult[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[sliceInd,:,:,:])

            # PredictedSeg = prediction_Mult[0,...,1] > 0.1
            try:
                Thresh_Mult = max(filters.threshold_otsu(prediction_Mult[0,...,1]),0.2)
            except:
                Thresh_Mult = 0.2

            PredictedSeg_Mult = prediction_Mult[0,...,1] > Thresh_Mult

        try:
            Thresh = max(filters.threshold_otsu(prediction2[0,...,1]),0.2)
        except:
            Thresh = 0.2

        PredictedSeg = prediction2[0,...,1] > Thresh

        if MultByThalamusFlag != 0:
            Prediction3D_Mult[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = prediction_Mult[0,...,1]
            Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = PredictedSeg_Mult

        Prediction3D_PureNuclei[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = prediction2[0,...,1]
        Prediction3D_PureNuclei_logical[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceInd]) ] = PredictedSeg

        # unet.error_rate(prediction_Mult, util.crop_to_shape(label, prediction_Mult.shape))

        sz = label.shape

        A = int((padSize/2))
        imgCombined = util.combine_img_prediction(data, label, prediction2)
        DiceCoefficient[sliceInd] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(imgCombined, Directory_Test_Results_Nuclei+"prediction_slice" + str(sliceInd) +'_'+ str(SliceNumbers[sliceInd]) + ".jpg")

        Loss = unet.error_rate(prediction2,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[sliceInd] = np.log10(Loss)

        if MultByThalamusFlag != 0:
            AA  = prediction_Mult*0
            AA[0,:,:,:] = PredictionFull_Thalamus[sliceInd,:,:,:]

            A = (padSize/2)
            imgCombined = util.combine_img_prediction(data, label, prediction_Mult)
            imgCombined2 = util.combine_img_prediction(data, label, AA)
            DiceCoefficient_Mult[sliceInd] = DiceCoefficientCalculator(PredictedSeg_Mult,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
            util.save_image(imgCombined, Directory_Test_Results_Thalamus +"prediction_slice" + str(sliceInd) +'_'+ str(SliceNumbers[sliceInd]) + ".jpg")
            util.save_image(imgCombined2, Directory_Test_Results_Thalamus+"prediction_slice" + str(sliceInd) +'_'+ str(SliceNumbers[sliceInd]) + "2.jpg")

            Loss = unet.error_rate(prediction_Mult,label[:,A:sz[1]-A,A:sz[2]-A,:])
            LogLoss_Mult[sliceInd] = np.log10(Loss)


    np.savetxt(Directory_Nuclei_Test0 + 'DiceCoefficient.txt',DiceCoefficient)
    np.savetxt(Directory_Nuclei_Test0 + 'LogLoss.txt',LogLoss)

    Prediction3D_nifti = nib.Nifti1Image(Prediction3D_PureNuclei,Affine)
    Prediction3D_nifti.get_header = Header
    nib.save(Prediction3D_nifti,Directory_Nuclei_Test0 + subFolders + '_' + NucleusName + '.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_PureNuclei_logical,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti,Directory_Nuclei_Test0 + subFolders + '_' + NucleusName + '_Logical.nii.gz')

    if MultByThalamusFlag != 0:
        np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient.txt',DiceCoefficient_Mult)
        np.savetxt(Directory_Test_Results_Thalamus + 'LogLoss.txt',LogLoss_Mult)

        Prediction3D_nifti = nib.Nifti1Image(Prediction3D_Mult,Affine)
        Prediction3D_nifti.get_header = Header
        nib.save(Prediction3D_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '.nii.gz')

        Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_Mult_logical,Affine)
        Prediction3D_logical_nifti.get_header = Header
        nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '_Logical.nii.gz')

    return Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical

def TestData3(net , MultByThalamusFlag, Directory_Nuclei_Test0 , Directory_Nuclei_Train0 , OriginalSeg , subFolders, CropDim , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName , SliceNumbers , gpuNum):


    Directory_Nuclei_Train_Model_cpkt = Directory_Nuclei_Train0 + 'model/model.cpkt'
    # Directory_Nuclei_Test_Results   = Directory_Nuclei_Test  + 'results/'

    if MultByThalamusFlag != 0:
        Directory_Test_Results_Thalamus = Directory_Nuclei_Test0 + 'Results_MultByManualThalamus/'
        try:
            os.stat(Directory_Test_Results_Thalamus)
        except:
            os.makedirs(Directory_Test_Results_Thalamus)

    Directory_Test_Results_Nuclei = Directory_Nuclei_Test0 + 'Results/'

    try:
        os.stat(Directory_Test_Results_Nuclei)
    except:
        os.makedirs(Directory_Test_Results_Nuclei)

    OriginalSeg_Data = OriginalSeg.get_data()
    Header = OriginalSeg.header
    Affine = OriginalSeg.affine

    Prediction3D_Mult_logical = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_Mult = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_PureNuclei = np.zeros(OriginalSeg_Data.shape)
    Prediction3D_PureNuclei_logical = np.zeros(OriginalSeg_Data.shape)
    sz = OriginalSeg_Data.shape
    PredictionFull_Thalamus = np.zeros((len(SliceNumbers),148,148,2))

    trainer = unet.Trainer(net)

    if MultByThalamusFlag == 1:
        try:
            PredictionFull_Thalamus = ThalamusExtraction(net , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , subFolders, CropDim , padSize)
            outputFolder = 'Results_MultByPredictedThalamus/'
        except:
            MultByThalamusFlag = 0
            print('Error: Could not load Thalamus')

    elif MultByThalamusFlag == 2:
        AA = OriginalSeg_Data[50:198 , 130:278 , SliceNumbers]
        L = len(SliceNumbers)
        for sliceInd in range(L):
            PredictionFull_Thalamus[sliceInd,:,:,1] = AA[:,:,sliceInd]
            PredictionFull_Thalamus[sliceInd,:,:,0] = 1 - AA[:,:,sliceInd]


    TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test0 + '*.tif',shuffle_data=False)

    L = len(SliceNumbers)
    DiceCoefficient  = np.zeros(L)
    DiceCoefficient_Mult  = np.zeros(L)
    LogLoss  = np.zeros(L)
    LogLoss_Mult  = np.zeros(L)

    SliceIdx = np.zeros(L)

    for sliceNum in range(L):
        Stng = TestData.data_files[sliceNum]
        d = Stng.find('_Slice')
        SliceIdx[sliceNum] = int(Stng[d+6:].split('.')[0])

    SliceIdxArg = np.argsort(SliceIdx)
    Data , Label = TestData(L)


    szD = Data.shape
    szL = Label.shape

    data  = np.zeros((1,szD[1],szD[2],szD[3]))
    label = np.zeros((1,szL[1],szL[2],szL[3]))

    shiftFlag = 0

    # PredictionFull_Thalamus = ThalamusExtraction(net , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , subFolders, CropDim , padSize)

    for sliceNum in SliceIdxArg:

        data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
        label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

        if shiftFlag == 1:
            shiftX = 0
            shiftY = 0
            data = np.roll(data,[0,shiftX,shiftY,0])
            label = np.roll(label,[0,shiftX,shiftY,0])

        prediction2 = net.predict( Directory_Nuclei_Train_Model_cpkt, data, GPU_Num=gpuNum)

        if MultByThalamusFlag != 0:

            prediction_Mult = np.zeros(prediction2.shape)
            # prediction_Mult[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[ SliceNumbers[sliceNum] ,:,:,:])
            prediction_Mult[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[sliceNum,:,:,:])

            # PredictedSeg = prediction_Mult[0,...,1] > 0.1
            try:
                Thresh_Mult = max(filters.threshold_otsu(prediction_Mult[0,...,1]),0.2)
            except:
                Thresh_Mult = 0.2

            PredictedSeg_Mult = prediction_Mult[0,...,1] > Thresh_Mult

        try:
            Thresh = max(filters.threshold_otsu(prediction2[0,...,1]),0.2)
        except:
            Thresh = 0.2

        PredictedSeg = prediction2[0,...,1] > Thresh

        if MultByThalamusFlag != 0:
            Prediction3D_Mult[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceNum]) ] = prediction_Mult[0,...,1]
            Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceNum]) ] = PredictedSeg_Mult

        Prediction3D_PureNuclei[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceNum]) ] = prediction2[0,...,1]
        Prediction3D_PureNuclei_logical[ CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],int(SliceNumbers[sliceNum]) ] = PredictedSeg

        # prediction = np.zeros(prediction2.shape)
        # # prediction[0,:,:,:] = np.multiply(prediction2[0,:,:,:],PredictionFull_Thalamus[sliceNum,:,:,:])
        #
        # # PredictedSeg = prediction[0,...,1] > 0.2
        # Thresh = max(filters.threshold_otsu(prediction[0,...,1]),0.2)
        # PredictedSeg = prediction[0,...,1] > Thresh

        # Prediction3D_Mult[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = prediction[0,...,1]
        # Prediction3D_Mult_logical[CropDim[0,0]:CropDim[0,1],CropDim[1,0]:CropDim[1,1],CropDim[2,0] + int(SliceIdx[sliceNum])] = PredictedSeg

        # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))

        sz = label.shape

        A = int((padSize/2))
        imgCombined = util.combine_img_prediction(data, label, prediction2)
        DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
        util.save_image(imgCombined, Directory_Test_Results_Nuclei+"prediction_slice"+ str(SliceNumbers[sliceNum]) + ".jpg")

        Loss = unet.error_rate(prediction2,label[:,A:sz[1]-A,A:sz[2]-A,:])
        LogLoss[sliceNum] = np.log10(Loss)
        np.savetxt(Directory_Test_Results_Nuclei + 'DiceCoefficient.txt',DiceCoefficient)
        np.savetxt(Directory_Test_Results_Nuclei + 'LogLoss.txt',LogLoss)


        if MultByThalamusFlag != 0:
            AA  = prediction_Mult*0
            AA[0,:,:,:] = PredictionFull_Thalamus[sliceNum,:,:,:]

            A = (padSize/2)
            imgCombined = util.combine_img_prediction(data, label, prediction_Mult)
            imgCombined2 = util.combine_img_prediction(data, label, AA)
            DiceCoefficient_Mult[sliceNum] = DiceCoefficientCalculator(PredictedSeg_Mult,label[0,A:sz[1]-A,A:sz[2]-A,1])  # 20 is for zero padding done for input
            util.save_image(imgCombined, Directory_Test_Results_Thalamus +"prediction_slice"+ str(SliceNumbers[sliceNum]) + ".jpg")
            util.save_image(imgCombined2, Directory_Test_Results_Thalamus+"prediction_slice"+ str(SliceNumbers[sliceNum]) + "2.jpg")

            Loss = unet.error_rate(prediction_Mult,label[:,A:sz[1]-A,A:sz[2]-A,:])
            LogLoss_Mult[sliceNum] = np.log10(Loss)

            np.savetxt(Directory_Test_Results_Thalamus + 'DiceCoefficient.txt',DiceCoefficient_Mult)
            np.savetxt(Directory_Test_Results_Thalamus + 'LogLoss.txt',LogLoss_Mult)


    Prediction3D_nifti = nib.Nifti1Image(Prediction3D_PureNuclei,Affine)
    Prediction3D_nifti.get_header = Header
    nib.save(Prediction3D_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '.nii.gz')

    Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_PureNuclei_logical,Affine)
    Prediction3D_logical_nifti.get_header = Header
    nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Nuclei + subFolders + '_' + NucleusName + '_Logical.nii.gz')

    if MultByThalamusFlag != 0:

        Prediction3D_nifti = nib.Nifti1Image(Prediction3D_Mult,Affine)
        Prediction3D_nifti.get_header = Header
        nib.save(Prediction3D_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '.nii.gz')

        Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_Mult_logical,Affine)
        Prediction3D_logical_nifti.get_header = Header
        nib.save(Prediction3D_logical_nifti,Directory_Test_Results_Thalamus + subFolders + '_' + NucleusName + '_Logical.nii.gz')


    return Prediction3D_PureNuclei, Prediction3D_PureNuclei_logical

