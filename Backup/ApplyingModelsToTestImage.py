from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import pickle
import nibabel as nib
from TestData import TestData , TestData2_MultipliedByWholeThalamus
import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util

NeucleusFolder = 'CNN_VLP2' #'CNN_Thalamus' #
NucleusName = '6-VLP' #'1-THALAMUS' #

A = [[4,3],[6,1],[1,2],[1,3],[4,1]] # [0,0]] # ,
SliceNumbers = range(107,140)

Directory_VLP = '/media/data1/artin/data/Thalamus/' + NeucleusFolder
Directory_Thalamus = '/media/data1/artin/data/Thalamus/' + 'CNN_Thalamus'

for ii in range(0,len(A)):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Test_Directory_VLP = Directory_VLP + '/' + TestName + '/'
    Test_Directory_Thalamus = Directory_Thalamus + '/' + TestName + '/'

    subFolders = os.listdir(Test_Directory_VLP)

    print subFolders[3]

    print range(len(subFolders)-15)
    for sFi in range(len(subFolders)-15):

        print('-----------------------------------------------------------')
        print('Test: ' + str(A[ii]) + 'Subject: ' + str(subFolders[sFi]))
        print('-----------------------------------------------------------')

        Orig_Segments_VLP_Directory = Directory_VLP + '/OriginalDeformedPriors/' +  subFolders[sFi] + '/ManualDelineation'
        Orig_Segment_VLP_Address = Orig_Segments_VLP_Directory +'/' + NucleusName + '_Deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

        Orig_Segments_Thalamus_Directory = Directory_Thalamus + '/OriginalDeformedPriors/' +  subFolders[sFi] + '/ManualDelineation'
        Orig_Segment_Thalamus_Address = Orig_Segments_Thalamus_Directory +'/1-THALAMUS_Deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

        # print Test_Path_VLP
        Test_Path_VLP = Test_Directory_VLP + subFolders[sFi] + '/Test/'
        Test_Path_Thalamus = Test_Directory_Thalamus + subFolders[sFi] +  '/Test/'

        Trained_Model_Path_VLP = Test_Directory_VLP + subFolders[sFi] +  '/Train/model/'
        Trained_Model_Path_Thalamus = Test_Directory_Thalamus + subFolders[sFi] +  '/Train/model/'
        # Directory_OriginalData = Directory + 'OriginalData/'
        # with open(Directory_OriginalData + "subFolderList.txt" ,"rb") as fp:
        #     subFolders = pickle.load(fp)
        # p = 0
        #
        #
        # MainDirectory_FullThalamus = 'ForUnet_Test8_Enhanced/TestSubject'+str(p)
        # MainDirectory = 'ForUnet_Test8_Enhanced/TestSubject'+str(p)

        # Test_Path  = Directory + MainDirectory + '/test/'
        # Test_Path_Thalamus  = Directory + MainDirectory_FullThalamus + '/test/'
        # print Test_Path_Thalamus
        # Trained_Model_Path = Directory + MainDirectory + '/train/
        # Trained_Model_Path_Thalamus = Directory + MainDirectory_FullThalamus + '/train/model/'

        OriginalSeg = nib.load(Orig_Segment_Thalamus_Address) # ThalamusSegDeformed_Croped PulNeucleusSegDeformed_Croped
        net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2)


        CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

        padSize = 90
        [data,label,prediction,OriginalSeg] = TestData2_MultipliedByWholeThalamus(net , Test_Path_VLP , Trained_Model_Path_VLP , OriginalSeg , subFolders[sFi] , CropDimensions , padSize , Test_Path_Thalamus , Trained_Model_Path_Thalamus , NucleusName)



### $$$$$$$$$$ -------------------------------------------



    # Trained_Model_Path = Train_Path + 'model.cpkt'
    # TestResults_Path   = Test_Path  + 'results/'
    #
    # try:
    #     os.stat(TestResults_Path)
    # except:
    #     os.makedirs(TestResults_Path)
    #
    #
    # OriginalSeg_Data = OriginalSeg.get_data()
    # Header = OriginalSeg.header
    # Affine = OriginalSeg.affine
    #
    #
    # Prediction3D_logical = np.zeros(OriginalSeg_Data.shape)
    # Prediction3D = np.zeros(OriginalSeg_Data.shape)
    #
    # trainer = unet.Trainer(net)
    #
    # TestData = image_util.ImageDataProvider(  Test_Path + '*.tif',shuffle_data=False)
    #
    # L = len(TestData.data_files)
    # DiceCoefficient  = np.zeros(L)
    # Error  = np.zeros(L)
    # SliceIdx = np.zeros(L)
    #
    # for sliceNum in range(L):
    #     Stng = TestData.data_files[sliceNum]
    #     d = Stng.find('slice')
    #     SliceIdx[sliceNum] = int(Stng[d+5:].split('.')[0])
    #
    # SliceIdxArg = np.argsort(SliceIdx)
    # Data , Label = TestData(len(SliceIdx))
    #
    # szD = Data.shape
    # szL = Label.shape
    #
    # data  = np.zeros((1,szD[1],szD[2],szD[3]))
    # label = np.zeros((1,szL[1],szL[2],szL[3]))
    #
    # shiftFlag = 0
    # for sliceNum in SliceIdxArg:
    #
    #     data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
    #     label[0,:,:,:] = Label[sliceNum,:,:,:].copy()
    #
    #     if shiftFlag == 1:
    #         shiftX = 0
    #         shiftY = 0
    #         data = np.roll(data,[0,shiftX,shiftY,0])
    #         label = np.roll(label,[0,shiftX,shiftY,0])
    #
    #     prediction = net.predict( Trained_Model_Path, data)
    #     Prediction3D[1:,41:40+93,int(SliceIdx[sliceNum])] = prediction[0,...,1]
    #     Predinneriction3D_logical[1:,41:40+93,int(SliceIdx[sliceNum])] = prediction[0,...,1] > 0.2
    #
    #     # unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
    #     PredictedSeg = prediction[0,...,1] > 0.5
    #     sz = label.shape
    #
    #     imgCombined = util.combine_img_prediction(data, label, prediction)
    #     Error[sliceNum] = np.sum(label[:,20:sz[1]-20,20:sz[2]-20,0]-prediction[:,:,:,0])
    #     DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,20:sz[1]-20,20:sz[2]-20,1])  # 20 is for zero padding done for input
    #     util.save_image(imgCombined, TestResults_Path+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")
    #
    #
    # np.savetxt(TestResults_Path+'DiceCoefficient.txt',DiceCoefficient)
    # np.savetxt(TestResults_Path+'Error.txt',Error)
    #
    # Prediction3D_nifti = nib.Nifti1Image(Prediction3D,Affine)
    # Prediction3D_nifti.get_header = Header
    #
    # nib.save(Prediction3D_nifti,TestResdata,label,prediction,OriginalSegults_Path + subFolders[p] + '_ThalamusSegDeformed_Croped_Predicted.nii.gz')
    #
    # Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_logical,Affine)
    # Prediction3D_logical_nifti.get_header = Header
    #
    # nib.save(Prediction3D_logical_nifti,TestResults_Path + subFolders[p] + '_ThalamusSegDeformed_Croped_Predicted_logical.nii.gz')
    #
    # # fig,ax = plt.subplots(1,4)
    # # ax[0].imshow(data[0,20:sz[1]-20,20:sz[2]-20,0], aspect="auto",cmap = 'gray')
    # # ax[1].imshow(label[0,20:sz[1]-20,20:sz[2]-20,1], aspect="auto",cmap = 'gray')
    # # ax[2].imshow(prediction[0,...,1], aspect="auto",cmap = 'gray')
    # # ax[3].imshow(OriginalSeg, aspect="auto",cmap = 'gray')
    # # ax[0].set_title("Input")
    # # ax[1].set_title("Ground truth")
    # # ax[2].set_title("Prediction")
    # # ax[3].set_title("Prediction OriginalSeg")
    # # plt.show()
