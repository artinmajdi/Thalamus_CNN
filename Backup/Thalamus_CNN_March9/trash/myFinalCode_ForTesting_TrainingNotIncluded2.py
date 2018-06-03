from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import pickle
import nibabel as nib
from TestData import TestData
import os
import numpy as np
import nibabel as nib
from tf_unet import unet, util, image_util

def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

Directory = '/media/data1/artin/data/Thalamus/'
Directory_OriginalData = Directory + 'OriginalData/'
with open(Directory_OriginalData + "subFolderList.txt" ,"rb") as fp:
    subFolders = pickle.load(fp)

for p in range(1): #
    MainDirectory = 'ForUnet_Test4_IncreasedMiddleSlicesToThirty/TestSubject'+str(p)
    Test_Path  = Directory + MainDirectory + '/test/'
    Trained_Model_Path = Directory + MainDirectory + '/train/model/'

    OriginalSeg = nib.load(Directory_OriginalData+subFolders[p]+'/PulNeucleusSegDeformed_Croped.nii.gz') # ThalamusSegDeformed_Croped
    net = unet.Unet(layers=3, features_root=16, channels=1, n_class=2)



Trained_Model_Path2 = Trained_Model_Path + 'model.cpkt'
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
# for sliceNum in SliceIdxArg:
sliceNum = SliceIdxArg[2]

data[0,:,:,:]  = Data[sliceNum,:,:,:].copy()
label[0,:,:,:] = Label[sliceNum,:,:,:].copy()

if shiftFlag == 1:
    shiftX = 0
    shiftY = 0
    data = np.roll(data,[0,shiftX,shiftY,0])
    label = np.roll(label,[0,shiftX,shiftY,0])

prediction = net.predict( Trained_Model_Path2, data)
Prediction3D[1:,41:40+93,int(SliceIdx[sliceNum])] = prediction[0,...,1]
Prediction3D_logical[1:,41:40+93,int(SliceIdx[sliceNum])] = prediction[0,...,1] > 0.2

# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
PredictedSeg = prediction[0,...,1] > 0.5
sz = label.shape

imgCombined = util.combine_img_prediction(data, label, prediction)
DiceCoefficient[sliceNum] = DiceCoefficientCalculator(PredictedSeg,label[0,20:sz[1]-20,20:sz[2]-20,1])  # 20 is for zero padding done for input
util.save_image(imgCombined, TestResults_Path+"prediction_slice"+ str(SliceIdx[sliceNum]) + ".jpg")

print prediction.shape
print label.shape
Loss = unet.error_rate(prediction[:,:,:,:],label[:,20:sz[1]-20,20:sz[2]-20,:])
LogLoss[SliceIdx[sliceNum]] = np.log10(Loss)

print SliceIdx

np.savetxt(TestResults_Path+'DiceCoefficient.txt',DiceCoefficient)
np.savetxt(TestResults_Path+'LogLoss.txt',LogLoss)

Prediction3D_nifti = nib.Nifti1Image(Prediction3D,Affine)
Prediction3D_nifti.get_header = Header

nib.save(Prediction3D_nifti,TestResults_Path + subFolders + '_ThalamusSegDeformed_Croped_Predicted.nii.gz')

Prediction3D_logical_nifti = nib.Nifti1Image(Prediction3D_logical,Affine)
Prediction3D_logical_nifti.get_header = Header

nib.save(Prediction3D_logical_nifti,TestResults_Path + subFolders + '_ThalamusSegDeformed_Croped_Predicted_logical.nii.gz')
