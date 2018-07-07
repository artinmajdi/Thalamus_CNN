from tf_unet import unet, util, image_util
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import nibabel as nib
import shutil
from collections import OrderedDict
import logging
from TestData_V4 import TestData
from tf_unet import unet, util, image_util
import multiprocessing

def DiceCoefficientCalculator(msk1,msk2):
    intersection = np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

def ReadMasks(DirectoryMask,SliceNumbers):

    mask = nib.load(DirectoryMask)
    maskD = mask.get_data()

    Header = mask.header
    Affine = mask.affine

    msk = maskD

    msk[maskD<0.5]  = 0
    msk[msk>=0.5] = 1

    return msk , Header , Affine

def SumMasks(DirectorySubFolders):

    i = 1
    Directory_Nuclei_Label = '/6_VLP_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD = np.zeros(msk.shape)
    maskD[msk == 1] = 10*i

    i = i + 1
    Directory_Nuclei_Label = '/7_VPL_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    i = i + 1
    Directory_Nuclei_Label = '/8_Pul_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    i = i + 1
    Directory_Nuclei_Label = '/12_MD_Pf_NeucleusSegDeformed.nii.gz'
    msk , Header , Affine = ReadMasks(DirectorySubFolders+Directory_Nuclei_Label)
    maskD[msk == 1] = 10*i

    return maskD , Header , Affine

# define worker function
def calculate(process_name, tasks, results):  
    print('[%s] evaluation routine starts' % process_name)

    while True:
        new_value = tasks.get()
        if new_value < 0:
            print('[%s] evaluation routine quits' % process_name)

            # Indicate finished
            results.put(-1)
            break
        else:
            # Compute result and mimic a long-running task
            compute = new_value * new_value
            sleep(0.02*new_value)

            # Output which process received the value
            # and the calculation result
            print('[%s] received value: %i' % (process_name, new_value))
            print('[%s] calculated value: %i' % (process_name, compute))

            # Add result to the queue
            results.put(compute)

    return

def Main(sFi):

    NeucleusFolder = 'CNN_VLP5_2D_PerSlice_Thalamus' #'CNN_Thalamus' #
    NucleusName = '1-THALAMUS' #'6-VLP' #

    A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]]
    SliceNumbers = range(107,140)

    Directory_main = '/localhome/msmajdi_local/Tests/Thalamus_CNN/'
    Directory_Nuclei_Full = Directory_main + NeucleusFolder
    # Directory_Thalamus_Full = Directory_main + 'CNN_Thalamus'

    ii = 1
    # for ii in range(1,len(A)):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Directory_Nuclei = Directory_Nuclei_Full + '/' + TestName + '/'
    # print Directory_Nuclei
    # with open(Directory_Nuclei_Full + '/OriginalDeformedPriors/subFolderList.txt' ,"rb") as fp:
    #     subFolders = pickle.load(fp)
    subFolders = os.listdir(Directory_Nuclei)
    # print len(subFolders)


    for sFi in range(len(subFolders)):

        Directory_Nuclei_Label = '/localhome/msmajdi_local/data/priors_forCNN/' +  subFolders[sFi] + '/ManualDelineation/' + NucleusName + '_Deformed.nii.gz'   # ThalamusSegDeformed  ThalamusSegDeformed_Croped    PulNeucleusSegDeformed  PulNeucleusSegDeformed_Croped

        for sliceInd in range(33):
            print('------------------------------------------------------')
            print('Test: ' + str(A[ii]) + ' Subject: ' + str(subFolders[sFi]) + ' Slide ' + str(sliceInd) )
            print('------------------------------------------------------')

            # sliceInd = 25
            Directory_Nuclei_Test  = Directory_Nuclei + subFolders[sFi] + '/Test/Slice' + str(sliceInd) + '/'
            Directory_Nuclei_Train = Directory_Nuclei + subFolders[sFi] + '/Train/Slice' + str(sliceInd) + '/'
            Directory_Nuclei_Train_Model = Directory_Nuclei_Train + 'model/'

            try:
                os.stat(Directory_Nuclei_Train_Model)
            except:
                os.makedirs(Directory_Nuclei_Train_Model)

            try:
                os.stat(Directory_Nuclei_Test  + 'results/')
            except:
                os.makedirs(Directory_Nuclei_Test  + 'results/')


            TrainData = image_util.ImageDataProvider(Directory_Nuclei_Train + "*.tif")

            # print Directory_Nuclei_Test
            # TestData = image_util.ImageDataProvider(  Directory_Nuclei_Test + '*.tif',shuffle_data=False)
            # print len(TrainData.data_files)
            # data , label = TrainData(len(TrainData.data_files))

            # plt.imshow(label[1,:,:,1],cmap='gray')
            # plt.show()


            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

            net = unet.Unet(layers=4, features_root=16, channels=1, n_class=2 , summaries=True) #  , cost="dice_coefficient"

            trainer = unet.Trainer(net)
            path = trainer.train(TrainData, Directory_Nuclei_Train_Model, training_iters=100, epochs=100, display_step=500) #, training_iters=100, epochs=100, display_step=10


            OriginalSegFull = nib.load(Directory_Nuclei_Label)
            # Header = OriginalSegFull.header
            # Affine = OriginalSegFull.affine
            # OriginalSeg = OriginalSegFull.get_data()


            CropDimensions = np.array([ [50,198] , [130,278] , [SliceNumbers[0] , SliceNumbers[len(SliceNumbers)-1]] ])

            padSize = 90
            [data,label,prediction] = TestData(net , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSegFull , subFolders[sFi] , CropDimensions , padSize)
            #   [data,label,prediction,OriginalSeg] = TestData2_MultipliedByWholeThalamus(net , Directory_Nuclei_Test , Directory_Nuclei_Train , OriginalSeg , subFolders[sFi] , CropDimensions , padSize , Directory_Thalamus_Test , Directory_Thalamus_TrainedModel , NucleusName)


if __name__ == "__main__":

    manager = multiprocessing.Manager()
    tasks = manager.Queue()
    results = manager.Queue()

    # sFi = range(19)
    # agents = 5
    # chunksize = 3
    # with Pool(processes=agents) as pool:
    #     result = pool.map(Main, sFi, chunksize)

    num_processes = 2
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    for i in range(num_processes):
        process_name = 'P%i' % i
        new_process = multiprocessing.Process(target=calculate, args=(process_name,tasks,results))
        processes.append(new_process)
        new_process.start()

