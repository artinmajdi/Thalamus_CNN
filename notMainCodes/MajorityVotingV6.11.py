import os
import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# from skimage import filters
import pickle
import sys

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'new'):

    A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

    if ind == 1:
        NucleusName = '1-THALAMUS'
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


    if 'local' in mode:

        if 'old' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/newPriors/7T_MS'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'

    elif 'server' in mode:

        if 'old' in dataset:
            Dir_Prior = '/array/hdd/msmajdi/data/priors_forCNN_Ver2'
        elif 'new' in dataset:
            Dir_Prior = '/array/hdd/msmajdi/data/newPriors/7T_MS'

        Dir_AllTests  = '/array/hdd/msmajdi/Tests/Thalamus_CNN/' + 'oldDataset' # dataset + 'Dataset_' + method +'Method'


    NeucleusFolder = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'

    return NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, A, SliceNumbers

def subFolderList(dir):
    subFolders = os.listdir(dir + '/Test_WMnMPRAGE_bias_corr_Deformed/')

    listt = []
    for i in range(len(subFolders)):
        if subFolders[i][:4] == 'vimp':
            listt.append(subFolders[i])

    return listt

def input_GPU_Ix():

    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = 1
    UserEntries['dataset'] = 'old'
    UserEntries['method'] = 'old'
    UserEntries['testMode'] = 'EnhancedSeperately' # 'AllTrainings'

    for input in sys.argv:
        if input.split('=')[0] == 'nuclei':
            UserEntries['IxNuclei'] = int(input.split('=')[1])
        elif input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'method':
            UserEntries['method'] = input.split('=')[1]

    return UserEntries

def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def testNme(A,ii):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    return TestName


UserEntries = input_GPU_Ix()


for ind in [UserEntries['IxNuclei']]: # [1,2,8,9,10,13]:

    NucleusName, NeucleusFolder, ThalamusFolder, Dir_AllTests, Dir_Prior, A, SliceNumbers = initialDirectories(ind = ind, mode = 'server' , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Dir_SaveMWFld = mkDir( Dir_AllTests + 'Folder_MajorityVoting/' )
    subFolders = subFolderList(Dir_AllTests +  NeucleusFolder)

    for reslt in ['Results']:

        print(reslt + '--------------->>>>>---------------')
        Dice = np.zeros((len(subFolders), len(A)+1))

        Directory_Nuclei_Label = Dir_Prior +  subFolders[0] + '/Manual_Delineation_Sanitized/' + NucleusName + '_deformed.nii.gz'
        Label = nib.load(Directory_Nuclei_Label)
        Label = Label.get_data()
        sz = Label.shape

        for sFi in range(len(subFolders)):
            print(str(sFi) + ': ' + str(subFolders[sFi]))
            Directory_Nuclei_Label = Dir_Prior +  subFolders[sFi] + '/Manual_Delineation_Sanitized/' + NucleusName + '_deformed.nii.gz'
            Label = nib.load(Directory_Nuclei_Label)
            Label = Label.get_data()

            Dir_save = mkDir( Dir_SaveMWFld + NeucleusFolder + '/' + subFolders[sFi] + '/' )


            Prediction_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
            Er = 0
            for ii in range(len(A)):
                TestName = testNme(A,ii)

                Dir_AllTests_nucleiFld_Ehd   = Dir_AllTests +  NeucleusFolder + '/' + TestName + '/'
                Dir_AllTests_ThalamusFld_Ehd = Dir_AllTests + 'CNN1_THALAMUS_2D_SanitizedNN/' + TestName + '/'
                Directory_Nuclei_Test  = Dir_AllTests_nucleiFld_Ehd + subFolders[sFi] + '/Test/' + reslt + '/'

                # try:
                PredictionF = nib.load( Directory_Nuclei_Test + subFolders[sFi] + '_' + NucleusName + '_Logical.nii.gz' )
                Prediction = PredictionF.get_data()

                # Thresh = max(filters.threshold_otsu(Prediction),0.2)
                Dice[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)

                Prediction_full[:,:,:,ii] = Prediction > 0.5
                np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')
                # except:
                #     Er = Er + 1

            Prediction2 = np.sum(Prediction_full,axis=3)
            predictionMV = np.zeros(Prediction2.shape)
            predictionMV[:,:,:] = Prediction2 > 2-Er

            Dice[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV)
            # np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')


            Header = PredictionF.header
            Affine = PredictionF.affine

            predictionMV_nifti = nib.Nifti1Image(predictionMV,Affine)
            predictionMV_nifti.get_header = Header
            nib.save(predictionMV_nifti , Dir_save + subFolders[sFi] + '_' + NucleusName + '_MW.nii.gz' )

        Dice2 = np.zeros( ( len(subFolders)+1 , len(A)+1 ) )
        Dice2[:len(subFolders),:] = Dice
        Dice2[len(subFolders),:] = np.mean(Dice,axis=0)

        np.savetxt(Dir_SaveMWFld + NeucleusFolder + '/DiceCoefsAll_' + reslt + '.txt',100*Dice2, fmt='%2.1f')

        with open(Dir_SaveMWFld + NeucleusFolder + "/subFoldersList_MW_"+reslt+".txt" ,"wb") as fp:
            pickle.dump(subFolders,fp)
