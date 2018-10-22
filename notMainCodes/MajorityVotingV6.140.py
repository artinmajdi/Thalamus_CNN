import os
import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# from skimage import filters
import pickle
import sys
from skimage import filters

A = [[0,0],[6,1],[1,2],[1,3],[4,1]]

def DiceCoefficientCalculator(msk1,msk2):
    intersection = msk1*msk2  # np.logical_and(msk1,msk2)
    DiceCoef = intersection.sum()*2/(msk1.sum()+msk2.sum())
    return DiceCoef

def initialDirectories(ind = 1, mode = 'local' , dataset = 'old' , method = 'new'):

    Params = {}

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
    elif ind == 14:
        NucleusName = '14-MTT'
        SliceNumbers = range(104,135)

    Params['modelFormat'] = 'ckpt'
    if 'localLT' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/artin/dataLocal1/dataThalamus/Unlabeled'

        Dir_AllTests  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + dataset + 'Dataset_' + method +'Method'
        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/artin/dataLocal1/dataThalamus/AllTests/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    elif 'localPC' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/media/data1/artin/thomas/priors/Unlabeled'

        Dir_AllTests  = '/media/data1/artin/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/media/data1/artin/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'

    elif 'server' in mode:

        if '20priors' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/20priors'
        elif 'MS' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/7T_MS'
        elif 'ET_3T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/3T'
        elif 'ET_7T' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/ET/7T'
        elif 'Unlabeled' in dataset:
            Dir_Prior = '/array/ssd/msmajdi/data/Unlabeled'

        Dir_AllTests  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + dataset + 'Dataset_' + method +'Method'

        if 'Unlabeled' in dataset:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + '20priors' + 'Dataset_' + 'old' +'Method'
        else:
            Params['Dir_AllTests_restore']  = '/array/ssd/msmajdi/Tests/Thalamus_CNN/' + 'Unlabeled' + 'Dataset_' + 'old' +'Method'


    Params['Dir_Prior']    = Dir_Prior
    Params['Dir_AllTests'] = Dir_AllTests
    Params['NucleusName'] = NucleusName
    Params['NeucleusFolder'] = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Params['registrationFlag'] = 0


    if Params['registrationFlag'] == 1:
        Params['SliceNumbers'] = SliceNumbers
    else:
        Params['SliceNumbers'] = range(129,251)

    return Params

def subFolderList(dir):
    subFolders = os.listdir(dir)

    listt = []
    for i in range(len(subFolders)):
        if subFolders[i][:4] == 'vimp':
            listt.append(subFolders[i])

    return listt

def input_GPU_Ix():



    UserEntries = {}
    UserEntries['gpuNum'] =  '4'  # 'nan'  #
    UserEntries['IxNuclei'] = [1]
    UserEntries['dataset'] = 'old' #'oldDGX' #
    UserEntries['method'] = 'old'
    UserEntries['testMode'] = 'EnhancedSeperately' # 'AllTrainings'
    UserEntries['enhanced_Index'] = range(len(A))
    UserEntries['mode'] = 'server'

    for input in sys.argv:

        if input.split('=')[0] == 'gpu':
            UserEntries['gpuNum'] = input.split('=')[1]
        elif input.split('=')[0] == 'testMode':
            UserEntries['testMode'] = input.split('=')[1] # 'AllTrainings'
        elif input.split('=')[0] == 'dataset':
            UserEntries['dataset'] = input.split('=')[1]
        elif input.split('=')[0] == 'mode':
            UserEntries['mode'] = input.split('=')[1]
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

        # elif input.split('=')[0] == 'training_iters':
        #     UserEntries['training_iters'] = input.split('=')[1] # 'AllTrainings'
        # elif input.split('=')[0] == 'epochs':
        #     UserEntries['epochs'] = input.split('=')[1] # 'AllTrainings'
        # elif input.split('=')[0] == 'temp_Slice':
        #     UserEntries['temp_Slice'] = input.split('=')[1] # 'AllTrainings'
    print('enhanced_Index: ',UserEntries['enhanced_Index'])
    return UserEntries

def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def testNme(Params,ii):

    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr'
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(Params['A'][ii][0]) + '_Contrast_' + str(Params['A'][ii][1])

    if Params['registrationFlag'] == 1:
        TestName = TestName + '_Deformed'

    return TestName

UserEntries = input_GPU_Ix()

for ind in UserEntries['IxNuclei']: # [1,2,8,9,10,13]:

    print('nuclei: ',ind)
    Params = initialDirectories(ind = ind, mode = UserEntries['mode'] , dataset = UserEntries['dataset'] , method = UserEntries['method'])
    Dir_SaveMWFld = mkDir( Params['Dir_AllTests'] + '/Folder_MajorityVoting/' )


    if Params['registrationFlag'] == 1:
        subFolders = subFolderList( Params['Dir_AllTests'] + '/' + Params['NeucleusFolder'] + '/Test_WMnMPRAGE_bias_corr_Deformed/' )
    else:
        subFolders = subFolderList( Params['Dir_AllTests'] + '/' + Params['NeucleusFolder'] + '/Test_WMnMPRAGE_bias_corr_Deformed/OneTrain_MultipleTest/TestCases/' )

    for reslt in ['Results']:

        Dice = np.zeros((len(subFolders), len(A)+1))

        # if Params['registrationFlag'] == 1:
        #     Directory_Nuclei_Label = Params['Dir_Prior'] + '/' + subFolders[0] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'
        # else:
        #     Directory_Nuclei_Label = Params['Dir_Prior'] + '/' + subFolders[0] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz'

        # Label = nib.load(Directory_Nuclei_Label)
        # Label = Label.get_data()


        for sFi in range(len(subFolders)):

            if Params['registrationFlag'] == 1:
                Directory_Nuclei_Label = Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz'
            else:
                Directory_Nuclei_Label = Params['Dir_Prior'] + '/' + subFolders[sFi] + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '.nii.gz'


            Label = nib.load(Directory_Nuclei_Label)
            Label = Label.get_data()
            sz = Label.shape

            Dir_save = mkDir( Dir_SaveMWFld + Params['NeucleusFolder'] + '/' + subFolders[sFi] + '/' )

            Prediction_full = np.zeros((sz[0],sz[1],sz[2],len(A)))
            Er = 0

            L = [0] if UserEntries['testMode'] == 'AllTrainings' else UserEntries['enhanced_Index'] # len(Params['A'])  # [1,4]: #
            for ii in L:
                TestName = testNme(A,ii)

                if Params['registrationFlag'] == 1:
                    Dir_AllTests_nucleiFld_Ehd   = Params['Dir_AllTests'] + '/' + Params['NeucleusFolder'] + '/' + TestName + '/'
                    Dir_AllTests_ThalamusFld_Ehd = Params['Dir_AllTests'] + '/CNN1_THALAMUS_2D_SanitizedNN/' + TestName + '/'
                else:
                    Dir_AllTests_nucleiFld_Ehd   = Params['Dir_AllTests'] + '/' + Params['NeucleusFolder'] + '/' + TestName + '/OneTrain_MultipleTest/TestCases/'
                    Dir_AllTests_ThalamusFld_Ehd = Params['Dir_AllTests'] + '/CNN1_THALAMUS_2D_SanitizedNN/' + TestName + '/OneTrain_MultipleTest/TestCases/'

                Directory_Nuclei_Test  = Dir_AllTests_nucleiFld_Ehd + subFolders[sFi] + '/Test/' + reslt + '/'

                # try:
                PredictionF = nib.load( Directory_Nuclei_Test + subFolders[sFi] + '_' + Params['NucleusName'] + '_Logical.nii.gz' )
                # PredictionF = nib.load( Directory_Nuclei_Test + subFolders[sFi] + '_' + Params['NucleusName'] + '.nii.gz' )
                Prediction = PredictionF.get_data()

                Thresh = 0.2
                try:
                    Thresh = max( filters.threshold_otsu(Prediction) ,Thresh)
                except:
                    print('---------------------------error Thresholding------------------')

                # Prediction = Prediction > Thresh


                Dice[sFi,ii] = DiceCoefficientCalculator(Label > 0.5 ,Prediction > 0.5)

                Prediction_full[:,:,:,ii] = Prediction > 0.5
                np.savetxt(Dir_SaveMWFld + Params['NeucleusFolder'] + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')
                # except:
                #     Er = Er + 1

            Prediction2 = np.sum(Prediction_full,axis=3)
            predictionMV = np.zeros(Prediction2.shape)
            predictionMV[:,:,:] = Prediction2 > 2-Er
            Dice[sFi,len(A)] = DiceCoefficientCalculator(Label > 0.5 ,predictionMV)
            np.savetxt(Dir_SaveMWFld + Params['NeucleusFolder'] + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')
            print(str(sFi) + ': ' + str(subFolders[sFi]).split('vimp2_')[1] , 'MW Dice: ' , Dice[sFi,len(A)])
            # np.savetxt(Dir_SaveMWFld + Params['NeucleusFolder'] + '/' + 'DiceCoefsAll_' + reslt + '.txt',100*Dice, fmt='%2.1f')


            Header = PredictionF.header
            Affine = PredictionF.affine

            predictionMV_nifti = nib.Nifti1Image(predictionMV,Affine)
            predictionMV_nifti.get_header = Header
            nib.save(predictionMV_nifti , Dir_save + subFolders[sFi] + '_' + Params['NucleusName'] + '_MW.nii.gz' )

        Dice2 = np.zeros( ( len(subFolders)+1 , len(A)+1 ) )
        Dice2[:len(subFolders),:] = Dice
        Dice2[len(subFolders),:] = np.mean(Dice,axis=0)

        np.savetxt(Dir_SaveMWFld + Params['NeucleusFolder'] + '/DiceCoefsAll_' + reslt + '.txt' , 100*Dice2 , fmt='%2.1f')
        with open(Dir_SaveMWFld + Params['NeucleusFolder'] + "/subFoldersList_MW_" + reslt + ".txt" ,"wb") as fp:
            pickle.dump(subFolders,fp)
