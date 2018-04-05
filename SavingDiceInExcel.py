import os
from glob import glob
import numpy as np
import xlwt
Dire = '/media/data1/artin/StThomas_Tests/DiceValues'
subDire = glob(Dire+'/Dice_*')
print subDire

neucli = ['6-VLP']
A = [[0,0],[4,3],[6,1],[1,2],[1,3],[4,1]] #
SliceNumbers = range(107,140)

Directory = '/media/data1/artin/data/Thalamus/CNN_VLP2'
Dice = np.zeros((6 , 5 , 33))

for ii in range(len(A)):
    if ii == 0:
        TestName = 'Test_WMnMPRAGE_bias_corr_Deformed' # _Deformed_Cropped
    else:
        TestName = 'Test_WMnMPRAGE_bias_corr_Sharpness_' + str(A[ii][0]) + '_Contrast_' + str(A[ii][1]) + '_Deformed'

    Test_Directory = Directory + '/' + TestName + '/'

    subFolders = os.listdir(Test_Directory)

    for testNum in range(len(subFolders)-15):
        try:
            ff = open(Test_Directory + subFolders[testNum] +'/Test/results/DiceCoefficient.txt','r')

            FullText = ff.read()
            Lines = FullText.split('\n')
            for SliceNum in range(33):
                if Lines[SliceNum] == 'nan':
                    Dice[ii,testNum,SliceNum] = 0
                else:
                    Dice[ii,testNum,SliceNum] = float(Lines[SliceNum])
            ff.close()
        except:
            print Test_Directory + subFolders[testNum] +'/Test/results/DiceCoefficient.txt'

results1 = xlwt.Workbook(encoding="utf-8")
results2 = xlwt.Workbook(encoding="utf-8")
for ii in range(len(A)):
    sheet1 = results1.add_sheet('Sharp_' + str(A[ii][0]) + '_Ctrst_' + str(A[ii][1]))
    for testNum in range(len(subFolders)-15):
        sheet1.write(testNum+1, 0, subFolders[testNum]  )

    sheet1.write(0, 0, 'Slice: ')
    for SliceNum in range(33):
        sheet1.write(0, SliceNum+1, str(SliceNum))
        for testNum in range(len(subFolders)-15):
            sheet1.write(testNum+1,SliceNum+1,Dice[ii,testNum,SliceNum]  )
            # sheet1.write(SliceNum+1,testNum+1, -1111 )

results1.save(Directory + '/Dice_CNN_VLP_Enhanced.xls')
