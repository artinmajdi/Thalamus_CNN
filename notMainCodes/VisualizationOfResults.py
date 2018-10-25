import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm, tqdm_notebook
import cv2
import skimage
from skimage import feature
from imageio import imwrite
from matplotlib import colors, cm
from scipy.misc import imrotate
from PIL import ImageEnhance , Image , ImageFilter


def nonZeroIndex(mask):
    a = np.zeros(mask.shape[2])
    for d in range(mask.shape[2]):
        a[d] = mask[...,d].sum()

    indexes = np.where(a>0)[0]

    if len(indexes) > 20:
        stepSize = 4
    else:
        stepSize = 2

    indexes = range(indexes[0],indexes[len(indexes)-1],stepSize)

    return indexes

def readingImages(dirr):

    nulceiName = directory.split('Results_Temp/CNN')[1].split('_2D_SanitizedNN/')[0]
    nulceiName = nulceiName.replace('_','-')

    Directory_Orig = dirr + '/original/'
    Directory_Pred = dirr + '/prediction/'
    for Pred_name in os.listdir(Directory_Pred):
        if 'Logical.nii.gz' in Pred_name:
            break

    im_OrigF = nib.load(Directory_Orig + 'WMnMPRAGE_bias_corr_Deformed.nii.gz')
    seg_OrigF = nib.load(Directory_Orig + 'Manual_Delineation_Sanitized/' + nulceiName + '_deformed.nii.gz')
    seg_PredF = nib.load(Directory_Pred + Pred_name)

    seg_Pred = seg_PredF.get_data()
    seg_Orig = seg_OrigF.get_data()
    im_Orig = im_OrigF.get_data()
    im_Orig = im_Orig/im_Orig.max()

    return seg_Pred, seg_Orig, im_Orig

def edgeDetection(mask):
    sz = mask.shape
    edge_pred = np.zeros(sz)
    for d in range(sz[2]):
        edge_pred[...,d] = feature.canny(mask[...,d])

    return edge_pred

def colorImageMaker(edge_pred, seg_Orig):
    sz = seg_Orig.shape
    RGB_im = np.zeros((sz[0],sz[1],3,sz[2]))
    for d in range(sz[2]):
        RGB_im[...,0,d] = edge_pred[...,d]
        RGB_im[...,1,d] = seg_Orig[...,d] > 0.5
        RGB_im[...,2,d] = seg_Orig[...,d] > 0.5

    return RGB_im

def concantenateImageMask(im, RGB_im):

    sz = im.shape
    im2 = np.zeros((sz[0],sz[1],3))

    # rgbIm = RGB_im[80:144 , 160:270 , :]
    # imCropped = im[80:144 , 160:270 , :]


    rgbIm = RGB_im[140:250 , 80:144,:]
    imCropped = im[140:250 , 80:144,:]
    sz = rgbIm.shape
    rgbIm = skimage.transform.resize(rgbIm,(sz[0]*4,sz[1]*4,3))
    imCropped = skimage.transform.resize(imCropped,(sz[0]*4,sz[1]*4,3))


    FinalImage = np.concatenate((im,imCropped,rgbIm),axis=1) #
    FinalImage = FinalImage*255/FinalImage.max()
    FinalImage = np.array(FinalImage,dtype=np.uint8)

    return FinalImage

def saveImages(dirr,FinalImage):

    try:
        os.makedirs(dirr)
    except:
        aaaa = 10

    for d in range(len(indexes)):
        imwrite(dirr + 'Slice' + str(indexes[d]) + '.jpg',FinalImage[...,d])

def initialDirectories(ind = 1):

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

    Params['NucleusName'] = NucleusName
    Params['NeucleusFolder'] = 'CNN' + NucleusName.replace('-','_') + '_2D_SanitizedNN'
    Params['SliceNumbers'] = SliceNumbers

    return Params

def subFolderList(dir):
    subFolders = os.listdir(dir + '/Test_WMnMPRAGE_bias_corr_Deformed/')

    listt = []
    for i in range(len(subFolders)):
        if subFolders[i][:4] == 'vimp':
            listt.append(subFolders[i])

    return listt

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

def enhancing(im , scaleEnhance):
    # im = im.astype(dtype=np.int16)
    im = Image.fromarray(im)
    im = im.convert('L')

    if scaleEnhance[0] != 1:
        im2 = ImageEnhance.Sharpness(im)
        im = im2.enhance(scaleEnhance[0])

    if scaleEnhance[1] != 1:
        im2 = ImageEnhance.Contrast(im)
        im = im2.enhance(scaleEnhance[1])

    return im

# dir = '/media/artin/D0E2340CE233F576/Folder_MajorityVoting'
dir = '/media/data1/artin/Tests/Folder_Visualization'

slc = 125
im = nib.load(dir + '/origtemplate.nii.gz' ).get_data()
im_Orig2 = im[...,slc]
im_Orig = (im_Orig2 - im_Orig2.min())/ ( im_Orig2.max() - im_Orig2.min() )

im_OrigE = im_Orig.copy()
im = Image.fromarray(im_Orig)
im = im.convert('L')
A = ImageEnhance.Contrast(im)
im_OrigE[:,:] = A.enhance(1.4)
im_OrigE = im_OrigE/2

plt.imshow(im_OrigE,cmap='gray')
plt.show()


sz = im_Orig.shape
RGB_im = np.zeros((sz[0],sz[1],3))
RGB_im[...,0] = im_Orig
RGB_im[...,1] = im_Orig
RGB_im[...,2] = im_Orig

im_Orig_RGB = RGB_im.copy()

def random_color():
    levels = range(32,256,32)
    return tuple(np.random.choice(levels) for _ in range(3))

for ind in [1,2,4,5,6,7]: # ,8,9,10,11,12,13]:
    print('ind',ind)
    Params = initialDirectories(ind = ind)
    pred = nib.load(dir + '/vimp2_ctrl_925_07152013_LS_' + Params['NucleusName'] + '_Logical.nii.gz').get_data()
    Label = nib.load(dir + '/Manual_Delineation_Sanitized/' + Params['NucleusName'] + '_deformed.nii.gz').get_data()

    seg_Pred = pred[...,slc]
    seg_Orig = Label[...,slc]
    edge_orig = feature.canny(seg_Orig)

    A = random_color()
    scale = 100
    edge_origColor = np.concatenate( ( (A[0]/scale)*edge_orig[...,np.newaxis], (A[1]/scale)*edge_orig[...,np.newaxis], (A[2]/scale)*edge_orig[...,np.newaxis]),axis=2)
    RGB_im = RGB_im + edge_origColor

plt.imshow(RGB_im)
plt.show()

# im_Orig2 = np.transpose(im_Orig,[1,0])
RGB_im2 = np.transpose(RGB_im,[1,0,2])
im_Orig_RGB2 = np.transpose(im_Orig_RGB,[1,0,2])

im_Orig_RGB2.shape
RGB_im2.shape
# FinalImage = concantenateImageMask(im_Orig_RGB, RGB_im)
FinalImage = concantenateImageMask(im_Orig_RGB2, RGB_im2)
RGB_im.max()
plt.imshow(FinalImage)
plt.show()
imwrite(dir + '/im.jpg',FinalImage)
