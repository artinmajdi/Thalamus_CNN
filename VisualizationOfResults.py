import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm, tqdm_notebook
import cv2
import skimage
from skimage import feature
from imageio import imwrite

Tests = ['CNN1_THALAMUS_2D_SanitizedNN', 'CNN4567_VL_2D_SanitizedNN', 'CNN6_VLP_2D_SanitizedNN', 'CNN8_Pul_2D_SanitizedNN',  'CNN10_MGN_2D_SanitizedNN', 'CNN12_MD_Pf_2D_SanitizedNN']
vimp = ['vimp2_668_02282013_CD', 'vimp2_845_05312013_VZ', 'vimp2_964_08092013_TG', 'vimp2_ANON724_03272013']

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

def concantenateImageMask(im_Orig, edge_pred, RGB_im):

    sz = im_Orig.shape
    im2 = np.zeros((sz[0],sz[1],3,sz[2]))
    for d in range(sz[2]):

        rgbIm = RGB_im[80:144,160:270,:,d]
        sz = rgbIm.shape
        rgbIm = skimage.transform.resize(rgbIm,(sz[0]*4,sz[1]*4,3))

        im = im_Orig[...,d]
        im = np.concatenate((im[...,np.newaxis],im[...,np.newaxis],im[...,np.newaxis]),axis=2)
        im[...,2] = im[...,2] + edge_pred[...,d]
        im2[...,d] = im
        RGB_im[...,d] = rgbIm

    FinalImage = np.concatenate((im2,RGB_im),axis=1)
    FinalImage = FinalImage*255/FinalImage.max()
    FinalImage = np.array(FinalImage,dtype=np.uint8)

    return FinalImage

def saveImages(dirr,FinalImage,indexes):

    try:
        os.makedirs(dirr)
    except:
        aaaa = 10

    for d in range(len(indexes)):
        imwrite(dirr + 'Slice' + str(indexes[d]) + '.jpg',FinalImage[...,d])


for t in range(len(Tests)):
    for v in tqdm(range(len(vimp)),desc='vimp'):

        directory = '/media/artin/D0E2340CE233F576/Results_Temp' + '/' + Tests[t] + '/Test_WMnMPRAGE_bias_corr_Deformed/' + vimp[v]
        seg_Pred, seg_Orig, im_Orig = readingImages(directory)

        indexes = nonZeroIndex(seg_Orig)

        seg_Pred = seg_Pred[...,indexes]
        seg_Orig = seg_Orig[...,indexes]
        im_Orig  = im_Orig[...,indexes]

        edge_pred = edgeDetection(seg_Pred)

        RGB_im = colorImageMaker(edge_pred, seg_Orig)

        FinalImage = concantenateImageMask(im_Orig, edge_pred, RGB_im)

        directory = '/media/artin/D0E2340CE233F576/Results_Temp2/' + Tests[t] + '/Test_WMnMPRAGE_bias_corr_Deformed/' + vimp[v] + '/'
        saveImages(directory,FinalImage,indexes)
