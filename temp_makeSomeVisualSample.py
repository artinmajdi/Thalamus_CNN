import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import cv2
import skimage
from skimage import feature
from imageio import imwrite

directory = '/media/artin/D0E2340CE233F576/toSend/'

def rootSubDir():
    samplesTemp = os.listdir(directory)
    samples = []
    for f in samplesTemp:
        if ('.py' not in f) & ('Results' not in f):
            samples = np.append(f,samples)

    return samples

def nonZeroIndex(mask):
    a = np.zeros(mask.shape[2])
    for d in range(mask.shape[2]):
        a[d] = mask[...,d].sum()

    indexes = np.where(a>0)[0]
    indexes = range(indexes[0],indexes[len(indexes)-1],4)

    return indexes

def readingImages(samples):

    Directory_Orig = directory + samples + '/original/'
    Directory_Pred = directory + samples + '/prediction/'
    for Pred_name in os.listdir(Directory_Pred):
        if 'Logical.nii.gz' in Pred_name:
            break

    for OrigSeg_name in os.listdir(Directory_Orig):
        if 'WMnMPRAGE_bias_corr_Deformed' in OrigSeg_name:
            im_OrigF = nib.load(Directory_Orig + OrigSeg_name)
        else:
            seg_OrigF = nib.load(Directory_Orig + OrigSeg_name)

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

def saveImages(directory,samples,FinalImage,indexes):

    dirr = directory + 'Results/' + samples + '/'
    try:
        os.makedirs(dirr)
    except:
        aaaa = 10

    for d in range(len(indexes)):
        imwrite(dirr + 'Slice' + str(indexes[d]) + '.jpg',FinalImage[...,d])

Samples = rootSubDir()

for samplesInd in tqdm(range(len(Samples))):
    # samplesInd = 0
    seg_Pred, seg_Orig, im_Orig = readingImages(Samples[samplesInd])

    indexes = nonZeroIndex(seg_Orig)

    seg_Pred = seg_Pred[...,indexes]
    seg_Orig = seg_Orig[...,indexes]
    im_Orig  = im_Orig[...,indexes]

    edge_pred = edgeDetection(seg_Pred)

    RGB_im = colorImageMaker(edge_pred, seg_Orig)

    FinalImage = concantenateImageMask(im_Orig, edge_pred, RGB_im)

    saveImages(directory,Samples[samplesInd],FinalImage,indexes)

