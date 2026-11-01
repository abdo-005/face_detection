# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:38:55 2026

@author: moham
"""
import cv2 as cv
import numpy as np
from pathlib import Path

PHOTOS_DIR = Path(__file__).resolve().parent / "Photos"
def get_image_path(category, imgname):
    path = PHOTOS_DIR / category / imgname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return path
def load_img(path):
        img = cv.imread(path)
        return img
def Gray_scale(img):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        return img
def rescaleimg(img,scale,interpolation):
        img = cv.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation=interpolation)
        return img
def Gaussianfilter(img,size,sigma):
        img = cv.GaussianBlur(img,(size,size),sigma)
        return img
def averagefilter(img,size):
        img = cv.blur(img,(size,size))
        return img
def medianfilter(img,size):
        img = cv.medianBlur(img,size)
        return img
def imgmask(img,lowerend,highend):
        img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        mask = cv.inRange(img,lowerend,highend)
        return mask
def creat_blob(img):
        blob = cv.resize(img ,(300,300))
        blob = blob.astype(np.float32)
        blob -= (104.0, 177.0, 123.0)
        blob = blob.transpose(2,0,1)
        blob = np.expand_dims(blob,axis=0)
        return blob



