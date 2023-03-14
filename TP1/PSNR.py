#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:55:37 2022

@author: ckervazo
"""

from math import log10, sqrt
import cv2
import numpy as np

#%%
def PSNR(im1,im2):
    """
    Computes the PSNR between im1 and im2. The two images must have the same size.

    Parameters
    ----------
    im1, im2 : nparray
        Two images.

    Returns
    -------
    psnr : float
    """
    mse = np.mean((im1 - im2) ** 2)
    if(mse == 0):
        return 100
    max_pixel = float(255.0)
    psnr = 20 * log10(max_pixel / sqrt(mse))

    return psnr
