import numpy as np
from scipy.ndimage import median_filter

def medianFilter(img, ksize):
    # Apply median filter to the image
    filtered = median_filter(img, size=ksize)
    return filtered
