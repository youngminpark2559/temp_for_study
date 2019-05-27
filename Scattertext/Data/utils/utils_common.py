# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

# ================================================================================
from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
from sklearn.model_selection import RepeatedKFold
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import timeit
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math
import traceback

# ================================================================================
def get_file_list(path):
    file_list=glob.glob(path)
    file_list=natsort.natsorted(file_list,reverse=False)
    return file_list

def return_path_list_from_txt(txt_file):
    txt_file=open(txt_file, "r")
    read_lines=txt_file.readlines()
    num_file=int(len(read_lines))
    txt_file.close()
    return read_lines,num_file

def chunks(l,n):
  # For item i in range that is length of l,
  for i in range(0,len(l),n):
    # Create index range for l of n items:
    yield l[i:i+n]

def divisorGenerator(n):
    large_divisors=[]
    for i in range(1,int(math.sqrt(n)+1)):
        if n%i==0:
            yield i
            if i*i!=n:
                large_divisors.append(n/i)
    for divisor in reversed(large_divisors):
        yield int(divisor)
# list(divisorGenerator(1024))

