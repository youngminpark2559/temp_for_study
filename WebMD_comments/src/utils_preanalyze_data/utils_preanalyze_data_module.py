import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth',-1);pd.set_option('display.max_columns',None)
import argparse
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from scipy.stats import norm

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
import pickle
import re

# ================================================================================
from src.utils import utils_image as utils_image
from src.utils import utils_common as utils_common

# ================================================================================
Condition_type_2_diabetes_mellitus="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus"
Condition_prevention_of_type2_diabetes_mellitus="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_prevention_of_type2_diabetes_mellitus"
Condition_other="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_other"
Condition_disease_of_ovaries_with_cysts="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_disease_of_ovaries_with_cysts"
Condition_diabetes_during_pregnancy="/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_diabetes_during_pregnancy"

comments_data=[
  Condition_type_2_diabetes_mellitus,
  Condition_prevention_of_type2_diabetes_mellitus,
  Condition_other,
  Condition_disease_of_ovaries_with_cysts]

# ================================================================================
def load_csv_file(path):
  loaded_csv=pd.read_csv(path,encoding='utf8')
  return loaded_csv

def check_nan(df):
  ret=df.isna()
  print("ret",ret)
  #      PatientID   Resp  PR Seq  RT Seq  VL-t0  CD4-t0
  # 0    False      False  False   False   False  False 
  # 1    False      False  False   False   False  False 

  # ================================================================================
  sum_nan=ret.sum()
  print("sum_nan",sum_nan)
  # PatientID    0 
  # Resp         0 
  # PR Seq       80
  # RT Seq       0 
  # VL-t0        0 
  # CD4-t0       0 

def estimated_PDF_by_using_mean_and_var(loaded_csv,args):
  # ================================================================================
  means=loaded_csv.describe().iloc[1,:]
  stds=loaded_csv.describe().iloc[2,:]
  vars=stds**2
  # print("means",means)
  # Pregnancies                 3.845052  
  # Glucose                     120.894531
  # BloodPressure               69.105469 
  # SkinThickness               20.536458 
  # Insulin                     79.799479 
  # BMI                         31.992578 
  # DiabetesPedigreeFunction    0.471876  
  # Age                         33.240885 
  # Outcome                     0.348958  

  # print("stds",stds)
  # Pregnancies                 3.369578  
  # Glucose                     31.972618 
  # BloodPressure               19.355807 
  # SkinThickness               15.952218 
  # Insulin                     115.244002
  # BMI                         7.884160  
  # DiabetesPedigreeFunction    0.331329  
  # Age                         11.760232 
  # Outcome                     0.476951  

  # print("vars",vars)
  # Pregnancies                 11.354056   
  # Glucose                     1022.248314 
  # BloodPressure               374.647271  
  # SkinThickness               254.473245  
  # Insulin                     13281.180078
  # BMI                         62.159984   
  # DiabetesPedigreeFunction    0.109779    
  # Age                         138.303046  
  # Outcome                     0.227483    

  params=list(zip(means,vars))
  # print("params",params)
  # [(3.8450520833333335, 11.354056320621417), (120.89453125, 1022.2483142519558), (69.10546875, 374.64727122718375), (20.536458333333332, 254.47324532811953), (79.79947916666667, 13281.180077955283), (31.992578124999977, 62.159983957382565), (0.4718763020833327, 0.10977863787313936), (33.240885416666664, 138.30304589037362), (0.3489583333333333, 0.22748261625380098)]

  # print("",len(params))
  # 9

  nb_h=len(params)/3
  nb_w=len(params)/3

  
  features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
  for i,one_feat in enumerate(params):
    # print("one_feat",one_feat)
    # (3.8450520833333335, 3.3695780626988623)

    x_values=np.arange(-1000,1000,0.001)
    plt.subplot(nb_h,nb_w,i+1)
    plt.title(features[i]+" / mean: "+str(round(one_feat[0],2))+" / variance: "+str(round(one_feat[1],2)))
    plt.subplots_adjust(wspace=None,hspace=0.3)
    plt.plot(x_values,norm.pdf(x_values,one_feat[0],one_feat[1]))
  plt.show()
  # /home/young/Pictures/2019_05_10_06:48:15.png

def check_nan_vals(loaded_csv,args):
  print("loaded_csv.info()",loaded_csv.info())

  # Pregnancies                 768 non-null int64
  # Glucose                     768 non-null int64
  # BloodPressure               768 non-null int64
  # SkinThickness               768 non-null int64
  # Insulin                     768 non-null int64
  # BMI                         768 non-null float64
  # DiabetesPedigreeFunction    768 non-null float64
  # Age                         768 non-null int64
  # Outcome                     768 non-null int64

def see_correlations_on_features(loaded_csv,args):
  train_csv_wo_id=loaded_csv
  # print("train_csv_wo_id",train_csv_wo_id.shape)
  # (920, 5)

  # ================================================================================
  cor_mat=train_csv_wo_id.corr()
  # print("cor_mat",cor_mat.shape)
  # (9, 9)

  # print("cor_mat",cor_mat)
  #                           Pregnancies   Glucose  BloodPressure  SkinThickness  \
  # Pregnancies               1.000000     0.129459  0.141282      -0.081672        
  # Glucose                   0.129459     1.000000  0.152590       0.057328        
  # BloodPressure             0.141282     0.152590  1.000000       0.207371        
  # SkinThickness            -0.081672     0.057328  0.207371       1.000000        
  # Insulin                  -0.073535     0.331357  0.088933       0.436783        
  # BMI                       0.017683     0.221071  0.281805       0.392573        
  # DiabetesPedigreeFunction -0.033523     0.137337  0.041265       0.183928        
  # Age                       0.544341     0.263514  0.239528      -0.113970        
  # Outcome                   0.221898     0.466581  0.065068       0.074752        

  #                            Insulin       BMI  DiabetesPedigreeFunction  \
  # Pregnancies              -0.073535  0.017683 -0.033523                   
  # Glucose                   0.331357  0.221071  0.137337                   
  # BloodPressure             0.088933  0.281805  0.041265                   
  # SkinThickness             0.436783  0.392573  0.183928                   
  # Insulin                   1.000000  0.197859  0.185071                   
  # BMI                       0.197859  1.000000  0.140647                   
  # DiabetesPedigreeFunction  0.185071  0.140647  1.000000                   
  # Age                      -0.042163  0.036242  0.033561                   
  # Outcome                   0.130548  0.292695  0.173844                   

  #                                Age   Outcome  
  # Pregnancies               0.544341  0.221898  
  # Glucose                   0.263514  0.466581  
  # BloodPressure             0.239528  0.065068  
  # SkinThickness            -0.113970  0.074752  
  # Insulin                  -0.042163  0.130548  
  # BMI                       0.036242  0.292695  
  # DiabetesPedigreeFunction  0.033561  0.173844  
  # Age                       1.000000  0.238356  
  # Outcome                   0.238356  1.000000  

  # ================================================================================
  # * Normalize data
  # * https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

  cor_mat_np=np.array(cor_mat,dtype="float16")
  cor_mat_np_sh=cor_mat_np.shape

  min_in_arr=np.min(cor_mat_np.reshape(-1))
  max_in_arr=np.max(cor_mat_np.reshape(-1))
  # print("min_in_arr",min_in_arr)
  # print("max_in_arr",max_in_arr)
  # -0.4272
  # 1.0

  # c norm_corr_mat_np: normalized correlation matrix in np
  norm_corr_mat_np=(cor_mat_np-min_in_arr)/(max_in_arr-min_in_arr)
  # print("norm_corr_mat_np",norm_corr_mat_np)
  # [[1.    0.554 0.21 ]
  #  [0.554 1.    0.   ]
  #  [0.21  0.    1.   ]]

  # norm_corr_mat_df=pd.DataFrame(norm_corr_mat_np)
  norm_corr_mat_df=pd.DataFrame(cor_mat_np)
  # print("norm_corr_mat_df",norm_corr_mat_df)
  #           0         1         2
  # 0  1.000000  0.554199  0.209961
  # 1  0.554199  1.000000  0.000000
  # 2  0.209961  0.000000  1.000000

  new_col_name={0:"Pregnancies",1:"Glucose",2:"BloodPressure",3:"SkinThickness",4:"Insulin",5:"BMI",6:"DiabetesPedigreeFunction",7:"Age",8:"Outcome"}
  new_idx_name={0:"Pregnancies",1:"Glucose",2:"BloodPressure",3:"SkinThickness",4:"Insulin",5:"BMI",6:"DiabetesPedigreeFunction",7:"Age",8:"Outcome"}
  norm_corr_mat_df_a=norm_corr_mat_df.rename(columns=new_col_name,index=new_idx_name,inplace=False)
  # print("norm_corr_mat_df_a",norm_corr_mat_df_a)
  #                           Pregnancies   Glucose  BloodPressure  SkinThickness  \
  # Pregnancies               1.000000     0.218506  0.229004       0.028976        
  # Glucose                   0.218506     1.000000  0.239258       0.153687        
  # BloodPressure             0.229004     0.239258  1.000000       0.288330        
  # SkinThickness             0.028976     0.153687  0.288330       1.000000        
  # Insulin                   0.036255     0.399658  0.182129       0.494385        
  # BMI                       0.118103     0.300537  0.355225       0.454346        
  # DiabetesPedigreeFunction  0.072205     0.225464  0.139404       0.267334        
  # Age                       0.590820     0.338623  0.317383       0.000000        
  # Outcome                   0.301514     0.520996  0.160645       0.169312        

  #                           Insulin       BMI  DiabetesPedigreeFunction  \
  # Pregnancies               0.036255  0.118103  0.072205                   
  # Glucose                   0.399658  0.300537  0.225464                   
  # BloodPressure             0.182129  0.355225  0.139404                   
  # SkinThickness             0.494385  0.454346  0.267334                   
  # Insulin                   1.000000  0.279785  0.268311                   
  # BMI                       0.279785  1.000000  0.228516                   
  # DiabetesPedigreeFunction  0.268311  0.228516  1.000000                   
  # Age                       0.064392  0.134766  0.132324                   
  # Outcome                   0.219360  0.364990  0.258301                   

  #                                Age   Outcome  
  # Pregnancies               0.590820  0.301514  
  # Glucose                   0.338623  0.520996  
  # BloodPressure             0.317383  0.160645  
  # SkinThickness             0.000000  0.169312  
  # Insulin                   0.064392  0.219360  
  # BMI                       0.134766  0.364990  
  # DiabetesPedigreeFunction  0.132324  0.258301  
  # Age                       1.000000  0.316162  
  # Outcome                   0.316162  1.000000  

  sns.heatmap(norm_corr_mat_df_a)
  plt.show()
  # /home/young/Pictures/2019_05_09_21:27:16.png

  # Meaning:
  # 1.. Negative correlation: Insulin-Pregnancies, SkinThickness-Pregnancies, Age-SkinThickness,
  # 2.. Positive correlation: Age-Pregnancies, BMI-SkinThickness, Insluin-SkinThickness, Insulin-Glucose

def create_diabete_label_by_120_glucose(loaded_csv,args):
  label_col=np.zeros((loaded_csv.shape[0]))
  # print("label_col",label_col.shape)
  # (768,)

  glucose_data=loaded_csv.iloc[:,1]
  label_col[glucose_data>=120]=1
  # print("label_col",label_col)

  loaded_csv["glucose_over_120"]=label_col
  # print("loaded_csv",loaded_csv.head(2))
  #    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI   DiabetesPedigreeFunction  Age  Outcome  glucose_over_120  
  # 0  6            148      72             35             0        33.6   0.627                     50   1        1.0               
  # 1  1            85       66             29             0        26.6   0.351                     31   0        0.0               

  return loaded_csv

def get_cond_probability_preg(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  pregnancies_data=loaded_csv.iloc[:,0]
  # print("pregnancies_data",pregnancies_data)
  # 0      6 
  # 1      1 
  # 2      8 
  # 3      1 

  uniq_vals=np.unique(pregnancies_data)
  # print("uniq_vals",uniq_vals)
  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 17]

  num_ele_in_one_group=int(len(uniq_vals)/4)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 4
  
  intervals=[
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15,17]]

  for i in range(4):
    first_val_for_inter=intervals[i][0]
    last_val_for_inter=intervals[i][-1]+1
    # print("first_val_for_inter",first_val_for_inter)
    # print("last_val_for_inter",last_val_for_inter)
    # 0 
    # 4 

    mask_for_each_interval=np.logical_and(first_val_for_inter<=pregnancies_data,pregnancies_data<last_val_for_inter)
    # print("mask_for_each_interval",mask_for_each_interval)
    # 0      False
    # 1      True 
    # 2      False

    masked_loaded_csv=loaded_csv[mask_for_each_interval]
    # print("masked_loaded_csv",masked_loaded_csv.shape)
    # (424, 10)

    masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
    # print("masked_by_1_label",masked_by_1_label.shape)
    # (173, 10)

    # ================================================================================
    # @ Let's apply following conditional probability fomular
    # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

    print("interval information: ["+str(first_val_for_inter)+","+str(last_val_for_inter-1)+"]")

    P_high_glucose_AND_preg_each_interval=round(masked_by_1_label.shape[0]/768,2)
    print("P_high_glucose_AND_preg_each_interval",P_high_glucose_AND_preg_each_interval)
    # 0.23

    P_preg_each_interval=round(masked_loaded_csv.shape[0]/768,2)
    print("P_preg_each_interval",P_preg_each_interval)
    # 0.55

    cond_prob_of_high_glucose_when_preg_0to3_is_given=round(P_high_glucose_AND_preg_each_interval/P_preg_each_interval,2)
    print("cond_prob_of_high_glucose_when_preg_0to3_is_given",cond_prob_of_high_glucose_when_preg_0to3_is_given)
    # 0.42
  
  # ================================================================================
  # interval information: [0,3]
  # P_high_glucose_AND_preg_each_interval 0.23
  # P_preg_each_interval 0.55
  # cond_prob_of_high_glucose_when_preg_0to3_is_given 0.42
  
  # interval information: [4,7]
  # P_high_glucose_AND_preg_each_interval 0.15
  # P_preg_each_interval 0.29
  # cond_prob_of_high_glucose_when_preg_0to3_is_given 0.52
  
  # interval information: [8,11]
  # P_high_glucose_AND_preg_each_interval 0.08
  # P_preg_each_interval 0.13
  # cond_prob_of_high_glucose_when_preg_0to3_is_given 0.62
  
  # interval information: [12,17]
  # P_high_glucose_AND_preg_each_interval 0.02
  # P_preg_each_interval 0.03
  # cond_prob_of_high_glucose_when_preg_0to3_is_given 0.67

def get_cond_probability_BP(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  BP_data=loaded_csv.iloc[:,2]
  # print("BP_data",BP_data)
  # 0      72
  # 1      66
  # 2      64

  uniq_vals=np.unique(BP_data)
  # print("uniq_vals",uniq_vals)
  # [  0  24  30  38  40  44  46  48  50  52  54  55  56  58  60  61  62  64
  #   65  66  68  70  72  74  75  76  78  80  82  84  85  86  88  90  92  94
  #   95  96  98 100 102 104 106 108 110 114 122]

  num_ele_in_one_group=int(len(uniq_vals)/8)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 5

  splited_uniq_vals=list(utils_common.chunks(uniq_vals,4))
  # print("splited_uniq_vals",splited_uniq_vals)
  # [array([ 0, 24, 30, 38]),
  #  array([40, 44, 46, 48]),
  #  array([50, 52, 54, 55]),
  #  array([56, 58, 60, 61]),
  #  array([62, 64, 65, 66]),
  #  array([68, 70, 72, 74]),
  #  array([75, 76, 78, 80]),
  #  array([82, 84, 85, 86]),
  #  array([88, 90, 92, 94]),
  #  array([ 95,  96,  98, 100]),
  #  array([102, 104, 106, 108]),
  #  array([110, 114, 122])]

  intervals=splited_uniq_vals

  for i in range(len(splited_uniq_vals)):
    first_val_for_inter=intervals[i][0]
    last_val_for_inter=intervals[i][-1]+1
    # print("first_val_for_inter",first_val_for_inter)
    # print("last_val_for_inter",last_val_for_inter)
    # 0
    # 39

    mask_for_each_interval=np.logical_and(first_val_for_inter<=BP_data,BP_data<last_val_for_inter)
    # print("mask_for_each_interval",mask_for_each_interval)
    # 0      False
    # 1      True 
    # 2      False

    masked_loaded_csv=loaded_csv[mask_for_each_interval]
    # print("masked_loaded_csv",masked_loaded_csv.shape)
    # (424, 10)

    masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
    # print("masked_by_1_label",masked_by_1_label.shape)
    # (173, 10)

    # ================================================================================
    # @ Let's apply following conditional probability fomular
    # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

    print("interval information: ["+str(first_val_for_inter)+","+str(last_val_for_inter-1)+"]")

    P_high_glucose_AND_preg_each_interval=masked_by_1_label.shape[0]/768
    print("P_high_glucose_AND_preg_each_interval",P_high_glucose_AND_preg_each_interval)
    # 0.23

    P_preg_each_interval=masked_loaded_csv.shape[0]/768
    print("P_preg_each_interval",P_preg_each_interval)
    # 0.55

    cond_prob_of_high_glucose_when_preg_each_interval_is_given=P_high_glucose_AND_preg_each_interval/P_preg_each_interval
    print("cond_prob_of_high_glucose_when_preg_each_interval_is_given",cond_prob_of_high_glucose_when_preg_each_interval_is_given)
    # 0.42
  
  # ================================================================================
  # interval information: [0,38]
  # P_high_glucose_AND_preg_each_interval 0.018229166666666668
  # P_preg_each_interval 0.05078125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.358974358974359
  
  # interval information: [40,48]
  # P_high_glucose_AND_preg_each_interval 0.006510416666666667
  # P_preg_each_interval 0.015625
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.4166666666666667
  
  # interval information: [50,55]
  # P_high_glucose_AND_preg_each_interval 0.016927083333333332
  # P_preg_each_interval 0.048177083333333336
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.3513513513513513
  
  # interval information: [56,61]
  # P_high_glucose_AND_preg_each_interval 0.03125
  # P_preg_each_interval 0.09244791666666667
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.3380281690140845
  
  # interval information: [62,66]
  # P_high_glucose_AND_preg_each_interval 0.041666666666666664
  # P_preg_each_interval 0.1484375
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.2807017543859649
  
  # interval information: [68,74]
  # P_high_glucose_AND_preg_each_interval 0.125
  # P_preg_each_interval 0.2578125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.48484848484848486
  
  # interval information: [75,80]
  # P_high_glucose_AND_preg_each_interval 0.08854166666666667
  # P_preg_each_interval 0.171875
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.5151515151515151
  
  # interval information: [82,86]
  # P_high_glucose_AND_preg_each_interval 0.06770833333333333
  # P_preg_each_interval 0.10416666666666667
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.6499999999999999
  
  # interval information: [88,94]
  # P_high_glucose_AND_preg_each_interval 0.052083333333333336
  # P_preg_each_interval 0.07942708333333333
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.6557377049180328
  
  # interval information: [95,100]
  # P_high_glucose_AND_preg_each_interval 0.0078125
  # P_preg_each_interval 0.014322916666666666
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.5454545454545455
  
  # interval information: [102,108]
  # P_high_glucose_AND_preg_each_interval 0.0078125
  # P_preg_each_interval 0.010416666666666666
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.75
  
  # interval information: [110,122]
  # P_high_glucose_AND_preg_each_interval 0.005208333333333333
  # P_preg_each_interval 0.006510416666666667
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.7999999999999999

def get_cond_probability_SkinThickness(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  SkinThickness_data=loaded_csv.iloc[:,3]
  # print("SkinThickness_data",SkinThickness_data)
  # 0      35
  # 1      29
  # 2      0 

  uniq_vals=np.unique(SkinThickness_data)
  # print("uniq_vals",uniq_vals)
  # [ 0  7  8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
  #   31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 54 56
  #   60 63 99]

  num_ele_in_one_group=int(len(uniq_vals)/11)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 4

  splited_uniq_vals=list(utils_common.chunks(uniq_vals,4))
  # print("splited_uniq_vals",splited_uniq_vals)
  # [array([ 0,  7,  8, 10]),
  #  array([11, 12, 13, 14]),
  #  array([15, 16, 17, 18]),
  #  array([19, 20, 21, 22]),
  #  array([23, 24, 25, 26]),
  #  array([27, 28, 29, 30]),
  #  array([31, 32, 33, 34]),
  #  array([35, 36, 37, 38]),
  #  array([39, 40, 41, 42]),
  #  array([43, 44, 45, 46]),
  #  array([47, 48, 49, 50]),
  #  array([51, 52, 54, 56]),
  #  array([60, 63, 99])]

  intervals=splited_uniq_vals

  for i in range(len(splited_uniq_vals)):
    first_val_for_inter=intervals[i][0]
    last_val_for_inter=intervals[i][-1]+1
    # print("first_val_for_inter",first_val_for_inter)
    # print("last_val_for_inter",last_val_for_inter)
    # 0
    # 11

    mask_for_each_interval=np.logical_and(first_val_for_inter<=SkinThickness_data,SkinThickness_data<last_val_for_inter)
    # print("mask_for_each_interval",mask_for_each_interval)
    # 0      False
    # 1      True 
    # 2      False

    masked_loaded_csv=loaded_csv[mask_for_each_interval]
    # print("masked_loaded_csv",masked_loaded_csv.shape)
    # (424, 10)

    masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
    # print("masked_by_1_label",masked_by_1_label.shape)
    # (173, 10)

    # ================================================================================
    # @ Let's apply following conditional probability fomular
    # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

    print("interval information: ["+str(first_val_for_inter)+","+str(last_val_for_inter-1)+"]")

    P_high_glucose_AND_preg_each_interval=round(masked_by_1_label.shape[0]/768,2)
    print("P_high_glucose_AND_preg_each_interval",P_high_glucose_AND_preg_each_interval)
    # 0.23

    P_preg_each_interval=masked_loaded_csv.shape[0]/768
    print("P_preg_each_interval",P_preg_each_interval)
    # 0.55

    cond_prob_of_high_glucose_when_preg_each_interval_is_given=round(P_high_glucose_AND_preg_each_interval/P_preg_each_interval,2)
    print("cond_prob_of_high_glucose_when_preg_each_interval_is_given",cond_prob_of_high_glucose_when_preg_each_interval_is_given)
    # 0.42
  
  # ================================================================================
  # interval information: [0,10]
  # P_high_glucose_AND_preg_each_interval 0.15
  # P_preg_each_interval 0.3072916666666667
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.49
  
  # interval information: [11,14]
  # P_high_glucose_AND_preg_each_interval 0.01
  # P_preg_each_interval 0.0390625
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.26
  
  # interval information: [15,18]
  # P_high_glucose_AND_preg_each_interval 0.02
  # P_preg_each_interval 0.0703125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.28
  
  # interval information: [19,22]
  # P_high_glucose_AND_preg_each_interval 0.02
  # P_preg_each_interval 0.07421875
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.27
  
  # interval information: [23,26]
  # P_high_glucose_AND_preg_each_interval 0.04
  # P_preg_each_interval 0.0859375
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.47
  
  # interval information: [27,30]
  # P_high_glucose_AND_preg_each_interval 0.05
  # P_preg_each_interval 0.11328125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.44
  
  # interval information: [31,34]
  # P_high_glucose_AND_preg_each_interval 0.05
  # P_preg_each_interval 0.1015625
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.49
  
  # interval information: [35,38]
  # P_high_glucose_AND_preg_each_interval 0.04
  # P_preg_each_interval 0.06770833333333333
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.59
  
  # interval information: [39,42]
  # P_high_glucose_AND_preg_each_interval 0.04
  # P_preg_each_interval 0.078125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.51
  
  # interval information: [43,46]
  # P_high_glucose_AND_preg_each_interval 0.02
  # P_preg_each_interval 0.032552083333333336
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.61
  
  # interval information: [47,50]
  # P_high_glucose_AND_preg_each_interval 0.01
  # P_preg_each_interval 0.018229166666666668
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.55
  
  # interval information: [51,56]
  # P_high_glucose_AND_preg_each_interval 0.0
  # P_preg_each_interval 0.0078125
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.0
  
  # interval information: [60,99]
  # P_high_glucose_AND_preg_each_interval 0.0
  # P_preg_each_interval 0.00390625
  # cond_prob_of_high_glucose_when_preg_each_interval_is_given 0.0

def get_cond_probability_Insulin(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  Insulin_data=loaded_csv.iloc[:,4]
  # print("Insulin_data",Insulin_data)
  # 0      35
  # 1      29
  # 2      0 

  uniq_vals=np.unique(Insulin_data)
  # print("uniq_vals",uniq_vals)
  # [  0  14  15  16  18  22  23  25  29  32  36  37  38  40  41  42  43  44
  #   45  46  48  49  50  51  52  53  54  55  56  57  58  59  60  61  63  64
  #   65  66  67  68  70  71  72  73  74  75  76  77  78  79  81  82  83  84
  #   85  86  87  88  89  90  91  92  94  95  96  99 100 105 106 108 110 112
  # 114 115 116 119 120 122 125 126 127 128 129 130 132 135 140 142 144 145
  # 146 148 150 152 155 156 158 159 160 165 166 167 168 170 171 175 176 178
  # 180 182 183 184 185 188 190 191 192 193 194 196 200 204 205 207 210 215
  # 220 225 228 230 231 235 237 240 245 249 250 255 258 265 270 271 272 274
  # 275 277 278 280 284 285 291 293 300 304 310 318 321 325 326 328 330 335
  # 342 360 370 375 387 392 402 415 440 465 474 478 480 485 495 510 540 543
  # 545 579 600 680 744 846]

  num_ele_in_one_group=int(len(uniq_vals)/11)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 4

  splited_uniq_vals=list(utils_common.chunks(uniq_vals,4))
  # print("splited_uniq_vals",splited_uniq_vals)
  # [array([ 0, 14, 15, 16]),
  #  array([18, 22, 23, 25]),
  #  array([29, 32, 36, 37]),
  #  array([38, 40, 41, 42]),
  #  array([43, 44, 45, 46]),
  #  array([48, 49, 50, 51]),
  #  array([52, 53, 54, 55]),
  #  array([56, 57, 58, 59]),
  #  array([60, 61, 63, 64]),
  #  array([65, 66, 67, 68]),
  #  array([70, 71, 72, 73]),
  #  array([74, 75, 76, 77]),
  #  array([78, 79, 81, 82]),
  #  array([83, 84, 85, 86]),
  #  array([87, 88, 89, 90]),
  #  array([91, 92, 94, 95]),
  #  array([ 96,  99, 100, 105]),
  #  array([106, 108, 110, 112]),
  #  array([114, 115, 116, 119]),
  #  array([120, 122, 125, 126]),
  #  array([127, 128, 129, 130]),
  #  array([132, 135, 140, 142]),
  #  array([144, 145, 146, 148]),
  #  array([150, 152, 155, 156]),
  #  array([158, 159, 160, 165]),
  #  array([166, 167, 168, 170]),
  #  array([171, 175, 176, 178]),
  #  array([180, 182, 183, 184]),
  #  array([185, 188, 190, 191]),
  #  array([192, 193, 194, 196]),
  #  array([200, 204, 205, 207]),
  #  array([210, 215, 220, 225]),
  #  array([228, 230, 231, 235]),
  #  array([237, 240, 245, 249]),
  #  array([250, 255, 258, 265]),
  #  array([270, 271, 272, 274]),
  #  array([275, 277, 278, 280]),
  #  array([284, 285, 291, 293]),
  #  array([300, 304, 310, 318]),
  #  array([321, 325, 326, 328]),
  #  array([330, 335, 342, 360]),
  #  array([370, 375, 387, 392]),
  #  array([402, 415, 440, 465]),
  #  array([474, 478, 480, 485]),
  #  array([495, 510, 540, 543]),
  #  array([545, 579, 600, 680]),
  #  array([744, 846])]

  intervals=splited_uniq_vals

  prob_intercept=[]
  prob_new_popu=[]
  prob_cond=[]
  for i in range(len(splited_uniq_vals)):
    first_val_for_inter=intervals[i][0]
    last_val_for_inter=intervals[i][-1]+1
    # print("first_val_for_inter",first_val_for_inter)
    # print("last_val_for_inter",last_val_for_inter)
    # 0
    # 11

    mask_for_each_interval=np.logical_and(first_val_for_inter<=Insulin_data,Insulin_data<last_val_for_inter)
    # print("mask_for_each_interval",mask_for_each_interval)
    # 0      False
    # 1      True 
    # 2      False

    masked_loaded_csv=loaded_csv[mask_for_each_interval]
    # print("masked_loaded_csv",masked_loaded_csv.shape)
    # (424, 10)

    masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
    # print("masked_by_1_label",masked_by_1_label.shape)
    # (173, 10)

    # ================================================================================
    # @ Let's apply following conditional probability fomular
    # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

    print("interval information: ["+str(first_val_for_inter)+","+str(last_val_for_inter-1)+"]")

    P_high_glucose_AND_Insulin_each_interval=masked_by_1_label.shape[0]/768
    print("P_high_glucose_AND_Insulin_each_interval",P_high_glucose_AND_Insulin_each_interval)
    # 0.23
    prob_intercept.append(P_high_glucose_AND_Insulin_each_interval)

    P_Insulin_each_interval=masked_loaded_csv.shape[0]/768
    print("P_Insulin_each_interval",P_Insulin_each_interval)
    # 0.55
    prob_new_popu.append(P_Insulin_each_interval)

    cond_prob_of_high_glucose_when_Insulin_each_interval_is_given=P_high_glucose_AND_Insulin_each_interval/P_Insulin_each_interval
    print("cond_prob_of_high_glucose_when_Insulin_each_interval_is_given",cond_prob_of_high_glucose_when_Insulin_each_interval_is_given)
    # 0.42
    prob_cond.append(cond_prob_of_high_glucose_when_Insulin_each_interval_is_given)
  
  # ================================================================================
  plt.subplot(1,3,1)
  plt.title("probability of A and B intercept occurring")
  plt.plot(prob_intercept)
  plt.subplot(1,3,2)
  plt.title("probability of new population like P(B) occuring")
  plt.plot(prob_new_popu)
  plt.subplot(1,3,3)
  plt.title("conditional probability of high glucose occuring when each insulin interval is given")
  plt.plot(prob_cond)
  plt.show()
  # /home/young/Pictures/2019_05_10_14:06:37.png

  # Meaning:
  # - According to 3rd plot, high insulin feature must result in high glucose?
  
def get_cond_probability_BMI(loaded_csv,args):
  # print("loaded_csv",loaded_csv.shape)
  # 768

  # print("loaded_csv.columns",loaded_csv.columns)
  # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'glucose_over_120']

  glucose_over_120_data=loaded_csv.iloc[:,-1]
  # print("glucose_over_120_data",glucose_over_120_data)
  # 0      1.0
  # 1      0.0

  num_high_glucose=glucose_over_120_data.sum()
  # print("num_high_glucose",num_high_glucose)
  # 360.0

  # ================================================================================
  # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

  BMI_data=loaded_csv.iloc[:,5]
  # print("BMI_data",BMI_data)
  # 0      35
  # 1      29
  # 2      0 

  uniq_vals=np.unique(BMI_data)
  # print("uniq_vals",uniq_vals)
  # [  0  14  15  16  18  22  23  25  29  32  36  37  38  40  41  42  43  44
  #   45  46  48  49  50  51  52  53  54  55  56  57  58  59  60  61  63  64
  #   65  66  67  68  70  71  72  73  74  75  76  77  78  79  81  82  83  84
  #   85  86  87  88  89  90  91  92  94  95  96  99 100 105 106 108 110 112
  # 114 115 116 119 120 122 125 126 127 128 129 130 132 135 140 142 144 145
  # 146 148 150 152 155 156 158 159 160 165 166 167 168 170 171 175 176 178
  # 180 182 183 184 185 188 190 191 192 193 194 196 200 204 205 207 210 215
  # 220 225 228 230 231 235 237 240 245 249 250 255 258 265 270 271 272 274
  # 275 277 278 280 284 285 291 293 300 304 310 318 321 325 326 328 330 335
  # 342 360 370 375 387 392 402 415 440 465 474 478 480 485 495 510 540 543
  # 545 579 600 680 744 846]

  num_ele_in_one_group=int(len(uniq_vals)/11)
  # print("num_ele_in_one_group",num_ele_in_one_group)
  # 4

  splited_uniq_vals=list(utils_common.chunks(uniq_vals,4))
  # print("splited_uniq_vals",splited_uniq_vals)
  # [array([ 0, 14, 15, 16]),
  #  array([18, 22, 23, 25]),
  #  array([29, 32, 36, 37]),
  #  array([38, 40, 41, 42]),
  #  array([43, 44, 45, 46]),
  #  array([48, 49, 50, 51]),
  #  array([52, 53, 54, 55]),
  #  array([56, 57, 58, 59]),
  #  array([60, 61, 63, 64]),
  #  array([65, 66, 67, 68]),
  #  array([70, 71, 72, 73]),
  #  array([74, 75, 76, 77]),
  #  array([78, 79, 81, 82]),
  #  array([83, 84, 85, 86]),
  #  array([87, 88, 89, 90]),
  #  array([91, 92, 94, 95]),
  #  array([ 96,  99, 100, 105]),
  #  array([106, 108, 110, 112]),
  #  array([114, 115, 116, 119]),
  #  array([120, 122, 125, 126]),
  #  array([127, 128, 129, 130]),
  #  array([132, 135, 140, 142]),
  #  array([144, 145, 146, 148]),
  #  array([150, 152, 155, 156]),
  #  array([158, 159, 160, 165]),
  #  array([166, 167, 168, 170]),
  #  array([171, 175, 176, 178]),
  #  array([180, 182, 183, 184]),
  #  array([185, 188, 190, 191]),
  #  array([192, 193, 194, 196]),
  #  array([200, 204, 205, 207]),
  #  array([210, 215, 220, 225]),
  #  array([228, 230, 231, 235]),
  #  array([237, 240, 245, 249]),
  #  array([250, 255, 258, 265]),
  #  array([270, 271, 272, 274]),
  #  array([275, 277, 278, 280]),
  #  array([284, 285, 291, 293]),
  #  array([300, 304, 310, 318]),
  #  array([321, 325, 326, 328]),
  #  array([330, 335, 342, 360]),
  #  array([370, 375, 387, 392]),
  #  array([402, 415, 440, 465]),
  #  array([474, 478, 480, 485]),
  #  array([495, 510, 540, 543]),
  #  array([545, 579, 600, 680]),
  #  array([744, 846])]

  intervals=splited_uniq_vals

  prob_intercept=[]
  prob_new_popu=[]
  prob_cond=[]
  for i in range(len(splited_uniq_vals)):
    first_val_for_inter=intervals[i][0]
    last_val_for_inter=intervals[i][-1]+1
    # print("first_val_for_inter",first_val_for_inter)
    # print("last_val_for_inter",last_val_for_inter)
    # 0
    # 11

    mask_for_each_interval=np.logical_and(first_val_for_inter<=BMI_data,BMI_data<last_val_for_inter)
    # print("mask_for_each_interval",mask_for_each_interval)
    # 0      False
    # 1      True 
    # 2      False

    masked_loaded_csv=loaded_csv[mask_for_each_interval]
    # print("masked_loaded_csv",masked_loaded_csv.shape)
    # (424, 10)

    masked_by_1_label=masked_loaded_csv[masked_loaded_csv.iloc[:,-1]==1.0]
    # print("masked_by_1_label",masked_by_1_label.shape)
    # (173, 10)

    # ================================================================================
    # @ Let's apply following conditional probability fomular
    # p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)

    print("interval information: ["+str(first_val_for_inter)+","+str(last_val_for_inter-1)+"]")

    P_high_glucose_AND_BMI_each_interval=masked_by_1_label.shape[0]/768
    print("P_high_glucose_AND_BMI_each_interval",P_high_glucose_AND_BMI_each_interval)
    # 0.23
    prob_intercept.append(P_high_glucose_AND_BMI_each_interval)

    P_BMI_each_interval=masked_loaded_csv.shape[0]/768
    print("P_BMI_each_interval",P_BMI_each_interval)
    # 0.55
    prob_new_popu.append(P_BMI_each_interval)

    cond_prob_of_high_glucose_when_BMI_each_interval_is_given=P_high_glucose_AND_BMI_each_interval/P_BMI_each_interval
    print("cond_prob_of_high_glucose_when_BMI_each_interval_is_given",cond_prob_of_high_glucose_when_BMI_each_interval_is_given)
    # 0.42
    prob_cond.append(cond_prob_of_high_glucose_when_BMI_each_interval_is_given)
  
  # ================================================================================
  plt.subplot(1,3,1)
  plt.title("probability of A and B intercept occurring")
  plt.plot(prob_intercept)
  plt.subplot(1,3,2)
  plt.title("probability of new population like P(B) occuring")
  plt.plot(prob_new_popu)
  plt.subplot(1,3,3)
  plt.title("conditional probability of high glucose occuring when each insulin interval is given")
  plt.plot(prob_cond)
  plt.show()
  # /home/young/Pictures/2019_05_10_14:49:11.png

  # Meaning:
  # - According to 3rd plot, high insulin feature must result in high glucose?

def get_actual_distribution_using_histogram(loaded_csv,args):
  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
  preg_data=np.array(loaded_csv.iloc[:,0])
  # print("preg_data",preg_data)
  # [ 6  1  8  1  0  5  3 10  2  8  4 10 10  1  5  7  0  7  1  1  3  8  7  9

  for i in range(len(columns)):
    one_feat_data=np.array(loaded_csv.iloc[:,i])
    plt.subplot(len(columns)/3,len(columns)/3,i+1)
    plt.title(columns[i])
    plt.subplots_adjust(wspace=None,hspace=0.3)
    n,bins,patches=plt.hist(one_feat_data,bins=50)
  plt.show()

def see_pair_scatter_plot(loaded_csv,loaded_csv_label,args):
  # sns.pairplot(loaded_csv,diag_kind='hist')
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind='hist',hue="glucose_over_120")
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind='kde',hue="glucose_over_120",palette='bright') # pastel, bright, deep, muted, colorblind, dark
  # plt.title("With estimated distribution via Kernel Density Estimation")
  # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  # plt.show()

  # sns.pairplot(loaded_csv_label,diag_kind="kde",kind='reg',hue="glucose_over_120",palette='bright') # pastel, bright, deep, muted, colorblind, dark
  sns.pairplot(loaded_csv_label,kind='reg') # pastel, bright, deep, muted, colorblind, dark
  plt.title("Linear regression line which can explain distributional pattern of each 2 feature data")
  plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
  plt.show()
  # /home/young/Pictures/2019_05_10_08:05:39.png

  # Meaning
  # 1.. See Glucose-Insulin
  # - Less Glucose results in less Insulin
  # - Increase Glucose results in More Insulin
  # 2.. See BMI-Glucose
  # - Even if low BMI, it can have high Glucose
  # 3.. At least according this plotting, there is no strong correlation between all features and glucose level
  # - It means low BMI can have high glucose, high BMI can have high glucose.
  # - Low SkinThickness can have high glucose, high SkinThickness can have high glucose.

def get_overall_stat(one_condition_entire_data,args):
  overall_stat=one_condition_entire_data.pop(0)
  # print("overall_stat",overall_stat)
  # ['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)']

  overall_stat_refined=[]
  for one_score in overall_stat:
    replaced=one_score.replace("Effectiveness@@@@@Current Rating: 0@@@@@(","").replace("Ease of Use@@@@@Current Rating: 0@@@@@(","").replace("Satisfaction@@@@@Current Rating: 0@@@@@(","").replace(")","")
    # print("replaced",replaced)
    # 3.29

    overall_stat_refined.append(replaced)
  
  return overall_stat_refined

def process_analyze_scores_data(one_condition_entire_data,args):

  # one_condition_entire_data_popped=one_condition_entire_data.pop(0)

  all_pages_data=[]
  for one_page_data in one_condition_entire_data:
    # print("one_page_data",one_page_data)
    # ['@@@@@Page0', ['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease 

    # ================================================================================
    page_num=one_page_data.pop(0)
    # print("page_num",page_num)

    # ================================================================================
    # print("one_page_data",one_page_data)
    
    # ================================================================================
    one_page_data_np=np.array(one_page_data)
    # print("one_page_data_np",one_page_data_np)
    # print("one_page_data_np",one_page_data_np.shape)

    if one_page_data_np.shape[0]!=5:
      continue
    
    # [['@@@@@Post0' 'Condition: Type 2 Diabetes Mellitus'
    #   '5/24/2019 5:54:59 PM'
    #   'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)'
    #   'Effectiveness@@@@@1' 'Ease of Use@@@@@5' 'Satisfaction@@@@@1'
    #   'Comment:after a week----mouth ulccers,cudnt talk,eat,drink for 5 days....whole body burnt,headache, fatigue....quit---am slowly getting better, wudnt give to my worst enemy.']
    # ['@@@@@Post1' 'Condition: Type 2 Diabetes Mellitus'
    #   '5/21/2019 9:27:08 PM'
    #   'Reviewer: MamaKin, 55-64 Female on Treatment for 6 months to less than 1 year (Caregiver)'
    #   'Effectiveness@@@@@2' 'Ease of Use@@@@@3' 'Satisfaction@@@@@1'
    #   'Comment:My elderly mother was prescribed this medication in addition to insulin. Her blood sugar dropped so low she wound up in the ER after passing out. Also had been very nauseous with little appetite after a few months of use. I don’t like her using this at all.']
    # ['@@@@@Post2' 'Condition: Type 2 Diabetes Mellitus'
    #   '4/25/2019 11:04:35 AM'
    #   'Reviewer: Judy in TX, 55-64 on Treatment for less than 1 month (Patient)'
    #   'Effectiveness@@@@@3' 'Ease of Use@@@@@3' 'Satisfaction@@@@@1'
    #   'Comment:On this medication for 2 weeks. Gas had been the biggest change I noticed right away. For the past 2 days my skin got itchy and I am now covered in several patches of hives! Getting off this RX. Tried because my Janumet RX was $600 for 90 days with the coupon1 So my Dr. tried Metformin which is a generic. Unfortunately Metformin not going to work for me.']
    # ['@@@@@Post3' 'Condition: Type 2 Diabetes Mellitus'
    #   '4/7/2019 12:48:29 PM'
    #   'Reviewer: tinz, 55-64 Female on Treatment for less than 1 month (Patient)'
    #   'Effectiveness@@@@@4' 'Ease of Use@@@@@5' 'Satisfaction@@@@@5'
    #   "Comment:i have been taking this meds 500 mg 2x/day for a week. no side effect at all for me. i've lost 1.5 lbs in a week!"]
    # ['@@@@@Post4' 'Condition: Type 2 Diabetes Mellitus'
    #   '2/23/2019 9:38:24 AM'
    #   'Reviewer: Miss Nick, 35-44 Female on Treatment for 1 to 6 months (Patient)'
    #   'Effectiveness@@@@@2' 'Ease of Use@@@@@5' 'Satisfaction@@@@@1'
    #   'Comment:This medication caused my blood sugar to drop so low that I ended up in the ER not knowing who I was or what I was doing. I hated it.']]

    # ================================================================================
    # print("one_page_data_np",one_page_data_np.shape)

    all_pages_data.append(one_page_data_np)
  
  # print("all_pages_data",all_pages_data)
  
  columns=['Effectiveness','Ease_of_Use', 'Satisfaction']
  df=pd.Series(all_pages_data,index=None)

  all_pages_data=np.stack(all_pages_data,0)
  # print("all_pages_data",all_pages_data.shape)
  # (245, 5, 8)

  all_pages_data_rs=all_pages_data.reshape(all_pages_data.shape[0],-1)
  # print("all_pages_data_rs",all_pages_data_rs.shape)
  # (245, 40)

  # ================================================================================
  scores_data=all_pages_data_rs[:,4:7]
  # print("scores_data",scores_data)
  # [['Effectiveness@@@@@1' 'Ease of Use@@@@@5' 'Satisfaction@@@@@1']

  # ================================================================================
  all_data_grp_temp=[]
  for one_data in scores_data:
    # print("one_data",one_data)
    # ['Effectiveness@@@@@1' 'Ease of Use@@@@@5' 'Satisfaction@@@@@1']

    one_data_grp_temp=[]
    for one_score in one_data:
      regex=r"[a-zA-Z].+[^0-9]"

      test_str=one_score

      matches=re.finditer(regex,test_str,re.MULTILINE)

      useless_strs=[]
      for one_match in matches:
        useless_str=one_match.group()
        # print("useless_str",useless_str)
        # Effectiveness@@@@@
        # Ease of Use@@@@@
        # Satisfaction@@@@@

        # ================================================================================
        refined=one_score.replace(useless_str,"")
        # print("refined",refined)

        one_data_grp_temp.append(float(refined))

    # print("one_data_grp_temp",one_data_grp_temp)
    # [1.0, 5.0, 1.0]

    all_data_grp_temp.append(one_data_grp_temp)
  
  # ================================================================================
  # print("all_data_grp_temp",all_data_grp_temp)
  # [[1.0, 5.0, 1.0], [3.0, 4.0, 1.0], [5.0, 5.0, 5.0], [4.0, 4.0, 4.0], [2.0, 5.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [4.0, 4.0, 1.0], [2.0, 2.0, 2.0], [1.0, 3.0, 1.0], [5.0,
  
  columns=['Effectiveness','Ease_of_Use', 'Satisfaction']
  df_score=pd.DataFrame(all_data_grp_temp,index=None,columns=columns)

  # print("df_score",df_score.describe())
  #        Effectiveness  Ease_of_Use  Satisfaction
  # count  245.000000     245.000000   245.000000  
  # mean   3.236735       3.804082     2.865306    
  # std    1.312408       1.346818     1.502123    
  # min    1.000000       1.000000     1.000000    
  # 25%    2.000000       3.000000     1.000000    
  # 50%    3.000000       4.000000     3.000000    
  # 75%    4.000000       5.000000     4.000000    
  # max    5.000000       5.000000     5.000000
  
  return df_score

def histogram_score(df_score,cond_name,args):

  # print("df_score.shape",df_score.shape)
  # (245, 3)

  columns=df_score.columns
  # print("columns",columns)
  # ['Effectiveness', 'Ease_of_Use', 'Satisfaction']

  # ================================================================================
  for i in range(df_score.shape[1]):
    one_feat_data=np.array(df_score.iloc[:,i])
    plt.subplot(len(columns)/3,3,i+1)
    plt.suptitle(cond_name)
    plt.title(columns[i])
    plt.subplots_adjust(wspace=None,hspace=0.3)
    n,bins,patches=plt.hist(one_feat_data,bins=50)
  plt.show()
  
  # ================================================================================
  # /home/young/Pictures/2019_05_25_17:14:36.png
  # Meaning:
  # - This is score data of "condition type 2 diabetes mellitus"
  # - Effectiveness is almost Normal distribution
  # - Many custormers for this condition feel Metformin-oral is easy to use
  # - Satisfaction shows many low scores
  # - Why low satisfaction even if Metformin-oral is easy to use for this condition?
  # - My personal guess is customers feel Metformin-oral doesn't work for their purpose (condition) even if Metformin-oral is easy to use
  # - So, customers who feel unsatifaction can find other treatment

  # ================================================================================
  # /home/young/Pictures/2019_05_25_17:22:36.png
  # - This is score data of "condition prevention of type 2 diabetes mellitus"
  # - Effectiveness is also Normal distribution
  # - Ease_of_Use and Satisfaction feature also have simialr pattern with the one of "condition type 2 diabetes mellitus"

  # ================================================================================
  # /home/young/Pictures/2019_05_25_17:25:46.png
  # - This is score data of "condition other"
  # - I don't know what other means
  # - Anyway, Effectiveness has more frequency in high scores than above 2 conditions
  # - Ease_of_Use has also high score, which is the confirmed consistent advantage of Metformin-oral so far
  # - In Satisfaction, except for lower frequence in lowest value (1), customer's evaluation is not clear (some is positive, some is negative)

  # ================================================================================
  # /home/young/Pictures/2019_05_25_18:10:01.png
  # - Effectiveness is high for condition of "disease of ovaries with cysts"
  # - Ease_of_Use is also high
  # - Satisfaction is somewhat high

def process_analyze_reviewer_data(one_condition_entire_data,cond_name,args):
  # print("one_condition_entire_data",one_condition_entire_data)
  # [[['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease of Use@@@@@5', 

  one_condition_entire_data_np=np.array(one_condition_entire_data)
  # print("one_condition_entire_data_np",one_condition_entire_data_np.shape)
  # (245, 5, 8)

  one_condition_entire_data_reviewer_np=one_condition_entire_data_np[:,:,3]
  # print("one_condition_entire_data_reviewer_np",one_condition_entire_data_reviewer_np)
  # [['Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)'
  #   'Reviewer: MamaKin, 55-64 Female on Treatment for 6 months to less than 1 year (Caregiver)'
  #   'Reviewer: Judy in TX, 55-64 on Treatment for less than 1 month (Patient)'

  one_condition_entire_data_reviewer_np=one_condition_entire_data_reviewer_np.reshape(-1)
  # print("one_condition_entire_data_reviewer_np",one_condition_entire_data_reviewer_np)
  # ['Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)'
  #  'Reviewer: MamaKin, 55-64 Female on Treatment for 6 months to less than 1 year (Caregiver)'

  data_li=[]
  for one_reviewer in one_condition_entire_data_reviewer_np:
    rep1=one_reviewer.replace("years","1212121212")\
                     .replace("year","1212121212")\
                     .replace("Male","1111111111")\
                     .replace("Female","2222222222")\
                     .replace("Female","2222222222")\
                     .replace(" months to ","-")\
                     .replace(" to ","-")
                     
    

    # regex=r"[a-zA-Z]+[^0-9]"
    regex=re.compile('[a-zA-Z]+[^0-9]')
    test_str=rep1

    aaa=regex.sub('',test_str).replace("(","").split(",")[-1]
    # print("aaa",aaa)

    if aaa!="":
      data_li.append(aaa)

    # print("one_reviewer",one_reviewer)

    # for one_match in matches:
    #   useless_str=one_match.group()
    #   print("useless_str",useless_str)
    #   # Effectiveness@@@@@
    #   # Ease of Use@@@@@
    #   # Satisfaction@@@@@

    #   # ================================================================================
    #   refined=test_str.replace(useless_str,"")
    #   print("refined",refined)

    # print("rep1",rep1)
  
  print("data_li",data_li)
  

  # print("one_condition_entire_data_reviewer_np",one_condition_entire_data_reviewer_np.shape)
  # (245, 5)
  
  

def process_analyze_comment_data(one_condition_entire_data,cond_name,args):
  # print("one_condition_entire_data",one_condition_entire_data)
  # [['@@@@@Page0', ['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease 

  all_pages_data=[]
  for one_page_data in one_condition_entire_data:
    # print("one_page_data",one_page_data)
    # ['@@@@@Page0', ['@@@@@Post0', 'Condition: Type 2 Diabetes Mellitus', '5/24/2019 5:54:59 PM', 'Reviewer: 65-74 Male on Treatment for less than 1 month (Patient)', 'Effectiveness@@@@@1', 'Ease 

    page_num=one_page_data.pop(0)
    all_pages_data.append(one_page_data)
  
  all_pages_data_np=np.stack(all_pages_data,0)
  # print("all_pages_data_np",all_pages_data_np.shape)
  # (245, 5, 8)

  all_pages_comment_data=all_pages_data_np[:,:,-1]
  print("all_pages_comment_data",all_pages_comment_data)
  afaf





  




def analyze_data(args):
  # print("comments_data",comments_data)
  # ['/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_prevention_of_type2_diabetes_mellitus', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_other', 
  #  '/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_disease_of_ovaries_with_cysts']

  for one_cond in comments_data:

    # print("one_cond",one_cond)
    # /mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus

    cond_name=one_cond.split("/")[-1]

    # ================================================================================
    loaded_path=utils_common.get_file_list(one_cond+"/*.pkl")
    # print("loaded_path",loaded_path)
    # ['/mnt/1T-5e7/Companies/Sakary/Sakary_project/Crawling/WebMD_metformin_oral_reviews/Crawled_data/Condition_type_2_diabetes_mellitus/comment_texts_entire_stat.pkl', 
    #  '/mnt/1T-5e7/Companies/

    one_condition_entire_data=[]
    for one_path in loaded_path:
      with open(one_path,'rb') as f:
        mynewlist=pickle.load(f)
        # print("mynewlist",mynewlist)
        # ['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)']

        one_condition_entire_data.append(mynewlist)
    # print("one_condition_entire_data",one_condition_entire_data)
    # [['Effectiveness@@@@@Current Rating: 0@@@@@(3.29)', 'Ease of Use@@@@@Current Rating: 0@@@@@(3.91)', 'Satisfaction@@@@@Current Rating: 0@@@@@(2.90)'], ['@@@@@Page0', ['@@@@@Post0', 'Condition: 

    # ================================================================================
    overall_stat_refined=get_overall_stat(one_condition_entire_data,args)
    # print("overall_stat_refined",overall_stat_refined)
    # ['3.29', '3.91', '2.90']

    # ================================================================================
    # df_score=process_analyze_scores_data(one_condition_entire_data,args)
    # print("df_score",df_score)
    #      Effectiveness  Ease_of_Use  Satisfaction
    # 0    1.0            5.0          1.0         
    # 1    3.0            4.0          1.0         

    # histogram_score(df_score,cond_name,args)

    # ================================================================================
    # df_reviewer=process_analyze_reviewer_data(one_condition_entire_data,cond_name,args)

    # ================================================================================
    df_comment=process_analyze_comment_data(one_condition_entire_data,cond_name,args)
