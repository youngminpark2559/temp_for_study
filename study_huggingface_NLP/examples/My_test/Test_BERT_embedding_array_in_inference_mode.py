# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/huggingface/examples/My_test && \
# rm e.l && python Test_BERT_embedding_array_in_inference_mode.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import torch
import torch.nn as nn
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import re
import glob,natsort
import pickle
import traceback
import pandas as pd

# ================================================================================
def load_checkpoint_file():
    # weights_path="/home/young/Downloads/biobert_v1.1_pubmed/pytorch_BioBERT_model.bin"
    weights_path="/mnt/1T-5e7/mycodehtml/NLP/huggingface/Output/simple_lm_finetuning_medical/pytorch_model.bin"

    state_dict = torch.load(weights_path, map_location='cpu')
    # print("state_dict",state_dict.keys())

    word_embedding_weights=state_dict["bert.embeddings.word_embeddings.weight"]

    return word_embedding_weights

def create_nn_embedding(vocabulary_size,embedding_size):
    embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size)
    return embedding

# def get_closest_to_vector(vector, n=1):

#     nn_indices = self.index.get_nns_by_vector(vector, n)
#     # print("nn_indices",nn_indices)
#     # [67, 18, 787, 332]
    
#     near_vecs=[]
#     for neighbor in nn_indices:
#         word_str=self.index_to_word[neighbor]
#         near_vecs.append(word_str)
#     # print("near_vecs",near_vecs)
#     # ['she', 'he', 'woman', 'never']

#     return near_vecs
    
# ================================================================================
word_embedding_weights=load_checkpoint_file()
# print("word_embedding_weights",word_embedding_weights.shape)
# torch.Size([28996, 768])

nn_word_embedding_arr=create_nn_embedding(vocabulary_size=28996,embedding_size=768)
# print("nn_word_embedding_arr",nn_word_embedding_arr)
# Embedding(28996, 768)

word_diabetes=word_embedding_weights[17973,:]
print("word_diabetes",word_diabetes)
# tensor([-2.2269e-02,  3.1729e-02, -4.0278e-02, -3.0624e-02, -3.8160e-02,
#          5.2429e-02,  2.2211e-03, -9.6097e-04,  1.0720e-02,  3.7002e-02,
#          2.1434e-02, -3.4800e-02, -8.9445e-03, -6.4320e-02, -9.3679e-02,
# tensor([-2.1786e-02,  3.3197e-02, -4.5260e-02, -2.7862e-02, -3.8446e-02,
#          5.5215e-02,  5.8009e-03,  3.3425e-03,  1.2459e-02,  3.9836e-02,
#          1.8082e-02, -3.7331e-02, -8.6600e-03, -6.0633e-02, -9.5151e-02,
word_insulin=word_embedding_weights[26826,:]

# ================================================================================
min_dist = None
result_vec = None
dist_dict={}
for one_word_arr in range(word_embedding_weights.shape[0]):
    distance = np.linalg.norm(word_insulin-word_embedding_weights[one_word_arr,:])
    # print("distance",distance)
    # 1.4241166

    distance = abs(distance)
    
    if (min_dist == None) or (min_dist > distance):
        min_dist = distance
        print("min_dist",min_dist)
        # result_vec = ref_vec
        print("one_word_arr",one_word_arr)

# ================================================================================
# from scipy.spatial.distance import cdist

# aa=cdist(word_embedding_weights, np.atleast_2d(word_insulin)).argmin()
# print("aa",aa)
# afaf

# array([2, 2])

# ================================================================================
# from sklearn.neighbors import NearestNeighbors

# nbrs = NearestNeighbors(n_neighbors=3).fit(word_embedding_weights)
# aaa=nbrs.kneighbors(np.atleast_2d(word_insulin))
# print("aaa",aaa)
# aaa (array([[0.        , 0.91239193, 0.91989911]]), array([[26826, 22222, 28365]]))

# closest_vec = matrix[nbrs.kneighbors(np.atleast_2d(search_vec))[1][0,0]]












# def get_file_list(path):
#   file_list=glob.glob(path)
#   file_list=natsort.natsorted(file_list,reverse=False)
#   return file_list

# def return_path_list_from_txt(txt_file):
#   txt_file=open(txt_file, 'r')
#   read_lines=txt_file.readlines()
#   num_file=int(len(read_lines))
#   txt_file.close()
#   return read_lines,num_file

# path="/mnt/1T-5e7/Companies/Sakary/Management_by_maps/00004_Crawling/Crawl_data_based_on_spreadsheet/diabetes.co.uk/Crawled_data/CSV_Diabetes.co.uk_www.diabetes.co.uk_forum_category_type-2-diabetes.25/*.csv"
# file_list=get_file_list(path)
# # print("file_list",file_list)
# # ['/mnt/1T-5e7/Companies/Sakary/Management_by_maps/00004_Crawling/Crawl_data_based_on_spreadsheet/diabetes.co.uk/Crawled_data/CSV_Diabetes.co.uk_www.diabetes.co.uk_forum_category_type-2-diabetes.25/text_page_1_post_1.csv', 

# # print("file_list",len(file_list))
# # 9973

# text_data_li=[]
# for one_csv in file_list:
#     # print("one_csv",one_csv)
#     # /mnt/1T-5e7/Companies/Sakary/Management_by_maps/00004_Crawling/Crawl_data_based_on_spreadsheet/diabetes.co.uk/Crawled_data/CSV_Diabetes.co.uk_www.diabetes.co.uk_forum_category_type-2-diabetes.25/text_page_1_post_1.csv

#     train=pd.read_csv(one_csv,index_col=False)
#     # print("train",train)
#     # afaf
#     # afaf

#     train_np=np.array(train)
#     # print("train_np",train_np.shape)
#     # (1909, 2)

#     text_data=train_np[1:,1]

#     # print("train",train.shape)
#     # (1909, 2)

#     text_data_li.append(text_data)

# with open("./ffff.txt",'a') as f:
#     for items in text_data_li:
#         f.write("\n\n")
#         for item in items:
#             print("item",item)
#             f.write(item+"\n")

