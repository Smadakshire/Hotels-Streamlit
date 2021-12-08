# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 00:08:39 2021

@author: JMada
"""
import pickle as pkl
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import pickle as pkl
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import matplotlib.pyplot as plt


with open("corpus_embeddings.pkl" , "rb") as file1:
    corpus_embeddings=pkl.load(file1)
  
with open("corpus.pkl" , "rb") as file2:
    corpus=pkl.load(file2)
  
with open("reviews_combined.pkl" , "rb") as file3:
    reviews_combined=pkl.load(file3)

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title('Austin City Hotel Recommendation System')

st.text('Describe what type of hotel you are looking for below:')

#User input
user_input = st.text_input("Description")

if not user_input:
    st.text('please enter a hotel Description')
#converting input into string
else:
    query =str(user_input)

  # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))
     
    query_embedding = model.encode(query, convert_to_tensor=True)
      
      # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
      
      
    #print("\n\n======================\n\n")
    #print("Query:", query)
    #print("\nTop 5 most similar sentences in corpus:")
      
    for score, idx in zip(top_results[0], top_results[1]):
        st.write('(Score: {:.4f})'.format(score))
         # print("(Score: {:.4f})".format(score))
         # print(corpus[idx], "(Score: {:.4f})".format(score))
        row_dict = reviews_combined.loc[reviews_combined['all_review']== corpus[idx]]
        st.write("Hotel:  " , row_dict['Hotel'].to_string(index=False), "\n")
      #print("Hotel:  " , row_dict['Hotel'] , "\n")
        st.write("Hotel Summary:  " , row_dict['summary'].values, "\n")
          #print("Hotel Summary:  " , row_dict['summary'] , "\n")
      # for idx, distance in results[0:closest_n]:
      #     print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
      #     print("Paragraph:   ", corpus[idx].strip(), "\n" )
      #     row_dict = df.loc[df['all_review']== corpus[idx]]
      #     print("paper_id:  " , row_dict['Hotel'] , "\n")