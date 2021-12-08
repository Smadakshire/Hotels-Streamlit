# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:50:37 2021

@author: JMada
"""

#importing required packages
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


#Load and read hotel reviews file
reviews=pd.read_csv('hotelReviewsInAustin__en2019100120191005.csv')

#Clean Dataframe
reviews.rename(columns={'hotelName':'Hotel','hotelUrl':'link','review_body':'review'},inplace=True) #change column names

reviews['Hotel']=reviews['Hotel'].str[5:] #Remove extra characters in hotel name
reviews['Hotel']=reviews['Hotel'].str[:-32] #Remove extra characters in hotel name

reviews=reviews.drop(columns=['Unnamed: 0','review_date']) #drop unused columns

reviews['Hotel'].drop_duplicates()

#Create dataframe that has all the reviews per hotel. Column is called all_review (punctuated)
reviews_combined = reviews.sort_values(['Hotel']).groupby('Hotel', sort=False).review.apply(''.join).reset_index(name='all_review')


#Create funtion that summarizes the punctuated all_review data 
def make_summary(text):

  words=word_tokenize(text)


  sentence_list = nltk.sent_tokenize(text)
  stopwords = nltk.corpus.stopwords.words('english')

  word_frequencies = {}
  for word in nltk.word_tokenize(text):
      if word not in stopwords:
          if word not in word_frequencies.keys():
              word_frequencies[word] = 1
          else:
              word_frequencies[word] += 1

  maximum_frequncy = max(word_frequencies.values())

  for word in word_frequencies.keys():
      word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
      
  sentence_scores = {}
  for sent in sentence_list:
      for word in nltk.word_tokenize(sent.lower()):
          if word in word_frequencies.keys():
              if len(sent.split(' ')) < 30:
                  if sent not in sentence_scores.keys():
                      sentence_scores[sent] = word_frequencies[word]
                  else:
                      sentence_scores[sent] += word_frequencies[word]
                      
  summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

  summary = ' '.join(summary_sentences)
  return summary

#Calls the summary function for each all_review
l=[]
for i in range(len(reviews_combined)):
  l.append(make_summary(reviews_combined['all_review'][i]))
  
#Makes the summary into a dataframe called reviews_summary  
reviews_summary=pd.DataFrame(l)
 
#Adds the summary dataframe as a column next to punctuated all_review
reviews_combined['summary']=reviews_summary


#Formats the punctuated all_review to remove punctuations and formatting
reviews_combined['all_review'] = reviews_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

reviews_combined['all_review']= reviews_combined['all_review'].apply(lambda x: lower_case(x))

#Create corpus embeddings for all_review
reviews_sentences = reviews_combined.set_index("all_review")
reviews_sentences.head()

reviews_sentences = reviews_sentences["Hotel"].to_dict()
reviews_sentences_list = list(reviews_sentences.keys())
len(reviews_sentences_list)


reviews_sentences_list = [str(d) for d in tqdm(reviews_sentences_list)]

corpus = reviews_sentences_list

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

#Create pickle file to load into the streamlit app file
with open("corpus_embeddings.pkl" , "wb") as file1:
  pkl.dump(corpus_embeddings,file1)
  
with open("corpus.pkl" , "wb") as file2:
  pkl.dump(corpus,file2)
  
with open("reviews_combined.pkl" , "wb") as file3:
  pkl.dump(reviews_combined,file3)

