#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import json
#Import the libraries
import pandas as pd
import re
import emoji
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import contractions
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from transformers import (
   AutoModel,
   AutoConfig,
   AutoTokenizer,
   TFAutoModelForSequenceClassification,
   AdamW,
   glue_convert_examples_to_features
)
import tensorflow as tf
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Choose model
model_name = 'digitalepidemiologylab/covid-twitter-bert' #@param ["digitalepidemiologylab/covid-twitter-bert", "bert-large-uncased", "bert-base-uncased"]

# Initialise tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_prediction(preds, label_mapping, label_name):
    preds = tf.nn.softmax(preds, axis=1)
    formatted_preds = []
    for pred in preds.numpy():
        # convert to Python types and sort
        pred = {label: float(probability) for label, probability in zip(label_mapping.values(), pred)}
        pred = {k: v for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)}
        formatted_preds.append({label_name: list(pred.keys())[0], f'{label_name}_probabilities': pred})
    return formatted_preds

model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    

@app.get("/predict-sentiment")
def rule_check_bert(post: str):
    rule_list = [('error', 'Negative'), ('wrongly', 'Negative'), ('tackle the pandemic', 'Positive'), ('tested positive', 'Negative'), ('reports of suspected adverse effects', 'Negative'), ('encourage', 'Positive'), ('protection in patients was achieved', 'Positive'), ('unusually heavy period', 'Negative'), ('breakthrough', 'Positive'), ('life-threatening', 'Negative'), ('slowest', 'Negative'), ('infected despite sinovac covid-19 vaccinations', 'Negative'), ('encouraging', 'Positive'), ('thankful', 'Positive'), ('warning', 'Negative'), ('failure to stop virus spread ', 'Negative'), ('growing ties', 'Positive'), ('93% vaccine efficacy', 'Positive'), ('“chronological connection” between the vaccination of the patient and the onset of symptoms of the disease', 'Negative'), ('suspended', 'Negative'), ('successfully', 'Positive'), ('contributed', 'Positive'), ('thoughtful', 'Positive'), ('significant milestone', 'Positive'), ('ramp up the pace of vaccinations', 'Positive'), ('recognised as vaccinated', 'Positive'), ('efficacy of their COVID-19 vaccine dropped', 'Negative'), ('easier to distribute', 'Positive'), ('support national vaccination programmes', 'Positive'), ('vaccination drive is in full swing', 'Positive'), ('grateful for their contributions', 'Positive'), ('enhance overall vaccination coverage', 'Positive'), ('has enough to meet its current needs', 'Positive'), ('ramp up vaccination ', 'Positive'), ('conveniently', 'Positive'), ('discovery of contaminants', 'Negative'), ('suffered', 'Negative'), ('late in procuring vaccines', 'Negative'), ('not enough', 'Negative'), ('good progress', 'Positive'), ('increase immunity', 'Positive'), ('painless', 'Positive'), ('win-win arrangement', 'Positive'), ('vaccine is safe', 'Positive'), ('100% per cent success rate', 'Positive'), ('misleading', 'Negative'), ('donating', 'Positive'), ('higher rate of the South African variant', 'Negative'), ('100% per cent success rate', 'Positive'), ('accused', 'Negative'), ('counterfeit versions', 'Negative'), ('remain effective', 'Positive'), ('absurd', 'Negative'), ('heart inflammation', 'Negative'), ('on track', 'Positive'), ('higher neutralising potency', 'Positive'), ('higher concentrations of antibodies', 'Positive'), ('good news', 'Positive'), ('pleased', 'Positive'), ('less effective', 'Negative'), ('able to stop the virus from replicating', 'Positive'), ('high efficacy', 'Positive'), ('disappointed', 'Negative'), ('twice as many neutralising antibodies', 'Positive'), ('optimistic', 'Positive'), ('mistake', 'Negative'), ('bilateral cooperation', 'Positive'), ('high vaccination rate', 'Positive'), ('higher vaccination coverage', 'Positive'), ('situation under control', 'Positive'), ('satisfied', 'Positive'), ('cheer', 'Positive'), ('long-standing friendship','Positive')]
    i = 0
    while (i < len(rule_list)):
        if  f' {rule_list[i][0]} ' in f' {post} ':
            return {"sentiment": rule_list[i][1]}
            #return {"sentiment": "ok"}
        i = i+1         
    tf_batch = tokenizer(post, max_length=128, padding=True, truncation=True, return_tensors='tf')   # we are tokenizing before sending into our trained model
    tf_outputs = model(tf_batch)                                  
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)       # axis=-1, this means that the index that will be returned by argmax will be taken from the *last* axis.
    labels = ['Negative','Positive']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    #return {"sentiment": "ok"}
    return {"sentiment": labels[label[0]]}

#Import cleaned data
data_cleaned = pd.read_excel('cleaned_data.xlsx', sheet_name="Raw data")

def get_data(post, vaccine):
    result = "False"
    msgInPost = str(post).lower()
    vaccineList = vaccine.replace(" ", "").split(",")
    if any(word in msgInPost for word in vaccineList):
        result = "True"
    return result

#Generate DataFrame for each vaccine
vaccines = ["astrazeneca", "sinovac,sinopharm", "moderna" ,"pfizer"]
dataFrameName = {}
for vaccine in vaccines:
    vaccine = vaccine.lower()
    vaccineName = vaccine.replace(",", "_")
    dataFrameName[vaccineName] = data_cleaned.copy()
    post_len = list(dataFrameName[vaccineName].Message.apply(lambda p: get_data(p, vaccine)))
    dataFrameName[vaccineName].loc[:,'Clean'] = post_len
    dataFrameName[vaccineName] = dataFrameName[vaccineName].loc[dataFrameName[vaccineName]['Clean'] == 'True']
    dataFrameName[vaccineName].drop_duplicates(subset = "Message", keep = 'first', inplace = True)
    dataFrameName[vaccineName] = dataFrameName[vaccineName].drop(columns=['Clean'])
locals().update(dataFrameName)

@app.get("/get-data")
def get_data(key):
    if (key == "all"):
        df = data_cleaned
    else:
        df =dataFrameName[key]
    js = df.to_json(orient = 'columns')    
    return js
