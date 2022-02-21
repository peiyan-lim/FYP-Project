import json
from IPython.display import display
import pandas as pd
import requests
import plotly
import plotly.express as px
import re
import string
import numpy as np
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime

def chk_sentiment(message):
    reqMessage = message.replace(" ","%20")
    try:
        response = requests.get("http://127.0.0.1:8000/predict-sentiment?post="+reqMessage, timeout=10)
        response.raise_for_status()
        dictRes = response.json()
        sentiment = dictRes["sentiment"]
        return sentiment
    except requests.exceptions.HTTPError as errh:
        print(errh)
        return "error"
    except requests.exceptions.ConnectionError as errc:
        print(errc)
        return "error"
    except requests.exceptions.Timeout as errt:
        print(errt)
        return "error"
    except requests.exceptions.RequestException as err:
        print(err)
        return "error"

def get_data(message="all"):
    try:
        response = requests.get("http://127.0.0.1:8000/get-data?key="+message, timeout=10)
        response.raise_for_status()
        data_json = response.json()
        return data_json
    except requests.exceptions.HTTPError as errh:
        print(errh)
        return "error"
    except requests.exceptions.ConnectionError as errc:
        print(errc)
        return "error"
    except requests.exceptions.Timeout as errt:
        print(errt)
        return "error"
    except requests.exceptions.RequestException as err:
        print(err)
        return "error"

def convert_to_date(timestamp):
    your_dt = datetime.datetime.fromtimestamp(int(timestamp)/1000)  # using the local timezone
    return your_dt.strftime("%Y-%m-%d")

def json_to_df(data_json):
    a_json = json.loads(data_json)
    dataframe = pd.DataFrame.from_dict(a_json, orient="columns")
    dataframe['Post Created Date'] =  dataframe['Post Created Date'].apply(lambda x: convert_to_date(x))
    dataframe['Post Created Date'] =  pd.to_datetime(dataframe['Post Created Date'])
    return dataframe

#Methods for cleaning messages
stop_words = list(set(stopwords.words('english')))
stop_words.extend(['-', '--', 'able', 'access', 'according', 'aci', 'aestheticclinic', 'aestheticdoctor', 
                   'aesthetictreatment', 'africa', 'afternoon', 'ago', 'ah', 'aic', 'also', 'amongst', 'ang', 
                   'app', 'articles', 'associate', 'based', 'batok', 'beautysg', 'become', 'began' , 'c', 'calvin', 
                   'cc', 'cent', 'centre', 'check', 'cheng', 'chinese', 'ci', 'cibukit', 'come', 'coming', 'continue',
                   'count', 'currently', 'daily', 'day', 'day', 'days', 'dermalfillers', 'done', 'dr', 'drganleeping',
                   'drsiew', 'drsiewtuckwah', 'eg', 'either', 'end', 'enough', 'entire', 'even', 'every', 'everyone', 
                   'facialcontouring', 'facialfillers', 'facialinjection', 'facing', 'feel', 'film', 'first', 'followed',
                   'foo', 'fro', 'get', 'getting', 'give', 'given', 'giving', 'go', 'goes', 'going', 'got', 'hafillers', 
                   'hedged', 'hen', 'hk', 'hsa', 'hsien', 'hyaluronicacidfiller', 'ii', 'iii', 'including', 'initial', 
                   'instead', 'ist', 'its', 'iv', 'jamaluddin', 'janil', 'joined', 'keep', 'kenneth', 'lee', 'let', 'li',
                   'lii', 'like', 'line', 'list', 'look', 'loong', 'made', 'mak', 'make', 'man', 'many', 'marks',
                   'may', 'mdm', 'mohamad', 'month', 'move', 'mr', 'ms', 'msexplains', 'na', 'nccs', 'ncss', 'need', 'needs', 
                   'neighbouring', 'new', 'ng', 'note', 'number', 'occur', 'on', 'one', 'one-upped', 'ooi', 'our', 'own', 'people',
                   'per', 'piece', 'pls', 'portfolio', 'put', 'puthucheary', 'radiummed', 'radiummedical', 'rainey', 'rd', 'read', 
                   'ren', 'republic', 'sa', 'said', 'sandra', 'say', 'says', 'sector', 'see', 'sgbeauty', 'share', 'showing', 'side',
                   'siew', 'skinboosters', 'st', 'start', 'state', 'still', 'stored', 'take', 'talking', 'task', 'teenager','telegram',
                   'the', 'though', 'took', 'twitter', 'us', 'use', 'used', 'uses', 'using', 'versions', 'very', 'via', 'video', 
                   'want', 'watch', 'weeks', 'well', 'wife', 'within','word', 'would', 'year', 'york', 'zaqy', 'x'])

extra_words = ['vaccine', 'vaccines', 'vaccinated', 'vaccination', 'vaccinations', 'singapore', 'covid-19', 
               'astrazeneca', 'sinovac', 'sinopharm', 'biontech', 'pfizer', 'moderna', 'pfizer-biontech', 'doses', 
               'dose', 'health', 'coronavirus']

#Replace contractions with words
def replace_contractions(text):
    text = re.sub(r"\'s", " is", text)
    return contractions.fix(text)

#Remove all punctuations except '-'' from text
def remove_punctuation(text):
    remove = string.punctuation.replace("-", "") #don't remove hyphens
    pattern = r"[{}]".format(remove) #create the pattern
    return re.sub(pattern, "", text) 

#To convert messages into a list of tokens
def tokenize_word(message, cleanWordCloud):
    msg = str(np.char.lower(message)) #convert message to lowercase
    msg = replace_contractions(msg) #replace contractions
    msg = remove_punctuation(msg) #remove punctuation 
    tokens = word_tokenize(msg) #Tokenize word
    clean_tokens = []
    wordList = []
    wordList.extend(stop_words)
    if(cleanWordCloud == True):
        wordList.extend(extra_words)
    for w in tokens:
        if w not in wordList:
            hypen = False
            pattern1 = re.compile("[a-zA-Z0-9-/](?=\-)")
            pattern2 = re.compile("(?<=\-)[a-zA-Z0-9-/]")
            result1 = pattern1.search(w)
            result2 = pattern2.search(w)
            if (result1!=None and result2!=None):
                hypen = True
            if w.isalpha() or hypen:
                clean_tokens.append(w)
    return clean_tokens

def get_wordcloud(df, senti, dataName):
    sentidf = df[df["ImproveCTBERTSentiment"]==senti]
    if senti == "Positive":
        color = "Greens"
        extension = dataName+"_pos_wc.png"
    elif senti == "Negative":
        color = "Reds"
        extension = dataName+"_neg_wc.png"
    clean_msgs = tokenize_word(sentidf.Message.str.cat(sep=' '), False)
    tfIdfVectorizer = TfidfVectorizer(use_idf=True,token_pattern =r'\S+',  stop_words='english')
    tfIdf = tfIdfVectorizer.fit_transform(clean_msgs)
    dense = tfIdf.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=tfIdfVectorizer.get_feature_names())
    wordcloud = WordCloud(background_color="white", colormap=color, max_words=100).generate_from_frequencies(df.T.sum(axis=1))
    wordcloud .to_file("./static/"+extension)

def get_scatter(df):
    graphdf = (df.groupby(['Post Created Date', 'ImproveCTBERTSentiment']).size() 
            .sort_values(ascending=False) 
            .reset_index(name='No of Posts'))
    fig = px.scatter(graphdf, x="Post Created Date", y="No of Posts", color="ImproveCTBERTSentiment")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def top_3(df, senti):
    tabledf = df[df["ImproveCTBERTSentiment"]==senti]
    tabledf = tabledf.loc[:,['Message','Likes']]
    dftop3 = tabledf.nlargest(3,['Likes'])
    dftop3 =dftop3.reset_index().drop(columns=['index'])
    dftop3.index = dftop3.index + 1
    return dftop3.to_html(header="true", table_id="table")

def get_pie(df):
    piedf = (df.groupby(['ImproveCTBERTSentiment']).size() 
   .sort_values(ascending=False) 
   .reset_index(name='No of Posts'))
    fig = px.pie(piedf, values='No of Posts', names='ImproveCTBERTSentiment', title='Posts by Sentiment')   
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

